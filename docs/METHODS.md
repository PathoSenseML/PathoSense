# Methods

How the deployed result is produced, end to end. Code map: `pipelines/README.md`.
Evaluation harness: `analysis/harness.py`.

## 1. The constraints

Binary classification of missense variants (pathogenic = 1, benign = 0) across four
gene panels, scored by F1 on the pathogenic class and averaged over panels.

Two facts drive every design decision.

*The test prevalence is inverted.* Training data is 70-83% pathogenic; the test set is
mostly benign, at per-panel pathogenic priors of 0.143 (general), 0.167 (cancer),
0.286 (pah) and 0.167 (cftr). These live in `utils.TEST_PRIORS`. Thresholds are chosen
at those priors, never at 0.5.

*The features are de-identified.* `AA_1`/`AA_2` are the reference and alternate amino
acid; `EK_*` are conservation scores (phyloP, phastCons, GERP); `AL_*` are roughly 300
transformed allele-frequency columns; `CAT_*` are population and region flags. Gene,
position, coordinate, sequence context and in-silico scores are all absent. Row-level
re-identification against dbNSFP and gnomAD fails: 0 of 280 PAH variants matched even
when restricted to the correct gene, because conservation values drift between dbNSFP
releases and the allele frequencies have been transformed below the 1/AN floor. This
rules out protein language models and any external per-variant lookup.

## 2. Data preparation

`schema.py` normalizes sentinel missing-codes to NaN and drops all-NaN rows.

`feature_engineering.py` derives the substitution chemistry the anonymization removed,
from `AA_1 -> AA_2`: BLOSUM62, Grantham distance, and deltas in hydrophobicity,
polarity, molecular weight and volume (Kyte-Doolittle and Zimmerman scales). Per-panel
extras are gated by `recipe.py`: ACMG flags for pah, amino-acid by conservation
interactions, score agreement.

`utils.detect_leaky_columns` drops a column if its single-feature AUC exceeds 0.99, if
it is constant, or if its missingness is too pure to be chance under the panel base
rate. The last test is binomial rather than a `nunique == 1` check, so a sparse column
that happens to be single-class is not thrown away.

`preprocess_v2.fit_keep_nan` is the general recipe: drop constant and sparse columns,
rank-transform the frequency columns, clip conservation at the 1st and 99th
percentiles, add missing-indicators for columns between 5% and 95% NaN, then drop
correlations above 0.95 using a NaN-aware correlation. Numeric NaN survives all of it.
The small panels use median imputation plus mRMR instead (`models.preprocess_data`
with `fast_mrmr`), because with a few hundred rows the selection matters more than the
missingness.

### Why NaN is kept

Missingness here is structural. A population's allele-frequency column is absent when
that cohort was not measured, which correlates with rarity, which correlates with
pathogenicity. Trees route NaN down a learned branch and use it.

Head-to-head on the general panel, relative to keeping NaN: median -5.4σ, KNN -6.1σ,
MICE -7.9σ, and imputation with explicit missingness indicators still -4σ. Scaling and
power transforms are inert for trees. UMAP and PCA reduction lose 22σ.

## 3. Models

### 3.1 Cross-panel transfer (`transfer.py`)

The small panels are data-starved. A model trained on `general` scores their variants
better than their own models do.

The leakage control matters: 63-69% of a small panel's variants also appear in
`general`, so the scorer for a panel is trained on `general` with all of that panel's
`Variant_ID`s excluded. Under that control the scorer AUCs are cftr 0.96, cancer 0.94,
pah 0.80, each at or above the panel's own model. For cftr and cancer the scorer is
itself a LightGBM + RF + CatBoost blend; for pah it is a single LightGBM.

Raw `AL_` values do not transfer. The organizers re-extracted allele frequencies per
panel, so the same variant can have correlation 0.43 (cancer), 0.014 (pah) or 0.017
(cftr) between its panel value and its general value. `EK_` conservation is stable at
about 0.98 and is used directly. Cancer therefore drops the `AL_` values and keeps only
the missingness family-flags, which are panel-independent. Pah keeps the values but
re-ranks them on its own training distribution first.

### 3.2 Per-panel deployment (`predict.py`, weights in `transfer.TRANSFER_PANELS`)

`probs = w * transfer + (1 - w) * own`.

- **general** (the transfer source, no `w`): native NaN, all features, no mRMR. The
  deployed mRMR-60 was over-pruning the 334-column `AL_` block. LightGBM with
  `scale_pos_weight=1` and bagging active, decorrelated against RF, CatBoost and
  XGBoost in an equal-weight blend; RF and CatBoost are Optuna-tuned. The LightGBM
  member alone drops 510 columns to 243 by removing near-duplicates above 0.99
  correlation; the other members keep the full matrix.
- **cancer** (`w = 0.5`): half transfer, half own median+mRMR LightGBM.
- **pah** and **cftr** (`w = 1.0`): transfer only, no own model.

### 3.3 Calibration and threshold

Each probability stream is Platt-scaled on its out-of-sample scores
(`calibration.py`). The threshold is F1-optimal at the panel's test prior
(`utils.optimize_threshold_f1_balanced`, bootstrapped to that prior).

The per-draw selector can overshoot on a single full-train fit, so where per-fold CV
shows a fixed threshold doing better, that value is pinned in `THRESHOLD_OVERRIDE`:
general 0.81, cancer 0.80, pah 0.82, cftr 0.90.

## 4. Evaluation

`analysis/harness.py` computes F1 at the test prior analytically from per-fold TPR and
FPR, under paired `RepeatedStratifiedKFold`, fitting preprocessing, selection,
calibration and threshold inside each fold.

The earlier single-split numbers were inflated because feature selection was fit on the
rows it then scored. That is worth stating plainly: it put the mean F1* near 0.64 when
the honest figure was 0.558, and the correction is what made cross-panel transfer
visible.

A change is accepted only if it clears 2σ on paired folds, holds across seeds, and can
be justified from training data alone. That bar has rejected own-model swaps, blend
reweighting, imputation, dimensionality reduction, transformer and masked-autoencoder
models, TabPFN, and protein-LM embeddings. `tests/test_batch_invariance.py` checks that
a row's probability does not depend on what else is in its batch.

## 5. Where the ceiling is

Cross-validated F1 at the test prior: general 0.537, cancer 0.725, pah 0.669,
cftr 0.851. Transfer is what carries the small panels.

The general panel is limited by its features rather than by the model. AUC sits at
about 0.86, and F1 = 0.70 at a prior of 0.143 would need roughly 0.94. Model families,
hyperparameters, blends, preprocessing, dimensionality reduction, neural models and
joint optimization all move AUC by approximately zero. Positional and in-silico signal
would move it, and the anonymization removed both.

## 6. Reproducing

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
.venv/bin/python pipelines/predict.py train --panel all
.venv/bin/python pipelines/predict.py infer --panel cftr --input test.csv --output preds.csv
.venv/bin/python tests/test_batch_invariance.py
```

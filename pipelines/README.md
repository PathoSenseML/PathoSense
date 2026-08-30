# `pipelines/` — code map

`predict.py` is the deployed model, for both training and inference. Everything else
here is a module it composes. Evaluation lives in `analysis/harness.py`.
`run_pipeline.py` is the old single-split evaluation; it stays because the analysis
scripts import shared helpers through it, but its metrics are inflated and should not
be quoted.

```
predict.py            train (fit and save four models) / infer (score a CSV)
├── schema.py         column partition (AA_/EK_/AL_/CAT_), sentinels, row drops
├── feature_engineering.py   BLOSUM62, Grantham, biochemical deltas, ACMG, interactions
├── preprocess_v2.py  native-NaN preprocessing, AL_ family features, fast_mrmr
├── models.py         preprocess_data (median + mRMR), model params, multi-seed LGBM
├── general_blend.py  RF + CatBoost + XGBoost blend for general; shared CatBoost
├── transfer.py       cross-panel transfer scorer and per-panel weights
├── calibration.py    Platt scaling on out-of-sample scores
├── utils.py          leak detection, threshold-at-prior, metrics, TEST_PRIORS
├── recipe.py         per-panel preprocessing and feature flags
└── config.py         seed, output directories
```

## Training (`predict.py train --panel <p>`)

1. Load the panel CSV, normalize sentinels to NaN, drop all-NaN rows.
2. Engineer amino-acid features, plus whatever `recipe.PANEL_RECIPE` enables for that
   panel (ACMG flags, interactions, score agreement).
3. Drop leaky columns: single-feature AUC above 0.99, constant, or missingness too
   pure for the panel base rate, union `KNOWN_LEAKY`.
4. Fit the panel's own model, if its transfer weight is below 1. General uses native
   NaN with all features; the small panels use median plus mRMR. Five-seed LightGBM
   with nested early stopping, then Platt calibration. General additionally prunes
   columns correlated above 0.99 off the LightGBM only, and blends RF, CatBoost and
   XGBoost on top.
5. Fit the transfer scorer, if the panel has a weight. `fit_transfer_scorer` trains on
   general with the target panel's `Variant_ID`s removed, `transfer_score` projects the
   target rows out of sample, then Platt calibration.
6. Blend: `probs = w * transfer + (1 - w) * own`.
7. Pick the threshold at the panel's test prior, or use a validated
   `THRESHOLD_OVERRIDE`.
8. Save everything to `models/<panel>/artifacts.joblib`.

`infer` replays the frozen artifacts, so a row scores the same alone as in a batch.
`tests/test_batch_invariance.py` locks that. Training is seeded.

## Deployed configuration

| panel | transfer w | own model | scorer | threshold | test prior |
|---|---|---|---|---|---|
| general | (source) | native NaN, all features, LGBM + RF + CatBoost + XGB | | 0.81 | 0.143 |
| cancer | 0.5 | median + mRMR LightGBM | LGBM + RF + CatBoost, AL_ as family-flags | 0.80 | 0.167 |
| pah | 1.0 | none | single LightGBM, target-ranked AL_ | 0.82 | 0.286 |
| cftr | 1.0 | none | LGBM + RF + CatBoost | 0.90 | 0.167 |

`transfer.TRANSFER_PANELS` is the only place these weights are defined. Cross-validated
F1 at the test prior: general 0.537, cancer 0.725, pah 0.669, cftr 0.851. The reasoning
is in `docs/METHODS.md`.

# `analysis/` — harness and experiments

`pipelines/` is the deployed system. This directory is how we decided what went into
it: the cross-validation harness, and one script per idea we tried.

## Core

| file | what it does |
|---|---|
| `harness.py` | F1 at the test prior, computed per fold, paired CV, reports Δ and σ |
| `meeting_lib.py`, `exp_transfer.py`, `exp_conditional_threshold.py` | shared per-fold pipeline replications that many experiments import |
| `explainability.py`, `make_shap_figure.py` | SHAP and group permutation importance |

## Experiments by theme

- Cross-panel transfer: `exp_transfer*.py`, covering the leakage control, the blend
  members, and how `AL_` is handled per panel.
- General-panel model search: `exp_fair_gbdt.py`, `exp_proper_bakeoff.py`,
  `exp_nontree_fair.py`, `exp_blend_*.py`, `exp_joint_opt.py`.
- Re-auditing the small panels' own models: `exp_own_model_bakeoff.py`, then
  `exp_{pah,cancer}_own_confirm.py`, `exp_cancer_own_power.py`, `exp_recalibrate_all.py`.
- Preprocessing: `exp_teacher_preprocessing.py`, `exp_teacher_fair.py`,
  `exp_teacher_fair_blend.py`.
- Thresholds: `exp_threshold_check_all.py`, `exp_threshold_recalibrate.py`,
  `exp_*_threshold_*.py`.
- Where the ceiling is: `exp_step0_headroom.py` through `exp_step3_learning_curve.py`,
  `exp_addendum_checks.py`.
- Correctness: `exp_audit_fix_validate.py`, `exp_audit_es_nested.py`, and
  `../tests/test_batch_invariance.py`.
- Model classes that did not work: `exp_bidirectional_transformer.py`,
  `exp_mae_autoencoder.py`, `exp_tabpfn_*.py`, `exp_nontree_bakeoff.py`.

## `studies/`

The experiments above answer one question each and live flat in this directory. The
longer campaigns, which ran over many scripts and several days, are grouped by topic
under `studies/`:

| directory | what it covers |
|---|---|
| `amino_acid_features/` | biochemical and BLOSUM62 substitution feature ablations |
| `allele_freq_ensembles/` | `AL_` log transforms, PCA and model-family ensembles |
| `baseline_model_search/` | the first fair comparison of model families on the general panel |
| `categorical_features/` | `CAT_*` row filters, encodings and redundancy |
| `categorical_rework/` | rebuilding the `CAT_new` datasets and rescanning the cancer blend |
| `candidate_selection/` | the final-candidate ladder: screens, smoke tests, weight microsearches |
| `transfer_suppressor/` | cross-ensemble transfer and false-positive suppression |
| `threshold_and_weights/` | no-refit threshold, calibration and ensemble-weight audits |
| `cancer_threshold_grid/` | cancer weight/threshold grid and calibration comparison |
| `preprocessing_audit/` | leakage audit, duplicate validation, historical metric alignment |
| `early_feature_selection/` | the first EDA and LightGBM feature-selection pass |
| `correlation_pruning/` | correlation-0.95 column pruning runs |
| `error_analysis/` | false-positive/false-negative and fold-stability analysis |
| `lightgbm_packaging/` | build and inference scripts for a standalone model package |

## Convention

Each `exp_*.py` opens with its hypothesis, protocol and run command, and prints a
paired Δ and σ against the deployed baseline. A change ships at 2σ or better, stable
across seeds, justified without looking at test data.

Most of these scripts record a failure. They are kept anyway. Reading them is the
fastest way to see that the general panel is limited by its features rather than by
anything the model does, and it saves the next person from re-running them.

Experiments read the panels from `data/panels/` and write their result tables to
`analysis/output/`. Neither directory is in the repository: the competition data is not
ours to redistribute, and the result tables are produced by running the scripts.

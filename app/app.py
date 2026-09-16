"""PathoSense — interactive demo and explainability interface.

Run from the repository root (models must exist in models/, see README):
    .venv/bin/python -m streamlit run app/app.py

This is a presentation layer over pipelines/predict.py and pipelines/explain.py. It never
retrains anything and never touches the scored path: every probability shown here is the one
`predict.py infer` would write.
"""
from __future__ import annotations
import os
import sys
import io
import html
import json
import re
from urllib.parse import quote
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipelines"))
import warnings; warnings.filterwarnings("ignore")  # noqa: E702

from predict import load_artifacts, score_frame                        # noqa: E402
from explain import (explain_frame, top_contributions, feature_value, global_importance,   # noqa: E402
                     family_importance, family_of, readable, af_columns, confidence_flags,
                     f1_at_prior, fpr_for_f1, FAMILY_LABELS, FLAG_KEYS)
from schema import normalize_sentinels, drop_all_nan_rows               # noqa: E402
from utils import PANELS, TEST_PRIORS                                   # noqa: E402

# Canonical CV means: analysis/output/PDR_CANONICAL_RESULTS.csv + panel_cm_rates.csv (2026-09-07).
# Artifact f1_estimate is full-train OOF, NOT a replacement for these CV results.
CANON = {
    "general": dict(train=0.7367673179396093, F1=0.537, SE=0.009, MCC=0.460, AUC=0.864, PR=0.507, TPR=0.7353, FPR=0.1714, thr=0.81),
    "cancer":  dict(train=0.6883116883116883, F1=0.725, SE=0.007, MCC=0.671, AUC=0.940, PR=0.772, TPR=0.8216, FPR=0.0953, thr=0.80),
    "pah":     dict(train=0.8365122615803815, F1=0.669, SE=0.007, MCC=0.517, AUC=0.802, PR=0.615, TPR=0.8325, FPR=0.2801, thr=0.82),
    "cftr":    dict(train=0.8108108108108109, F1=0.851, SE=0.009, MCC=0.830, AUC=0.965, PR=0.879, TPR=0.8403, FPR=0.0407, thr=0.90),
}
PANEL_ORDER = ["general", "cancer", "pah", "cftr"]
# Overview's prevalence band is ONE control: a continuous prevalence whose track carries the three known
# populations as named marks. Prevalence is a property of the set being screened, and not one of the three
# marks is a value anybody here chose — the contest hands us its test set (the default, so every screenshot
# opens there), the curated mark is this panel's own training data (the table's train column), and the
# screening scenario is the low-prevalence end every variant classifier is weakest at. What makes a continuum
# safe to drag along this axis is beside it and not in a caption: recall is labelled fixed and does not move,
# and the per-1000 picture visibly worsens wherever F1 improves. Never a bare percentage: the marks are what
# say where the meaningful points are, and a track with no maximum worth reaching is not a knob to turn up.
COHORTS = ("screen", "test", "curated")
SCREEN_PRIOR = 0.01   # 10 pathogenic per 1000 screened: a scenario, stated as a composition and not as an epidemiological claim
PREV_MIN, PREV_MAX, PREV_STEP = 0.01, 0.90, 0.005   # 0.5% is ~2.6px of track at 1366: it drags as a continuum, and every stop is a set the curve can be read at
PREV_DOMAIN = (0.0, 0.9)   # the curve's x domain, written once: the live drag maps a prevalence to a pixel through the same two numbers the axis is drawn from
PK_CELLS = ("tp", "fn", "fp", "tn")   # the confusion matrix in the order the per-1000 picture reads it: found, missed, wrongly flagged, correctly left
SAMPLE_N, SAMPLE_SEED = 300, 42   # the demo sample: fixed and deterministic, so the same rows are shown every run
NARROW = 1100   # the one breakpoint: below it the paired columns stack (CSS @media, the subject chart, the input list)
TICK_FLIP = 0.85   # above this threshold the flow scale's tick label is drawn left of its rule, or it overhangs the scale
# Compact chart heights, in design pixels; H() sizes their minimum layout. Taller viewports
# flex the main chart bands instead. A bar chart is its rows times a pitch,
# never tighter than MIN_BAND (two 12px labels have to stay apart at the type floor),
# and a band chart is sized to the band it fills. The two ranked charts — Overview's top ten and the
# review page's family totals — take MIN_BAND as their pitch: they had a RANK_BAND of their own, and
# H() shrank it below the floor at every viewport this app is specified for, so the floor was already
# what set both heights and the second constant was a lever the code believed it was pulling.
MIN_BAND = 20
GAUGE_BAND, GAUGE_AXIS = 44, 40   # one score bar and its gap; the shared 0-1 axis drawn once under them
BAND = 220                        # a chart level with the controls beside it (Overview's prevalence curves)
BAND_TALL = 250                   # a chart that owns its row (Predictions' histogram)
SUBJECT, SUBJECT_NARROW = 560, 360   # Variant review's contribution chart: the page's subject, wide and stacked
                                     # (560: the family chart above it took 90 of its 650 so the page stays under the fold)
BORDER_MARGIN = 0.05              # half-width of the borderline window around a panel's threshold, passed into
                                  # confidence_flags rather than left to its default: the band the histogram
                                  # draws, the ◐ column, the legend and the filter's count are then one rule
                                  # read four ways, and cannot drift apart
CONTRIB_BARS = 10                 # ranked drivers the contribution chart draws (plus a tail bar), and the
                                  # same drivers the Variant review input list carries with their values
CHEM_BARS = 4                     # ranked substitutions the batch-character block draws under the histogram
LABEL_LIMIT = 430                 # widest axis label any chart draws before Vega truncates it (styled())
# One palette: the CSS block and the Vega charts read the same names, so a colour is written once.
INK, MUTED, LINE, SURF, GRID = "#223042", "#526172", "#D8E0E8", "#F8FAFC", "#E8EDF2"
# The brand teal of the team's wordmark (measured off the title card, 6.1:1 on white). It is the colour of the
# interface's own state and structure — the wordmark, the selected segment, the active page, the marked row,
# focus, the flow diagram's strokes — and is written once more where a Python name cannot reach: as the fill
# of app/logo.svg and as primaryColor in .streamlit/config.toml (the framework's slider, chips and focus
# rings), which the tests hold equal to this value. It is never a model call: RED and BLUE below are the two
# classes on every bar, gauge, tag and histogram, and no chart mark, swatch or cell is ever drawn in BRAND.
BRAND = "#0F6E66"
RED, BLUE, GREEN, GREY, FAINT = "#C0392B", "#2471A3", "#16A085", "#7F8C8D", "#B3B6B7"
# The agreement mark's three tones (truth_label): a traffic light, read as trust and not as class. The word
# beside them is ink, so the cell has no other colour in it and the failure tone can be the one everybody
# already reads as failure — RED itself, the call colour, rather than a second red a few degrees away.
# The trap in the other two is that the glyph is knocked out of the chip: what shows through it is the page,
# which is white, so the contrast that decides whether the mark is legible is the fill against white — not
# the fill against the cell around it. A bright yellow chip is a near-white glyph on a near-white fill and
# vanishes. Yellow here is therefore a deep gold, fully saturated at hue 44 and 3.85:1 on white: still
# yellow at 16px, and the lightest of the three, which is also what tells it from the green in greyscale
# (relative luminance 0.222 against 0.152). The green stays dark at 5.19:1 for that same reason: green and
# gold are the pair that share a glyph, so tone is the only thing standing between them.
TRUST_OK, TRUST_WARN, TRUST_FAIL = "#3F7A34", "#A87A00", RED
# The wider of the panel's two class words at --fs/600, per language. The truth cell reserves exactly this
# much for the word so that every agreement mark lands on one x — the column still reads straight down for
# the model's mistakes — one word's length from the word instead of at the far edge of the column.
TRUTH_W = {"tr": "4.375rem", "en": "5.125rem"}   # Patojenik 54.7px, Pathogenic 64.9px, measured at 13px root
FAMILY_COLORS = {"ek": "#E67E22", "al": BLUE, "amino_acid": GREEN,
                 "cat": "#8E44AD", "missingness": GREY, "other": FAINT}
# Prevalence curves use the pooled cross-validated recall (TPR) and an FPR back-solved so that the curve
# passes through the canonical F1 at the test prevalence: the canonical value is a mean of per-fold F1,
# which the F1 of pooled confusion counts does not reproduce on the small panels (cftr 0.822 vs 0.851).
for _p, _c in CANON.items():
    _c["FPR_anchor"] = fpr_for_f1(_c["TPR"], TEST_PRIORS[_p], _c["F1"])

# ── strings ───────────────────────────────────────────────────────────────────────────────────
S = {
"en": {
    "panel": "Gene panel",
    "panel_general": "General (MASTER)", "panel_cancer": "Hereditary cancer",
    "panel_pah": "PAH (phenylketonuria)", "panel_cftr": "CFTR (cystic fibrosis)",
    "upload_btn": 'Choose file', "upload_limit": 'Up to 200 MB', "upload_title": 'Upload CSV',
    "row_id": 'row {i}',
    "thr_short": 'threshold',
    "no_models": "No trained model for this panel. From the repository root run:", "threshold": "Decision threshold", "prior_test": "Test prevalence",
    "prior_train": "Training n · prevalence", "recall": "Recall",
    "scoring": "Scoring {n} variants with the {panel} model…",
    "tab_pred": "Predictions", "tab_explain": 'Variant review',

    "n_path": "Called pathogenic",
    "expected_rate": 'Expected call rate at test prevalence ({p}): {r}. An extrapolation, not a measurement.',
    "hist_title": "Model score distribution",
    "card_labelled": "{n} labelled",   # the batch's own composition, on the card whose count it is misread against
    # The four verdicts of the truth column, as the card that is that column's key (verdict_card). The
    # subject stays the review warning: the columns are the warning, the marks are the agreement.
    "quad_title": "Warning and label agreement",
    "quad_warned": "Warned", "quad_clean": "No warning",
    "v_ok_clean": "Correct, no warning", "v_ok_warn": "Correct, warned",
    "v_bad_warn": "Wrong, warned", "v_bad_clean": "Wrong, nothing warned",
    "quad_note": "Columns: whether a review flag fired. Marks: whether the call matches the file's own Label column. The four cells are all {n} rows.",
    # Which warnings the batch actually raised, in the flag vocabulary and with no trust mark anywhere —
    # the middle card on both paths, and the only place the breakdown is drawn (flag_quad_card).
    "flag_quad_title": "Warning types",
    "flag_quad_note": "Counts overlap; {k} of {n} rows raised at least one.",
    "silent_note": "Counted against the file's own Label column; not a measure of the model.",
    "labelled_note": "The file's own Label column; not an output of the model.",
    "hist_y": "Variants", "variant": "Variant", "score": "Model score",
    "chem_central": 'median {parts}',   # the qualifier of the batch-character heading, whose subject is flow_chem
    "col_aa": "Change",   # the substitution as a table column; the review page's own row says it in full (aa_change)
    "col_flags": "Review flags", "col_label": "True label",
    "pathogenic": "Pathogenic", "benign": "Benign",
    # The one file the table leaves as. The label names the file and the values it carries the call in;
    # the rest of the columns are the help, which is also where the reader learns the file is more than
    # the two confirmed columns it leads with.
    "export_calls": "Predictions (0/1)", "export_cols": "Variant_ID · prediction 0/1 · probability · flag columns",
    "export_label": " · label",
    "flag_ek7_missing": "EK_7 missing", "flag_complex_aa": "non-standard AA code",
    "flag_cat6_flag": "CAT_6 present", "flag_borderline": "borderline",
    # The trust key (truth_label). It names the tones in words because a tooltip is text the framework
    # renders and the app's marks are drawn; the legend under the table has no room for it (table_foot).
    "trust_key": "Check: the call matches the label. Cross: it does not."
                 " Green: unflagged. Gold: a review flag fired. Red: wrong, and nothing warned.",

    "eval_f1_prior": 'F1 (test prevalence {p})',
    "eval_auc": "ROC AUC", "eval_mcc": "MCC",
    "f1_prior_withheld": 'F1 at test prevalence withheld: at least 30 rows in each class are needed ({p} pathogenic / {b} benign).',
    "aa_change": "Amino-acid change",
    "verdict_margin": "Relative to threshold", "margin_above": '{d} above', "margin_below": '{d} below',
    "margin_at": 'At threshold', "margin_tiny_above": 'Less than 0.001 above', "margin_tiny_below": 'Less than 0.001 below',
    "member_own": "Panel's own model", "member_transfer": "Transfer scorer", "weight": "weight",
    "contrib_title": 'Features that moved the LightGBM score',
    "contrib_help": 'TreeSHAP ({members}), seed-averaged and member-weighted; not a decomposition of the calibrated or blended probability.'
                    ' Column names are anonymised by the competition.',

    "bad_csv": 'The file could not be scored.',
    "low_coverage": "Feature coverage is only {pct}; these scores are not reliable.",
    "missing": "missing", "yes": "yes", "no": "no",
    "contrib_x": "Contribution (log-odds)", "fam_x": "Family total (log-odds)", "value": "Input value", "feature_id": "Feature ID",
    "flagtext_ek7_missing": "EK_7 is missing; on such rows discrimination fell from AUC 0.86 to 0.67 (general panel).",
    "flagtext_complex_aa": "Multi-residue or non-standard amino-acid code: the substitution chemistry features are undefined.",
    "flagtext_cat6_flag": "The row carries a CAT_6 value; the model measurably loses discrimination on such rows.",
    "flagtext_borderline": "Within 0.05 of the threshold.",
    "global_fam": 'Feature families', "global_top": "Top 10 features (LightGBM gain)",
    "share": "Share", "family": "Family",
    "op_cohort": "The set being screened",
    "coh_screen": "Broad screen", "coh_test": "Contest test set", "coh_curated": "Curated set",
    "coh_help": 'The axis changes the composition of the set being screened; the model and its {thr:.2f} '
                'threshold are the same at every point. The three marks: a screening scenario (10 pathogenic '
                'per 1000), the contest test set for this panel ({test}), and this panel\'s own training data '
                '({train} pathogenic). Every point between them is arithmetic, not a measurement.',
    "op_chart": 'Expected F1 for the same model in different sets',
    "op_extrap": "extrapolation",
    "op_x": "Prevalence of pathogenic variants", "op_panel": "Panel",
    "op_prec": "Precision",
    "op_prec_help": 'The share of the variants called pathogenic that really are pathogenic (positive predictive value).',
    "op_fixed": "fixed",   # the chip that stops the prevalence control being read as a threshold
    "op_fixed_help": 'Recall measured in cross-validation at the {thr:.2f} threshold. The same number in every set.',
    "op_marker": "test prevalence",
    "col_prec": "Precision",   # the same quantity as op_prec, as a table column: the mean's other term, beside recall
    "single_input": 'single input',
    "results_title": 'Cross-validated results',
    "chip_test_prior": 'at test prevalence',   # claim 6, drawn as the chip beside that heading
    "results_help": 'Binary F1, positive class pathogenic: 3 seeds, fold mean ± standard error, every step refitted inside every fold. Not a clinical validation.',
    "g_ek": "Anonymised EK_ columns and the features read off them.",
    "g_al": "Anonymised AL_ columns and the features read off them.",
    "g_amino_acid": "Grantham, BLOSUM62; volume, charge, hydrophobicity differences.",
    "g_cat": "Anonymised categorical CAT_ columns.",
    "g_missingness": "Indicators of unobserved input values.",
    "g_other": "Mean and spread of scaled features (score agreement).",
    "tab_overview": 'Overview',
    "remove": 'Remove',
    "filter": 'Filter',
    "wrong": 'wrong call',
    "short_general": 'General',
    "short_cancer": 'Cancer',
    "short_pah": 'PAH',
    "short_cftr": 'CFTR',
    "waiting": 'CSV input requirements',
    "empty_csv": 'The file could not be scored: no variant rows. Remove this file and upload a CSV containing at least one variant row.',
    "missing_columns": 'The file could not be scored: {miss} of {total} model inputs are missing, {n} of them required: {names}',
    "id_optional": 'Optional; falls back to row ID',
    "label_optional": 'Optional: 0 = Benign, 1 = Pathogenic. Compared only when every row has one of these values.',
    "column_group": 'Column group',
    "count": 'Count',
    "example": 'Example',
    "sample_title": 'Training-set sample',   # the fixed demo sample: SAMPLE_N rows; the seed is in the heading's tooltip
    "sample_seed": 'Fixed sample, seed {seed}.',
    "training_batch": 'Every row here is training data for this model; batch metrics would not measure performance.',
    "training_batch_part": 'The panel’s own model — the {o} weighted share of the score — was fitted on these rows; batch metrics would not measure performance.',
    "transfer_caveat": 'The threshold and calibration{rank} were fitted on these rows, so F1 and MCC here are optimistic, not held-out validation.',
    "transfer_rank": ' (and the AL_ rank transform)',
    "tag_label_ignored": 'Label column ignored',
    "label_ignored": 'Label values must be 0 or 1; this file has “{v}”.',
    "eval_title": 'Label comparison',
    "eval_not_validation": 'not validation',   # claim 1 on this heading, as a chip
    "eval_not_heldout": 'not held-out',        # claim 3 on the transfer form of it, as a chip
    "overlap_note": "Checked against this panel's training IDs only: a non-matching ID is not evidence of independence.",
    "unknown_note": 'No Variant_ID column, so the rows cannot be checked against the training set.',
    "unknown_suppressed": 'Metrics are withheld: without Variant_ID these rows cannot be told from training data. Add a Variant_ID column and upload the file again.',
    "unseen_note": "Checked against the IDs in this panel's training set; a non-matching ID is not on its own evidence of independence.",
    "single_class": 'Single-class batch: AUC, MCC and F1 at test prevalence are undefined.',
    "canonical_anchor": "Cross-validated F1 {f1:.3f} ± {se:.3f}",
    "table_count": '{shown} / {total} rows',
    "no_matches": 'No variants match this filter.',
    "previous": 'Previous variant',
    "next": 'Next variant',
    "candidate_count": '{m}/{n} variants',
    "input_values": 'Input values',
    "tail": 'Remaining {n} features (net)',
    "tail_value": 'sum over {n} features',
    "tail_fam": 'Multiple families',
    "contrib_note": 'base {base:+.2f} + bars = {total:+.2f} → sigmoid {sig:.3f} · LightGBM score before calibration (the {p:.3f} above is the calibrated blend)',
    "tag_in_sample": 'in-sample',
    "tag_weight": '{o} weight',
    "tag_transfer": 'transfer from General',
    "tag_transfer_cal": 'only calibration in-sample',
    "tag_unseen": 'not in the training set',
    "tag_overlap_part": '{k}/{n} IDs in the training set',
    "tag_overlap_unknown": 'overlap unknown',
    "tag_upload_panel": 'file was uploaded for {p}',
    "tag_coverage": 'feature coverage {pct}',
    "pk_cap": 'Per 1000 variants screened: {p} pathogenic, {b} benign',
    "pk_pos": "Called pathogenic", "pk_neg": "Left benign",
    "pk_tp": '{n} pathogenic found', "pk_fn": '{n} pathogenic missed',
    "pk_fp": '{n} benign wrongly flagged', "pk_tn": '{n} benign correctly left',
    "pk_short_tp": '{n} found', "pk_short_fn": '{n} missed', "pk_short_fp": '{n} wrongly flagged',
    "op_note": 'Recall ({tpr}) and the false-positive rate are held fixed: an extrapolation, not a measurement.',
    "flow_chem": "Substitution chemistry",
    "flow_prep": "Preprocessing",
    "flow_models": "Models → calibrated score",
    "flow_rank": "AL_ → rank",
    "flow_clip": "EK_ → clip",
    "flow_missing": "NaN + flag",
    "flow_median": "NaN → median",
    "flow_keep_nan": "NaN kept",
    "flow_interactions": "Interactions",
    "flow_agreement": "Score agreement",
    "flow_flags": "AL_ → present/absent",
    "flow_target": "AL_ → in-panel rank",
    "flow_own": "Own",
    "flow_transfer": "Transfer",
    "flow_excluded": "IDs excluded",   # n_excluded counts IDs; flow_matched is the number of source rows they removed
    "flow_matched": "{n} rows matched",
    "flow_train": "training rows",
    "flow_target_fit": "Platt calibration and threshold, fitted on this panel's training rows; it rescales the score to a probability.",
    "flow_title": 'Scoring flow',
    "seeds": 'seeds',
    "features": 'features',
    "description": 'Description',
    "imp_note": 'Normalized gain importance of the LightGBM members only: {members}.',
    "imp_own": "the panel's own LightGBM",
    "imp_transfer": 'the transfer LightGBM',
    "imp_general": "Trained on the General panel, so largely the General panel's importance.",
},
"tr": {
    "panel": "Gen paneli",
    "panel_general": "Genel (MASTER)", "panel_cancer": "Kalıtsal kanser",
    "panel_pah": "PAH (fenilketonüri)", "panel_cftr": "CFTR (kistik fibrozis)",
    "upload_btn": 'Dosya seç', "upload_limit": 'En çok 200 MB', "upload_title": 'CSV yükle',
    "row_id": 'satır {i}',
    "thr_short": 'eşik',
    "no_models": "Bu panel için eğitilmiş model yok. Depo kökünden şunu çalıştırın:", "threshold": "Karar eşiği", "prior_test": "Test prevalansı",
    "prior_train": "Eğitim n · prevalans", "recall": "Duyarlılık",
    "scoring": "{n} varyant {panel} modeliyle skorlanıyor…",
    "tab_pred": "Tahminler", "tab_explain": 'Varyant incelemesi',

    "n_path": "Patojenik karar",   # the app's own phrase for the call (expected_rate); the long "olarak sınıflandırılan" form wraps the card's label off its own qualifier
    "expected_rate": 'Test prevalansında ({p}) beklenen patojenik karar oranı: {r}. Ölçüm değil, ekstrapolasyon.',
    "hist_title": "Model skoru dağılımı",
    "card_labelled": "etikette {n}",   # the batch's own composition, on the card whose count it is misread against
    # The four verdicts of the truth column, as the card that is that column's key (verdict_card). The
    # subject stays the review warning: the columns are the warning, the marks are the agreement.
    "quad_title": "Uyarı ve etiket uyumu",
    "quad_warned": "Uyarı var", "quad_clean": "Uyarı yok",
    "v_ok_clean": "Doğru karar, uyarı yok", "v_ok_warn": "Doğru karar, uyarılı",
    "v_bad_warn": "Uyarılı hatalı karar", "v_bad_clean": "Uyarısız hatalı karar",
    "quad_note": "Sütunlar: inceleme uyarısı var mı. İşaretler: karar dosyanın kendi Label sütunuyla uyuşuyor mu. Dört hücre {n} satırın tamamı.",
    # Which warnings the batch actually raised, in the flag vocabulary and with no trust mark anywhere —
    # the middle card on both paths, and the only place the breakdown is drawn (flag_quad_card).
    "flag_quad_title": "Uyarı türleri",
    "flag_quad_note": "Sayılar örtüşür; {n} satırın {k} tanesi en az bir uyarı taşıyor.",
    "silent_note": "Dosyanın kendi Label sütununa göre sayılır; modelin başarım ölçüsü değil.",
    "labelled_note": "Dosyanın kendi Label sütunu; model çıktısı değil.",
    "hist_y": "Varyant", "variant": "Varyant", "score": "Model skoru",
    "chem_central": 'medyan {parts}',   # the qualifier of the batch-character heading, whose subject is flow_chem
    "col_aa": "Değişim",   # the substitution as a table column; the review page's own row says it in full (aa_change)
    "col_flags": "İnceleme uyarıları", "col_label": "Gerçek etiket",
    "pathogenic": "Patojenik", "benign": "Benign",
    "export_calls": "Tahminler (0/1)", "export_cols": "Variant_ID · tahmin 0/1 · olasılık · uyarı sütunları",
    "export_label": " · etiket",
    "flag_ek7_missing": "EK_7 eksik", "flag_complex_aa": "standart dışı AA kodu",
    "flag_cat6_flag": "CAT_6 dolu", "flag_borderline": "sınırda",
    # The trust key (truth_label). It names the tones in words because a tooltip is text the framework
    # renders and the app's marks are drawn; the legend under the table has no room for it (table_foot).
    "trust_key": "Onay: karar etiketle aynı. Çarpı: farklı."
                 " Yeşil: uyarısız. Sarı: inceleme uyarısı var. Kırmızı: yanlış ve hiçbir uyarı yok.",

    "eval_f1_prior": 'F1 (test prevalansı {p})',
    "eval_auc": "ROC AUC", "eval_mcc": "MCC",
    "f1_prior_withheld": 'Test prevalansında F1 hesaplanmadı: her sınıfta en az 30 satır gerekir ({p} patojenik / {b} benign).',
    "aa_change": "Amino asit değişimi",
    "verdict_margin": "Eşiğe göre konum", "margin_above": '{d} üzerinde', "margin_below": '{d} altında',
    "margin_at": 'Eşikte', "margin_tiny_above": "Eşiğin 0.001'den az üzerinde", "margin_tiny_below": "Eşiğin 0.001'den az altında",
    # Two whole sentences rather than margin_above(d=margin_tiny): composed, the Turkish came out as
    # "0.001'den az üzerinde", two fragments welded together where the language wants one clause.
    "member_own": "Panelin kendi modeli", "member_transfer": "Aktarım modeli", "weight": "ağırlık",
    "contrib_title": 'LightGBM skorunu etkileyen özellikler',
    "contrib_help": 'TreeSHAP ({members}), tohumlar üzerinden ortalanmış, üye ağırlıklı; kalibre edilmiş veya harmanlanmış olasılığın ayrıştırması değildir.'
                    ' Sütun adları yarışmada anonimleştirilmiştir.',

    "bad_csv": 'Dosya skorlanamadı.',
    "low_coverage": "Özellik kapsamı yalnızca {pct}; skorlar güvenilir değil.",
    "missing": "eksik", "yes": "evet", "no": "hayır",
    "contrib_x": "Katkı (log-odds)", "fam_x": "Aile toplamı (log-odds)", "value": "Girdi değeri", "feature_id": "Özellik kimliği",
    "flagtext_ek7_missing": "EK_7 eksik; bu tür satırlarda ayırt etme gücü AUC 0.86'dan 0.67'ye düşmüştü (genel panel).",
    "flagtext_complex_aa": "Birden fazla kalıntı veya standart dışı amino asit kodu: değişim kimyası özellikleri tanımsız.",
    "flagtext_cat6_flag": "Bu satırda CAT_6 değeri var; model bu tür satırlarda ölçülebilir biçimde ayırt etme gücü kaybediyor.",
    "flagtext_borderline": "Karar eşiğine 0.05'ten az uzaklıkta.",
    "global_fam": 'Özellik aileleri', "global_top": "En etkili 10 özellik (LightGBM kazancı)",
    "share": "Pay", "family": "Aile",
    "op_cohort": "Taranan küme",
    "coh_screen": "Geniş tarama", "coh_test": "Yarışma test kümesi", "coh_curated": "Küratörlü küme",
    "coh_help": 'Eksen taranan kümenin bileşimini değiştirir; model ve {thr:.2f} karar eşiği her noktada '
                'aynıdır. İşaretli üç nokta: tarama senaryosu (1000 varyantta 10 patojenik), yarışmanın bu '
                'panel için verdiği test kümesi ({test}) ve bu panelin kendi eğitim verisi ({train} patojenik). '
                'Aralarındaki her nokta hesaptır, ölçüm değildir.',
    "op_chart": 'Aynı modelin farklı kümelerde beklenen F1 değeri',
    "op_extrap": "ekstrapolasyon",
    "op_x": "Patojenik varyant prevalansı", "op_panel": "Panel",
    "op_prec": "Kesinlik",   # one line in the card, with its gloss in the help: at three cards a wrapped label costs 12px, and cancer Overview has 8
    "op_prec_help": 'Patojenik denen varyantların gerçekten patojenik olan payı (pozitif öngörü değeri, precision).',
    "op_fixed": "sabit",   # the chip that stops the prevalence control being read as a threshold
    "op_fixed_help": '{thr:.2f} karar eşiğinde çapraz doğrulamayla ölçülen duyarlılık. Her kümede aynı sayı.',
    "op_marker": "test prevalansı",
    "col_prec": "Kesinlik",   # the same quantity as op_prec, as a table column: the mean's other term, beside recall
    "single_input": 'tek girdi',
    "results_title": 'Çapraz doğrulama sonuçları',
    "chip_test_prior": 'test prevalansında',   # claim 6, drawn as the chip beside that heading
    "results_help": "İkili F1, pozitif sınıf patojenik: 3 tohum, kat ortalaması ± standart hata; her adım her katta yeniden belirlenir. Klinik doğrulama değildir.",
    "g_ek": "Anonimleştirilmiş EK_ sütunları ve bunlardan türetilen özellikler.",
    "g_al": "Anonimleştirilmiş AL_ sütunları ve bunlardan türetilen özellikler.",
    "g_amino_acid": "Grantham, BLOSUM62; hacim, yük, hidrofobiklik farkları.",
    "g_cat": "Anonimleştirilmiş kategorik CAT_ sütunları.",
    "g_missingness": "Hangi girdilerin eksik olduğunu gösteren bayraklar.",
    "g_other": "Ölçeklenmiş özelliklerin ortalaması ve yayılımı (skor uyumu).",
    "tab_overview": 'Genel bakış',
    "remove": 'Kaldır',
    "filter": 'Filtre',
    "wrong": 'hatalı karar',
    "short_general": 'Genel',
    "short_cancer": 'Kanser',
    "short_pah": 'PAH',
    "short_cftr": 'CFTR',
    "waiting": 'CSV girdi gereksinimleri',
    "empty_csv": 'Dosya skorlanamadı: varyant satırı yok. Bu dosyayı kaldırıp en az bir varyant satırı içeren bir CSV yükleyin.',
    "missing_columns": 'Dosya skorlanamadı: {total} model girdisinin {miss} tanesi eksik, {n} tanesi zorunlu: {names}',
    "id_optional": 'İsteğe bağlı; yoksa satır kimliği kullanılır',
    "label_optional": 'İsteğe bağlı: 0 = Benign, 1 = Patojenik. Yalnızca her satırda bu değerlerden biri varsa karşılaştırılır.',
    "column_group": 'Sütun grubu',
    "count": 'Sayı',
    "example": 'Örnek',
    "sample_title": 'Eğitim kümesinden örneklem',   # the fixed demo sample: SAMPLE_N rows; the seed is in the heading's tooltip
    "sample_seed": 'Sabit örneklem, tohum {seed}.',
    "training_batch": 'Buradaki satırların tamamı modelin eğitim verisi; küme metrikleri performans ölçmez.',
    "training_batch_part": 'Panelin kendi modeli — skorun {o} ağırlıklı payı — bu satırlar üzerinde eğitildi; küme metrikleri performans ölçmez.',
    "transfer_caveat": 'Eşik ve kalibrasyon{rank} bu satırlarda belirlendi; buradaki F1 ve MCC bu yüzden iyimserdir. Dış doğrulama değildir.',
    "transfer_rank": ' (ve AL_ rank dönüşümü)',
    "tag_label_ignored": 'Label sütunu yok sayıldı',
    "label_ignored": 'Label değerleri 0 veya 1 olmalı; bu dosyada “{v}” var.',
    "eval_title": 'Etiket karşılaştırması',
    "eval_not_validation": 'doğrulama değil',   # claim 1 on this heading, as a chip
    "eval_not_heldout": 'dış doğrulama değil',  # claim 3 on the transfer form of it, as a chip
    "overlap_note": 'Yalnızca bu panelin eğitim kimlikleriyle karşılaştırıldı: eşleşmeyen kimlik bağımsızlık kanıtı değildir.',
    "unknown_note": 'Variant_ID sütunu yok; satırlar eğitim kümesiyle karşılaştırılamıyor.',
    "unknown_suppressed": 'Metrikler gösterilmiyor: Variant_ID olmadan bu satırlar eğitim verisinden ayırt edilemez. Dosyaya Variant_ID sütunu ekleyip yeniden yükleyin.',
    "unseen_note": 'Bu panelin eğitim kümesindeki kimliklerle karşılaştırıldı; eşleşmeyen kimlik tek başına bağımsızlık kanıtı değildir.',
    "single_class": 'Tek sınıflı küme: AUC, MCC ve test prevalansında F1 tanımsız.',
    "canonical_anchor": "Çapraz doğrulama F1 {f1:.3f} ± {se:.3f}",
    "table_count": '{shown} / {total} satır',
    "no_matches": 'Bu filtreyle eşleşen varyant yok.',
    "previous": 'Önceki varyant',
    "next": 'Sonraki varyant',
    "candidate_count": '{m}/{n} varyant',
    "input_values": 'Girdi değerleri',
    "tail": 'Kalan {n} özellik (net)',
    "tail_value": '{n} özelliğin toplamı',
    "tail_fam": 'Birden çok aile',
    "contrib_note": 'taban {base:+.2f} + çubuklar = {total:+.2f} → sigmoid {sig:.3f} · kalibrasyon öncesi LightGBM skoru (yukarıdaki {p:.3f} kalibre edilmiş harman)',
    "tag_in_sample": 'örneklem içi',
    "tag_weight": '{o} ağırlık',
    "tag_transfer": "Genel'den aktarım",
    "tag_transfer_cal": 'yalnızca kalibrasyon örneklem içi',
    "tag_unseen": 'eğitim kümesinde yok',
    "tag_overlap_part": '{k}/{n} kimlik eğitim kümesinde',
    "tag_overlap_unknown": 'örtüşme bilinmiyor',
    "tag_upload_panel": 'dosya {p} paneli için yüklendi',
    "tag_coverage": 'özellik kapsamı {pct}',
    "pk_cap": 'Taranan 1000 varyantta {p} patojenik, {b} benign',
    "pk_pos": "Patojenik işaretlenen", "pk_neg": "Benign bırakılan",
    "pk_tp": '{n} patojenik bulundu', "pk_fn": '{n} patojenik kaçırıldı',
    "pk_fp": '{n} benign yanlış işaretlendi', "pk_tn": '{n} benign doğru bırakıldı',
    "pk_short_tp": '{n} bulundu', "pk_short_fn": '{n} kaçırıldı', "pk_short_fp": '{n} yanlış işaretlendi',
    "op_note": 'Duyarlılık ({tpr}) ve yanlış pozitif oranı sabit tutulur; bu bir ekstrapolasyondur, ölçüm değildir.',
    "flow_chem": "Değişim kimyası",
    "flow_prep": "Ön işleme",
    "flow_models": "Modeller → kalibre skor",
    "flow_rank": "AL_ → rank",
    "flow_clip": "EK_ → kırp",
    "flow_missing": "NaN + bayrak",
    "flow_median": "NaN → medyan",
    "flow_keep_nan": "NaN korunur",
    "flow_interactions": "Etkileşimler",
    "flow_agreement": "Skor uyumu",
    "flow_flags": "AL_ → var/yok",
    "flow_target": "AL_ → panel içi rank",
    "flow_own": "Kendi",
    "flow_transfer": "Aktarım",
    "flow_excluded": "kimlik dışlandı",   # n_excluded counts IDs; flow_matched is the number of source rows they removed
    "flow_matched": "{n} satır eşleşti",
    "flow_train": "eğitim satırı",
    "flow_target_fit": "Platt kalibrasyonu ve eşik, bu panelin eğitim satırlarında belirlendi; skoru olasılığa çevirir.",
    "flow_title": 'Skorlama akışı',
    "seeds": 'tohum',
    "features": 'özellik',
    "description": 'Açıklama',
    "imp_note": 'Yalnızca LightGBM üyelerinin normalize kazanç önemi: {members}.',
    "imp_own": "panelin kendi LightGBM'i",
    "imp_transfer": "aktarım LightGBM'i",
    "imp_general": 'Genel panelde eğitildiği için önem büyük ölçüde Genel panelinkiyle aynıdır.',
},
}


def t(key: str, **kw) -> str:
    s = S[st.session_state.get("lang", "tr")].get(key) or S["en"][key]
    return s.format(**kw) if kw else s


# ── cached resources ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_art(panel: str):
    return load_artifacts(panel)


@st.cache_data(show_spinner=False)
def load_panel_sample(panel: str, n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> bytes:
    """The demo batch: the same rows every run, so the demo is reproducible."""
    df = pd.read_csv(PANELS[panel])
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df.to_csv(index=False).encode()


@st.cache_data(show_spinner=False, max_entries=4)
def run_scoring(panel: str, csv_bytes: bytes) -> dict:
    """Score + explain one batch. Keyed on the panel and the exact bytes, so re-runs are free."""
    art = get_art(panel)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if df.empty:
        raise ValueError("the file has a header but no variant rows")
    expected = expected_columns(art)
    missing = expected - set(df.columns)
    coverage = len(expected & set(df.columns)) / max(len(expected), 1)
    # Only categorical columns indexed directly by the own native-NaN replay are mandatory.
    # Transfer replay reindexes missing inputs; engineered AA columns are not CSV requirements.
    own = art.get("own") or {}
    prep = own.get("prep_art") or {}
    required = set(prep.get("ppf", {}).get("cat_cols", [])) if prep.get("kind") == "keep_nan" else set()
    req_missing = required - set(df.columns)
    if coverage < 0.5 or req_missing:
        # The named columns are the ones to add first, but the count beside them is every input the file is
        # missing: naming eight when 294 are absent reads as "add eight columns".
        raise InputError(sorted(req_missing or missing, key=lambda c: (c.startswith("AL_"), c)),   # tolerated AL_ last
                         len(expected), len(missing), len(req_missing))
    raw = normalize_sentinels(df.copy())
    sc = score_frame(df, art)
    ex = explain_frame(art, sc)
    flags = confidence_flags(raw, sc["probs"], art["threshold"], BORDER_MARGIN)
    mats = [m.reset_index(drop=True) for m in (sc.get("X_lgbm"), sc.get("Xs")) if m is not None]
    matrix = pd.concat(mats, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    return {"raw": raw, "probs": np.asarray(sc["probs"]), "own": sc["own_probs"],
            "transfer": sc["transfer_probs"], "contribs": ex["contribs"], "base": ex["base"],
            "members": ex["members"], "flags": flags, "matrix": matrix, "af_cols": af_columns(raw), "coverage": coverage}


@st.cache_data(show_spinner=False)
def get_importance(panel: str) -> pd.Series:
    return global_importance(get_art(panel))


@st.cache_data(show_spinner=False)
def training_rows(panel: str):
    """What predict.train_and_save fitted on: the panel CSV after drop_all_nan_rows (the raw file also carries
    labelled rows with no features), and the strongest single raw EK_ input on those rows. A single
    input has no fitted parameter, so its AUC on the training rows is the same number it would be out of sample."""
    if not os.path.exists(PANELS[panel]):
        return None
    from sklearn.metrics import roc_auc_score
    df = drop_all_nan_rows(pd.read_csv(PANELS[panel]), verbose=False)
    y = df["Label"].astype(int)
    best = ("", 0.0)
    for col in (c for c in df.columns if c.startswith("EK_")):
        v = pd.to_numeric(df[col], errors="coerce"); seen = v.notna()
        if y[seen].nunique() == 2:
            auc = roc_auc_score(y[seen], v[seen]); auc = max(auc, 1 - auc)   # orientation is not a parameter either
            best = max(best, (col, auc), key=lambda b: b[1])
    return {"n": len(df), "best": best}


# ── helpers ───────────────────────────────────────────────────────────────────────────────────
# A categorical column's value is printed exactly as the file carries it. The codes inside CAT_1 / CAT_2
# look like database and cohort codes, but the organizers do not say what they are, so the app does not
# expand them into names it cannot stand behind.

_FLAG_PREFIXES = ("ALfam_", "AA_to_stop", "AA_is_", "AA_glycine_loss", "AA_proline_", "AA_cysteine_loss", "AA_same",
                  "AA_grantham_high", "AA_charge_flipped", "AA_blosum_", "acmg_ba1", "acmg_bs1", "acmg_conserved", "Q_t", "Q_h")


E = html.escape


def sync_upload():
    """Mirror the uploader into a plain session key: Streamlit forgets a widget that is not rendered, and
    the uploader is only rendered while the upload dialog is open. Taking a file is what that dialog is for,
    so a file closes it; the app-scoped rerun in its body is what scores the batch behind it.

    The panel showing in the header at that moment is kept with the bytes. That control can be switched
    afterwards, and then the same bytes are scored by another panel's model with nothing to say so
    (render_pred tags the heading). It is what the file was declared for, not a second panel control: only a
    new file re-declares."""
    up = st.session_state.get("upload_w")
    if up is None:
        st.session_state.pop("upload_keep", None)
    else:
        st.session_state["upload_keep"] = (up.name, up.getvalue(), st.session_state.get("panel_keep", "general"))
        st.session_state.pop("upload_open", None)
    st.session_state["upload_dirty"] = True


def upload_panel():
    """The panel the loaded file was declared for, or None when there is no file (or a session that predates
    the declaration)."""
    keep = st.session_state.get("upload_keep")
    return keep[2] if keep and len(keep) > 2 else None


def variant_ids(raw: pd.DataFrame):
    # Rows are counted from 1: the operator reading this id is matching it back to a row of their own file.
    return raw["Variant_ID"].astype(str).values if "Variant_ID" in raw else [t("row_id", i=i + 1) for i in range(len(raw))]


def call_name(pathogenic) -> str:
    """The word for a model call, wherever one is written."""
    return t("pathogenic") if pathogenic else t("benign")


def is_flag(feat: str, af_cols=None) -> bool:
    if feat.endswith("_missing") or feat.startswith(_FLAG_PREFIXES):
        return True
    return feat.startswith("AL_") and af_cols is not None and feat not in af_cols


def fmt_val(v, feat: str = "", af_cols=None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return t("missing")
    if is_flag(feat, af_cols) and float(v) in (0.0, 1.0):
        return t("yes") if float(v) == 1.0 else t("no")
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.3g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.3f}"   # fixed precision: 1.000 and 0.964 are the same column
    return str(v)


def fmt_int(v, feat: str = "") -> str:
    """A whole-number input (Grantham, BLOSUM62) reads as one; a missing value reads as fmt_val renders it."""
    return fmt_val(v, feat) if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.0f}"


def fmt_p(p: float) -> str:
    """A calibrated probability must never read as certainty: 0.999849 is >0.999, not 1.000."""
    return f"{p:.3f}" if 0.0005 <= p < 0.9995 else (">0.999" if p > .5 else "<0.001")


def flag_text(row: pd.Series) -> str:
    return ", ".join(t(f"flag_{k}") for k in FLAG_KEYS if bool(row[k]))


# The app's own marks are drawn, not typed. Every one of them is a path in this one set: 16 by 16, and all
# but the agreement pair are unfilled, one 1.5px stroke in the current text colour, round caps and joins —
# the two exceptions are below. They were Unicode codepoints, and the
# page's font stack has none of them — every one arrived from a fallback face nothing here chose, at one
# advance and at ink heights spanning 13 to 21px, one of them a solid mass among hairlines. Drawn, the set
# has one weight and one box, and the app stops running a second icon vocabulary beside Material Symbols.
#
# Each flag's form is still its meaning — the empty set for a value that is not there, the exchange arrows
# for a substitution that is not one clean swap, the hazard triangle for an unreliable region, and a shape
# divided by a line for a score sitting on the threshold, the same line the table draws between the two
# classes. Four base silhouettes (circle, arrows, triangle, square), so the column still reads in greyscale,
# out of focus and across a hall; the triangle is outlined like the rest, because a fill among hairlines is
# what made the old set ragged, and its outline still says hazard.
# They are drawn as CSS masks rather than as inline <svg>, because st.html runs its payload through
# DOMPurify with USE_PROFILES {html: true} — the SVG profile is off and an <svg> element is dropped whole
# (verified in the rendered DOM, and in the frontend bundle). A mask keeps the authored path and takes its
# paint from the element, so a mark still follows the colour of the text it sits in.
#
# `yes` and `no` are the exception, and deliberately so: the agreement mark carries a trust tone, and a tone
# on a hairline is a thread of colour. They are therefore inverted — a filled 15px rounded square with the
# glyph cut out of it, chip and glyph in one path under fill-rule evenodd, so the second subpath punches a
# hole rather than adding ink. The mask reads alpha, so the hole is genuinely transparent and shows whatever
# the row behind it is painted (white, --surf when hovered, the brand tint on tr.now). Two consequences the
# next editor should keep: the glyph's gap is 2px, not the set's 1.5px stroke, because negative space at
# 16px closes up below about 1.8 while 2.5 gave the knockout more of the chip than the tone; and the four
# review flags stay hairline, because a solid chip beside outlined flags is what stops the agreement mark
# reading as a fifth flag. Both glyphs are offsets of one centreline each — the check
# (4,8.1)→(6.8,10.9)→(12,4.9) mitred, the cross (4.25,4.25)–(11.75,11.75) crossed — so a future weight is
# the same two outlines recomputed at the new gap, not a redrawn shape.
MARKS = {
    "ek7_missing": "<circle cx='8' cy='8' r='5.5'/><path d='M4.1 11.9 11.9 4.1'/>",
    "complex_aa": "<path d='M2.5 5.5h11m-2-2 2 2-2 2M13.5 10.5h-11m2-2-2 2 2 2'/>",
    "cat6_flag": "<path d='M8 2.6 14.4 13.4H1.6Z'/>",
    "borderline": "<rect x='2.5' y='2.5' width='11' height='11' rx='0.5'/><path d='M8 2.5v11'/>",
    "info": "<circle cx='8' cy='8' r='6'/><path d='M8 7.4v4.2'/><path d='M8 4.8v0.01'/>",
    "yes": ("<path fill='black' fill-rule='evenodd' stroke='none' d='M5 0.5H11A4.5 4.5 0 0 1 15.5 5V11"
            "A4.5 4.5 0 0 1 11 15.5H5A4.5 4.5 0 0 1 0.5 11V5A4.5 4.5 0 0 1 5 0.5Z"
            "M3.29 8.81L6.85 12.37L12.76 5.55A1 1 0 0 1 11.24 4.25L6.75 9.43L4.71 7.39"
            "A1 1 0 0 1 3.29 8.81Z'/>"),
    "no": ("<path fill='black' fill-rule='evenodd' stroke='none' d='M5 0.5H11A4.5 4.5 0 0 1 15.5 5V11"
           "A4.5 4.5 0 0 1 11 15.5H5A4.5 4.5 0 0 1 0.5 11V5A4.5 4.5 0 0 1 5 0.5Z"
           "M9.41 8L12.46 11.04A1 1 0 0 1 11.04 12.46L8 9.41L4.96 12.46A1 1 0 0 1 3.54 11.04"
           "L6.59 8L3.54 4.96A1 1 0 0 1 4.96 3.54L8 6.59L11.04 3.54A1 1 0 0 1 12.46 4.96Z'/>"),
    "chevron": "<path d='M6 3.5 10.5 8 6 12.5'/>",
}


def mark_url(name: str) -> str:
    """One mark as a CSS mask source. The stroke colour inside the data URI is irrelevant — a mask reads the
    alpha channel — so the paint is the element's own (background:currentColor in the .mk rule)."""
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='black'"
           " stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'>" + MARKS[name] + "</svg>")
    return 'url("data:image/svg+xml,' + quote(svg, safe="/:=' ") + '")'


def mark(name: str) -> str:
    """One drawn mark. Presentational everywhere it is used: the accessible name is on the element that
    carries the mark (the flag cell's role=img, the hint's aria-label, the true label's own word)."""
    return f"<span class='mk mk-{name}'></span>"


def flag_mark(key: str) -> str:
    """One flag's mark. The legend's mark and the cell's mark are this one call, so they cannot drift apart;
    the fixed box it is drawn in is what puts every mark on one baseline and one footprint (CSS .fl)."""
    return f"<span class='fl'>{mark(key)}</span>"


def flag_symbols(row: pd.Series) -> str:
    """The fired flags as symbols, carrying their full names as the cell's tooltip (as the review list does)
    and as its accessible name: the symbols are an image of the flag set, not text to be read aloud.

    Every flag has a fixed slot, in FLAG_KEYS order — the legend's order — so the same mark sits at the same
    x on every row and the column reads down as well as across: a flag that is not fired leaves its slot
    empty, an empty box of the mark's own width and nothing drawn in it. The slots live inside the role=img
    span, whose children are presentational, so an empty one is silent. A row with no flag draws nothing."""
    fired = [k for k in FLAG_KEYS if bool(row[k])]
    names = E(flag_text(row))
    return (f"<span class='fls' role='img' aria-label='{names}' data-tip='{names}'>"
            + "".join(flag_mark(k) if k in fired else "<span class='fl'></span>" for k in FLAG_KEYS) + "</span>") if fired else ""


def truth_label(truth, agree, flagged=False) -> str:
    """The true label as a word in plain ink, with the agreement mark after it.

    The word was painted in its class colour until the chip became a traffic light: red and blue still mean
    a clinical call everywhere they are drawn — the histogram and its legend, the per-row score bars, the
    decision scale on Overview — but not here, because a red chip beside a red "Patojenik" would be one
    colour carrying two meanings a character apart. The cell states the class in words and its reliability
    in colour; the score bar on the same row is where that row's call is still red or blue.

    The mark states what the competition actually asks about a false positive or a false negative: not only
    that the model was wrong, but whether the row's review flags warned anyone. Four states out of two facts,
    in three tones (TRUST_OK / TRUST_WARN / TRUST_FAIL): a green check where the call matches the label and
    nothing fired, gold on either mark where a flag did fire — the row is not to be trusted in either
    direction — and red on a cross where the call is wrong and no flag fired at all. The last is the case a
    reviewer would never have been sent to look at, and nothing else on this page distinguishes it.

    The tone is carried by a filled chip with the glyph knocked out of it, not by a hairline: at 16px a
    stroked glyph gives the tone a few dozen pixels of ink, and the state has to read across a hall. The
    chip is also what keeps this mark out of the flag vocabulary — solid against four outlined shapes, so
    it is not read as a fifth flag — and it is why the trust set stays two silhouettes rather than four.

    Green and gold are the same glyph, so that pair rests on tone alone — which is why the tone is never the
    only carrier: the flag column on the same row already says symbolically whether anything fired, and the
    check and the cross carry the agreement on their own."""
    tone = "ok" if agree and not flagged else "warn" if flagged else "bad"
    return (f"<b class='cls'>{E(call_name(truth))}</b> "
            + f"<span class='tm tm-{tone}'>{mark('yes' if agree else 'no')}</span>")


def table_foot(count: str) -> str:
    """The one line under the results table: the four symbols named on the left, what the table is showing
    on the right. Not the flags' explanations — those are on Variant review. If the pair cannot share the
    line, the count keeps the right edge on a line of its own rather than moving back above the table."""
    return ("<div class='key'>" + "".join(f"<span>{flag_mark(k)}{E(t('flag_' + k))}</span>"
                                          for k in FLAG_KEYS) + f"<span class='count'>{E(count)}</span></div>")


# Custom blocks go through st.html, not st.markdown: markdown containers carry a -1rem bottom margin
# and add anchor links to headings, which broke the block rhythm and put invisible stops in the tab order.
def lbl(text, help=None, chips=()):
    """A band heading, read as `subject · qualifier`: the subject is set heavy and everything after the first
    separator muted — the brand's dark display line over its muted second line, on the one line the page can
    afford. `help` is the app's route for a fact that qualifies the numbers under the heading without needing
    to be read by everyone: it hangs off the marker, not the layout.

    `chips` are the app's one badge treatment (`.chp`), and they are what a fact takes when it is a state of
    the thing named rather than more of its name: a claim the page must carry without a hover reads as a
    marker on the title, not as another clause of it. Everything that is not that is deleted instead."""
    subject, sep, rest = text.partition(" · ")
    sub = f"<span class='sub'>{sep}{E(rest)}</span>" if sep else ""
    tags = "".join(f"<span class='chp'>{E(c)}</span>" for c in chips)
    st.html(f"<h3 class='lab'>{E(subject)}{sub}{tags}{hint_mark(help)}</h3>")


def hint_mark(help) -> str:
    """The marker a fact hangs off, wherever the fact qualifies the line rather than belonging on it. It is a
    tab stop, like the framework's own help icons, so the fact is reachable from the keyboard (focus shows the
    tooltip); its accessible name is the fact itself, not a bare glyph."""
    return (f" <span class='hint' role='note' tabindex='0' data-tip='{E(help)}' aria-label='{E(help)}'>{mark('info')}</span>"
            if help else "")


def note(text, help=None):
    """A muted line under the block it belongs to. It takes `help` for the same reason a heading does: the line
    states the one fact a reader carries away, and what qualifies that fact hangs off the marker."""
    st.html(f"<div class='small'>{E(text)}{hint_mark(help)}</div>")


def html_table(columns, rows, marked=None, tall=False, ids=None, foot="", divider=None, empty=""):
    """Cells are pre-escaped; only the scrollable (tall) table enters the tab order. With `ids` the rows are
    the way into Variant review: each carries its variant id for the click bridge to read, and its first cell
    becomes a button — the row is the pointer target, the button the keyboard one. `foot` is drawn inside this
    same element: the line under a table describes that table, so it follows the table's own height rather
    than the column's, which a filter that leaves four rows would otherwise leave it 400px below.

    `empty` is what the body says when there are no rows: a filter that matches nothing still draws its
    table — the header, the border, the foot line and the binding to the fold — with one muted line across
    the columns where the rows would be, so the page keeps its shape and the filter above it stays the
    recovery. A message in place of the table would switch the whole flex-fill chain off.

    `divider` is `(index, (left, middle, right))`: a labelled line drawn across every column before data row
    `index`, for a table whose rows are sorted so that a property of the row changes exactly once. Its parts
    are plain text and are escaped here. It is not a row of the data — it carries no id, so the click bridge
    passes over it, it is outside the caller's row cap, and nothing built from `rows` (the export) sees it.

    A column given as `(name, fact)` carries the fact on the heading's own marker, the way a band heading
    does: what a column's cells mean qualifies the column, not any one row in it."""
    head = ''.join(f'<th scope="col">{E(str(c[0])) + hint_mark(c[1]) if isinstance(c, tuple) else E(str(c))}</th>'
                   for c in columns)
    body = f"<tr class='none'><td colspan='{len(columns)}'>{E(empty)}</td></tr>" if not rows and empty else ''
    for i, r in enumerate(rows):
        if divider is not None and divider[0] == i:
            body += (f"<tr class='split'><td colspan='{len(columns)}'><div class='splitline'>"
                     + "".join(f"<span>{E(part)}</span>" for part in divider[1]) + "</div></td></tr>")
        cells = list(r)
        vid = f' data-vid="{E(str(ids[i]))}"' if ids is not None else ''
        if ids is not None:
            cells[0] = f"<button type='button' class='row'>{cells[0]}</button>"
        body += ((f"<tr class='now'{vid}>" if i == marked else f"<tr{vid}>")
                 + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    st.html(f"<div class='tablewrap{' tall' if tall else ''}'{" tabindex='0'" if tall else ''}><table class='grid'>"
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{foot}")


def metric_row(items):
    """items: (label, value, help) and, for at most one of them, a fourth truthy element: the chip. A metric
    label is a framework string and cannot carry the app's own `.chp` span, so the chip is drawn by CSS on a
    keyed container (`.st-key-metric_chip`), in the one treatment every other chip uses. No deltas:
    Streamlit always draws an arrow next to them."""
    for c, item in zip(st.columns(len(items)), items):
        label, value, help_ = item[:3]
        with c:
            if len(item) > 3 and item[3]:
                with st.container(key="metric_chip"):
                    st.metric(label, value, help=help_)
            else:
                st.metric(label, value, help=help_)


def styled(chart):
    """One typography/config treatment for every chart (applied after layering)."""
    return (chart.configure_view(stroke=None)
            .configure_axis(labelFontSize=TXT, titleFontSize=TXT, labelColor=MUTED, titleColor=MUTED,
                            titleFontWeight=500, gridColor=GRID, titlePadding=H(8), labelLimit=H(LABEL_LIMIT))
            .configure_legend(labelFontSize=TXT, labelColor=MUTED, titleFontSize=TXT, padding=H(8)))


def prob_hist(probs, thr):
    """The band's label is the threshold and nothing else: how many rows fall inside the band is one of the
    four figures the warning-types card prints a few hundred pixels away, and was said twice."""
    edges = np.round(np.linspace(0, 1, 41), 4)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, 39)
    d = pd.DataFrame({"lo": edges[idx], "hi": edges[idx + 1],
                      "call": np.where(probs >= thr, call_name(True), call_name(False))})
    d = d.groupby(["lo", "hi", "call"]).size().reset_index(name="n")
    base = alt.Chart(d).mark_bar(opacity=0.9).encode(
        x=alt.X("lo:Q", bin="binned", title=t("score"), scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFlushOffset=-2)),   # the flush-aligned last label otherwise overflows the clipped SVG
        x2="hi:Q",
        # A count axis has no half: on a batch small enough that the tallest bin is under about twenty — the
        # size of file this page is handed live — Vega's nice ticks otherwise step by 0.5 of a variant.
        y=alt.Y("n:Q", title=t("hist_y"), stack=True, axis=alt.Axis(tickMinStep=1, format="d"),
                scale=alt.Scale(domain=[0, int(d.groupby("lo")["n"].sum().max()) + 1])),
        color=alt.Color("call:N", scale=alt.Scale(domain=[call_name(True), call_name(False)], range=[RED, BLUE]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("lo:Q", title=t("score"), format=".3f"), alt.Tooltip("n:Q", title=t("hist_y"))])
    rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color="black", strokeDash=[6, 4], size=2).encode(x="t:Q", tooltip=alt.value(None))
    # The decision-uncertainty band: the strip the borderline flag marks, drawn full height behind the bars so
    # the distribution says how many calls sit close enough to the threshold to flip. It is a rule, not a
    # measurement, so it takes the grid grey — a call colour or the brand here would read as a verdict.
    band = alt.Chart(pd.DataFrame({"lo": [max(0.0, thr - BORDER_MARGIN)], "hi": [min(1.0, thr + BORDER_MARGIN)]})).mark_rect(
        color=GRID).encode(x="lo:Q", x2="hi:Q", tooltip=alt.value(None))
    # The word, not the ◐: a font draws these geometric glyphs at about half their em — which is why the table
    # renders the flags at 2em — so at a 12px annotation the same codepoint is a ~6px speck across a hall. The
    # word is the filter option's own string, so the label and the dropdown cannot say different things. It is
    # wider than the plot leaves right of any panel's threshold, so it is drawn left of the rule on all four:
    # the four screenshots then read as one picture, which matters more than which side of the rule it takes.
    label = f"{t('thr_short')} {thr:.2f}"
    # It is drawn inside the plot, along the top, ending left of the band rather than left of the rule. Above
    # the frame it floated in the legend's padding with the rule's tip touching its corner, reading as a
    # stray heading over the chart; anchored to the band's left edge it labels the thing it counts and the
    # line that thing is drawn around, and it clears both. The tallest bin is at the top of the score range
    # on every panel — right of the threshold — so the top of the plot left of the band is the free strip.
    text = alt.Chart(pd.DataFrame({"t": [max(0.0, thr - BORDER_MARGIN)], "lbl": [label]})).mark_text(
        align="right", dx=-6, dy=4, baseline="top",
        fontWeight="bold", fontSize=TXT).encode(x="t:Q", y=alt.value(0), text="lbl:N")
    return styled((band + base + rule + text).properties(height=H(BAND_TALL)))


def family_chart(row, lang):
    """The whole before the detail: each family's net contribution over every feature, as a small diverging
    chart drawn above the ten-bar chart in the same colours and the same orientation. It is also that chart's
    legend — the family names are its axis — so the ten bars carry no key of their own. The totals add up to
    the `+ bars` term in the note under the chart, tail included, which ten bars and a tail cannot be summed
    to by eye on a projector; and a family total is larger than any one of its features, so this chart has
    its own axis rather than borrowing the scale of the one below. A family whose net is zero at this
    precision is not drawn; the order is the taxonomy's, so the picture is the same on every variant."""
    sums = row.groupby(row.index.map(family_of)).sum()
    fams = [k for k in FAMILY_COLORS if round(float(sums.get(k, 0.0)), 3)]
    d = pd.DataFrame({"family": [FAMILY_LABELS[lang][k] for k in fams], "contrib": [float(sums[k]) for k in fams],
                      "color": [FAMILY_COLORS[k] for k in fams]})
    base = alt.Chart(d).encode(
        y=alt.Y("family:N", sort=d["family"].tolist(), title=None, axis=alt.Axis(labelOverlap=False, labelFlush=False)),
        # The axis is titled: this chart and the one below share a column, an orientation and a colour scheme
        # but not a scale (a family total is larger than any one feature), and two labelled axes read as two
        # charts where one continued picture invites the bar lengths to be compared across them.
        x=alt.X("contrib:Q", title=t("fam_x"), stack=None, axis=alt.Axis(tickCount=8),   # a small chart: fewer ticks than the subject's
                scale=alt.Scale(domain=[min(0, d.contrib.min()) - max(.15, d.contrib.abs().max()*.25),
                                        max(0, d.contrib.max()) + max(.2, d.contrib.abs().max()*.4)])),
        tooltip=[alt.Tooltip("family:N", title=t("family")), alt.Tooltip("contrib:Q", title=t("contrib_x"), format="+.3f")])
    bars = base.mark_bar().encode(color=alt.Color("color:N", scale=None))
    label = alt.Text("contrib:Q", format="+.3f")
    pos = base.transform_filter(alt.datum.contrib >= 0).mark_text(align="left", dx=4, fontSize=TXT).encode(text=label)
    neg = base.transform_filter(alt.datum.contrib < 0).mark_text(align="right", dx=-4, fontSize=TXT).encode(text=label)
    rule = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK).encode(x="z:Q")
    # The height carries its own axis (the framework fits the axis inside the height it is given), at the
    # same allowance the gauge's axis takes.
    return styled((bars + pos + neg + rule).properties(height=chart_rows(len(fams), MIN_BAND, GAUGE_AXIS)))


def contrib_chart(row, top, raw_row, mat_row, lang, af_cols):
    """The chart for the already-ranked drivers `top`. Its legend is the family chart drawn above it
    (family_chart), not a Vega legend: Vega centres a top-oriented legend across the plot width, which
    strands it in empty space. This chart is where a driver is named — its axis labels carry the column id,
    and what this project computed off that column where there is something to say; an organizer column
    carries the id alone, because its meaning is not disclosed. The bar labels carry the contribution; the
    feature's own value is often missing here (an unobserved input is exactly the kind of thing that moves
    the score), so it stays in the list and the tooltip rather than decorating every bar with "eksik"."""
    rows = []
    for f, c in top.items():
        v = fmt_val(feature_value(f, raw_row, mat_row), f, af_cols)
        name = readable(f, lang)
        rows.append(dict(feature=f, name=f if name == f else f"{f} · {name}", contrib=float(c), value=v,
                         family=FAMILY_LABELS[lang][family_of(f)]))
    # The tail is the residual of every feature outside the ten bars, from any family — not the `other` family,
    # whose net the key now states beside it. It borrowed that family's word until then; one screen carrying
    # "Diğer" twice for two quantities is a defect, and the taxonomy label is fixed (Overview and explain.py
    # read the same one), so the tail is the side that renames — its bar and its tooltip's family together.
    # The tail's tooltip says what its length is the net of: read as one quantity beside a named driver it is
    # the biggest thing on the chart on PAH, and a juror would take it for a single unnamed factor.
    rows.append(dict(feature="—", name=t("tail", n=len(row) - len(top)), contrib=float(row.sum() - top.sum()),
                     value=t("tail_value", n=len(row) - len(top)), family=t("tail_fam")))
    d = pd.DataFrame(rows)
    base = alt.Chart(d).encode(
        y=alt.Y("name:N", sort=d["name"].tolist(), title=None,   # labelFlush would baseline-shift the first and last labels into their neighbours
                axis=alt.Axis(labelOverlap=False, labelFlush=False)),
        x=alt.X("contrib:Q", title=t("contrib_x"), stack=None,
                scale=alt.Scale(domain=[min(0, d.contrib.min()) - max(.15, d.contrib.abs().max()*.25),
                                        max(0, d.contrib.max()) + max(.2, d.contrib.abs().max()*.4)])),
        tooltip=[alt.Tooltip("feature:N", title=t("feature_id")), alt.Tooltip("family:N", title=t("family")),
                 alt.Tooltip("value:N", title=t("value")), alt.Tooltip("contrib:Q", title=t("contrib_x"), format="+.3f")])
    # The tail keeps the faint grey it has always had — the palette's other grey is missingness — and it is in
    # the scale under its own name rather than by borrowing a family's entry.
    fam_colors = {**{FAMILY_LABELS[lang][k]: FAMILY_COLORS[k] for k in FAMILY_LABELS[lang]}, t("tail_fam"): FAINT}
    bars = base.mark_bar().encode(color=alt.Color("family:N", title=None, legend=None,
        scale=alt.Scale(domain=list(fam_colors), range=list(fam_colors.values()))))
    label = alt.Text("contrib:Q", format="+.3f")
    pos = base.transform_filter(alt.datum.contrib >= 0).mark_text(align="left", dx=4, fontSize=TXT).encode(text=label)
    neg = base.transform_filter(alt.datum.contrib < 0).mark_text(align="right", dx=-4, fontSize=TXT).encode(text=label)
    rule = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK).encode(x="z:Q")
    # The page's subject takes the screen the rest of the page leaves; under NARROW the two columns
    # stack, and a chart sized for the space beside the verdict card would only add scroll.
    #
    # This chart and the family chart above it do not share a plot origin — each pads its y-axis by its own
    # longest label, so the two plot areas start 105px apart at 1366 and 74px at 1706, drifting with the
    # language and the panel. `alt.vconcat(..., bounds="flush")` is the fix on paper and was tried: a concat
    # spec cannot take `width: "container"`, so the pair sized itself from its data — 1262px of SVG inside a
    # 986px column at 1706, the family chart's axis title dropped, and the page's flex-fill binding replaced
    # by a full-viewport block with the note at the bottom. Reverted. Any second attempt has to solve the
    # container width first (a measured pixel width for the concat), not the alignment.
    return styled((bars + pos + neg + rule).properties(height=H(SUBJECT if VW > NARROW else SUBJECT_NARROW)))


def top_feature_chart(imp: pd.Series, lang: str, k: int = 10):
    top = imp.sort_values(ascending=False).head(k)
    fams = [family_of(f) for f in top.index]
    # The axis carries the column id alone. The contribution chart labels a feature "id · name" because it
    # stands by itself; here the name is the widest thing on the chart and the least of what it adds — the
    # bar's colour and the family table beside it already say which block the column is from, ten times over
    # for the AL_ block, while the id is the only part no neighbour supplies. The bars, which are the
    # quantity this chart exists to show, get the width back.
    d = pd.DataFrame({"feature": top.index, "name": [readable(f, lang) for f in top.index], "share": top.values,
                      "family": [FAMILY_LABELS[lang][f] for f in fams], "color": [FAMILY_COLORS.get(f, GREY) for f in fams]})
    # An organizer column's name is its id, which the axis already carries: the description line is in the
    # tooltip only when some feature on the chart has something else to say.
    tips = [alt.Tooltip("family:N", title=t("family")), alt.Tooltip("share:Q", title=t("share"), format=".2%")]
    if (d["name"] != d["feature"]).any():
        tips.insert(0, alt.Tooltip("name:N", title=t("description")))
    return styled(alt.Chart(d).mark_bar().encode(
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelFlush=False)),
        x=alt.X("share:Q", title=t("share"), stack=None, scale=alt.Scale(domain=[0, float(top.max()) * 1.05]), axis=alt.Axis(format="%")),
        color=alt.Color("color:N", scale=None),
        tooltip=tips,
    ).properties(height=chart_rows(k, MIN_BAND)))


def expected_columns(art):
    cols = set(art["feature_cols"]) | set(((art.get("transfer") or {}).get("art") or {}).get("raw_cols", []))
    return {c for c in cols if re.fullmatch(r"(?:AL|EK|CAT|AA)_\d+", c)}


class InputError(ValueError):
    """Missing model inputs, safe to describe without exposing a traceback: the columns to name, then the
    number of inputs the model takes, the number this file is missing and how many of those are required."""


@st.cache_data(show_spinner=False)
def panel_ids(panel):
    if not os.path.exists(PANELS[panel]):
        return None
    return set(pd.read_csv(PANELS[panel], usecols=["Variant_ID"])["Variant_ID"].dropna().astype(str))


def training_overlap(raw):
    """Rows the deployed artifacts were fitted on (predict.train_and_save trains on the whole panel
    CSV), or None when the batch cannot be compared to it."""
    ids = panel_ids(panel)
    if ids is None or "Variant_ID" not in raw or not raw["Variant_ID"].notna().all():
        return None
    return int(raw["Variant_ID"].astype(str).isin(ids).sum())


def batch_labels(raw):
    lab = pd.to_numeric(raw["Label"], errors="coerce") if "Label" in raw else None
    return lab.astype(int).to_numpy() if lab is not None and lab.notna().all() and lab.isin([0, 1]).all() else None


def rejected_label(raw):
    """The first Label value that is not 0/1: clinical strings are reported to the user, never coerced."""
    bad = raw["Label"][~pd.to_numeric(raw["Label"], errors="coerce").isin([0, 1])] if "Label" in raw else []
    return str(bad.iloc[0]) if len(bad) else None


def batch_mask(res, y, filters):
    calls = res["probs"] >= thr
    mask = np.ones(len(calls), dtype=bool)
    selected = [calls if k == "pathogenic" else ~calls for k in filters if k in ("pathogenic", "benign")]
    if selected:
        mask &= np.logical_or.reduce(selected)
    selected = [res["flags"][k].to_numpy() for k in FLAG_KEYS if "flag_" + k in filters]
    if selected:
        mask &= np.logical_or.reduce(selected)
    if "wrong" in filters and y is not None:
        mask &= calls != y
    return mask


def option(key):
    """A filter option's word. The list mixes model calls with review flags, and the flags are already
    written under the table in lower case: one case for the whole list, or the lower-case items read as
    sub-items of the one above them. A word that opens with a column id (EK_7, CAT_6) is left alone —
    lower-casing an id makes it a different string, and the id is the only thing naming that flag."""
    word = t(key)
    return word if word[1:2].isupper() else word[:1].lower() + word[1:]


def active_filter():
    keys = ss.get("filter_keep", ())
    return [t("filter") + ": " + ", ".join(option(k) for k in keys)] if keys else []


def mirror(key):
    value = st.session_state[key + "_w"]
    if value is not None:   # None: a single-select control was clicked off; the kept choice re-seeds it below
        st.session_state[key + "_keep"] = value



def cohort_prior(key, sel_panel):
    """The prevalence of a named set. Two of the three are numbers the page already carries in the table above
    the band — the contest's test prior and the panel's own training prior — so the marks are read against a
    row a juror can see, and none of them is a value this team picked."""
    return {"screen": SCREEN_PRIOR, "test": TEST_PRIORS[sel_panel], "curated": CANON[sel_panel]["train"]}[key]


def prev_options(sel_panel):
    """The stops the prevalence control travels through: a 0.5% grid over the curve's whole domain, with the
    three known prevalences inserted exactly. A select_slider rather than st.slider because the anchors are
    then values the control can actually land on — the contest's 14.2857% is on no round grid, and a reader
    who drags away has to be able to come back to it — and because the readout is then this app's own
    localized percent instead of a printf format. Grid stops within half a step of an anchor are dropped, so
    two indistinguishable stops never sit 1px apart."""
    anchors = sorted({cohort_prior(k, sel_panel) for k in COHORTS})
    grid = [round(PREV_MIN + i * PREV_STEP, 4) for i in range(int(round((PREV_MAX - PREV_MIN) / PREV_STEP)) + 1)]
    return tuple(sorted(anchors + [g for g in grid if all(abs(g - a) > PREV_STEP / 2 for a in anchors)]))


def anchor_rail(sel_panel, opts, p_sel):
    """The three known prevalences, drawn on the track the thumb travels along: this is what keeps the control
    from being the bare percentage slider it replaces, and what keeps the high-prevalence end labelled as this
    panel's own curated training data rather than as a setting available to us.

    Positions are computed from the option INDEX, not from the value: the thumb travels in equal steps through
    the option list, and the three inserted anchors make that list very slightly non-uniform in value. The mark
    the reader sees therefore sits exactly under the thumb that selects it.

    The labels are a flex row whose slots are the gaps between the marks, so each name starts at its own mark
    and no two can overlap. Where the last mark is close to the right edge (PAH's curated set is at 83.7% of a
    90% axis) its name cannot shrink below its text, so flexbox takes the room out of the wide slot in the
    middle instead and the name hangs left of its mark rather than off the end of the rail.

    The rail also carries the band's whole answer table (`data-prev`), which is what lets PREV_JS repaint the
    bars, the cards, the label and the curve's rule while the thumb is moving — see prev_answers()."""
    n, marks = len(opts) - 1, [(k, cohort_prior(k, sel_panel)) for k in COHORTS]
    fracs = [opts.index(v) / n for _, v in marks]
    slots = [b - a for a, b in zip(fracs, fracs[1:])] + [1 - fracs[-1]]
    ticks = "".join(f"<i style='left:{f * 100:.3f}%'></i>" for f in fracs)
    # The mark the thumb is standing on takes the ink: a named set is being read, not a point between two of
    # them. It is a colour change and not a weight change, so nothing on the rail moves when the thumb arrives.
    names = "".join(f"<span class='rk{' on' if v == p_sel else ''}' style='flex-basis:{s * 100:.3f}%'>{E(t('coh_' + k))}</span>"
                    for (k, v), s in zip(marks, slots))
    return (f"<div class='rail small' data-prev='{E(prev_answers(sel_panel, opts))}'>"
            f"<span class='rt'>{ticks}</span><span class='rl'>{names}</span></div>")


def prev_answers(sel_panel, opts):
    """Every value the band can display, one row per stop the control can stand on, as JSON on the rail.

    The point of a table rather than a formula is that there is then only ONE implementation of the
    arithmetic. A live drag has to repaint faster than a server round trip, and the obvious way — send the
    model's constants and let the browser evaluate f1_at_prior — puts a second copy of this project's
    arithmetic in JavaScript, where it can drift from the Python one without anybody noticing. The control has
    at most ~180 stops, so Python can simply answer all of them in advance: the browser looks a row up and
    writes it out, and what it writes is by construction what the server would have rendered on release.

    Recall is one string, not a column, for the same reason it is labelled `fixed`: there is nothing here for
    it to vary with, and the live path writes that same string back on every move."""
    c, rows = CANON[sel_panel], []
    for v in opts:
        precision, recall, f1 = f1_at_prior(c["TPR"], c["FPR_anchor"], v)
        rows.append(list(outcome_counts(v, recall, c["FPR_anchor"]))
                    + [f"{precision:.3f}", f"{f1:.3f}", percent(v), round(v, 6)])
    return json.dumps({"rows": rows,
                       "anch": [opts.index(cohort_prior(k, sel_panel)) for k in COHORTS],
                       "lab": t("op_cohort") + " · ", "recall": f"{c['TPR']:.3f}", "dom": list(PREV_DOMAIN),
                       "cap": t("pk_cap", p="{p}", b="{b}"),
                       "seg": {k: t("pk_" + k, n="{n}") for k in PK_CELLS},
                       "key": {k: t("pk_short_" + k, n="{n}") for k in PK_CELLS[:3]}},
                      ensure_ascii=False, separators=(",", ":"))


def prevalence_chart(sel_panel, p_sel):
    rows = [{"panel": t("panel_" + p), "prev": pi, "f1": f1_at_prior(CANON[p]["TPR"], CANON[p]["FPR_anchor"], pi)[2], "sel": p == sel_panel}
            for p in PANEL_ORDER for pi in np.linspace(0.01, 0.9, 160)]   # to 0.9: the curated set is a panel's training prior, and PAH's is 83.7%
    d = pd.DataFrame(rows)
    base = alt.Chart(d).encode(x=alt.X("prev:Q", title=t("op_x"), scale=alt.Scale(domain=list(PREV_DOMAIN)),
                                     axis=alt.Axis(format="%", tickCount=9)),
                               y=alt.Y("f1:Q", title="F1", scale=alt.Scale(domain=[0, 1])), detail="panel:N")
    lines = base.mark_line().encode(color=alt.condition(alt.datum.sel, alt.value(INK), alt.value(FAINT)),
        strokeWidth=alt.condition(alt.datum.sel, alt.value(3), alt.value(1.5)),
        tooltip=[alt.Tooltip("panel:N", title=t("op_panel")), alt.Tooltip("prev:Q", title=t("op_x"), format=".1%"),
                 alt.Tooltip("f1:Q", title="F1", format=".3f")])
    # Stagger direct labels along the curves instead of colliding at their converging ends, and give each label the
    # emphasis of its own curve: where the curves converge, a uniformly dark label reads as belonging to the bold one.
    label_rows = [dict(x=x, y=f1_at_prior(CANON[p]["TPR"], CANON[p]["FPR_anchor"], x)[2], name=t("short_" + p), sel=p == sel_panel)
                  for p, x in zip(PANEL_ORDER, (.78, .40, .58, .88))]   # spread over the wider domain: at the old spacing PAH and general sat 47px apart on converging curves
    ends = alt.Chart(pd.DataFrame(label_rows)).mark_text(align="right", dy=-8, fontSize=TXT).encode(
        x="x:Q", y="y:Q", text="name:N",
        color=alt.condition(alt.datum.sel, alt.value(INK), alt.value(FAINT)))
    marks = pd.DataFrame([dict(prev=TEST_PRIORS[p], f1=CANON[p]["F1"], panel=t("panel_" + p), sel=p == sel_panel) for p in PANEL_ORDER])
    pts = alt.Chart(marks).mark_point(shape="diamond", size=H(90), filled=True, stroke=MUTED).encode(
        x="prev:Q", y="f1:Q", fill=alt.condition(alt.datum.sel, alt.value(INK), alt.value("white")),
        tooltip=[alt.Tooltip("panel:N", title=t("op_panel")), alt.Tooltip("prev:Q", format=".1%", title=t("op_marker")),
                 alt.Tooltip("f1:Q", title="F1", format=".3f")])
    # The rule alone: its reading was a word-for-word copy of the metric card beside it at every position.
    # (Marking the other two sets on the curve as well was tried and dropped: unlabelled they are dots, and
    # labelled or not they land on the panel names at some panel — PAH's curated set at 83.7% sits under the
    # CFTR label. Which set is chosen is the control's job; the rule only has to say where it falls.)
    rule = alt.Chart(pd.DataFrame({"x": [p_sel]})).mark_rule(color=INK, strokeDash=[6, 4]).encode(x="x:Q", tooltip=alt.value(None))
    return styled((lines + ends + pts + rule).properties(height=H(BAND)))


def outcome_bars(p_sel, tpr, fpr):
    """The per-1000 sentence as a picture, because it was already a confusion matrix in prose. Two bars on one
    1000-variant scale — what the model calls pathogenic, what it leaves benign — each split by what those
    variants actually are: solid means the call was right, hatched means it was wrong, in the red and blue the
    rest of the app already reads as pathogenic and benign.

    It is drawn beside the F1 because it carries what a rising F1 hides: the top bar's solid-red share IS the
    precision, so a set with more pathogenic variants in it visibly reddens the bar — and the same move makes
    the bottom bar's red segment, the variants this model misses, bigger every time. Nothing about the model
    improved between two of these pictures."""
    tp, fn, fp, tn = outcome_counts(p_sel, tpr, fpr)
    # data-cell on every part a value reaches: the live drag (PREV_JS) writes these same cells while the thumb
    # is moving, and a named hook is what keeps that repaint pointed at exactly what the server rendered.
    def seg(cell, n):   # a shape carrying a fact: named for a screen reader, as the review flags' cells are
        phrase = E(t("pk_" + cell, n=n))
        return (f"<i class='pk-{cell}' data-cell='{cell}' style='width:{n / 10}%' role='img'"
                f" aria-label='{phrase}' data-tip='{phrase}'></i>")
    row = lambda label, cells: (f"<div class='pk-row'><span class='pk-lab'>{E(label)}</span>"
                                f"<span class='pk-track'>{''.join(seg(*cell) for cell in cells)}</span></div>")
    key = "".join(f"<span data-cell='{cell}' aria-description='{E(t('pk_' + cell, n=n))}' data-tip='{E(t('pk_' + cell, n=n))}'>"
                  f"<i class='pk-{cell}'></i><span class='pkv'>{E(t('pk_short_' + cell, n=n))}</span></span>"
                  for cell, n in (("tp", tp), ("fn", fn), ("fp", fp)))
    return (f"<div class='pk small'><div class='pk-cap'>{E(t('pk_cap', p=tp + fn, b=fp + tn))}</div>"
            + row(t("pk_pos"), (("tp", tp), ("fp", fp)))
            + row(t("pk_neg"), (("fn", fn), ("tn", tn)))
            + f"<div class='pk-key'>{key}</div></div>")


def outcome_counts(p_sel, tpr, fpr):
    """The four cells of the confusion matrix per 1000 variants screened. The bars, the answer table the live
    drag reads and anything else that counts variants come through here, so there is one rounding rule."""
    tp, fn = round(1000 * p_sel * tpr), round(1000 * p_sel * (1 - tpr))
    fp = round(1000 * (1 - p_sel) * fpr)
    return tp, fn, fp, 1000 - tp - fn - fp


def percent(value, decimals=1):
    return f"%{value * 100:.{decimals}f}" if lang == "tr" else f"{value:.{decimals}%}"


def others(blend):
    """Non-LightGBM members of a blend, read from the artifact."""
    return (["RF", "CatBoost"] + (["XGBoost"] if blend.get("xgb_model") else [])) if blend else []


def importance_note(art):
    """Which members the gain is read from, and on a transfer panel whose importance it therefore mostly is.
    "Only" already says the blend's other members are not in it; naming them again was a second sentence."""
    own, ta, w = art["own"], (art["transfer"] or {}).get("art"), art["transfer_w"]
    members = []
    if own:
        members.append((percent(1 - w, 0) + " " if ta else "") + t("imp_own"))
    if ta:
        members.append((percent(w, 0) + " " if own else "") + t("imp_transfer"))
    return " ".join([t("imp_note", members=" + ".join(members))] + ([t("imp_general")] if not own else []))


def pipeline_html(panel, art):
    """Widths encode weights; parallel lanes join only after their actual Platt boundary."""
    own, transfer = art["own"], art["transfer"]
    lanes = []
    for kind, model in ([("own", own)] if own else []) + (
            [("transfer", transfer["art"])] if transfer else []):
        is_own = kind == "own"
        prep = model["prep_art"] if is_own else model
        ops = []
        if prep.get("kind") == "median":
            ops = [t("flow_median"), t("flow_interactions")]
            if prep.get("sa"):
                ops.append(t("flow_agreement"))
            ops.append("mRMR")
        else:
            ops = [t("flow_keep_nan")]
            for key, label in (("pop_grids", "flow_rank"), ("cons_clip", "flow_clip"), ("miss_cols", "flow_missing")):
                if prep["ppf"].get(key):
                    ops.append(t(label))
            if prep.get("alf"):
                ops.append(t("flow_flags"))
        if not is_own and transfer.get("target_grids"):
            ops.append(t("flow_target"))
        models = model.get("models") or [model["model"]]
        names = ["LightGBM"] + others(model.get("blend"))
        weights = [1 / len(names)] * len(names) if is_own or not model.get("blend") else model["blend_weights"]
        members = []
        for i, (name, w) in enumerate(zip(names, weights)):
            n = len(models) if i == 0 else 1
            feats = len(models[0].booster_.feature_name()) if i == 0 else len(prep["feats"])
            members.append(f"<div class='flow-member' style='flex:{w:.8f}' data-weight='{w:.8f}'>"
                           f"<b data-tip='{n} {E(t('seeds'))}'>{E(name)} ×{n} <span>{E(percent(w, 0))}</span></b>"
                           f"{('<small>' + E(str(feats) + ' ' + t('features') + ' → Platt') + '</small>') if is_own else ''}</div>")
        provenance = ""
        if not is_own:
            sources = " + ".join(t("short_" + source) for source in model["sources"])
            # n_excluded counts target IDs, not the number of source rows removed, so the ID count alone made this
            # a subtraction that does not close (2815 - 367 != 2562). The rows those IDs actually matched are the
            # difference between the source panels' fitted row count and the scorer's: both sides load the same
            # panel CSV through drop_all_nan_rows, so it is a row count, not an estimate (253 / 244 / 77, each
            # verified as an isin count against the General training frame). Drawn beside the ID count, the
            # arithmetic closes. A source panel whose file is absent has no row count, and then none is drawn.
            fitted = [training_rows(source) for source in model["sources"]]
            matched = sum(f["n"] for f in fitted) - model["n_train"] if all(fitted) else None
            provenance = (f"<div class='flow-training'><b>{E(sources)}</b> ∖ "
                          f"<span class='flow-excluded'><s>{E(t('short_' + panel))}</s> {model['n_excluded']} {E(t('flow_excluded'))}"
                          f"{' · ' + E(t('flow_matched', n=matched)) if matched is not None and matched >= 0 else ''}</span>"
                          f" → <b>{model['n_train']} {E(t('flow_train'))}</b> · {len(model['feats'])} {E(t('features'))}</div>")
        calibration = "" if is_own else (f"<span class='flow-platt' tabindex='0' data-tip='{E(t('flow_target_fit'))}' "
                                          f"aria-description='{E(t('flow_target_fit'))}'>→ Platt</span>")
        lanes.append(f"<div class='flow-lane' data-branch='{kind}'>"
                     f"<div class='flow-prep'>" + "".join(f"<span>{E(op)}</span>" for op in ops) + "</div>"
                     f"<div class='flow-model'>{provenance}<div class='flow-members'>{''.join(members)}{calibration}</div></div></div>")
    blend = ""
    if own and transfer:
        blend = "<div class='flow-mix'>" + "".join(
            f"<span style='flex:{w:.8f}' data-weight='{w:.8f}'>{E(t('flow_' + name))} {E(percent(w, 0))}</span>"
            for name, w in (("own", 1-art["transfer_w"]), ("transfer", art["transfer_w"]))) + "</div>"
    chemistry = " · ".join(name for feature, name in (("AA_blosum62", "BLOSUM62"), ("AA_grantham", "Grantham"))
                           if feature in art["feature_cols"])
    threshold = art["threshold"]
    return (f"<div class='flow' role='group' aria-label='{E(t('flow_title'))}'>"
            f"<div class='flow-input'><b class='flow-file'>CSV</b><span>AA_1 → AA_2</span><b>{E(t('flow_chem'))}</b><small>{E(chemistry)}</small></div>"
            f"<div class='flow-pipeline'><div class='flow-head'><span>{E(t('flow_prep'))}</span><span>{E(t('flow_models'))}</span></div>"
            f"{''.join(lanes)}</div><div class='flow-decision'>{blend}"
            f"<b>{E(t('threshold'))}</b><div class='flow-scale' style='--threshold:{threshold*100:.8f}%' "
            f"role='img' aria-label='{E(t('threshold'))}: {threshold:.2f}; 0–{threshold:.2f} {E(t('benign'))}; ≥{threshold:.2f} {E(t('pathogenic'))}'>"
            f"<span class='flow-tick{' flip' if threshold > TICK_FLIP else ''}'>{threshold:.2f}</span></div>"
            f"<div class='flow-outcomes'><span>0 · {E(t('benign'))}</span><span>{E(t('pathogenic'))} · 1</span></div>"
            f"<small>{E(t('prior_test'))} {E(percent(TEST_PRIORS[panel]))}</small></div></div>")


# ── page ──────────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PathoSense", page_icon="🧬", layout="wide")
# The wordmark is the team's script mark, traced to a vector from the title card of the brand film so it
# stays sharp at any size; its fill is BRAND.
st.logo(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.svg"), size="large")
# The language is chosen when the interface is launched, not in the UI: PATHOSENSE_LANG=en for English,
# Turkish otherwise. It cannot change within a session, so widget keys are plain (no language scope).
st.session_state.setdefault("lang", "en" if os.environ.get("PATHOSENSE_LANG", "").lower().startswith("en") else "tr")
lang = st.session_state["lang"]

DESIGN_H = 940          # fallback viewport before the browser probe answers
FOLD_H = 768            # preserve the presentation layout and its cancer Overview safety margin
# The root is fixed, not fluid. What binds each page to the fold is the flex-fill chain and the results
# table's container query, never the reading size: across every viewport this product is specified for
# (1366 to ~1706) a bounded clamp moved the root by one pixel, while its own floors pinned the small role
# and let the body role float, leaving the two 0.25px apart. Compact geometry is authored against a 16px
# root, so H() carries the ratio (see below); the type roles are written in px, once, in the CSS block.
ROOT_PX = 13
VIEWPORT_JS = """(function () {
  if (!window.__psResize) {
    window.__psResize = true;
    window.parent.addEventListener('resize', function () {
      clearTimeout(window.__psT);
      window.__psT = setTimeout(function () { sendDataToPython({value: window.parent.innerWidth + 'x' + window.parent.innerHeight, dataType: 'json'}); }, 250);
    });
  }
  return window.parent.innerWidth + 'x' + window.parent.innerHeight;
})()"""

# The way into Variant review is a click on a row of the results table. That table is the app's own HTML in
# the main document, so the listener is one delegated handler on that document, installed from the probe
# component — a component's iframe is the only route a click has back to Python. An anchor would be simpler
# and is wrong: following a link is a page load, a page load is a new session, and the jury's uploaded CSV
# would be gone. Two rules the browser forces: a remounted iframe leaves a handler whose realm is detached
# and inert, so every mount replaces the previous one; and the value is an event, not state (take_row_click).
ROW_CLICK_JS = """(function () {
  var p = window.parent, d = p.document, h = p.history;
  p.__psClicks = p.__psClicks || 0;
  /* One address per screen. The framework pushes a history entry for every query-string write, so the
     variant page's own address (?panel=&variant=, rewritten on every step) became a trail Back walked
     through one entry at a time without ever leaving the page. A write that keeps the path is the same
     screen's address and replaces the entry; a new path is a page and still pushes. The wrapper is made in
     the parent's realm (p.Function) so it outlives this iframe, and is installed once. */
  if (!h.__psPush) {
    h.__psPush = true;
    h.pushState = new p.Function('push', 'replace',
      'return function (s, t, u) { if (u == null) return push.call(history, s, t, u);' +
      ' return (new URL(u, location.href).pathname === location.pathname ? replace : push).call(history, s, t, u); }'
    )(p.History.prototype.pushState, p.History.prototype.replaceState);
  }
  /* The list as it was left: the results table's scroll offset is kept on the way into a variant — a row
     click, the header's link, or the nav's entry — and put back when the table is next drawn, since a route rebuilds the
     page's DOM and a scroll offset is nothing the framework keeps. If the pick moved meanwhile (the
     steppers), the marked row is brought into the middle of the view instead. Only the return restores:
     a filter change also redraws the table, and starts it at the top as before. The way back is a page
     link too, clicked where there is no table: it must not overwrite the offset kept on the way in. */
  function keepScroll() { var tw = d.querySelector('.tablewrap.tall'); if (!tw) return; p.__psReturn = true; p.__psTableScroll = tw.scrollTop; }
  if (p.__psRowHandler) d.removeEventListener('click', p.__psRowHandler);
  p.__psRowHandler = function (ev) {
    if (!ev.target.closest) return;
    if (ev.target.closest('[data-testid="stPageLink"] a, [data-testid="stTopNavLink"][href$="/explain"]')) keepScroll();   /* the header's link, or the nav's entry for the same page */
    var row = ev.target.closest('tr[data-vid]');
    if (!row) return;
    keepScroll();
    p.__psClicks += 1;
    sendDataToPython({value: {vid: row.getAttribute('data-vid'), n: p.__psClicks}, dataType: 'json'});
  };
  d.addEventListener('click', p.__psRowHandler);
  if (p.__psTableWatch) p.__psTableWatch.disconnect();
  function settle(tw, saved) {
    if (saved != null) tw.scrollTop = saved;
    var now = tw.querySelector('tr.now'), head = tw.querySelector('th');
    if (!now) return;
    var r = now.getBoundingClientRect(), b = tw.getBoundingClientRect();
    if (r.top < b.top + (head ? head.offsetHeight : 0) || r.bottom > b.bottom) tw.scrollTop = now.offsetTop - (tw.clientHeight - now.offsetHeight) / 2;
  }
  p.__psTableWatch = new p.MutationObserver(function () {
    var tw = p.__psReturn ? d.querySelector('.tablewrap.tall') : null;
    if (!tw) return;
    p.__psReturn = false;
    var saved = p.__psTableScroll; p.__psTableScroll = null;
    settle(tw, saved);   /* now, so the list does not flash at the top ... */
    p.setTimeout(function () { settle(tw, saved); }, 250);   /* ... and once more when the fold has given the table its height, or the offset is clamped */
  });
  p.__psTableWatch.observe(d.body, {childList: true, subtree: true});
  /* Escape leaves the variant the way a drawer closes: through the page's own way back (the "← Tahminler"
     link, a route inside the session), so the batch, the filter and the marked row are all still there.
     Not while a list is open under the key — seen in the capture phase, before the list's own handler has
     closed it — and not on the press that dismissed a tooltip (TIP_JS, which runs first). */
  if (p.__psEsc) d.removeEventListener('keydown', p.__psEsc, true);
  p.__psEsc = function (ev) {
    if (ev.key !== 'Escape' || ev.defaultPrevented || ev.psTip || d.querySelector('[role="listbox"]')) return;
    var back = d.querySelector('.st-key-review_head [data-testid="stPageLink"] a');
    if (back) back.click();
  };
  d.addEventListener('keydown', p.__psEsc, true);
  return null;
})()"""

# The app's own tooltip, for every fact the interface hangs off a mark of its own — the ⓘ after a heading, a
# row's flag symbols, a review rule, a node of the flow diagram. A native title= is drawn by the operating
# system in its own face, at its own delay and in its own box, so it could never match the framework's help
# tooltips or the charts'; this one is a single element in the main document, styled by the same rule as
# those two (CSS #ps-tip). It escapes any clipping container, which a CSS ::after on a cell inside the
# scrolling results table could not, and it keeps the viewport: centred under its anchor, above it when the
# anchor is near the fold, and never past either edge. Timing is the framework's own — open after TIP_OPEN,
# stay TIP_CLOSE after the pointer leaves, open and close at once on focus — so the three surfaces behave
# as one. Installed like the click bridge: from the probe, replacing the previous mount's handlers.
TIP_OPEN, TIP_CLOSE = 200, 300   # ms: what the framework's help tooltips use (Tooltip.tsx delay/closeDelay)
TIP_JS = """(function () {
  var p = window.parent, d = p.document, OPEN = %d, CLOSE = %d;
  var tip = d.getElementById('ps-tip');
  if (!tip) { tip = d.createElement('div'); tip.id = 'ps-tip'; tip.setAttribute('role', 'tooltip'); d.body.appendChild(tip); }
  var old = p.__psTipHandlers || {};
  Object.keys(old).forEach(function (k) { d.removeEventListener(k, old[k], true); });
  var anchor = null, timer = null, shown = false;
  function clear() { if (timer) { p.clearTimeout(timer); timer = null; } }
  function place() {
    var r = anchor.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight, m = 8, gap = 6;
    var x = r.width > w ? r.left : r.left + r.width / 2 - w / 2;   /* centred on a mark; flush left under a full-width line */
    x = Math.max(m, Math.min(x, p.innerWidth - w - m));
    var y = r.bottom + gap;
    if (y + h > p.innerHeight - m) y = r.top - h - gap;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hide() {
    clear();
    if (anchor) anchor.removeAttribute('aria-describedby');
    anchor = null; shown = false;
    tip.classList.remove('on');
  }
  function reveal(el) {
    clear();
    if (!el.isConnected) { anchor = null; return; }
    anchor = el; shown = true;
    tip.textContent = el.getAttribute('data-tip');
    tip.classList.add('on');
    place();
    el.setAttribute('aria-describedby', 'ps-tip');
  }
  function open(el) { clear(); anchor = el; timer = p.setTimeout(function () { timer = null; reveal(el); }, OPEN); }
  function leave() { clear(); if (shown) timer = p.setTimeout(hide, CLOSE); else anchor = null; }
  function at(ev) { return ev.target && ev.target.closest ? ev.target.closest('[data-tip]') : null; }
  var on = {
    pointerover: function (ev) {
      var el = at(ev);
      if (!el) { if (anchor && !timer) leave(); return; }   /* an anchor a rerun removed under the pointer */
      if (el === anchor) { if (shown) clear(); else if (!timer) open(el); return; }
      if (shown) hide();
      open(el);
    },
    pointerout: function (ev) {
      var el = at(ev);
      if (!el || el !== anchor || (ev.relatedTarget && el.contains(ev.relatedTarget))) return;
      leave();
    },
    focusin: function (ev) { var el = at(ev); if (el) reveal(el); },
    focusout: function (ev) { if (at(ev) === anchor) hide(); },
    keydown: function (ev) { if (ev.key === 'Escape') { if (shown) ev.psTip = true; hide(); } },   /* the flag: this press was spent here */
    scroll: function () { if (shown) hide(); }
  };
  Object.keys(on).forEach(function (k) { d.addEventListener(k, on[k], true); });
  p.__psTipHandlers = on;
  return null;
})()""" % (TIP_OPEN, TIP_CLOSE)


# Overview's prevalence control repaints its own band while the thumb is moving. A Streamlit widget only
# reruns on release, so without this the curve, the bars and the cards all lag a drag by a round trip; with
# it the reader drags and the picture follows, which is the whole reason the control is continuous again.
#
# Python stays authoritative and stays the only place the arithmetic lives: every stop's answer is computed
# server-side and carried on the rail as `data-prev` (prev_answers), so this handler looks a row up and writes
# it out rather than evaluating f1_at_prior a second time in another language. On release the normal rerun
# re-renders the same values from the server.
#
# Two browser facts shape it. The thumb is a react-aria control: a pointer drag never fires an `input` event,
# it just writes the thumb's inline `left`, so the live signal is a MutationObserver on that style, attached
# only between pointerdown and pointerup and reading react-aria's own answer rather than re-deriving it from
# the pointer. Keyboard moves DO fire `input`, and are handled on the same path. And, as with the row-click
# bridge, a remounted probe iframe leaves a handler in a detached realm: every mount replaces the previous.
PREV_JS = r"""(function () {
  var p = window.parent, d = p.document;
  var old = p.__psPrev || {};
  Object.keys(old).forEach(function (k) { d.removeEventListener(k, old[k], true); });
  if (p.__psPrevObs) { p.__psPrevObs.disconnect(); p.__psPrevObs = null; }

  function cfg() {   /* the answer table as the last rerun left it; re-parsed only when the server changed it */
    var el = d.querySelector('.rail[data-prev]');
    if (!el) return null;
    if (el.__psRaw !== el.dataset.prev) { el.__psRaw = el.dataset.prev; el.__psCfg = JSON.parse(el.dataset.prev); }
    el.__psCfg.rail = el;
    return el.__psCfg;
  }
  function fill(s, o) { return s.replace(/\{(\w+)\}/g, function (m, k) { return o[k]; }); }
  function text(el, s) { if (el) { while (el.firstElementChild) el = el.firstElementChild; if (el.textContent !== s) el.textContent = s; } }
  function say(el, s) { if (el) { el.setAttribute('data-tip', s); el.setAttribute(el.hasAttribute('aria-label') ? 'aria-label' : 'aria-description', s); } }

  /* The curve's rule, moved in the SVG the chart already rendered: there is no handle on the Vega view from
     the page, but the axis is drawn from the same domain the server emitted, so its first and last gridline
     give the pixels of both ends of that domain and the rule can be placed between them. Anything unreadable
     here is left alone — the rerun on release draws it correctly. */
  function moverule(c, pi) {
    var svg = d.querySelector('[data-testid="stVegaLiteChart"] svg');
    var line = svg && svg.querySelector('g.mark-rule.role-mark line');
    if (!line) return;
    var lo = Infinity, hi = -Infinity, g = svg.querySelectorAll('g.mark-rule.role-axis-grid line'), n = 0;
    for (var i = 0; i < g.length; i++) {
      var r = g[i].getBoundingClientRect();
      if (r.width >= 1) continue;   /* the horizontal gridlines of the y axis */
      lo = Math.min(lo, r.left); hi = Math.max(hi, r.left); n++;
    }
    if (n < 3 || hi - lo < 1) return;
    var off = line.getBoundingClientRect().left - line.x1.baseVal.value;   /* the group's own translation, read off the mark itself */
    var x = lo + (pi - c.dom[0]) / (c.dom[1] - c.dom[0]) * (hi - lo) - off;
    line.setAttribute('x1', x); line.setAttribute('x2', x);
  }

  function paint(i) {
    var c = cfg(); if (!c) return;
    var r = c.rows[i]; if (!r) return;
    var n = {tp: r[0], fn: r[1], fp: r[2], tn: r[3]};
    text(d.querySelector('.st-key-prev_w label p'), c.lab + r[6]);
    var marks = c.rail.querySelectorAll('.rk');
    for (var k = 0; k < marks.length; k++) marks[k].classList.toggle('on', c.anch[k] === i);
    var pk = d.querySelector('.pk'); if (!pk) return;
    text(pk.querySelector('.pk-cap'), fill(c.cap, {p: n.tp + n.fn, b: n.fp + n.tn}));
    Object.keys(n).forEach(function (cell) {
      var seg = pk.querySelector('.pk-track [data-cell="' + cell + '"]'), phrase = fill(c.seg[cell], {n: n[cell]});
      if (seg) { seg.style.width = (n[cell] / 10) + '%'; say(seg, phrase); }
      var item = pk.querySelector('.pk-key [data-cell="' + cell + '"]');
      if (item) { text(item.querySelector('.pkv'), fill(c.key[cell], {n: n[cell]})); say(item, phrase); }
    });
    /* Recall is written back as the one string the server emitted, not recomputed: the card that must not
       move cannot move, even on the live path. Precision and F1 are the row's own. */
    var cards = d.querySelectorAll('.st-key-overview_explorer [data-testid="stMetricValue"]');
    text(cards[0], c.recall); text(cards[1], r[4]); text(cards[2], r[5]);
    moverule(c, r[7]);
  }

  function thumb() { return d.querySelector('.st-key-prev_w [data-orientation] > div > div[style*="left"]'); }
  function here() {
    var el = thumb(), input = d.querySelector('.st-key-prev_w input[type="range"]');
    if (!el || !input) return -1;
    var pct = parseFloat(el.style.left);   /* react-aria writes the thumb's position as index/max, so this IS the index */
    return isNaN(pct) ? -1 : Math.round(pct / 100 * (+input.max || 1));
  }
  function stop() { if (p.__psPrevObs) { p.__psPrevObs.disconnect(); p.__psPrevObs = null; } }
  var on = {
    pointerdown: function (ev) {
      stop();   /* first, not only on release: a pointer that left the window never sent one */
      if (!ev.target || !ev.target.closest || !ev.target.closest('.st-key-prev_w')) return;
      var el = thumb(); if (!el) return;
      p.__psPrevObs = new p.MutationObserver(function () { var i = here(); if (i >= 0) paint(i); });
      p.__psPrevObs.observe(el, {attributes: true, attributeFilter: ['style']});
    },
    pointerup: stop,
    pointercancel: stop,
    input: function (ev) { if (ev.target && ev.target.closest && ev.target.closest('.st-key-prev_w')) paint(+ev.target.value); }
  };
  Object.keys(on).forEach(function (k) { d.addEventListener(k, on[k], true); });
  p.__psPrev = on;
  return null;
})()"""


def take_row_click(value, ss):
    """The one new row click, or None. The component re-sends its last value on every later rerun and reports
    None after a remount, so the nonce — not the variant — says whether this run was caused by a click: the
    same row clicked twice in a row still drills in, and a re-sent value never does."""
    if isinstance(value, dict) and value.get("vid") and value.get("n") != ss.get("click_seen"):
        ss["click_seen"] = value["n"]
        return str(value["vid"])
    return None


# Two tabs and a subpage. Variant review is Tahminler' child: a nav entry of its own, so the drill-down is
# reachable from the band as well as from a row and from the table's header, but drawn nested under the
# tab it belongs to (CSS: the › marker, the smaller role, the closed gap) rather than as a third peer.
PAGES = [("overview", "overview", ":material/monitoring:", "visible"),
         ("pred", "predictions", ":material/table_chart:", "visible"),
         ("explain", "explain", ":material/manage_search:", "visible")]
ss = st.session_state
res = batch_error = panel = art = thr = pi_test = c = own_w = overlap = in_sample = None
# A fresh session — a reloaded or shared address — takes the panel the address names, before anything reads
# panel_keep. Later, the session's own choice wins and the address follows it (render_explain).
if "panel_keep" not in ss and st.query_params.get("panel") in PANEL_ORDER:
    ss["panel_keep"] = st.query_params["panel"]
with st.container(key="probe"):   # taken out of flow by CSS: it must run before the scale factor below
    _vh = streamlit_js_eval(js_expressions=VIEWPORT_JS, key="vh")
    # One probe installs all three document-level handlers: the app's tooltip, the prevalence band's live
    # repaint, and the row-click bridge. The last expression is the component's value, so the bridge's (null)
    # goes last.
    _click = streamlit_js_eval(js_expressions=TIP_JS + ";" + PREV_JS + ";" + ROW_CLICK_JS, key="rowclick")

try:
    VW, VH = (float(v) for v in str(_vh).split("x"))
    st.session_state["viewport"] = (VW, VH)
except (ValueError, TypeError):
    # the probe answers None on the rerun a button triggers; keep the last measurement instead of
    # falling back to the design size, which would rescale the whole page on every click
    VW, VH = st.session_state.get("viewport", (1920.0, float(DESIGN_H)))
FILL_CHARTS = VW > NARROW and VH > FOLD_H


def H(px: float) -> int:
    """Compact layout size; flexible subject charts can grow beyond this minimum. The named heights above
    are design pixels at a 16px root, and the root is ROOT_PX."""
    return int(px * ROOT_PX / 16)


def chart_rows(n: int, pitch: int, axis: int = 0) -> int:
    """Height of a horizontal bar chart with `n` bands, plus the x-axis furniture under it."""
    return max(H(pitch * n + axis), MIN_BAND * n + axis)


# Vega draws to a canvas and cannot read a CSS variable, so chart text repeats the --fs-sm rule: the small
# roles carry the evidence (contribution values, feature names) and are drawn at the same 12px.
TXT = 12

# One custom property per mark, so the set is written once and a rule only names the mark it draws.
MARK_RULES = "\n  ".join(f".mk-{name} {{--mk:{mark_url(name)}}}" for name in MARKS)
CHEVRON = mark_url("chevron")

st.html(f"""<style>
  html {{font-size:{ROOT_PX}px}}
  /* One type scale: every text role is --fs-sm (captions, control and metric labels, tags, the flow
     diagram, chart legends), --fs (body: tables and band headings) or --fs-lg (the one display size: a
     metric value, a variant id). Three fixed sizes, 12 → 13.5 → 20, so the small role and the body role
     carry different jobs at every width; a product UI is read at one DPI and gains nothing from a
     reading size that follows the window.
     --row-* are the results table's own metrics; its leading is pinned, because table density is a
     density decision and not a type role.
     One spacing scale, in rem so it follows the root font: 0.125 0.25 0.375 0.5 0.75 1 2 — every padding,
     margin and gap below is one of those seven, and nothing else is. (Widths and drawn shapes — swatches,
     bars, arrows — are sizes, not rhythm, and are set where they are drawn.) Two radii: --r for a box,
     --r-sm for a chip, a bar or a swatch. One border colour: --line. */
  :root {{--surf:{SURF}; --line:{LINE}; --ink:{INK}; --muted:{MUTED}; --brand:{BRAND}; --r:0.5rem; --r-sm:0.25rem; --header:2.5rem;
    --fs-sm:12px; --lh-sm:16px;
    --fs:13.5px; --lh:17px;
    --fs-lg:20px; --lh-lg:24px;
    --row-fs:var(--fs); --row-lh:16px; --row-pad:0.375rem;
    --export-w:9rem}}   /* the foot's export: a 1rem icon, a 0.25rem gap, its name in the small role and the pill's own padding and line (measured 113.6px in Turkish, 106px in English) */
  /* The nav bar lives in Streamlit's own header, which paints over the page: the block clears it and no more. */
  [data-testid="stHeader"] {{height:var(--header); min-height:var(--header)}}
  [data-testid="stHeaderLogo"] {{height:2rem}}   /* a script mark needs the height a sans did not: its x-height is a third of the box */
  [data-testid="stMain"] .block-container {{padding:calc(var(--header) + 0.25rem) 2rem 0.5rem}}
  [data-testid="stMain"] [data-testid="stVerticalBlock"] {{gap:0.375rem}}   /* the three pages are read from a screenshot: the rhythm is tuned to fit 1366x768 */

  /* Two kinds of modal reach this page and only one of them is the app's. The framework's is its English
     "page not found", raised by a deep link to a path that does not route and falling back to Overview
     anyway: it stays suppressed. The app's own is the upload dialog, and it is known by the container it
     draws its controls in, so the rule keys off that marker rather than off the framework's testid alone. */
  [data-testid="stDialog"]:not(:has(.st-key-upload_dialog)) {{display:none}}
  [data-testid="stLogoLink"] {{pointer-events:none}}   /* st.logo turns the wordmark into a second, English-labelled route to the default page */
  /* The nav is typographic: the page you are on is underlined, the rest are plain words. */
  [data-testid="stTopNavLink"] {{background:none; border-radius:0; color:var(--muted); font-weight:500}}
  /* Hover paints, it never re-measures: the framework's own hover rule drops the weight to 400, which
     re-shapes the label under the pointer and makes the whole band feel unstable. The weight is restated
     here so the only thing hover changes is the ink. */
  [data-testid="stTopNavLink"]:hover {{background:none; color:var(--ink); font-weight:500}}
  [data-testid="stTopNavLink"][aria-current="page"] {{background:none; color:var(--ink); font-weight:600; box-shadow:inset 0 -2px var(--brand)}}
  /* Variant review is Tahminler' child, and the band draws it as one: its entry follows the tab with the
     gap closed, behind a › where the tabs carry an icon, in the tabs' muted colour but never their weight
     while another page is current — a step down from the two tabs, not a third. On /explain the framework
     marks it current, so the underline says where you are without a rule pretending otherwise. */
  [data-testid="stTopNavLinkContainer"]:has(> [data-testid="stTopNavLink"][href$="/explain"]) {{margin-left:-0.375rem}}
  [data-testid="stTopNavLink"][href$="/explain"] {{padding-left:0.25rem}}
  [data-testid="stTopNavLink"][href$="/explain"] > span:first-child {{display:none}}   /* the icon: the › is its marker */
  /* The marker is drawn, not typed: a › in generated content is also a › in the link's accessible name. */
  [data-testid="stTopNavLink"][href$="/explain"]::before {{content:""; --mk:{CHEVRON}; display:inline-block; width:10px; height:16px;
    vertical-align:middle; margin:-8px 0; background:var(--muted);
    -webkit-mask:var(--mk) center/16px 16px no-repeat; mask:var(--mk) center/16px 16px no-repeat}}   /* a narrower box than the mark's own: the chevron's side bearings are not its gap */
  /* The gene panel is global state, the same class of thing as the nav, so it sits beside it rather than
     owning a row on every page. Streamlit exposes no API for its own header, so the container is lifted
     into the band the header already reserves: fixed, right-aligned, one layer above the header's own
     z-index (999990), and short enough that the band does not grow. Nothing else is in the flow there —
     the nav ends well left of it at every width, down to the point where it collapses. */
  div:has(> .st-key-panelbar) {{position:absolute; height:0}}   /* the wrapper leaves the flow too, or it spends a flex gap on a row it no longer has */
  .st-key-panelbar {{position:fixed; top:0; right:2rem; z-index:1000000; height:var(--header); justify-content:center}}
  .st-key-panelbar, .st-key-panelbar > * {{flex:0 0 auto !important; width:auto !important}}   /* !important: the block wrapper is width:100% inline */
  /* The gene-panel segmented control in the header band. */
  .st-key-panelbar [data-testid="stButtonGroup"] {{gap:0.125rem}}
  .st-key-panelbar [data-testid="stButtonGroup"] button {{min-height:0; padding:0.125rem 0.5rem; border-radius:var(--r-sm); color:var(--muted); font-weight:500}}
  .st-key-panelbar [data-testid="stButtonGroup"] button p {{font-size:var(--fs-sm); line-height:var(--lh-sm)}}
  /* The state pseudo-classes are part of the selector, not an afterthought: the framework repaints a
     focused or hovered segment in its own blue, and after every switch the chosen segment is the focused
     one — which is exactly the state a projector shows. The selection stays the filled brand one. */
  :is(.st-key-panelbar,.st-key-upload_dialog) [data-testid="stButtonGroup"] button[aria-checked="true"],
  :is(.st-key-panelbar,.st-key-upload_dialog) [data-testid="stButtonGroup"] button[aria-checked="true"]:is(:hover,:focus,:focus-visible,:active)
    {{background:var(--brand); color:#fff}}
  /* The same control, in the dialog: it is the declaration the file is taken against, so it is named, set at
     the app's reading size and written in full panel names — the band's four short words are what fits a
     header, not what a file is declared for. It wraps rather than overflowing the dialog at any width. */
  .st-key-upload_dialog [data-testid="stButtonGroup"] {{flex-wrap:wrap; gap:0.25rem}}   /* four full names do not fit a dialog's width on one line; a grid here takes the label out of its own row */
  .st-key-upload_dialog [data-testid="stButtonGroup"] button {{min-height:2rem; padding:0.125rem 0.625rem; border-radius:var(--r-sm); color:var(--muted); font-weight:500}}
  .st-key-upload_dialog [data-testid="stButtonGroup"] button p {{font-size:var(--fs-sm); line-height:var(--lh-sm)}}
  .st-key-upload_dialog label p {{font-size:var(--fs-sm); line-height:var(--lh-sm); font-weight:500; color:var(--muted)}}
  /* The upload control in the band: a button of the segments' own height and rhythm, never a bordered
     dropzone. Its label is the state a glance has to read — the loaded file's name, or the invitation to
     choose one — and the brand colour says the batch on screen is that file and not the demo sample. */
  :is(.st-key-upload_btn,.st-key-upload_file) button {{min-height:0; padding:0.125rem 0.5rem; border-radius:var(--r-sm); color:var(--muted); font-weight:500}}
  :is(.st-key-upload_btn,.st-key-upload_file) button:hover {{color:var(--ink); background:var(--surf)}}
  :is(.st-key-upload_btn,.st-key-upload_file) button p {{max-width:10rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:var(--fs-sm); line-height:var(--lh-sm)}}
  .st-key-upload_file button {{color:var(--brand)}}
  /* No page toolbar: each of the two list controls carries its own header row instead — above the results
     table the filter that scopes it and what leaves it, above the variant picker the way back —
     and every widget is sized to its content so the row is a line of controls and not a grid.
     !important: st.container(horizontal=True) gives every child flex:1 and a computed width inline. */
  .st-key-table_head > *, .st-key-table_actions > *, .st-key-table_actions {{flex:0 0 auto !important; width:auto !important}}
  /* The filter is the table's own chrome, not a peer of the action beside it: it takes the width the action
     leaves, from the table's left edge to the button, so the row is one bar over the table and not two
     controls at either end of a hole (17rem of well and a distributed row left ~300px of it at 1706). */
  .st-key-table_head > [data-testid="stLayoutWrapper"]:has(> .st-key-filters) {{flex:1 1 auto !important; min-width:17rem}}
  /* The picker's row: the link and the stepper pair at their content width, the picker taking the rest, and
     the picker drawn as the same well as the filter over the table it came from (the framework's flat grey). */
  .st-key-review_head > * {{flex:0 0 auto !important; width:auto !important}}
  .st-key-review_head > [data-testid="stElementContainer"]:has(> [data-testid="stSelectbox"]) {{flex:1 1 auto !important; min-width:0}}
  .st-key-review_head [data-testid="stSelectbox"] div:has(> [role="combobox"]) {{background:var(--surf); border:1px solid var(--line); border-radius:var(--r)}}
  .st-key-review_head [data-testid="stSelectbox"] div:has(> [role="combobox"]):focus-within {{border-color:var(--brand)}}
  /* Dropdown density, and its one exception, stated here and nowhere else. The filter lists the app's own
     rows, so it takes the app's own density: Streamlit's 2.5rem option for 1.4rem of text reads as a looser
     product than everything under it. Its list is virtualised with a row height computed in JS, so padding
     alone changes nothing — the rows must leave their absolute placement too, which is safe only because
     the filter offers at most seven options and the virtualiser renders them all. The variant selectbox
     therefore keeps the 40px row: it windows 300 options, where the same override would leave rows
     unrendered. The row is a full-width click target either way.
     What the variant list may change is its length. Its cap is a plain max-height on the listbox, which is
     the scroll box the virtualiser measures (the framework's own is 7.5 rows or 70vh), so a taller cap is
     honoured with more rows rendered, not with blank rows: 300 options render and scroll at every height
     (checked at 1366×768 and 1706×1045). Whole rows and a half — the half is the scroll cue — up to 23.5,
     within the viewport less the band and the row above the list and a margin under it. */
  [data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {{max-height:max(18.75rem, min(58.75rem, 100vh - 12rem));
    max-height:calc(round(down, max(18.75rem, min(58.75rem, 100vh - 12rem)), 2.5rem) + 1.25rem)}}
  /* A filter chip carries a class word ("Patojenik · 176"), so it takes no colour that could be read as one: not
     the framework's accent, which is the brand teal on every other control, and not a call colour. Ink, like text. */
  [data-testid="stMultiSelectTagsContainer"] [data-tag] {{background:var(--ink)}}
  [data-testid="stMultiSelectDropdown"] :is([role="option"], [role="option"] > div) {{height:auto; min-height:0}}
  [data-testid="stMultiSelectDropdown"] [role="option"] {{padding-top:0.25rem; padding-bottom:0.25rem}}
  [data-testid="stMultiSelectDropdown"] [role="presentation"]:has([role="option"])
    {{position:relative !important; top:0 !important; contain:none !important; height:auto !important}}
  /* The uploader, the loaded-file chip and the filter are one shape wherever they are drawn: the same
     2.5rem well in the card colour with the app's one border colour. Streamlit draws them mismatched — a
     taller tinted dropzone holding a white bordered button, a box inside a box, beside a flat borderless
     combobox — and they read as unrelated widgets that happen to belong to one app. */
  [data-testid="stFileUploaderDropzone"] {{min-height:2.5rem; padding:0 0.5rem; background:var(--surf); border:1px solid var(--line); border-radius:var(--r); flex-direction:row !important; align-items:center; gap:0.5rem}}   /* one line in the toolbar, not Streamlit's stacked column */
  [data-testid="stFileUploaderDropzone"] button {{min-height:0; border:0; background:none; padding:0 0.25rem}}   /* the well is the one box; the button is its label */
  [data-testid="stFileUploaderDropzoneInstructions"] {{flex:0 1 auto}}
  .st-key-filters [data-testid="stMultiSelect"] > div > div:first-child {{background:var(--surf); border:1px solid var(--line); border-radius:var(--r)}}
  .st-key-filters [data-testid="stMultiSelect"] > div > div:first-child:focus-within {{border-color:var(--brand)}}   /* the override is the resting state only: focus stays visible, in the app's focus colour */
  /* One chosen filter must not double the well's height. BaseWeb gives its text input a 39px minimum width,
     which the widest chip in this list overshoots by a fraction of a pixel, wrapping the input under the chip
     and taking the row from 33 to 59px — which then drops the table's header row out of line with the actions
     opposite it. The input keeps whatever the chip leaves instead. */
  .st-key-filters [data-testid="stMultiSelect"] input {{min-width:0}}
  /* Streamlit ships no locale for its own widget chrome; the app already draws button labels this way (see pick_prev). */
  [data-testid="stFileUploaderDropzone"] button p, [data-testid="stFileUploaderDropzoneInstructions"] span {{font-size:0}}
  [data-testid="stFileUploaderDropzone"] button p::after {{content:"{t('upload_btn')}"; font-size:var(--fs-sm)}}
  [data-testid="stFileUploaderDropzoneInstructions"] span::after {{content:"{t('upload_limit')}"; font-size:var(--fs-sm)}}
  [data-testid="stFileUploaderDropzoneInstructions"] [data-testid="stIconMaterial"] {{display:none}}
  .st-key-chip {{box-sizing:border-box; min-height:2.5rem; padding:0 0.5rem; background:var(--surf); border:1px solid var(--line); border-radius:var(--r); white-space:nowrap}}   /* the chip stands in for the well, and is one: the slot reads the same whichever state it is in */
  .st-key-chip button {{min-height:0}}   /* as in the dropzone: the well is the box, and Remove is a word inside it */
  /* The way in and the way back ride the header row of the list each page is about — the results table
     carries "Varyant incelemesi", the variant picker carries "← Tahminler" — so neither costs a row, and
     both keep a full-height target beside a 2.5rem control. st.page_link takes no key of its own.
     The way in is the page's one primary: the drill-down is the whole explainability story, so it is the
     one filled control on the page — the brand as a block, white label, the well's radius and height — and
     stays a real link: keyboard, middle-click, a focus ring of its own. Nothing else on the page is filled.
     The way back is drawn like the tertiary Remove button: no fill, ink text, the height of the well. */
  :is(.st-key-table_actions,.st-key-review_head) [data-testid="stPageLink"] a {{min-height:2.5rem; padding:0 0.5rem; background:none; color:var(--ink)}}
  :is(.st-key-table_actions,.st-key-review_head) [data-testid="stPageLink"] a span {{font-weight:500}}
  .st-key-review_head [data-testid="stPageLink"] a:hover {{background:none; color:var(--brand)}}
  .st-key-table_actions [data-testid="stPageLink"] a {{padding:0 0.75rem; border-radius:var(--r); background:var(--brand); color:#fff}}
  .st-key-table_actions [data-testid="stPageLink"] a :is(span,p) {{color:#fff}}   /* the icon and the label carry the framework's ink inline */
  .st-key-table_actions [data-testid="stPageLink"] a span {{font-weight:600}}
  .st-key-table_actions [data-testid="stPageLink"] a:hover {{background:color-mix(in srgb, var(--brand) 85%, #000); color:#fff}}
  .st-key-table_actions [data-testid="stPageLink"] a:focus-visible {{outline:2px solid var(--brand); outline-offset:2px; box-shadow:none}}
  /* The export keeps the foot line of the table it is the export of, and is drawn there as a control: the
     brand as a line and a tint with brand ink, one step under the filled drill-down above, which stays the
     page's loudest. The button is a widget of its own, so it cannot be a child of the foot the table draws;
     it is laid over the line instead — pulled up by the line's height, the column's gap and the foot's
     bottom margin, so it adds nothing to the column — and the count leaves it the room (.key .count).
     Its box is 1.5rem against the drill-down's 2.5rem: the container is held to the foot line's own
     var(--lh-sm) and the pill's 4px over that height is paid out of the negative margins, so the column is
     no taller and the table keeps every row it had. A 2.5rem block here would not fit — see the pass
     report: with a long filter name the count takes a second foot line and there is 12px left under it. */
  .st-key-table_export {{margin-top:calc(-1 * (var(--lh-sm) + 0.25rem + 0.375rem)); height:var(--lh-sm); align-self:flex-end; position:relative; z-index:1}}
  .st-key-table_export button {{min-height:1.75rem; padding:0 0.5rem; margin:-0.125rem 0 0; gap:0.25rem;
    border:1px solid var(--brand); border-radius:var(--r-sm); background:color-mix(in srgb, var(--brand) 7%, transparent);
    color:var(--brand); font-size:var(--fs-sm); line-height:var(--lh-sm); font-weight:600}}
  .st-key-table_export button:hover {{background:color-mix(in srgb, var(--brand) 14%, transparent); border-color:var(--brand); color:var(--brand)}}
  .st-key-table_export button:focus-visible {{outline:2px solid var(--brand); outline-offset:2px; box-shadow:none}}
  .st-key-table_export button [data-testid="stIconMaterial"] {{font-size:1rem; color:inherit}}
  .st-key-table_export button p {{font-size:var(--fs-sm); line-height:var(--lh-sm); font-weight:600}}
  div:has(> .st-key-probe) {{position:absolute; height:0; overflow:hidden; visibility:hidden}}   /* measures, out of the flow and out of the tab order */
  [data-testid="stMetric"], .card {{box-sizing:border-box; height:100%; display:flex; flex-direction:column; background:var(--surf); border:1px solid var(--line);
    border-radius:var(--r); padding:0.5rem; color:var(--ink); box-shadow:none}}
  [data-testid="stMetric"] > div {{flex:1; display:flex; flex-direction:column}}
  [data-testid="stElementContainer"]:has(> [data-testid="stMetric"]) {{flex:1}}   /* cards in one row share a height */
  [data-testid="stMetricLabel"] p, [data-testid="stSelectbox"] label p, .st-key-prev_w label p, .card .lab
    {{font-size:var(--fs-sm); line-height:var(--lh-sm); font-weight:500; color:var(--muted); margin:0}}
  [data-testid="stMetricValue"], .card .num {{margin-top:auto; font-weight:600; font-variant-numeric:tabular-nums}}   /* cards side by side are read as a pair: a label that wraps must not stagger the numbers */
  /* The batch's count (count_card): the framework's metric card, drawn by the app because two of the three
     things on it are beyond st.metric — the reference the count is read against, on the value line at the
     reading size, and that fraction as a bar. The track is the grid line the charts are ruled with, so the
     unfilled remainder is not read as a second quantity. */
  .cards {{display:flex; gap:0.5rem; align-items:stretch}}
  /* Content sets the shares and every card grows into what is left: the label is what a card cannot be
     narrower than — at 180px `Called pathogenic · 227 labelled` wraps onto a second line — and the row is
     one strip of surface, so a card stopping at its content width would leave the remainder as a gap
     against the column's edge. Most visible on the finals path, where two cards divide the row.
     One height: the tallest card sets it, and the shorter card fills it rather than hanging from its top
     edge — the count card's leftover used to pool under its label as 33px of nothing. */
  .cards .card {{flex:1 1 auto; min-width:0; height:auto}}
  .card .lab .sub {{font-weight:400}}   /* the qualifier after the separator, as a band heading sets it (lbl) */
  .card .base {{font-size:var(--fs); line-height:var(--lh-lg); font-weight:500; color:var(--muted)}}
  /* The count sits on the grids' first row: the drop is their column-head line, their row gap and their
     cell's own padding, so the three cards' first figures share one line however wide the row is drawn.
     Held against the card's bottom edge instead, as it was, the whole leftover pooled under the label. */
  .cards .card .num {{margin-top:calc(var(--lh-sm) + 0.1875rem)}}
  /* The fraction is the card's foot: full-bleed on the bottom edge, the width of the card, closing the
     height the grids beside it set. Drawn at the weight it always was — given the whole leftover instead,
     it stops being a proportion and becomes a slab of ink over half the card's area. */
  .card .prop {{display:flex; height:0.5rem; margin:auto -0.5rem -0.5rem; border-radius:0 0 var(--r) var(--r); overflow:hidden; background:var(--grid)}}
  .card .prop i {{background:var(--muted)}}
  /* The cards whose value is four counts (quad_card). The 2×2 is the shape of what is counted, and it is
     what gives the silent failure a cell the size of the other three instead of 7.7% of a bar. One grid,
     two cards: the warning types take the same 2×2 as the verdicts, so the row reads as two matched grids.
     Both are held to the bottom edge, which is what aligns their figures on the same two lines — the space
     the flag card's missing column heads would have taken is the space above its first row. A row of four
     instead put a wide card beside a tall one, with nothing lining up. */
  .card .quad {{display:grid; grid-template-columns:1fr 1fr; gap:0.125rem 0.5rem; margin-top:auto}}
  .card .qh {{font-size:var(--fs-sm); line-height:var(--lh-sm); color:var(--muted); white-space:nowrap}}
  .card .qc {{display:flex; align-items:center; gap:0.375rem; padding:0.0625rem 0.375rem; border-radius:var(--r-sm); width:max-content; margin-left:-0.375rem}}   /* the cell is its mark and its number: the tint below fits the pair, not the column it sits in */
  .card .qc b {{font-size:var(--fs-lg); line-height:var(--lh-lg); font-weight:600; font-variant-numeric:tabular-nums}}   /* the row's value size: four figures on one card, read against the count beside them */
  /* The marks sit in a row of text here rather than on a table row's baseline: the box is the mark's own
     16px again, so the four cells share one optical line and one left edge. */
  .card .qc :is(.tm,.fl) {{display:flex; align-items:center; margin:0}}
  .card .qc .mk {{margin:0}}
  /* The one tinted cell in the app: wrong, and nothing warned. Red is already the only red in the grid, but
     the cell has to be found across a hall from a screenshot, and a tint is what makes it the first thing
     read in a block of four equal cells rather than the last of two rows. Mixed from the trust red itself,
     not a second red picked by eye. */
  .card .qc.lead {{background:color-mix(in srgb, {TRUST_FAIL} 15%, #fff)}}
  [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div {{white-space:normal !important; overflow:visible !important; text-overflow:clip !important}}   /* !important: Streamlit truncates a metric label to one line, and these labels name a prevalence */
  .tablewrap {{overflow:auto; border:1px solid var(--line); border-radius:var(--r)}}
  table.grid {{border-collapse:collapse; width:100%; font-size:var(--fs); line-height:var(--lh); font-variant-numeric:tabular-nums}}   /* a column of measurements is read down: the ± signs and the decimal points line up */
  /* z-index: the flag cell is positioned (for the hover glyph below), which lifts it and everything in it
     above a sticky header of z-index auto, so a row scrolled under the header showed through it and its
     hover glyph painted over the column name. A header with a stacking level of its own paints over every
     row, hovered or not. */
  table.grid th {{text-align:left; font-weight:500; color:var(--muted); padding:0.25rem 0.5rem; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:1; background:#fff}}
  table.grid td {{padding:0.25rem 0.5rem; border-bottom:1px solid var(--line); color:var(--ink); white-space:nowrap}}
  /* The two edges that touch the border take the next step up: at 0.5rem the first cell's text began level
     with the corner's own radius, so the arc ran through it. The wrapper is not padded instead — th is
     sticky, and an inset would leave rows visible above the stuck header. The vertical padding is what
     buys the row count and is not touched. */
  table.grid :is(td,th):first-child {{padding-left:0.75rem}}
  table.grid :is(td,th):last-child {{padding-right:0.75rem}}
  table.grid tr:last-child td {{border-bottom:0}}
  table.grid tr.none td {{color:var(--muted); white-space:normal}}   /* the body of a table a filter left empty (html_table empty=) */
  /* Overview's family table is the one table in the app whose cells wrap: a description column reads as
     prose, so the rows are top-aligned and the pitch is a little looser. It was a `wrap` variant of the
     grid, and a variant that exists once is a rule with a longer name — its own first-cell width was dead
     under the id rule below it, and only the weight ever applied. padding-block, so the first and last
     cells keep the edge inset every grid takes. */
  .st-key-overview_importance table.grid td {{white-space:normal; vertical-align:top; padding-block:0.375rem}}
  .st-key-overview_importance table.grid td:first-child {{width:18rem; white-space:nowrap; font-weight:600}}
  .st-key-overview_importance table.grid td:nth-child(2) {{width:7rem; white-space:nowrap; text-align:right}}   /* the share, as a number: read down the column against its neighbours */
  /* One bar, wherever a quantity is drawn beside its own label: the results table's score, Overview's
     family shares, the batch's commonest substitutions. Its colour is the caller's, never a default call. */
  .bar {{display:inline-block; height:0.5rem; background:var(--bar,{BLUE}); border-radius:var(--r-sm); vertical-align:middle; margin-right:0.5rem}}
  /* Overview's per-1000 picture (outcome_bars). Two bars on one thousand-variant scale; the track behind a
     bar is the part of the thousand that call does not reach. Solid is a call the model got right, hatched
     is one it got wrong, and the hue is what the variant actually is — so the two hatched cells are the two
     kinds of error, and the solid-red share of the top bar is exactly the precision card beside it. No fourth
     palette: the figure is the app's pathogenic red and benign blue, broken by white where the call was wrong. */
  .pk-cap {{margin-bottom:0.25rem}}
  .pk-row {{display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem}}
  .pk-lab {{flex:0 0 9.5rem; text-align:right; white-space:nowrap}}   /* right-aligned: the call sits against the bar it labels, and the two bars share one left edge */
  .pk-track {{flex:1; display:flex; height:0.75rem; background:var(--surf); border:1px solid var(--line); border-radius:var(--r-sm); overflow:hidden}}
  .pk-track i {{height:100%}}
  .pk-key {{display:flex; flex-wrap:wrap; gap:0.125rem 0.75rem; margin-left:10rem}}   /* under the bars, not under their labels */
  .pk-key span {{display:inline-flex; align-items:center; gap:0.375rem}}
  .pk-key i {{width:0.75rem; height:0.75rem; border-radius:var(--r-sm)}}
  .pk-tp {{background:{RED}}}
  .pk-tn {{background:{BLUE}}}
  .pk-fn {{background:repeating-linear-gradient(45deg,{RED} 0 2px,#fff 2px 4px)}}
  .pk-fp {{background:repeating-linear-gradient(45deg,{BLUE} 0 2px,#fff 2px 4px)}}
  /* Overview's prevalence control: the framework's slider and the app's own anchor rail, drawn as one control
     in a gapless container. Three things about the framework's slider are changed and each is load-bearing.
     Its 1.25rem of padding around a 3px track is a form-page density and is cropped to the band's rhythm, so
     the whole control keeps the height the three segments it replaces had. Its Min/Max tick bar is dropped:
     the rail takes that row, and naming the three known prevalences there is what stops the control from
     being the bare percentage slider this band used to carry. Its value bubble over the thumb is dropped too
     — at this height it would sit in the label's line, and the label carries the value instead.
     The track is one flat grey and never a filled brand bar: a bar that grows as the reader drags right reads
     as a level being raised, which is precisely the misreading this whole band exists to prevent. The thumb
     is the only brand mark on it, and it is there as the interface's own state, not as a quantity. */
  [data-testid="stVerticalBlock"].st-key-cohort {{gap:0}}   /* the slider and its rail are one control, so the track and the marks under it are not two blocks */
  .st-key-prev_w [data-testid="stSlider"] > div:last-child {{margin-block:-0.75rem}}
  /* z-index: cropping the slider's padding lifts its own box over the lower half of the label row, and a
     positioned sibling drawn later takes the pointer with it — which left the label's help marker unhoverable.
     The label is raised above it; the track and thumb sit well below the label's box and are untouched. */
  .st-key-prev_w [data-testid="stSlider"] label {{min-height:0; margin-bottom:0; position:relative; z-index:1}}
  .st-key-prev_w :is([data-testid="stSliderTickBar"],[data-testid="stSliderThumbValue"]) {{display:none}}
  .st-key-prev_w [data-testid="stSlider"] > div:last-child > div > div:first-child {{background:var(--line) !important}}
  /* The rail. Its inline padding is the thumb's half-width, which is what makes a mark sit exactly under the
     thumb that selects it; the marks' own left offsets are computed from the option index in anchor_rail. */
  .rail {{padding:0 0.375rem}}
  .rail .rt {{position:relative; display:block; height:0.25rem}}
  .rail .rt i {{position:absolute; top:0; bottom:0; width:1px; background:var(--muted); transform:translateX(-50%)}}
  .rail .rl {{display:flex}}
  .rail .rk {{white-space:nowrap}}   /* a flex item's automatic minimum is its text: the names cannot be squeezed into each other, only pushed */
  .rail .rk:first-child {{flex-shrink:0}}   /* when the last name has to hang left of its mark, the room comes out of the wide middle slot alone, so the first two stay on their marks */
  .rail .rk.on {{color:var(--ink)}}
  /* The true label (truth_label): the word in plain ink, the agreement mark in its trust tone. The
     cell is sized to the wider of the two class words plus one fixed gap, not to the column — the marks
     still share one x, so the column reads straight down for the model's mistakes, but it is a smaller x
     and the mark sits with the word it qualifies instead of against the review flags four rems away. The
     flag column takes the difference as its own inset (below), so the two stop reading as one group. */
  .cls {{font-weight:600; color:var(--ink)}}   /* ink, not the class colour: the chip beside it is the cell's only colour, and on the review page the line around it is muted */
  .truth {{display:inline-flex; align-items:baseline; gap:0.375rem}}
  .truth .cls {{min-width:{TRUTH_W[lang]}}}
  .tm-ok {{color:{TRUST_OK}}}
  .tm-warn {{color:{TRUST_WARN}}}
  .tm-bad {{color:{TRUST_FAIL}}}
  .st-key-table_body table.grid :is(td,th):last-child {{padding-left:2.5rem}}   /* heading and cells together, so the column name still sits over its own marks */
  [data-testid="stAlert"] {{border-radius:var(--r); padding:0.75rem}}
  .vh {{display:flex; flex-wrap:wrap; align-items:center; gap:0.75rem}}
  [data-testid="stMetricValue"], .vh .vid {{font-size:var(--fs-lg); line-height:var(--lh-lg)}}
  .vh .vid {{font-weight:600; color:var(--ink)}}
  .tag {{color:#fff; border-radius:var(--r-sm); padding:0.125rem 0.5rem; font-weight:600}}
  /* The key/value list sits in a column beside the contribution chart, and its value column can be long (a
     categorical column's raw value), so the two share a narrow gutter and wrap from the top. */
  table.kv {{border-collapse:collapse}}
  table.kv td {{padding:0.125rem 0.75rem 0.125rem 0; vertical-align:top; color:var(--ink); border:0; font-variant-numeric:tabular-nums}}
  table.kv td:first-child {{color:var(--muted)}}
  table.kv td:last-child {{padding-right:0}}
  ul.checks {{list-style:none; margin:0; padding:0}}
  ul.checks li {{padding:0.375rem 0; border-top:1px solid var(--line); cursor:help}}   /* every rule carries the sentence behind it as its tooltip (data-tip) */
  ul.checks li:first-child {{border-top:0}}
  ul.checks .bad::before {{content:"!"; color:{RED}; font-weight:700; margin-right:0.5rem}}
  h3.lab {{font-size:var(--fs); line-height:var(--lh); margin:0; font-weight:600; color:var(--ink)}}
  h3.lab .sub {{font-weight:400; color:var(--muted)}}   /* the qualifiers after the subject, muted (lbl) */
  /* The app's one chip. A claim this interface may not hide in a hover used to ride its heading as another
     ` · clause`, which reads as more of the sentence; drawn as a bordered marker it reads as a state of the
     thing named, and the line goes back to being a title. One treatment at every site — heading, card label
     and, through the second selector, the one metric label the framework will not let carry a span — so the
     marker means the same thing wherever it turns up. Its box is 16px inside the 17px line it sits on, so a
     chipped heading is the height of a bare one; `vertical-align` sets it on the cap line rather than the
     baseline, where a bordered box hangs low. */
  .chp, .st-key-metric_chip [data-testid="stMetricLabel"] p::after
    {{display:inline-block; margin-left:0.5rem; padding:0 0.375rem; border:1px solid var(--line); border-radius:var(--r-sm);
      background:var(--surf); color:var(--muted); font-size:var(--fs-sm); line-height:1rem; font-weight:500;
      vertical-align:0.0625rem; white-space:nowrap}}
  .st-key-metric_chip [data-testid="stMetricLabel"] p::after {{content:"{t('op_fixed')}"}}
  .hint {{font-weight:400; color:var(--muted); cursor:help}}   /* the qualifier lives here, not in a paragraph under the heading */
  :is(.hint,ul.checks li,.flow-platt):focus-visible {{outline:2px solid var(--brand); outline-offset:2px; border-radius:var(--r-sm)}}
  /* One tooltip. Three things on this page answer a hover — the app's own marks (#ps-tip, placed by the probe
     script), the framework's help icons on widgets and metric cards, and the charts' Vega tooltip — and they
     are one object: the card surface and the one border colour, the box radius of a chip, the small text
     role at the 12px floor (a chart tooltip otherwise arrived at 9.8px), the table cell's padding, a measure
     that wraps the longest caveat into five readable lines rather than one 900px line, and the only shadow
     on the page, because a tooltip is the one thing that floats. Delay and dismissal are the framework's
     own numbers (TIP_OPEN / TIP_CLOSE, in JS for #ps-tip, as CSS transition delays for the Vega element,
     which its own script would otherwise show and hide at once). */
  :root {{--tip-open:{TIP_OPEN}ms; --tip-close:{TIP_CLOSE}ms; --tip-fade:80ms; --tip-w:28rem}}
  @keyframes ps-tip {{from {{opacity:0}} to {{opacity:1}}}}
  #ps-tip, [data-testid="stTooltipContent"], #vg-tooltip-element {{box-sizing:border-box; background:var(--surf); color:var(--ink);
    border:1px solid var(--line); border-radius:var(--r-sm); padding:0.375rem 0.5rem; max-width:var(--tip-w);
    font-size:var(--fs-sm); line-height:var(--lh-sm); font-weight:400; box-shadow:0 4px 12px color-mix(in srgb, var(--ink) 20%, transparent)}}
  #ps-tip {{position:fixed; left:0; top:0; width:max-content; z-index:1000002; pointer-events:none; white-space:normal;
    opacity:0; visibility:hidden; transition:opacity var(--tip-fade), visibility 0s var(--tip-fade)}}
  #ps-tip.on {{opacity:1; visibility:visible; transition:opacity var(--tip-fade), visibility 0s}}
  [data-testid="stTooltipContent"] {{animation:ps-tip var(--tip-fade) ease-out both}}   /* the framework already waited --tip-open before mounting it */
  /* !important: the markdown container inside carries an inline font-size, and a -1rem bottom margin meant to
     cancel its paragraph's; with the paragraph's margin gone the container's would swallow the last line. */
  [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"] {{font-size:var(--fs-sm) !important; line-height:var(--lh-sm); margin:0}}
  [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"] :is(p,li,span,code,strong,em) {{font-size:var(--fs-sm); line-height:var(--lh-sm); margin:0}}
  [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"] p + p {{margin-top:0.375rem}}   /* a tooltip that says two things says them as two paragraphs; the rule above zeroes every margin */
  #vg-tooltip-element {{opacity:0; transition:opacity var(--tip-fade) var(--tip-close), visibility 0s calc(var(--tip-close) + var(--tip-fade))}}
  #vg-tooltip-element.visible {{opacity:1; transition:opacity var(--tip-fade) var(--tip-open), visibility 0s var(--tip-open)}}
  /* Key beside value on one line, as the review page's own key/value list draws them (the framework's cell
     rules make each cell a block, which stacks the pair); the row pitch is the line height, as in every list here. */
  #vg-tooltip-element table tr td {{display:table-cell; vertical-align:baseline; padding:0; line-height:var(--lh-sm); max-width:none}}
  #vg-tooltip-element table tr td.key {{color:var(--muted); text-align:left; white-space:nowrap; padding-right:0.75rem; font-weight:400}}
  #vg-tooltip-element table tr td.value {{max-width:20rem; font-weight:600; white-space:normal}}
  [data-testid="stElementContainer"]:not(:first-child) .lab {{margin-top:0.25rem}}   /* a band heading groups what follows it */
  .tablewrap:focus-visible {{outline:2px solid var(--brand); outline-offset:2px}}
  /* The results table is the last band of its page, so it takes the height actually left under the
     bands above it instead of a constant reserved for a layout that has since changed: the page ends
     at the fold on every panel and in both languages with no number to re-tune. */
  [data-testid="stMain"] .block-container:has(.tablewrap.tall) {{height:100%; display:flex; flex-direction:column}}
  .block-container:has(.tablewrap.tall) > [data-testid="stVerticalBlock"] {{min-height:0}}   /* min-height:auto would let the flex child grow past the fold */
  /* The table and its export share one block (table_body), and that block is what takes the height left
     under the header row: a size container, so the table can cap itself against it (100cqh below) while
     keeping its own content height — the foot line then follows the table's last row, not the block's
     bottom, and the export laid over that line has a fixed anchor: the block's slack under the whole rows
     falls below the foot, never between the two. */
  [data-testid="stLayoutWrapper"]:has(> .st-key-table_body) {{flex:1; min-height:0; display:flex; flex-direction:column}}
  .st-key-table_body {{flex:1; min-height:14rem; container-type:size}}
  [data-testid="stHtml"]:has(> .tablewrap.tall) {{display:flex; flex-direction:column}}
  /* Predictions is a split, so that chain now runs through a column: every wrapper between the block
     container and the table joins the flex column, and both halves take the row's full height — the
     table measures the fold from inside the right column, the histogram fills what the left one leaves.
     A column is a block element by default; it has to become a flex column for its children to flex. */
  .block-container:has(.tablewrap.tall) [data-testid="stLayoutWrapper"]:has(.st-key-pred_split) {{flex:1; min-height:0}}
  .st-key-pred_split {{height:100%; min-height:0}}
  .st-key-pred_split > [data-testid="stLayoutWrapper"] {{flex:1; min-height:0}}
  .st-key-pred_split > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{height:100%; min-height:0}}
  /* height, not stretch: the row wraps, so a line takes its tallest item's content height and 300 rows
     of table would set it. A definite column height is what the table's max-height resolves against. */
  .st-key-pred_split > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]
    {{height:100%; min-height:0; display:flex; flex-direction:column}}
  .st-key-pred_split > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
    {{flex:1; min-height:0}}
  .st-key-pred_split [data-testid="stElementContainer"][height="100%"]:has([data-testid="stVegaLiteChart"]) {{flex:1; min-height:0}}   /* the histogram, when it is asked to stretch (see the fold rule below) */
  /* A cap, not a height: a filter that leaves three rows still draws a three-row box, and its legend follows
     it up the column because the two are one element. What the cap leaves is that legend's own line — two of
     them, because a filter with a long name takes the count onto a line of its own, and the room for it has
     to be reserved rather than left to whatever the row rounding happens to spare. The second value rounds
     down to whole rows so the last row is not a half-row, and is dropped where round() is unsupported. */
  .tablewrap.tall {{max-height:calc(100cqh - 2*var(--lh-sm) - 0.375rem);
    max-height:calc(round(down, 100cqh - 2*var(--lh-sm) - 0.375rem, calc(var(--row-lh) + 2*var(--row-pad) + 1px)) + 2px);
    /* The same rule while scrolling: the table settles on a row boundary under its header, so no row is ever
       half under it. The snap inset is the header's height, which is a row's. */
    scroll-snap-type:y mandatory; scroll-padding-top:calc(var(--row-lh) + 2*var(--row-pad) + 1px)}}
  .tall table.grid tbody tr {{scroll-snap-align:start}}
  /* Above the presentation fold, share the remaining space between Overview's two plot bands,
     or give it to the review subject. Native stretch charts measure these flex boxes themselves;
     only plot area grows, with no CSS zoom and no measured-height rerun loop. Which chart grows is
     the one Python asked to (height="stretch"), and the framework writes that request into the DOM
     as the container's height attribute — "100%" for stretch, "auto" for a content-height chart —
     so the rule names that element and not a column position: a second, content-height chart in
     the same column (the family totals over the review subject) keeps its own height, and the slack
     goes whole to the subject, in whichever column it is. */
  @media (min-width:{NARROW + 1}px) and (min-height:{FOLD_H + 1}px) {{
    [data-testid="stMain"] .block-container:has(:is(.st-key-overview_explorer,.st-key-review_body))
      {{height:100%; display:flex; flex-direction:column}}
    .block-container:has(:is(.st-key-overview_explorer,.st-key-review_body)) > [data-testid="stVerticalBlock"] {{min-height:0}}
    [data-testid="stLayoutWrapper"]:has(> :is(.st-key-overview_explorer,.st-key-overview_importance,.st-key-review_body)) {{flex:1; min-height:0}}
    :is(.st-key-overview_explorer,.st-key-overview_importance,.st-key-review_body) {{height:100%; min-height:0}}
    :is(.st-key-overview_explorer,.st-key-overview_importance,.st-key-review_body) > [data-testid="stLayoutWrapper"] {{flex:1; min-height:0}}
    :is(.st-key-overview_explorer,.st-key-overview_importance,.st-key-review_body) > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{height:100%; min-height:0}}
    :is(.st-key-overview_explorer,.st-key-overview_importance,.st-key-review_body) > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{height:100%; min-height:0}}
    :is(.st-key-overview_explorer,.st-key-overview_importance,.st-key-review_body)
      [data-testid="stElementContainer"][height="100%"]:has([data-testid="stVegaLiteChart"]) {{flex:1; min-height:0}}
  }}
  .tall table.grid {{font-size:var(--row-fs); line-height:var(--row-lh)}}
  .tall table.grid td, .tall table.grid th {{padding-top:var(--row-pad); padding-bottom:var(--row-pad)}}
  /* The results table is sorted by score, so its two classes are one block each and the call is a place in
     the list rather than a fact repeated on every row: it is stated once, as the line the list crosses. The
     line names both classes and the threshold between them, so the class of a row is still readable without
     colour — which side of the line it is on, and its score. It is not a row of the table: it has no id and
     no click, it is outside the 500-row cap, and it is not in the export. */
  table.grid tr.split td {{padding:0.125rem 0.5rem; background:var(--surf); color:var(--muted);
    border-top:1px solid var(--muted); border-bottom:1px solid var(--muted); font-size:var(--fs-sm); line-height:var(--lh-sm)}}
  table.grid tr.split .splitline {{display:flex; justify-content:space-between; gap:0.5rem}}
  /* The marked row is brand, not --surf: --surf is also what a hovered row takes, and the row the presenter
     last inspected and the row the pointer happens to be over were then the same object, told apart only by
     weight and by a 3px inset that is off-screen whenever the table is scrolled sideways. Brand is what the
     inset already says, and selection is one of the accent's own jobs. */
  table.grid tr.now td {{font-weight:600; background:color-mix(in srgb, var(--brand) 6%, #fff)}}
  table.grid tr.now td:first-child {{box-shadow:inset 3px 0 var(--brand)}}   /* the selected panel's row, and the variant last inspected: the same brand as the control that chose it */
  /* A results row is the way into Variant review: the whole row is the pointer target, and its first cell is
     a button so the column is keyboard-operable (Tab, then Enter or Space). The bridge reads data-vid. */
  .tall table.grid tr[data-vid] {{cursor:pointer}}
  .tall table.grid tr[data-vid]:hover td {{background:var(--surf)}}
  /* What the click does, said on the row: the same glyph the header's "Varyant incelemesi" link carries,
     in the interface colour, at the row's right end while it is hovered or its button has focus. Out of
     flow, so a hovered row is exactly as tall as the rest; not on the threshold line, which has no id. It is
     not a sixth flag: a different position, a different colour, a magnifier rather than a set glyph. */
  .tall table.grid tr[data-vid] td:last-child {{position:relative}}
  .tall table.grid tr[data-vid]:is(:hover,:has(button.row:focus-visible)) td:last-child::after {{content:"manage_search";
    font-family:"Material Symbols Rounded"; font-size:1.5rem; line-height:1; color:var(--brand);
    position:absolute; right:0.5rem; top:50%; transform:translateY(-50%)}}
  table.grid button.row {{all:unset; font:inherit; color:var(--ink); cursor:pointer}}
  /* The app's one text underline, so it is drawn rather than left to the browser: clear of the descenders
     in a variant id, a hairline, and in the muted ink so it reads as an affordance and not as a rule. */
  table.grid button.row:hover, table.grid button.row:focus-visible {{text-decoration:underline;
    text-underline-offset:0.15em; text-decoration-thickness:1px; text-decoration-color:var(--muted)}}
  table.grid button.row:focus-visible {{outline:2px solid var(--brand); outline-offset:1px}}
  /* The two steppers are one control with the picker: a bordered pair of the well's height and border,
     joined edge to edge (the second overlaps the shared border), rounded only on the outside. A hovered or
     focused half rises above its neighbour so its whole outline shows. */
  .st-key-pick_step {{flex-wrap:nowrap; gap:0}}
  .st-key-pick_step > * {{flex:0 0 auto !important; width:auto !important}}
  .st-key-pick_step button {{min-height:2.5rem; width:2.5rem; padding:0; background:var(--surf); border-color:var(--line); color:var(--muted)}}   /* two squares of the well's height, in the well's colours */
  .st-key-pick_step button:is(:hover,:focus-visible) {{position:relative; z-index:1; color:var(--brand); border-color:var(--brand)}}
  .st-key-pick_prev button {{border-radius:var(--r) 0 0 var(--r)}}
  .st-key-pick_next button {{border-radius:0 var(--r) var(--r) 0; margin-left:-1px}}
  /* The label is the accessible name; the mark is drawn. It is drawn as a box the label lays out as a flex
     item, not as an inline glyph: an inline mark in a zero-size label sits on that label's baseline, at the
     bottom of a line box with no height, which is what hung the triangle below the button's centre. The
     box is the whole triangle (no transparent side), 0.75 by 0.875rem — a third of the square — so its
     bounds are the shape's. A triangle's mass sits at its base, so a box-centred one reads as pushed
     toward the base: the outer margin, the smallest step of the scale, moves it a pixel toward the apex. */
  :is(.st-key-pick_prev,.st-key-pick_next) button p {{font-size:0; line-height:0; display:flex; align-items:center; justify-content:center}}
  :is(.st-key-pick_prev,.st-key-pick_next) button p::before {{content:""; display:block; width:0; height:0; border-top:0.4375rem solid transparent; border-bottom:0.4375rem solid transparent}}
  .st-key-pick_prev button p::before {{border-right:0.75rem solid currentColor; margin-right:0.125rem}}
  .st-key-pick_next button p::before {{border-left:0.75rem solid currentColor; margin-left:0.125rem}}
  .flow {{display:flex; gap:1rem; align-items:center; color:var(--ink)}}
  .flow small {{display:block; color:var(--muted)}}
  .flow-input {{flex:0 0 10rem; display:flex; flex-direction:column; align-items:center; position:relative}}
  /* The diagram's structure — the file glyph, the arrows, the members' top rules — is drawn in thin brand strokes;
     its one filled area is the decision scale, which keeps the two call colours. */
  .flow-file {{border:1px solid var(--brand); border-radius:0 var(--r) 0 0; padding:0 0.75rem; letter-spacing:0.1em}}
  .flow-input::after,.flow-pipeline::after {{content:"→"; position:absolute; right:-1rem; top:45%; color:var(--brand)}}
  .flow-pipeline {{flex:1; min-width:0; position:relative}}
  .flow-head,.flow-lane {{display:grid; grid-template-columns:17rem minmax(0,1fr); gap:1rem}}
  .flow-head {{color:var(--muted); margin-bottom:0.125rem}}
  .flow-lane + .flow-lane {{margin-top:0.125rem; padding-top:0.125rem; border-top:1px solid var(--line)}}
  .flow-prep {{display:flex; flex-wrap:wrap; gap:0 0.75rem; align-content:center; border-right:1px solid var(--line); padding-right:0.5rem}}
  .flow-prep span {{white-space:nowrap}}
  .flow-model {{min-width:0}}
  .flow-training {{display:flex; gap:0.375rem; align-items:center; margin-bottom:0.25rem}}
  .flow-excluded {{border:1px dashed var(--muted); padding:0 0.25rem}}
  .flow-members {{display:flex; align-items:stretch; gap:0.125rem}}
  .flow-member {{min-width:0; border:1px solid var(--line); border-top:1px solid var(--brand); background:var(--surf); padding:0 0.375rem}}   /* the same thin brand stroke as the file glyph and the arrows: at a single 100% member it was a 686px slab reading as a section rule */
  .flow-member b {{display:flex; justify-content:space-between; gap:0.25rem; white-space:nowrap; font-weight:600}}
  .flow-member b span {{font-weight:400}}
  .flow-member small {{white-space:nowrap}}
  .flow-platt {{align-self:center; white-space:nowrap; font-weight:600}}
  .flow-decision {{flex:0 0 15rem; border-left:1px solid var(--line); padding-left:1rem}}
  .flow-mix {{display:flex; gap:0.125rem; margin-bottom:0.25rem}}
  .flow-mix span {{min-width:0; padding:0.125rem; background:var(--surf); border:1px solid var(--line); text-align:center}}
  .flow-mix span:last-child {{background:var(--line)}}
  /* The tick spans its own rise plus the bar it marks, so the two stay locked together. */
  .flow-scale {{height:0.75rem; margin-top:1rem; position:relative; background:linear-gradient(to right,{BLUE} var(--threshold),{RED} var(--threshold))}}
  .flow-tick {{position:absolute; left:var(--threshold); top:-1rem; height:1.75rem; border-left:2px solid var(--ink); padding-left:0.25rem; font-weight:600; line-height:1rem}}
  /* A high threshold (CFTR's 0.90) would push the label past the scale's right edge: draw it left of its rule. */
  .flow-tick.flip {{transform:translateX(-100%); border-left:0; border-right:2px solid var(--ink); padding-left:0; padding-right:0.25rem}}
  .flow-outcomes {{display:flex; justify-content:space-between; margin-top:0.25rem}}
  .sw {{display:inline-block; width:0.75rem; height:0.75rem; margin-right:0.375rem; border-radius:var(--r-sm)}}
  /* The app's own icon set: one 16px box, one 1.5px stroke, no fill, painted in the current text colour
     (mark(), MARKS). The negative block margin is what the flags' line-height:0 used to do — the box
     contributes nothing to its line box, so a marked row, a heading with a hint and a legend line are
     exactly as tall as they are without one. */
  .mk {{display:inline-block; width:16px; height:16px; vertical-align:middle; margin:-8px 0; background:currentColor;
    -webkit-mask:var(--mk) center/16px 16px no-repeat; mask:var(--mk) center/16px 16px no-repeat}}
  {MARK_RULES}
  /* A review flag is drawn as a shape, not written as a phrase, and the same mark is drawn in the table
     cell and in the legend under the table, so the two are one mark (flag_mark). Shape carries it, never
     colour. Every mark takes the same fixed box: a flag that did not fire leaves that box empty, so the
     same mark sits at the same x on every row and the column reads down as well as across. */
  .fl {{display:inline-block; width:16px; height:16px; vertical-align:middle; margin:-8px 0; color:var(--ink)}}
  .fl .mk {{display:block; margin:0}}
  .fls {{cursor:help}}
  /* What is in the batch, under what the model made of it: the commonest substitutions, each bar a share of
     the commonest one. The bars are the amino-acid family's colour, the one Overview already draws that
     family in — this block is that family's content, and a call colour here would read as a verdict on the
     chemistry. The track takes the width the label and the count leave, so the counts line up as a column. */
  .chem {{display:flex; flex-direction:column; gap:0.125rem; font-size:var(--fs); line-height:var(--lh)}}
  .chem > span {{display:flex; align-items:center; gap:0.5rem; color:var(--muted)}}
  .chem b {{flex:0 0 4rem; color:var(--ink); font-weight:600}}
  .chem .track {{flex:1; min-width:0}}
  .chem .bar {{display:block; margin-right:0}}   /* the row's own gap spaces it; the bar fills its track */
  .key .fl {{margin-right:0.25rem}}
  /* The count takes the right edge of the legend's line, and keeps it if the pair has to wrap; the room it
     leaves at its right is the export's (.st-key-table_export, laid over the line), a gap and the control.
     A filter now names itself in a count wide enough to take that line on its own, which is the second
     foot line the table has always reserved — the rows do not move. */
  .key .count {{margin-left:auto; padding-right:calc(0.5rem + var(--export-w))}}
  .key {{display:flex; flex-wrap:wrap; gap:0.125rem 0.5rem; margin:0.125rem 0 0.25rem; color:var(--muted)}}
  /* The one breakpoint, written once here and once in Python (NARROW): the paired columns stack, and the
     flow diagram gives up the width it cannot have. Charts sized for a column read their own copy. */
  @media (max-width:{NARROW}px) {{
    :is(.st-key-review_body,.st-key-pred_split,.st-key-overview_explorer,.st-key-overview_importance) > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{flex-wrap:wrap}}
    :is(.st-key-review_body,.st-key-pred_split,.st-key-overview_explorer,.st-key-overview_importance) > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{flex:1 1 100%; width:100%; min-width:0}}
    .flow {{gap:0.75rem}}
    .flow-input {{flex-basis:8rem}}
    .flow-decision {{flex-basis:13rem; padding-left:0.5rem}}
    .flow-head,.flow-lane {{grid-template-columns:14rem minmax(0,1fr); gap:0.5rem}}
    .flow-member b {{flex-wrap:wrap; gap:0}}
    .flow-member small {{white-space:normal}}
  }}
  .small {{color:var(--muted)}}
  .base {{color:var(--muted)}}   /* a second measured number in a cell: the reading size of the number it stands beside, muted so the pair is still read in order */
  /* The small roles, in one place: captions, tags, key/value lists, review flags, the flow diagram and the
     contribution chart's legend, which has to match the chart labels TXT draws beside it. */
  .small, .hint, .tag, .key, .flow, .flow small, table.kv td, ul.checks li {{font-size:var(--fs-sm); line-height:var(--lh-sm)}}
  /* The surfaces the browser draws for us. They ship in colours that belong to no design system, and on
     this page the default selection blue and the framework's ink are both a class the reader has been
     taught to read: blue is benign. So they take the palette too.
     Not themed: accent-color. There is no checkbox, radio, progress or select element in the app, and the
     one range input is the framework's hidden input behind a track it already paints brand. */
  ::selection {{background:color-mix(in srgb, var(--brand) 18%, transparent); color:var(--ink)}}
  .tablewrap.tall, [data-testid="stSelectboxVirtualDropdown"] [role="listbox"]
    {{scrollbar-width:thin; scrollbar-color:var(--line) transparent}}   /* the results table keeps a gutter on screen at all times */
  .st-key-filters input, .st-key-review_head input {{caret-color:var(--brand)}}
</style>""")


def open_upload():
    ss["upload_open"] = True


def declare_panel():
    """The dialog's segments write the header's mirror, the only one there is (mirror), and mark the dialog
    dirty: the batch behind it is rescored by the panel just declared, so the file taken next is taken
    against what the whole screen is showing."""
    mirror("panel")
    ss["upload_dirty"] = True


def close_upload():
    """Dismissing the dialog is the app's business, not only the frontend's: the flag that keeps it open is
    cleared and the whole script runs again, so the page behind it follows the state the dialog left."""
    ss.pop("upload_open", None)


def drop_upload():
    ss.pop("upload_keep", None)
    ss.pop("batch_failed", None)   # removing the file is the way out of the failure: the control it put on the review page goes with it
    ss["upload_dirty"] = True


@st.dialog(t("upload_title"), on_dismiss=close_upload)
def upload_dialog():
    """The panel the file is for, and then the file. The declaration is made where the file is taken, in full
    panel names and one line above the well, because the bytes are scored by whatever panel is current at the
    moment they are taken (sync_upload) and a file dropped against the wrong one is a confident wrong answer.

    It is the same control as the header band's, not a copy of it: `panel_bar` draws the segments only while
    this dialog is shut, so one widget under one key writes one mirror (panel_keep) and two identical
    controls are never on a screen together. The band's own position is fixed to its right edge, so the
    upload button does not move when the segments leave it. No prose here either: the CSV's requirements are
    the rejected-upload state, which is where a file that does not meet them lands. A dialog is a fragment,
    so a widget in here reruns only this function: every change the app behind it must see ends in an
    app-scoped rerun, which the open flag survives so the dialog does not close under the user's hands. Its
    container is also what the CSS keys on to tell this dialog from the framework's own."""
    with st.container(key="upload_dialog"):
        ss["panel_w"] = ss.setdefault("panel_keep", "general")
        st.segmented_control(t("panel"), PANEL_ORDER, key="panel_w", on_change=declare_panel,
                             format_func=lambda p: t(f"panel_{p}"))
        if ss.get("upload_keep"):
            with st.container(horizontal=True, vertical_alignment="center", key="chip"):
                note(ss["upload_keep"][0])
                st.button(t("remove"), key="remove_upload", on_click=drop_upload, type="tertiary", icon=":material/close:")
        else:
            # No help: the collapsed label makes the framework draw its marker as a 0×0 button — a focusable
            # control with nothing to point at, carrying the longest string in the app to nobody. The same
            # facts are on the rejected-upload screen, visibly, which is where a reader who needs them is.
            st.file_uploader("CSV", type=["csv"], label_visibility="collapsed", key="upload_w", on_change=sync_upload)
    if ss.pop("upload_dirty", False):
        st.rerun()   # app scope: the header, the panel and the scored batch behind the dialog follow the choice


def panel_bar(uploads=True):
    """The header band's controls. The gene panel is global state that changes all three pages, so it is
    drawn once beside the nav (CSS lifts this container into the band) and no page spends a row on it.
    Deselectable, so it is seeded from its mirror every run. Short labels fit the band; every page names the
    panel in full. Beside it, the batch: one button whose label is the loaded file's name, and the dialog
    behind it carrying the panel declaration, the file input and Remove. The variant page inspects one row of
    a batch it does not choose, so it carries no upload control at all.

    While that dialog is open the segments are drawn inside it instead of here — one widget, one key, one
    mirror, and never the two identical controls that a copy in the dialog put on one screen."""
    in_dialog = uploads and ss.get("upload_open")
    with st.container(horizontal=True, vertical_alignment="center", key="panelbar"):
        if not in_dialog:
            ss["panel_w"] = ss.setdefault("panel_keep", "general")
            # No help: the collapsed label still names the radio group, and a tooltip icon would be a 0×0
            # tab stop in the header repeating the name the group already carries.
            st.segmented_control(t("panel"), PANEL_ORDER, key="panel_w", on_change=mirror, args=("panel",),
                                 format_func=lambda p: t(f"short_{p}"), label_visibility="collapsed")
        if uploads:
            loaded = ss.get("upload_keep")
            st.button(loaded[0] if loaded else t("upload_btn"), key="upload_file" if loaded else "upload_btn",
                      icon=":material/description:" if loaded else ":material/upload_file:",
                      type="tertiary", on_click=open_upload)
    if in_dialog:
        upload_dialog()


def shell(needs_data=False):
    """Everything a page needs before it draws its first band: the panel, its artifact and, on the two
    scoring pages, the scored batch. It draws nothing itself — there is no page toolbar. The upload control
    is in the header band, and the filter and the export are the results table's own header row."""
    global panel, art, thr, pi_test, c, res, batch_error, own_w, overlap, in_sample
    csv_bytes = None
    panel = ss.setdefault("panel_keep", "general")
    if needs_data:
        # The source is implicit: a file is scored when one is loaded, the demo sample when none is.
        # Removing the file returns to the sample, so there is no state with nothing to show.
        if ss.get("upload_keep"):
            csv_bytes = ss["upload_keep"][1]
        elif os.path.exists(PANELS[panel]):
            csv_bytes = load_panel_sample(panel)
    try:
        art = get_art(panel)
    except FileNotFoundError:
        st.error(t("no_models")); st.code(f"python pipelines/predict.py train --panel {panel}"); st.stop()
    thr = float(art["threshold"]); pi_test = TEST_PRIORS[panel]; c = CANON[panel]
    res, batch_error = None, None
    if csv_bytes is not None:
        with st.spinner(t("scoring", n=csv_bytes.count(b"\n") - 1, panel=t("panel_" + panel))):
            try:
                res = run_scoring(panel, csv_bytes)
            except InputError as exc:
                names, total, miss, n_required = exc.args
                batch_error = t("missing_columns", total=total, miss=miss, n=n_required,
                                names=", ".join(names[:8]) + (f" … (+{len(names)-8})" if len(names) > 8 else ""))
            except Exception as exc:
                print(f"Dashboard input: {type(exc).__name__}: {exc}", file=sys.stderr)
                batch_error = t("empty_csv") if isinstance(exc, ValueError) and "no variant rows" in str(exc) else t("bad_csv")
    # Which part of the deployed score was fitted on these very rows? Read it from the artifact: a panel
    # with no own model is scored by the General-trained transfer model, which excluded this panel's variants.
    own_w = 1 - (art["transfer_w"] or 0) if art["own"] else 0.0
    if needs_data:
        # The review page carries no upload control, so a file that fails there cannot be removed from the
        # page you are standing on: this is what puts it back (panel_bar). Both scoring pages set it, so the
        # flag is this batch's, one run old at worst — a file is uploaded from Predictions, never from here.
        ss["batch_failed"] = batch_error is not None
    overlap = training_overlap(res["raw"]) if res is not None else None
    in_sample = res is not None and own_w > 0 and overlap == len(res["probs"])   # an aggregate over rows the own model was fitted on measures nothing


def filter_row():
    """Model call, review flags, agreement with the label. The counts are whole-batch, so the list still
    says how many rows each option would match once a filter is on."""
    y = batch_labels(res["raw"]); calls = res["probs"] >= thr
    counts = {"pathogenic": int(calls.sum()), "benign": int((~calls).sum())}
    counts.update({"flag_" + k: int(res["flags"][k].sum()) for k in FLAG_KEYS})
    if y is not None:
        # Every option in this list counts the rows it would leave, this one included. It was withheld on a
        # training batch — where an error rate measures nothing — but the foot line one column over prints
        # the same number the moment the filter is on ("28 / 300 rows"), so the withholding hid it from
        # nobody and left two controls saying different things. What withheld metrics are is a decision about
        # aggregates that read as performance; a count of the rows a filter lists is neither, and the heading
        # above it already says these rows are the model's own training data.
        counts["wrong"] = int((calls != y).sum())
    kept = [k for k in ss.get("filter_keep", ()) if k in counts]
    ss["filter_keep"] = tuple(kept)
    ss["filter_w"] = kept   # seeded from the mirror every run: only a session-state write reaches the frontend
    st.multiselect(t("filter"), list(counts), key="filter_w", on_change=mirror, args=("filter",),
                   format_func=lambda k: f"{option(k)} · {counts[k]}".strip(" ·"),
                   label_visibility="collapsed", placeholder=t("filter"), select_all=False)   # its label is Streamlit's own English


def batch_status():
    if batch_error:
        st.error(batch_error)
    if res is not None:
        if res["coverage"] < .9:
            st.warning(t("low_coverage", pct=percent(res["coverage"])))
        return True
    lbl(t("waiting"))
    expected = expected_columns(art)
    rows = []
    for group in ("AL_", "EK_", "CAT_", "AA_"):
        cols = sorted((c for c in expected if c.startswith(group)), key=lambda c: int(c.split("_")[1]))   # AL_1, AL_2, AL_3 — not lexicographic
        rows.append([group, str(len(cols)), E(", ".join(cols[:3]))])
    rows += [["Variant_ID", "—", E(t("id_optional"))], ["Label", "—", E(t("label_optional"))]]
    html_table([t("column_group"), t("count"), t("example")], rows)
    return False


def trust_counts(y, preds, flags):
    """The four verdicts the truth column marks its rows with, as counts: whether the call agrees with the
    file's own Label column, crossed with whether any review flag fired on the row. Two facts, four states,
    and the one the competition's false-positive / false-negative question is about — wrong, and nothing
    warned — is the cell nothing else on this page distinguishes. Returns None where there are no labels to
    compare against, which is the finals batch: there the row is the two cards the labels are not needed for."""
    if y is None:
        return None
    agree = np.asarray(y) == np.asarray(preds)
    warned = flags[list(FLAG_KEYS)].any(axis=1).to_numpy()
    return {"ok_clean": int((agree & ~warned).sum()), "ok_warn": int((agree & warned).sum()),
            "bad_clean": int((~agree & ~warned).sum()), "bad_warn": int((~agree & warned).sum())}


def count_card(label, n, of, help=None, sub=None):
    """One of the batch's count cards: what is counted, the count, the reference it is read against, and that
    fraction drawn under the pair.

    It is the framework's metric card — one CSS rule, two callers — with the two things st.metric cannot
    hold, and both of them are the point of this row: a second number on the value line, so no card needs a
    neighbouring card to mean anything, and a bar, so the fraction reads before the digits do."""
    qualifier = f"<span class='sub'> · {E(sub)}</span>" if sub else ""
    return (f"<div class='card'><span class='lab'>{E(label)}{qualifier}{hint_mark(help)}</span>"
            f"<b class='num'>{n}<span class='base'> / {of}</span></b>"
            f"<span class='prop'><i style='width:{n / of:.1%}'></i></span></div>")


def quad_card(label, help, heads, cells):
    """The card at the end of the row, whose value is four counts rather than one: a 2×2 because that is the
    shape of what it counts, and because four cells give each state the same footprint whatever its count —
    a stacked bar would draw the silent failure, 7.7% of this batch, as a sliver of the one state the row
    exists for.

    `cells` are (mark html, count, base, accessible name, extra class) in reading order, `heads` the two
    column names or none. Each figure is drawn over the base it is a count of, the way the batch's count
    card draws its own: a card that states its denominator on every figure needs no clause under its title
    saying what the four numbers are out of, and no column head doubling as a total. Every cell carries its mark and its tone, never colour alone: two of the four verdicts
    are the same gold and are told apart by the glyph, so the pair that share a tone must also be the pair
    that share nothing else. The marks are `mark()`'s own output — the ones the table draws in the same
    tones — so this card cannot drift from the column it is the key for.

    Each cell is a role=img: its children are drawn shapes and a bare number, and what it is a number of is
    the name on the element, not something the reader has to reconstruct from the grid."""
    head = "".join(f"<span class='qh'>{E(h)}</span>" for h in heads) if heads else ""
    body = "".join(f"<span class='qc {cls}' role='img' aria-label='{E(name)}' data-tip='{E(name)}'>"
                   f"{glyph}<b>{k}<span class='base'> / {of}</span></b></span>" for glyph, k, of, name, cls in cells)
    # Two columns either way. Where the counts cross two facts the columns are named; where they are four of
    # one kind they are not, and the four cells are the legend's four marks in the legend's order, read
    # across and then down. The heads are the only difference between the two cards in this row.
    # The label is a title and nothing after it: what the figures are counted against is on the figures.
    return (f"<div class='card'><span class='lab'>{E(label)}{hint_mark(help)}</span>"
            f"<span class='quad'>{head}{body}</span></div>")


def verdict_card(counts, n):
    """The truth column's four marks, counted. The columns are the review warning — which is what keeps the
    card's subject the warnings and not the model's score — and each column head carries its own total, so
    no figure in the grid needs a neighbour to mean something. The rows are the agreement mark, check over
    cross, in the order the column reads down.

    The silent failure is the one cell that must be found first, and position alone would not do it: it is
    the only red among a green and two golds, it is the only cell the card tints, and it sits at the head of
    its own row — the left column, which is the larger half of the batch and the half nobody would look at."""
    clean, warned = counts["ok_clean"] + counts["bad_clean"], counts["ok_warn"] + counts["bad_warn"]
    cells = [("ok_clean", "yes", "ok", clean, ""), ("ok_warn", "yes", "warn", warned, ""),
             ("bad_clean", "no", "bad", clean, "lead"), ("bad_warn", "no", "warn", warned, "")]
    return quad_card(t("quad_title"), t("quad_note", n=n) + " " + t("silent_note"),
                     [t("quad_clean"), t("quad_warned")],
                     [(f"<span class='tm tm-{tone}'>{mark(glyph)}</span>", counts[key], of,
                       f"{t('v_' + key)}: {counts[key]} / {of}", cls) for key, glyph, tone, of, cls in cells])


def flag_quad_card(flags, n):
    """The middle card, on both paths: which of the four warnings fired, and on how many rows each. It is
    drawn in the flag vocabulary — the marks the table draws in its last column and the legend names
    underneath — with no trust mark and no agreement axis, because a warning is knowable whether or not the
    file carries labels.

    It replaced a card that printed how many rows are flagged, which the verdict card beside it already
    prints as the head of its warned column. The breakdown is not printed anywhere else, and it is what a
    reader wants next: 40 borderline scores and 4 low-quality regions are not the same batch.

    A 2×2 rather than a row of four, so this card and the verdict card are one shape and their figures sit
    on the same two lines; the space its missing column heads would take is what aligns them. The counts
    overlap — one row can raise several — so they do not add up to the flagged total, which is why that
    total is stated in the tooltip and never as a fifth figure here. A batch that raised no warning at all
    has nothing to say and the card is not drawn."""
    fired = {key: int(flags[key].sum()) for key in FLAG_KEYS}   # the legend's order, not ranked: the same four marks in the same four places
    if not any(fired.values()):
        return ""
    return quad_card(t("flag_quad_title"), t("flag_quad_note", n=n, k=int(flags[list(FLAG_KEYS)].any(axis=1).sum())), None,
                     [(flag_mark(key), fired[key], n, f"{t('flag_' + key)}: {fired[key]} / {n}", "")
                      for key in FLAG_KEYS])


def batch_character(raw, matrix, flags):
    """What is in the batch, under what the model made of it: the substitutions it is built from, and where
    it sits on the two scales a geneticist reads a substitution by.

    Descriptive only. This block says what the file contains and never how well the model did on it: it is
    drawn on every panel, directly above the place the provenance rule removes the metrics, so nothing here
    is computed against Label and nothing here behaves like one. Chemistry that is missing or unusable draws
    nothing at all — not an error, not an empty frame."""
    counts = {}
    if "AA_1" in raw and "AA_2" in raw:
        # The ◆ rule, read a second time: a multi-residue or non-standard code has no chemistry to count.
        usable = ~flags["complex_aa"].to_numpy()
        counts = (raw["AA_1"].astype(str) + " → " + raw["AA_2"].astype(str))[usable].value_counts().to_dict()
    central = []
    for feat, name in (("AA_grantham", "Grantham"), ("AA_blosum62", "BLOSUM62")):
        src = raw if feat in raw.columns else matrix   # feature_value's rule, taken a column at a time
        v = pd.to_numeric(src[feat], errors="coerce").dropna() if feat in src.columns else pd.Series(dtype=float)
        if len(v):
            central.append(f"{name} {float(v.median()):g}")
    if not counts and not central:
        return
    lbl(" · ".join([t("flow_chem")] + ([t("chem_central", parts=" · ".join(central))] if central else [])))
    if counts:
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:CHEM_BARS]   # ties by name: the same picture every run
        widest = top[0][1]
        st.html("<div class='chem'>" + "".join(
            f"<span><b>{E(change)}</b><span class='track'>"
            f"<span class='bar' style='--bar:{FAMILY_COLORS['amino_acid']};width:{k / widest:.1%}'></span></span>{k}</span>"
            for change, k in top) + "</div>")


def score_bars(rows, threshold, tlabel=True):
    d = pd.DataFrame([dict(name=name, p=p, color=color, start=0., end=1.) for name, p, color in rows])
    h = chart_rows(len(rows), GAUGE_BAND, GAUGE_AXIS)
    # A tighter limit than LABEL_LIMIT: these labels name two models, and the bar beside them is the point.
    base = alt.Chart(d).encode(y=alt.Y("name:N", sort=d["name"].tolist(), title=None, axis=alt.Axis(labelLimit=H(230))),
                              x=alt.X("start:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(values=[0, .2, .4, .6, .8, 1], format=".1f")))
    track = base.mark_bar(color=GRID, size=H(16)).encode(x2="end:Q")
    fill = base.mark_bar(size=H(16)).encode(x2="p:Q", color=alt.Color("color:N", scale=None),
                                            tooltip=[alt.Tooltip("p:Q", title=t("score"), format=".3f")])   # the bar's own label is the y axis
    mark = alt.Chart(pd.DataFrame({"t": [threshold], "label": [f"{t('thr_short')} {threshold:.2f}"]})).encode(x="t:Q", y=alt.value(0), tooltip=alt.value(None))
    tick = mark.encode(y2=alt.value(alt.ExprRef(expr="height")))   # explicit extent: a bare rule in this layered bar context has zero height
    chart = track + fill + tick.mark_rule(color="white", size=4) + tick.mark_rule(color=INK, size=2)
    if tlabel:   # the member bars sit under the gauge that already carries the threshold label
        chart += mark.mark_text(fontSize=TXT, align="right", dy=-4).encode(text="label:N")
    return styled(chart.properties(height=h))


def render_pred():
    shell(needs_data=True)
    if not batch_status():
        return
    raw, probs, flags = res["raw"], res["probs"], res["flags"]
    y = batch_labels(raw); ids = variant_ids(raw); preds = (probs >= thr).astype(int)
    n = len(probs); npos = int(preds.sum())
    show_eval = y is not None and not in_sample
    # Built before the page's first line, for the reason given in render_explain: the distribution is the
    # most expensive spec on this page, and constructing it after the heading and the counts have been sent
    # is what made those land a stage ahead of it.
    hist = prob_hist(probs, thr)
    # The tag is the visible half of the provenance and the tooltip is the rest: a screenshot of this batch
    # must never show counts or metrics without saying whose training rows they are. Every branch of the
    # comparison writes one, the unlabelled file included — that is the batch the jury hands over, and
    # "none of these variants is in our training set" is the strongest thing this project can say about it.
    tags = []
    if in_sample:
        tags.append(t("tag_in_sample"))
        if own_w != 1:   # which share of the score the in-sample half is: the tag beside it, not a clause inside it
            tags.append(t("tag_weight", o=percent(own_w, 0)))
        # The heading's tooltip is this claim's only home: nothing is drawn under the batch to carry it. The
        # tag beside it is the visible half — a screenshot of this batch says örneklem içi without it — and
        # the marker holds why that means the cards below carry no metrics of the model.
        provenance = t("training_batch") if own_w == 1 else t("training_batch_part", o=percent(own_w, 0))
    elif overlap == n:   # pure transfer: the scorer never saw these rows, but the calibration and threshold did
        tags += [t("tag_transfer"), t("tag_transfer_cal")]
        # The caveat is about the F1 and MCC this batch does show, so it rides the label comparison's heading
        # over those cards (transfer_caveat, below); here the tag says all the heading needs to.
        provenance = None
    elif overlap is None:
        # No Variant_ID: the comparison cannot be made, and this is the state in which deleting one column
        # would otherwise buy a full set of metrics with the caveat hidden in a tooltip.
        tags.append(t("tag_overlap_unknown"))
        provenance = t("unknown_note")
    elif y is None:   # the finals case: no labels to compare, so the comparison that can be made is the ID one
        tags.append(t("tag_unseen") if overlap == 0 else t("tag_overlap_part", k=overlap, n=n))
        provenance = t("unseen_note")   # not overlap_note: it ends "not validation results", and there is nothing here to validate against
    else:
        provenance = None   # a labelled batch: the label comparison's own heading below states the same fraction
    demo = not ss.get("upload_keep")
    origin = t("sample_title") if demo else ss["upload_keep"][0]
    # What file, and nothing else: the panel it is scored by is the highlighted segment in the header band on
    # this same screen, and naming it again here was the heading restating a control the reader is looking at.
    # It comes back only when the two disagree, as tag_upload_panel below. What the batch is made of — the
    # share its labels call pathogenic — was a clause here too and is now a card beside the call rate it
    # corrects. How many variants it holds was the last clause to go: the count card under it is drawn on
    # every path and states the batch as the base of its own fraction ("176 / 300").
    heading = origin
    # Which panel the file was declared for, when the header has since been switched to another: the same
    # bytes are then scored by a different panel's model, and nothing else on the page would say so.
    if (declared := upload_panel()) and declared != panel:
        tags.insert(0, t("tag_upload_panel", p=t("short_" + declared)))
    # Coverage rides the heading as well as the banner above it: cropped to the heading, a batch the model
    # could only half read must not look like any other.
    if res["coverage"] < .9:
        tags.append(t("tag_coverage", pct=percent(res["coverage"])))
    # A Label column the app could not read is a caveat about this batch, which is what the heading's tags
    # and its tooltip are for. It was a notice box: 65px of the framework's blue — four pixels of hue from
    # the blue that on these pages means a benign call — for one sentence that qualifies the numbers under
    # a heading four rems above it. With this gone and the empty state drawing its own table, stAlert only
    # ever paints an error.
    ignored = t("label_ignored", v=bad) if (bad := rejected_label(raw)) is not None else None
    if ignored:
        tags.append(t("tag_label_ignored"))
    # The seed is what makes the demo reproducible, but it was also the third word a juror read: it rides
    # the tooltip the heading already carries rather than the line.
    lbl(heading, chips=tags,   # a line, not a row: nothing else is on it — a title and the state it is in
        help=" ".join(filter(None, [provenance, ignored, t("sample_seed", seed=SAMPLE_SEED) if demo else None])))
    # The expected call rate belongs to an unknown batch scored at test prevalence; on the training sample, or
    # on a file that is the training set, the batch prevalence is not the test prior and the number means nothing.
    rate = t("expected_rate", p=percent(pi_test), r=percent(pi_test * c["TPR"] + (1 - pi_test) * c["FPR_anchor"])) \
        if ss.get("upload_keep") and overlap != n else None
    # The page is a split, not a stack: what the batch amounts to on the left — the counts, the score
    # distribution and, when the labels can be compared, that comparison — and the rows themselves on the
    # right. Even columns: the table's five columns need about half of 1366 and nothing more, and neither
    # half is the page's subject. The histogram fills the height the left column has left over.
    with st.container(key="pred_split"):
        summary, listing = st.columns(2)
    # The table's own header row, in the column the table is in: the filter that scopes it on the left, and
    # on the right what leaves it — the way into a single variant, and the export of exactly these rows.
    # The filter runs first because it decides which rows both of them are about.
    with listing, st.container(horizontal=True, vertical_alignment="center", key="table_head"):
        with st.container(key="filters"):
            filter_row()
        actions = st.container(horizontal=True, vertical_alignment="center", key="table_actions")
    mask = batch_mask(res, y, ss.get("filter_keep", ()))
    order = [i for i in np.argsort(-probs, kind="stable") if mask[i]]
    # The one file the table leaves as, built once here from the rows it is about: the filtered,
    # score-sorted set entire, not the 500 rows listed, and never the divider line or a flag symbol.
    #
    # It leads with the one shape that is known: a row per variant, its id and the call as 0/1. The
    # organizers have published no file spec — what is confirmed is binary 0/1 per variant and per-panel
    # scoring — so the first two columns claim nothing beyond that, and the panel it was scored by is in
    # its name. After them come what the app read the rows by: the probability the call was cut from, each
    # review flag as 0/1, and the file's own label where it carries one. Those ride the same file rather
    # than a second control — one table, one export — and a reader who wants the confirmed pair alone
    # takes the first two columns.
    table_csv = pd.DataFrame({"Variant_ID": [ids[i] for i in order], "prediction": preds[order],
                              "probability": probs[order].round(6)})
    for k in FLAG_KEYS:
        table_csv["flag_" + k] = flags[k].to_numpy()[order].astype(int)
    if y is not None:
        table_csv["label"] = y[order]
    with actions:
        # The way in, alone on this row: the drill-down is the explainability story and it is the page's
        # one filled block. The export is not here — it keeps the foot line of the table it is the export
        # of, where it costs the column no height at all.
        st.page_link(PAGE["explain"], label=t("tab_explain"), icon=":material/manage_search:")
    with summary:
        # One count and then the grids. A share of the rows called pathogenic describes the file and not the
        # model — a file can carry any prevalence at all — so it is stated as a count over the batch, with
        # what the file's own labels call pathogenic beside it as the qualifier that stops 58.7% being read
        # against the 14.3% test prevalence on the sibling page.
        # The middle card is the warnings the batch raised, one figure per flag, and it is drawn whether or
        # not the file carries labels — the breakdown is knowable either way, and the two paths are then the
        # same row with one card more or less. The card it replaced counted the flagged rows, which is the
        # number the verdict card prints as the head of its warned column a few hundred pixels along.
        # The last card is the one that needs the labels: the truth column's four verdicts, the agreement
        # with them crossed with whether anything warned. Its bottom-left cell — wrong, and no flag fired —
        # is the competition's false-positive / false-negative question, so the card is built around it: the
        # only colour in the row, and the only cell that is tinted.
        counts = trust_counts(y, preds, flags)
        st.html("<div class='cards'>"
                + count_card(t("n_path"), npos, n, sub=t("card_labelled", n=int(y.sum())) if y is not None else None,
                             help=" ".join(filter(None, [rate, t("labelled_note") if y is not None else None])))
                + flag_quad_card(flags, n)
                + (verdict_card(counts, n) if counts else "")
                + "</div>")
        lbl(t("hist_title"))
        st.altair_chart(hist, width="stretch", height="stretch")   # the table binds the page to the fold at any width, so the histogram takes the rest of its column
        # Under the distribution of what the model said, what the batch is made of. Its height comes out of
        # the histogram, which stretches and whose bar coverage does not depend on how tall it is drawn.
        batch_character(raw, res["matrix"], flags)
        if show_eval:   # the label comparison belongs with the summary numbers, not with the row-level table
            # Only the pure-transfer batch may say the model never saw these rows; on a partly-overlapping
            # upload to a panel with its own model, the same sentence would be false.
            # One title, and the state of the comparison as a chip on it. With no IDs the comparison cannot
            # be checked at all, so what it is not — validation — is the whole of what the chip can say; on a
            # pure-transfer batch the chip is that its threshold and calibration were fitted here; anywhere
            # else it is how much of the batch the training set already holds, which the heading is the only
            # place that states.
            lbl(t("eval_title"),
                chips=[t("eval_not_validation")] if overlap is None else
                      [t("eval_not_heldout")] if own_w == 0 and overlap == n else
                      [t("tag_overlap_part", k=overlap, n=n)],
                # No IDs: why the cards are missing, and the way out. Pure transfer: why the cards are optimistic.
                help=t("unknown_suppressed") if overlap is None else
                t("transfer_caveat", rank=t("transfer_rank") if art["transfer"].get("target_grids") else "") if own_w == 0 and overlap == n else
                t("overlap_note"))
        # Deleting the Variant_ID column is enough to make the overlap check impossible, and that is the one
        # path where an aggregate the model may have been fitted on could be captioned rather than withheld:
        # AUC 0.986 and MCC 0.781 for a panel whose published AUC is 0.864. Everywhere else in the app that
        # aggregate is withheld, so nothing is drawn here — not the cards, and not the confusion counts they
        # can be rebuilt from. The heading above is what says so, and its marker carries why. The rows
        # themselves, their flags and the export are untouched: nothing about a single variant depends on
        # knowing whether the batch is training data.
        if show_eval and overlap is not None:
            from sklearn.metrics import roc_auc_score, matthews_corrcoef
            tp = int(((preds == 1) & (y == 1)).sum()); fn = int(((preds == 0) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum()); tn = int(((preds == 0) & (y == 0)).sum())
            both = len(set(y)) > 1
            thin = min(tp + fn, fp + tn) < 30   # the prior-adjusted F1 rescales this batch's TPR and FPR: both need support
            # F1 at this batch's own prevalence is not drawn. It was the largest number on the page — 0.880 on
            # PAH, 0.916 on CFTR — computed at a prevalence the product will never meet (both batches are
            # over 80% pathogenic against a benign-dominant test set) and on the rows the threshold and the
            # Platt calibration were fitted on. What is left is the three a reader can use: the same F1
            # rescaled to the test prior, and the two that do not depend on the prevalence at all.
            cards = []
            if both:
                if not thin:
                    cards.append((t("eval_f1_prior", p=percent(pi_test)), f"{f1_at_prior(tp/(tp+fn), fp/(fp+tn), pi_test)[2]:.3f}", None))
                cards += [(t("eval_auc"), f"{roc_auc_score(y, probs):.3f}", None), (t("eval_mcc"), f"{matthews_corrcoef(y, preds):.3f}", None)]
            for i in range(0, len(cards), 2):   # two per row, and a lone card fills its row rather than leaving a void
                metric_row(cards[i:i + 2])
            if not both:
                note(t("single_class"))
            elif thin:
                note(t("f1_prior_withheld", p=tp + fn, b=fp + tn))
            note(f"TP {tp} · FN {fn} · FP {fp} · TN {tn}")
            # The panel's own result, as the second route to the F1 card above it. It used to end "the same
            # range as this batch's F1" on every batch that drew that card — an agreement the line asserted
            # without ever comparing the two numbers. Both are on this screen; the reader compares them.
            note(t("canonical_anchor", f1=c["F1"], se=c["SE"]))
    with listing, st.container(key="table_body"):   # the table, its foot line and the export of the rows it lists
        # The count is drawn only where the listing is not the batch: a filter is on, or the 500-row cap is
        # biting. "300 / 300" said nothing the absence of a filter did not already say, and the heading one
        # column over states how many rows the batch has. Where it does appear it names what did it.
        filtered = active_filter()
        count = " · ".join([t("table_count", shown=min(500, len(order)), total=n)] + filtered) \
            if filtered or len(order) > 500 else ""
        # On return from Variant review the variant last inspected keeps its row marked, so the presenter
        # comes back to where they were rather than to the top of the list.
        saved = ss.get("pick_keep", {}).get(panel)
        shown = order[:500]
        marked = next((k for k, i in enumerate(shown) if str(ids[i]) == saved), None)
        # The substitution the row is about, in the terms a geneticist reads a variant by rather than an
        # opaque id and a number. A multi-residue or missing code has no substitution to write and draws
        # nothing at all: the ⇄ on the same row is what says why, and it is already in the legend.
        aa = "AA_1" in raw and "AA_2" in raw
        rows = []
        for i in shown:
            color = RED if preds[i] else BLUE
            change = "" if not aa or flags["complex_aa"].iloc[i] else \
                f"{E(str(raw['AA_1'].iloc[i]))} → {E(str(raw['AA_2'].iloc[i]))}"
            row = [E(str(ids[i])), change,
                   f"<span class='bar' style='--bar:{color};width:{probs[i]*8:.2f}rem'></span>{E(fmt_p(probs[i]))}"]
            if y is not None:   # the truth stands beside the score it is read against; agreement is a mark on it, not a column
                # The word, one fixed gap, then the mark: the marks share one x and the column can be scanned
                # straight down for the model's mistakes (CSS .truth). The mark's tone is the row's flags read
                # together with its agreement — a wrong call nothing flagged is the one this page must not bury.
                row.append(f"<span class='truth'>{truth_label(y[i], y[i] == preds[i], bool(flags.iloc[i].any()))}</span>")
            row.append(flag_symbols(flags.iloc[i]))   # the annotations last: they qualify the row, they are not what it says
            rows.append(row)
        # Sorted by score, so the call is monotonic under every filter: the model's answer is the side of
        # this line a row falls on. A filter (or the row cap) that leaves one class means no crossing, and
        # then there is no line to draw — the foot line already names the filter that did it.
        cross = next((k for k, i in enumerate(shown) if not preds[i]), 0)
        divider = (cross, (f"{call_name(True)} ↑", f"{t('threshold')} {thr:.2f}", f"↓ {call_name(False)}")) \
            if 0 < cross < len(shown) else None
        # The trust key hangs off the truth column's own marker rather than joining the legend under the
        # table: measured at 1366 in Turkish the three entries want 265px on a line with 129px left on it,
        # and a coloured mark among the four ink flag marks would read as a fifth flag besides.
        html_table([t("variant"), t("col_aa"), t("score")] + ([(t("col_label"), t("trust_key"))] if y is not None else []) + [t("col_flags")],
                   rows, tall=True, marked=marked, ids=[ids[i] for i in shown], foot=table_foot(count), divider=divider,
                   empty=t("no_matches"))
        # The export, at the right end of the listing's own foot line — the line that names the four flags
        # its four flag columns carry — and drawn as a control rather than as a caption: brand ink over a
        # brand line and a tint, one step under the filled drill-down, at the foot line's own height.
        # The framework's download is what carries the file, whatever its size.
        st.download_button(t("export_calls"), table_csv.to_csv(index=False).encode(), file_name=f"pathosense_{panel}_predictions.csv",
                           mime="text/csv", key="table_export", icon=":material/download:",
                           help=t("export_cols") + (t("export_label") if y is not None else ""))


def keep_pick(variant_id):
    """The chosen variant is remembered per panel, by id: the only writer of pick_keep."""
    ss.setdefault("pick_keep", {})[panel] = str(variant_id)


def remember_pick(ids):
    keep_pick(ids[ss["pick_w"]])


def step_pick(order, ids, offset):
    # Runs before the script re-executes, so it reads the mirror rather than the widget key.
    saved = ss.get("pick_keep", {}).get(panel)
    current = next((i for i in order if str(ids[i]) == saved), order[0])
    keep_pick(ids[order[(order.index(current) + offset) % len(order)]])


def kv(rows):
    st.html("<table class='kv'>" + "".join(f"<tr><td>{E(k)}</td><td><b>{E(v)}</b></td></tr>" for k, v in rows) + "</table>")


def render_explain():
    shell(needs_data=True)   # the nav's own entry for this page is aria-current here: no rule has to say where you are
    if not batch_status():
        return
    raw, probs, flags = res["raw"], res["probs"], res["flags"]
    ids = variant_ids(raw); y = batch_labels(raw)
    mask = batch_mask(res, y, ss.get("filter_keep", ()))
    order = [int(i) for i in np.argsort(-probs, kind="stable") if mask[i]]
    if not order:
        # The filter that emptied this page belongs to the table on Predictions, so the way back is the
        # recovery and it keeps the row it always has. The message is the app's own muted line: the
        # framework's notice paints a blue that on these pages means benign.
        with st.container(horizontal=True, vertical_alignment="center", key="review_head"):
            st.page_link(PAGE["pred"], label=t("tab_pred"), icon=":material/arrow_back:")
        note(t("no_matches"))
        return
    # The pick is per panel and survives every route inside the session. The URL is consulted only when the
    # panel has none — a fresh session, where state is empty — so a deck screenshot's address reopens its screen.
    saved = ss.get("pick_keep", {}).get(panel) or st.query_params.get("variant")
    default = next((i for i in order if str(ids[i]) == saved), min(order, key=lambda i: abs(probs[i]-thr)))
    # The picker's own row is where the way back belongs, ahead of the control that moves within the batch:
    # the underlined nav entry reads as where you are, not as a way out, and a row of its own for one link
    # would cost this page a band. It is the same slot the way in occupies over the table it came from. The
    # two steppers close the row as one bordered pair at the picker's height (CSS .st-key-pick_step): the
    # picker takes the width the link and the pair leave, so the row ends where the page does.
    with st.container(horizontal=True, vertical_alignment="center", key="review_head"):
        st.page_link(PAGE["pred"], label=t("tab_pred"), icon=":material/arrow_back:")
        ss["pick_w"] = default
        st.selectbox(t("variant"), order, key="pick_w", label_visibility="collapsed", help=t("variant"),
            format_func=lambda i: f"{ids[i]} · {fmt_p(probs[i])} · {call_name(probs[i] >= thr)}" +
                (" · ! " + flag_text(flags.iloc[i]) if flags.iloc[i].any() else ""),
            on_change=remember_pick, args=(ids,))
        with st.container(horizontal=True, gap=None, key="pick_step"):   # content width: each label is a drawn ~10px triangle
            st.button(t("previous"), key="pick_prev", help=t("previous"), on_click=step_pick, args=(order, ids, -1))
            st.button(t("next"), key="pick_next", help=t("next"), on_click=step_pick, args=(order, ids, 1))
    pick = ss["pick_w"]
    keep_pick(ids[pick])
    # Written from Python: the address updates, nothing loads. The panel goes with the variant, or the address
    # would reopen the wrong batch: a fresh session starts on General, and a variant of another panel is not
    # in it. The bridge script turns these writes into history replacements (they are one screen's address,
    # not places to go back to), so Back leaves the variant page in one step.
    if st.query_params.get("variant") != str(ids[pick]) or st.query_params.get("panel") != panel:
        st.query_params.from_dict({"panel": panel, "variant": str(ids[pick])})
    # Unfiltered, "300 / 300 variants listed" states nothing this page does not already state. Filtered, how
    # many variants the picker is offering and what narrowed it is worth the line — but not how many of them
    # carry a flag or sit near the threshold: this page is about one variant, those counts are not about it,
    # and where the filter is a flag the count was the filter's own name said back as a number.
    if filtered := active_filter():
        note(" · ".join([t("candidate_count", m=len(order), n=len(probs))] + filtered))
    p = float(probs[pick]); color = RED if p >= thr else BLUE
    row_raw, row_mat, af = raw.iloc[pick], res["matrix"].iloc[pick], res["af_cols"]
    row = res["contribs"].iloc[pick]
    ranked = top_contributions(row, k=CONTRIB_BARS)   # the drivers the contribution chart draws, and names
    # Both charts are built here, before the page draws its first element. Streamlit sends each element to
    # the browser as the script produces it, so a spec built after the column beside it has already gone out
    # arrives that much later and the page assembles in stages. The work is the same; it is now spent ahead
    # of the first element instead of between two, and what the browser receives is one burst.
    fam, contrib = family_chart(row, lang), contrib_chart(row, ranked, row_raw, row_mat, lang, af)
    with st.container(key="review_body"):
        a, b = st.columns([2, 3])
    with a:
        st.html(f"<div class='vh'><span class='vid'>{E(str(ids[pick]))}</span><span class='tag' style='background:{color}'>{E(call_name(p >= thr))}</span></div>")
        if y is not None:
            # The same encoding as the results table, so the two pages cannot disagree about one variant: the
            # flags this page lists below are what decide whether the mark is gold.
            flagged = bool(res["flags"].iloc[pick].any())
            st.html(f"<div class='small'>{E(t('col_label'))}: {truth_label(y[pick], y[pick] == (p >= thr), flagged)}"
                    + hint_mark(t("trust_key")) + "</div>")
        st.altair_chart(score_bars([("", p, color)], thr), width="stretch")
        d = p - thr   # a signed number reads as a distance; the gauge above already carries the threshold itself
        rows = [(t("score"), fmt_p(p)),
                (t("verdict_margin"), t("margin_at") if d == 0 else
                 t("margin_above" if d > 0 else "margin_below", d=f"{abs(d):.3f}") if abs(d) >= 0.0005 else
                 t("margin_tiny_above" if d > 0 else "margin_tiny_below"))]
        # The substitution's chemistry in the units a geneticist reads (Grantham, BLOSUM62) goes on the amino-acid
        # row, its natural home; a multi-residue change has neither and reads as missing.
        # A multi-residue or non-standard code has no substitution to write and no chemistry to write it in:
        # the row is not drawn at all, as the results table's Change column already leaves it blank. The ⇄ in
        # the flag list below says why, once.
        chemistry = [(f, name) for f, name in (("AA_grantham", "Grantham"), ("AA_blosum62", "BLOSUM62")) if f in row_mat.index]
        if "AA_1" in raw and "AA_2" in raw and not flags["complex_aa"].iloc[pick]:
            rows.append((t("aa_change"), " · ".join([f"{fmt_val(row_raw['AA_1'])} → {fmt_val(row_raw['AA_2'])}"] +
                                                    [f"{name} {fmt_int(feature_value(f, row_raw, row_mat), f)}" for f, name in chemistry])))
        else:
            chemistry = []
        kv(rows)
        members = res["own"] is not None and res["transfer"] is not None
        if members:
            member_rows = [(f"{t('member_' + m)} · {t('weight')} {weight:.2f}", float(res[m][pick]), MUTED)
                           for m, weight in (("transfer", art["transfer_w"]), ("own", 1-art["transfer_w"]))]
            st.altair_chart(score_bars(member_rows, thr, tlabel=False), width="stretch")
        lbl(t("input_values"))
        # The chart beside this ranks the same drivers but carries their values only in a tooltip; the list
        # reads them as values, under the three standard inputs a reviewer looks up regardless (one that is
        # also a driver is listed once). One name per feature per screen: a driver is keyed here by the
        # column id its chart label starts with, and only a feature the chart does not draw takes the name.
        # Under NARROW the columns stack, so the list is beside nothing and falls back to the standard three.
        # A chemistry driver already read on the amino-acid row above is not listed a second time.
        inputs = [f for f in dict.fromkeys(["EK_7", "EK_6", "CAT_1"] + (list(ranked.index) if VW > NARROW else []))
                  if f not in {f for f, _ in chemistry}]
        kv([(f if f in ranked.index else readable(f, lang),
             fmt_val(feature_value(f, row_raw, row_mat), f, af)) for f in inputs])
        lbl(t("col_flags"))
        fired = [k for k in FLAG_KEYS if flags.iloc[pick][k]]
        if fired:   # the rule's name is the line; why it fires — and that it is a rule, not a confidence — is the tooltip
            st.html("<ul class='checks'>" + "".join(   # a tab stop with a description: the sentence is reachable without a pointer
                f"<li class='bad' tabindex='0' data-tip='{E(t('flagtext_' + k))}' aria-description='{E(t('flagtext_' + k))}'>"
                f"{E(t('flag_' + k))}</li>" for k in fired) + "</ul>")
    with b:
        # Which panel's model produced the score is the highlighted segment in the header band on this same
        # screen, so the heading no longer says it back.
        lbl(t("contrib_title"), help=t("contrib_help",
            members=", ".join(f"{t('member_' + name)} {t('weight')} {weight:.2f}" for name, weight, _, _ in res["members"])))
        # The whole, then the detail: the family totals first, on their own axis, then the ten drivers.
        st.altair_chart(fam, width="stretch")
        st.altair_chart(contrib, width="stretch", height="stretch" if FILL_CHARTS else "content")
        total = res["base"][pick] + row.sum()
        note(t("contrib_note", base=res["base"][pick], total=total, sig=1 / (1 + np.exp(-total)), p=p))


def render_overview():
    shell()   # no toolbar: Overview reads no batch, so it has neither data controls nor a page action
    lbl(t("results_title"), chips=[t("chip_test_prior")], help=t("results_help"))
    rows = []
    for p in PANEL_ORDER:
        fitted = training_rows(p)   # the row count the model was fitted on, and the single-input baseline the AUC is read against
        # Precision at the test prior, from the same TPR/FPR_anchor pair the curve and the cards read: it is
        # the mean's other term, so F1 becomes arithmetic the reader can do between two adjacent cells
        # (2·r·p/(r+p) reproduces every canonical F1 in this table to three decimals). On general it is 0.423,
        # the least flattering number in the project, and it belongs beside the score it explains.
        prec = f1_at_prior(CANON[p]["TPR"], CANON[p]["FPR_anchor"], TEST_PRIORS[p])[0]
        rows.append([E(t("panel_" + p)), (f"{fitted['n']} · " if fitted else "") + percent(CANON[p]["train"]), percent(TEST_PRIORS[p]),
                     f"{CANON[p]['thr']:.2f}", f"{CANON[p]['TPR']:.3f}", f"{prec:.3f}", f"{CANON[p]['F1']:.3f} ± {CANON[p]['SE']:.3f}",
                     # The single-input baseline is a measured number, not an annotation on one: a parameter-free
                     # comparison on the same rows is the cheapest evidence in the project, so it is set at the
                     # size of the AUC beside it and keeps only the muted colour that says which of the two is ours.
                     f"{CANON[p]['AUC']:.3f}" + (f" <span class='base'>· {E(t('single_input'))} {E(fitted['best'][0])} {fitted['best'][1]:.3f}</span>" if fitted and fitted["best"][0] else ""),
                     f"{CANON[p]['MCC']:.3f}", f"{CANON[p]['PR']:.3f}"])
    html_table([t("op_panel"), t("prior_train"), t("prior_test"), t("threshold"), t("recall"), t("col_prec"), "F1 ± SE", "AUC", "MCC", "PR-AUC"],
               rows, marked=PANEL_ORDER.index(panel))
    opts = prev_options(panel)
    # A prevalence means something different on each panel — the contest's own set is 14.3% here and 28.6% on
    # PAH — so the control returns to this panel's contest set when the panel changes, rather than carrying a
    # number that named one set and now names none. Within a panel the position survives navigation.
    if ss.setdefault("prev_panel", panel) != panel:   # setdefault, not get: a session that reaches this page
        ss["prev_panel"], ss["prev_keep"] = panel, TEST_PRIORS[panel]   # from a deep link has no previous panel, and has not changed panel either
    kept = ss.get("prev_keep")
    if not isinstance(kept, float):   # a cohort key from an older session or a deep link, or no value yet
        kept = TEST_PRIORS[panel]
    ss["prev_w"] = ss["prev_keep"] = p_sel = min(opts, key=lambda v: abs(v - kept))   # seeded from the mirror every run, like the filter, and snapped onto this panel's own stops
    # The curve is built before this row draws anything, for the reason given in render_explain: the three
    # metric cards beside it are small elements and would otherwise be on screen a stage before the chart
    # they annotate, on every change of set.
    curve = prevalence_chart(panel, p_sel)
    with st.container(key="overview_explorer"):
        left, right = st.columns([3, 2])
    with right:
        # One control, not two: the track is continuous and the three known prevalences are marks on it. The
        # rail is a sibling of the slider inside a gapless container and takes over the row Streamlit spends on
        # its own Min/Max labels (hidden in CSS), so naming the three sets costs the band nothing.
        with st.container(key="cohort", gap=None):
            # The value is read off the label rather than off the framework's bubble over the thumb: the bubble
            # needs a row above the track that this band does not have, and a label that already names the
            # variable can carry its value on the line it was going to spend anyway. Plain text and not
            # markdown, because the framework passes the label's source string to the input's aria-label and a
            # `**bold**` value would be read out as asterisks.
            st.select_slider(f"{t('op_cohort')} · {percent(p_sel)}", opts, key="prev_w", on_change=mirror,
                             args=("prev",), format_func=percent,
                             help=t("coh_help", test=percent(pi_test), train=percent(c["train"]), thr=c["thr"]))
            st.html(anchor_rail(panel, opts, p_sel))
        precision, recall, f1 = f1_at_prior(c["TPR"], c["FPR_anchor"], p_sel)
        # What the set does to 1000 variants, before what it does to the score: the picture is the reason the
        # choice exists at all, and the F1 under it is one summary of the same four counts.
        # st.html, like every other block on this page: the app's one st.markdown drew this block at the
        # framework's own 13.07/20.91 body, a fourth size on a page whose scale is 12 / 13.5 / 20, and
        # brought back the container margin and the heading anchors the custom blocks exist to avoid.
        st.html(outcome_bars(p_sel, recall, c["FPR_anchor"]))
        # Recall leads the row and is labelled fixed because it is the one number here that does not move
        # anywhere along the track: the reader drags, and watches it stand still while the precision beside it
        # changes, which is the whole of why F1 differs from set to set and the reason the track is safe.
        # Recall carries the chip because it is the one number here that does not move: the other two are of
        # the set the control above names, and saying so on each of them was the label repeating the control.
        metric_row([(t("recall"), f"{recall:.3f}", t("op_fixed_help", thr=c["thr"]), True),
                    (t("op_prec"), f"{precision:.3f}", t("op_prec_help")),
                    ("F1", f"{f1:.3f}", t("op_note", tpr=percent(recall)))])
    with left:
        # The subject, and the one claim this page must carry without a hover: that the curve is arithmetic
        # and not a measurement. The F1 this project is scored on was a clause here too — the results table at
        # the top of this same screen prints it per panel with this panel's row marked, so it was the heading
        # saying back a number the reader had just read.
        lbl(t("op_chart"), chips=[t("op_extrap")])
        st.altair_chart(curve, width="stretch", height="stretch" if FILL_CHARTS else "content")
    lbl(t("flow_title"))
    st.html(pipeline_html(panel, art))
    imp = get_importance(panel)
    top = top_feature_chart(imp, lang)   # built before its row, like the curve above
    with st.container(key="overview_importance"):
        left, right = st.columns([3, 2])
    with left:
        lbl(t("global_fam"), help=importance_note(art))   # the shares and the chart beside them have one base, so one note
        rows = []
        for family, share in family_importance(imp).items():
            color = FAMILY_COLORS[family]
            # The share is a number, not a bar: at this column's width the smallest families drew 0.78px and
            # 0.52px, which read as a stray mark rather than a quantity, and it is exactly the small share a
            # reader squints at. The swatch already ties the row to the chart's colour, and the ten-bar chart
            # beside it carries the same quantities as bars at full resolution.
            rows.append([f"<span class='sw' style='background:{color}'></span>{E(FAMILY_LABELS[lang][family])}",
                         percent(share), E(t('g_' + family))])
        html_table([t("family"), t("share"), t("description")], rows)
    with right:
        lbl(t("global_top"), help=importance_note(art))
        st.altair_chart(top, width="stretch", height="stretch" if FILL_CHARTS else "content")


# Registered here, where the page functions already exist: st.navigation only has to precede its own run().
PAGE = {key: st.Page(globals()["render_" + key], title=t("tab_" + key), icon=icon, url_path=path,
                     default=(key == "overview"), visibility=vis) for key, path, icon, vis in PAGES}
pg = st.navigation(list(PAGE.values()), position="top")
# A row click is dispatched here: after the pages are registered, so switch_page accepts one, and before the
# page runs, so the click is spent on the way in rather than on the page it came from.
if (vid := take_row_click(_click, ss)) is not None:
    ss.setdefault("pick_keep", {})[ss.setdefault("panel_keep", "general")] = vid
    st.switch_page(PAGE["explain"])
# After navigation, because which page is running decides whether the band carries the upload control, and
# before the page runs, because the band is the first thing drawn on every one of them. The variant page is
# the exception's exception: it carries no upload control, except while the loaded file is one that could not
# be scored, which would otherwise be removable only from the page it already failed on.
panel_bar(uploads=pg is not PAGE["explain"] or bool(ss.get("batch_failed")))
pg.run()

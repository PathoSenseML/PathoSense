"""Explainability for the deployed PathoSense models.

Per-variant feature contributions come from LightGBM's native ``pred_contrib`` (TreeSHAP in
log-odds space) on the gradient-boosting member(s) that scored the row, averaged over the seed
ensemble and weighted by the panel's transfer/own blend weight. Feature families, names for the
features this project engineers, and per-variant confidence flags. Consumed by app/app.py and
tests/test_explain.py.

What the organizers' columns (AL_, EK_, CAT_) measure is not disclosed, so nothing here names
one: they are grouped by prefix and shown by id. A feature this project computes is named by
what it computes, in terms of the column ids it reads.

Honesty note: the deployed probability for cftr/cancer/general is a blend of LightGBM with
RF/CatBoost/XGBoost members. The contributions here describe the LightGBM member(s) only,
which is the model family that carries the blend's decision structure; the other members are
decorrelation partners trained on the same matrix.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

STD_AA = set("ACDEFGHIKLMNPQRSTVWY") | {"*"}

# ── feature families ──────────────────────────────────────────────────────────
FAMILY_LABELS = {
    "en": {"ek": "EK columns",
           "al": "AL columns",
           "amino_acid": "Amino-acid substitution chemistry",
           "cat": "CAT columns",
           "missingness": "Missing-data pattern",
           "other": "Other"},
    "tr": {"ek": "EK sütunları",
           "al": "AL sütunları",
           "amino_acid": "Amino asit değişiminin kimyası",
           "cat": "CAT sütunları",
           "missingness": "Eksik veri örüntüsü",
           "other": "Diğer"},
}

# Only features this project computes. An organizer column is absent by design: see the module
# docstring — readable() falls through to the id for anything not listed here.
_EXACT = {
    "en": {
        "AA_1": "Reference amino acid", "AA_2": "Alternate amino acid",
        "AA_blosum62": "BLOSUM62 substitution score", "AA_grantham": "Grantham distance",
        "AA_hydro_delta": "Change in hydrophobicity", "AA_polarity_delta": "Change in polarity",
        "AA_mw_delta": "Change in molecular weight", "AA_volume_delta": "Change in residue volume",
        "AA_charge_change": "Change in charge", "AA_to_stop": "Substitution to a stop codon",
        "AA_glycine_loss": "Glycine lost", "AA_proline_intro": "Proline introduced",
        "AA_proline_loss": "Proline lost", "AA_cysteine_loss": "Cysteine lost (disulfide bond)",
        "AA_same": "Same amino acid", "AA_is_multiletter": "Multi-residue substitution",
        "AA_is_conservative": "Conservative substitution (BLOSUM ≥ 0)",
        "AA_is_radical": "Radical substitution (BLOSUM ≤ −3)",
        "AA_grantham_high": "Large Grantham distance (≥ 100)", "AA_charge_flipped": "Charge flipped",
        "AA_blosum_abs": "|BLOSUM62|", "AA_blosum_neg_flag": "BLOSUM62 negative",
        "AA_blosum_zero_flag": "BLOSUM62 zero", "AA_blosum_pos_flag": "BLOSUM62 positive",
        "FX_cons_x_disrupt": "EK_7 × chemical disruption",
        "FX_cons_x_conservative": "EK_7 × conservative change",
        "FX_score_agree_count": "Selected, scaled features above 0.5",
        "FX_score_agree_mean": "Mean of selected, scaled features",
        "FX_score_agree_std": "Spread of selected, scaled features",
        "acmg_max_af": "Highest AL_ value on the row", "acmg_ba1": "Highest AL_ value above 0.05",
        "acmg_bs1": "Highest AL_ value above 0.01", "acmg_phastcons": "Mean of EK_4, EK_5, EK_6",
        "acmg_conserved": "EK_7 or EK_2 above 2",
        "ALn_fams": "Number of missingness groups with data",
        "Q_typical_path": "High EK_7 + disruptive change", "Q_hypomorphic": "High EK_7 + mild change",
        "Q_tolerable": "Low EK_7 + disruptive change", "Q_typical_benign": "Low EK_7 + mild change",
        "Q_biochem_composite": "Biochemical disruption composite",
    },
    "tr": {
        "AA_1": "Referans amino asit", "AA_2": "Alternatif amino asit",
        "AA_blosum62": "BLOSUM62 değişim skoru", "AA_grantham": "Grantham uzaklığı",
        "AA_hydro_delta": "Hidrofobiklik değişimi", "AA_polarity_delta": "Polarite değişimi",
        "AA_mw_delta": "Moleküler ağırlık değişimi", "AA_volume_delta": "Yan zincir hacmi değişimi",
        "AA_charge_change": "Yük değişimi", "AA_to_stop": "Dur kodonuna dönüşüm",
        "AA_glycine_loss": "Glisin kaybı", "AA_proline_intro": "Prolin kazanımı",
        "AA_proline_loss": "Prolin kaybı", "AA_cysteine_loss": "Sistein kaybı (disülfid bağı)",
        "AA_same": "Aynı amino asit", "AA_is_multiletter": "Birden fazla amino asit kalıntısını içeren değişim",
        "AA_is_conservative": "Konservatif amino asit değişimi (BLOSUM ≥ 0)",
        "AA_is_radical": "Radikal amino asit değişimi (BLOSUM ≤ −3)",
        "AA_grantham_high": "Büyük Grantham uzaklığı (≥ 100)", "AA_charge_flipped": "Yük tersine döndü",
        "AA_blosum_abs": "|BLOSUM62|", "AA_blosum_neg_flag": "BLOSUM62 negatif",
        "AA_blosum_zero_flag": "BLOSUM62 sıfır", "AA_blosum_pos_flag": "BLOSUM62 pozitif",
        "FX_cons_x_disrupt": "EK_7 × kimyasal bozulma",
        "FX_cons_x_conservative": "EK_7 × konservatif amino asit değişimi",
        # One phrase for the trio, short enough that the longest of them still fits the
        # contribution chart's axis label beside its column id (it used to truncate).
        "FX_score_agree_count": "0.5 üzerindeki seçilmiş ölçeklenmiş özellik sayısı",
        "FX_score_agree_mean": "Seçilmiş ölçeklenmiş özellik ortalaması",
        "FX_score_agree_std": "Seçilmiş ölçeklenmiş özellik yayılımı",
        "acmg_max_af": "Satırdaki en yüksek AL_ değeri", "acmg_ba1": "En yüksek AL_ değeri 0,05 üzerinde",
        "acmg_bs1": "En yüksek AL_ değeri 0,01 üzerinde", "acmg_phastcons": "EK_4, EK_5, EK_6 ortalaması",
        "acmg_conserved": "EK_7 veya EK_2 değeri 2 üzerinde",
        "ALn_fams": "Veri içeren eksiklik grubu sayısı",
        "Q_typical_path": "Yüksek EK_7 + yıkıcı değişim", "Q_hypomorphic": "Yüksek EK_7 + hafif değişim",
        "Q_tolerable": "Düşük EK_7 + yıkıcı değişim", "Q_typical_benign": "Düşük EK_7 + hafif değişim",
        "Q_biochem_composite": "Bileşik biyokimyasal bozulma skoru",
    },
}

_TEMPLATES = {
    # ALfam_ groups AL_ columns by their missingness pattern — a pattern this project measured in the data,
    # not a property the organizers stated, so the name says missingness and nothing about the columns.
    "en": {"missing": "Missing: {base}", "alfam": "Missingness group {n} has data"},
    "tr": {"missing": "Eksik: {base}", "alfam": "Eksiklik grubu {n}: veri var"},
}

_AL_RE = re.compile(r"^AL_(\d+)$")
_ALFAM_RE = re.compile(r"^ALfam_(\d+)$")


def family_of(feat: str) -> str:
    """Map a model feature name to one of the FAMILY_LABELS ids."""
    base = feat[:-8] if feat.endswith("_missing") else feat
    if feat.endswith("_missing"):
        return "missingness"
    if base.startswith("EK_") or base in ("acmg_phastcons", "acmg_conserved") or base.startswith("FX_cons"):
        return "ek"   # the EK_ columns and the features read off them
    if base.startswith("ALfam_") or base == "ALn_fams":   # AL_ missingness-pattern groups: their own names say missingness
        return "missingness"
    if base.startswith("AL") or base in ("acmg_max_af", "acmg_ba1", "acmg_bs1"):
        return "al"
    if base.startswith("FX_score_agree"):   # an aggregate of AUC-selected numeric features, not substitution chemistry
        return "other"
    if base.startswith("AA_") or base.startswith("Q_") or base.startswith("FX_"):
        return "amino_acid"
    if base.startswith("CAT_"):
        return "cat"
    return "other"


def readable(feat: str, lang: str = "en") -> str:
    """Name for a feature this project computes. An organizer column (AL_, EK_, CAT_) is returned as its
    own id: what it measures is not disclosed, and a caller that shows this beside a number must not put
    a guess there."""
    ex, tp = _EXACT.get(lang, _EXACT["en"]), _TEMPLATES.get(lang, _TEMPLATES["en"])
    if feat.endswith("_missing"):
        return tp["missing"].format(base=readable(feat[:-8], lang))
    if feat in ex:
        return ex[feat]
    m = _ALFAM_RE.match(feat)
    if m:
        return tp["alfam"].format(n=m.group(1))
    return feat


def af_columns(df_raw: pd.DataFrame) -> set:
    """AL_ columns carrying something other than 0/1 in this batch. Observed from the batch itself and
    used only to decide how a value is printed (a number, or yes/no); it says nothing about what the
    column measures, which the organizers do not disclose."""
    out = set()
    for c in df_raw.columns:
        if not _AL_RE.match(str(c)):
            continue
        v = pd.to_numeric(df_raw[c], errors="coerce").dropna().unique()
        if len(v) and not set(v).issubset({0.0, 1.0}):
            out.add(c)
    return out


# ── contributions ─────────────────────────────────────────────────────────────

def lgbm_contribs(models, X: pd.DataFrame):
    """Mean TreeSHAP contributions (log-odds) over a list of LGBMClassifiers.
    Returns (contribs DataFrame n×features, base_value array n)."""
    C = np.mean([m.booster_.predict(X, pred_contrib=True) for m in models], axis=0)
    return pd.DataFrame(C[:, :-1], columns=list(X.columns), index=X.index), C[:, -1]


def explain_frame(art: dict, scored: dict) -> dict:
    """Blend-weighted contributions for every row of a score_frame() result.

    Returns {"contribs": DataFrame (rows × union of features), "base": array,
             "members": [(label, weight, contribs, base)]} where weights are the deployed
    transfer/own blend weights (own=1 when there is no transfer)."""
    w = art["transfer_w"]
    members = []
    if art.get("own") and scored.get("X_lgbm") is not None:
        c, b = lgbm_contribs(art["own"]["models"], scored["X_lgbm"])
        members.append(("own", 1.0 if w is None else 1.0 - w, c, b))
    if art.get("transfer") and scored.get("Xs") is not None:
        tart = art["transfer"]["art"]
        c, b = lgbm_contribs(tart.get("models") or [tart["model"]], scored["Xs"])
        members.append(("transfer", float(w), c, b))
    members = [m for m in members if m[1] > 0]
    cols = sorted(set().union(*[set(m[2].columns) for m in members]))
    total = sum(m[1] * m[2].reindex(columns=cols).fillna(0.0) for m in members)
    base = sum(m[1] * m[3] for m in members)
    return {"contribs": total, "base": np.asarray(base), "members": members}


def top_contributions(contribs_row: pd.Series, k: int = 12) -> pd.Series:
    """The k largest |contribution| features of one row, sorted by contribution."""
    nz = contribs_row[contribs_row != 0]
    return nz.reindex(nz.abs().sort_values(ascending=False).index[:k])


def feature_value(feat: str, row_raw: pd.Series, row_matrix: pd.Series):
    """Value to display next to a contribution: the raw input when the column exists in the input row
    (any AL_, EK_, CAT_ or AA_ column of the file), else the engineered / transformed value from the
    model matrix."""
    if feat in row_raw.index:
        return row_raw[feat]
    return row_matrix.get(feat, np.nan)


# ── global importance ─────────────────────────────────────────────────────────

def _gain_series(models, feats) -> pd.Series:
    g = np.zeros(len(feats))
    for m in models:
        gi = m.booster_.feature_importance(importance_type="gain").astype(float)
        g += gi / max(gi.sum(), 1e-12)
    return pd.Series(g / len(models), index=feats)


def global_importance(art: dict) -> pd.Series:
    """Blend-weighted, normalized gain importance of the LightGBM member(s). Sums to 1."""
    w = art["transfer_w"]
    parts = []
    if art.get("own"):
        models = art["own"]["models"]
        feats = models[0].booster_.feature_name()
        parts.append((1.0 if w is None else 1.0 - w, _gain_series(models, feats)))
    if art.get("transfer"):
        tart = art["transfer"]["art"]
        models = tart.get("models") or [tart["model"]]
        parts.append((float(w), _gain_series(models, models[0].booster_.feature_name())))
    parts = [p for p in parts if p[0] > 0]
    idx = sorted(set().union(*[set(p[1].index) for p in parts]))
    s = sum(p[0] * p[1].reindex(idx).fillna(0.0) for p in parts)
    return s / max(s.sum(), 1e-12)


def family_importance(imp: pd.Series) -> pd.Series:
    return imp.groupby(imp.index.map(family_of)).sum().sort_values(ascending=False)


# ── confidence flags ──────────────────────────────────────────────────────────

FLAG_KEYS = ("ek7_missing", "complex_aa", "cat6_flag", "borderline")


def confidence_flags(df_raw: pd.DataFrame, probs, threshold: float, margin: float = 0.05) -> pd.DataFrame:
    """Per-row reliability flags (boolean columns FLAG_KEYS). The first three mark subsets on
    which the deployed model measurably loses discrimination (analysis/exp_confidence.py);
    ``borderline`` marks probabilities within ``margin`` of the decision threshold."""
    n = len(df_raw); idx = df_raw.index
    ek7 = df_raw["EK_7"].isna() if "EK_7" in df_raw else pd.Series(False, index=idx)
    aa1 = df_raw["AA_1"].astype(object) if "AA_1" in df_raw else pd.Series([None] * n, index=idx)
    aa2 = df_raw["AA_2"].astype(object) if "AA_2" in df_raw else pd.Series([None] * n, index=idx)
    complex_aa = ~(aa1.isin(STD_AA) & aa2.isin(STD_AA))
    region = df_raw["CAT_6"].notna() if "CAT_6" in df_raw else pd.Series(False, index=idx)
    border = pd.Series(np.abs(np.asarray(probs, dtype=float) - threshold) < margin, index=idx)
    return pd.DataFrame({"ek7_missing": ek7.values, "complex_aa": complex_aa.values,
                         "cat6_flag": region.values, "borderline": border.values}, index=idx)


# ── operating point ───────────────────────────────────────────────────────────

def f1_at_prior(tpr: float, fpr: float, prior: float):
    """(precision, recall, F1) of a fixed classifier with the given TPR/FPR when the
    positive class has prevalence ``prior``. This is the analytic F1* used throughout the
    evaluation harness."""
    tp = prior * tpr; fn = prior * (1 - tpr); fp = (1 - prior) * fpr
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
    return prec, tpr, f1


def fpr_for_f1(tpr: float, prior: float, f1: float) -> float:
    """The FPR at which a classifier with recall ``tpr`` reaches ``f1`` at prevalence ``prior``
    (inverse of f1_at_prior in its FPR argument). Used to anchor the prevalence curves at the
    canonical cross-validated F1: the canonical value is a mean of per-fold F1, which the F1 of
    pooled confusion counts does not reproduce on the small panels."""
    tp = prior * tpr; fn = prior * (1 - tpr)
    return max(0.0, (2 * tp / f1 - 2 * tp - fn) / (1 - prior))


if __name__ == "__main__":
    # self-check: naming, families, the analytic F1 and a contribution round-trip on a tiny model
    assert family_of("EK_7") == "ek" and family_of("AL_3_missing") == "missingness"
    assert family_of("ALfam_2") == "missingness" and family_of("ALn_fams") == "missingness"
    assert family_of("AL_12") == "al" and family_of("CAT_6") == "cat"
    # No organizer column carries a name in any language: the id is the name.
    for _f in ("EK_7", "EK_6", "AL_12", "CAT_1", "CAT_6"):
        assert readable(_f) == _f and readable(_f, "tr") == _f, _f
    assert readable("AA_grantham") == "Grantham distance" and readable("ALfam_4").startswith("Missingness group 4")
    for tpr_, pi_, f1_ in [(0.8403, 0.167, 0.851), (0.8325, 0.286, 0.669), (0.7353, 0.143, 0.537)]:
        assert abs(f1_at_prior(tpr_, fpr_for_f1(tpr_, pi_, f1_), pi_)[2] - f1_) < 1e-9
    assert readable("EK_7_missing", "tr") == "Eksik: EK_7"
    assert af_columns(pd.DataFrame({"AL_1": [1.0, 0.0, np.nan], "AL_2": [1e-5, 1.0, np.nan]})) == {"AL_2"}
    assert af_columns(pd.DataFrame({"AL_1": [1e-40, 1.0]})) == {"AL_1"}
    p, r, f = f1_at_prior(1.0, 0.0, 0.2); assert (p, r, f) == (1.0, 1.0, 1.0)
    _, _, f = f1_at_prior(0.7353, 0.1714, 0.143); assert abs(f - 0.532) < 0.002, f
    import lightgbm as lgb
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=["EK_7", "AL_1", "AA_grantham", "CAT_1"])
    X["CAT_1"] = pd.Categorical(rng.choice(["a", "b"], 300))
    y = (X["EK_7"] + 0.5 * X["AL_1"] > 0).astype(int)
    m = lgb.LGBMClassifier(n_estimators=30, verbose=-1).fit(X, y)
    C, b = lgbm_contribs([m], X)
    logit = C.sum(axis=1).values + b
    assert np.allclose(1 / (1 + np.exp(-logit)), m.predict_proba(X)[:, 1], atol=1e-6)
    fl = confidence_flags(pd.DataFrame({"EK_7": [1.0, np.nan], "AA_1": ["A", "AG"], "AA_2": ["V", "S"],
                                        "CAT_6": [None, "x"]}), [0.83, 0.2], 0.81)
    assert fl["ek7_missing"].tolist() == [False, True] and fl["complex_aa"].tolist() == [False, True]
    assert fl["cat6_flag"].tolist() == [False, True] and fl["borderline"].tolist() == [True, False]
    print("explain.py self-check OK")

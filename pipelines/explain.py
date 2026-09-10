"""Explainability for the deployed PathoSense models.

Per-variant feature contributions come from LightGBM's native ``pred_contrib`` (TreeSHAP in
log-odds space) on the gradient-boosting member(s) that scored the row, averaged over the seed
ensemble and weighted by the panel's transfer/own blend weight. Readable names, feature
families and per-variant confidence flags turn the anonymized column ids into something a
clinician or a juror can read. Consumed by app/app.py and tests/test_explain.py.

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
    "en": {"conservation": "Evolutionary conservation",
           "allele_frequency": "Population allele frequency",
           "amino_acid": "Amino-acid substitution chemistry",
           "population_flags": "Population and region flags",
           "missingness": "Missing-data pattern",
           "other": "Other"},
    "tr": {"conservation": "Evrimsel korunmuşluk",
           "allele_frequency": "Popülasyon alel frekansı",
           "amino_acid": "Amino asit değişiminin kimyası",
           "population_flags": "Popülasyon ve bölge bayrakları",
           "missingness": "Eksik veri örüntüsü",
           "other": "Diğer"},
}

_EXACT = {
    "en": {
        "EK_1": "GERP++ neutral rate", "EK_2": "GERP++ rejected substitutions",
        "EK_3": "phyloP (sparse alignment)", "EK_4": "phastCons (100-way or 17-way)",
        "EK_5": "phastCons (17-way or 100-way)", "EK_6": "phastCons 470-way mammalian",
        "EK_7": "phyloP 100-way vertebrate", "EK_8": "phyloP 17-way primate",
        "EK_9": "phyloP 470-way mammalian",
        "AA_1": "Reference amino acid", "AA_2": "Alternate amino acid",
        "CAT_1": "gnomAD population with highest AF",
        "CAT_2": "All of Us population with highest AF",
        "CAT_3": "Ancestral genotype annotation A", "CAT_4": "Ancestral genotype annotation B",
        "CAT_5": "Ancestral genotype annotation C",
        "CAT_6": "Low-quality genomic region flag (LCR / segdup / decoy)",
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
        "FX_cons_x_disrupt": "Conservation × chemical disruption",
        "FX_cons_x_conservative": "Conservation × conservative change",
        "FX_score_agree_count": "Selected, scaled features above 0.5",
        "FX_score_agree_mean": "Mean of selected, scaled features",
        "FX_score_agree_std": "Spread of selected, scaled features",
        "acmg_max_af": "Maximum observed allele frequency", "acmg_ba1": "ACMG BA1 (AF > 5%)",
        "acmg_bs1": "ACMG BS1 (AF > 1%)", "acmg_phastcons": "Mean phastCons",
        "acmg_conserved": "Conserved site (PP3-style)",
        "ALn_fams": "Number of missingness groups with data",
        "Q_typical_path": "Conserved site + disruptive change", "Q_hypomorphic": "Conserved site + mild change",
        "Q_tolerable": "Variable site + disruptive change", "Q_typical_benign": "Variable site + mild change",
        "Q_biochem_composite": "Biochemical disruption composite",
    },
    "tr": {
        "EK_1": "GERP++ nötr değişim hızı", "EK_2": "GERP++ reddedilen nükleotit değişimleri",
        "EK_3": "phyloP (seyrek hizalama)", "EK_4": "phastCons (100 omurgalı / 17 primat ayrımı belirsiz)",
        "EK_5": "phastCons (17 primat / 100 omurgalı ayrımı belirsiz)", "EK_6": "phastCons 470 memeli",
        "EK_7": "phyloP 100 omurgalı", "EK_8": "phyloP 17 primat", "EK_9": "phyloP 470 memeli",
        "AA_1": "Referans amino asit", "AA_2": "Alternatif amino asit",
        "CAT_1": "En yüksek AF’li gnomAD popülasyonu",
        "CAT_2": "En yüksek AF’li All of Us popülasyonu",
        "CAT_3": "Atasal genotip anotasyonu A", "CAT_4": "Atasal genotip anotasyonu B", "CAT_5": "Atasal genotip anotasyonu C",
        "CAT_6": "Düşük kaliteli genom bölgesi bayrağı (LCR / segdup / decoy)",
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
        "FX_cons_x_disrupt": "Korunmuşluk × kimyasal bozulma",
        "FX_cons_x_conservative": "Korunmuşluk × konservatif amino asit değişimi",
        # One phrase for the trio, short enough that the longest of them still fits the
        # contribution chart's axis label beside its column id (it used to truncate).
        "FX_score_agree_count": "0.5 üzerindeki seçilmiş ölçeklenmiş özellik sayısı",
        "FX_score_agree_mean": "Seçilmiş ölçeklenmiş özellik ortalaması",
        "FX_score_agree_std": "Seçilmiş ölçeklenmiş özellik yayılımı",
        "acmg_max_af": "Gözlenen en yüksek alel frekansı", "acmg_ba1": "ACMG BA1 (AF > %5)",
        "acmg_bs1": "ACMG BS1 (AF > %1)", "acmg_phastcons": "Ortalama phastCons",
        "acmg_conserved": "Korunmuş konum (PP3 benzeri)",
        "ALn_fams": "Veri içeren eksiklik grubu sayısı",
        "Q_typical_path": "Korunmuş konum + yıkıcı değişim", "Q_hypomorphic": "Korunmuş konum + hafif değişim",
        "Q_tolerable": "Değişken konum + yıkıcı değişim", "Q_typical_benign": "Değişken konum + hafif değişim",
        "Q_biochem_composite": "Bileşik biyokimyasal bozulma skoru",
    },
}

_TEMPLATES = {
    # The AL_ values are organizer-transformed, not real frequencies (analysis/column_identification): the name says so,
    # because a value like 2.33e-14 is below any possible allele-count floor and a geneticist will notice.
    "en": {"missing": "Missing: {base}", "af": "Transformed AF · population column {n}",
           "al_flag": "Population flag column {n}", "alfam": "Missingness group {n} has data",
           "cat": "Categorical column {n}", "ek": "Conservation score {n}", "aa": "Amino-acid feature {n}"},
    "tr": {"missing": "Eksik: {base}", "af": "Dönüştürülmüş AF · popülasyon sütunu {n}",
           "al_flag": "Popülasyon bayrak sütunu {n}", "alfam": "Eksiklik grubu {n}: veri var",
           "cat": "Kategorik sütun {n}", "ek": "Korunmuşluk skoru {n}", "aa": "Amino asit özelliği {n}"},
}

_AL_RE = re.compile(r"^AL_(\d+)$")
_ALFAM_RE = re.compile(r"^ALfam_(\d+)$")


def family_of(feat: str) -> str:
    """Map a model feature name to one of the FAMILY_LABELS ids."""
    base = feat[:-8] if feat.endswith("_missing") else feat
    if feat.endswith("_missing"):
        return "missingness"
    if base.startswith("EK_") or base in ("acmg_phastcons", "acmg_conserved") or base.startswith("FX_cons"):
        return "conservation"
    if base.startswith("ALfam_") or base == "ALn_fams":   # AL_ missingness-pattern groups: their own names say missingness
        return "missingness"
    if base.startswith("AL") or base in ("acmg_max_af", "acmg_ba1", "acmg_bs1"):
        return "allele_frequency"
    if base.startswith("FX_score_agree"):   # an aggregate of AUC-selected numeric features, not substitution chemistry
        return "other"
    if base.startswith("AA_") or base.startswith("Q_") or base.startswith("FX_"):
        return "amino_acid"
    if base.startswith("CAT_"):
        return "population_flags"
    return "other"


def readable(feat: str, lang: str = "en", af_cols=None) -> str:
    """Plain-language name for a model feature. ``af_cols`` (optional set) marks which AL_
    columns are allele-frequency values (rank-transformed at training) as opposed to flags."""
    ex, tp = _EXACT.get(lang, _EXACT["en"]), _TEMPLATES.get(lang, _TEMPLATES["en"])
    if feat.endswith("_missing"):
        return tp["missing"].format(base=readable(feat[:-8], lang, af_cols))
    if feat in ex:
        return ex[feat]
    m = _AL_RE.match(feat)
    if m:
        key = "af" if (af_cols is None or feat in af_cols) else "al_flag"
        return tp[key].format(n=m.group(1))
    m = _ALFAM_RE.match(feat)
    if m:
        return tp["alfam"].format(n=m.group(1))
    if feat.startswith("CAT_"):
        return tp["cat"].format(n=feat[4:])
    if feat.startswith("EK_"):
        return tp["ek"].format(n=feat[3:])
    if feat.startswith("AA_"):
        return tp["aa"].format(n=feat[3:])
    return feat


def verified_name(feat: str, lang: str = "en") -> str | None:
    """The dictionary's confirmed name for a column, or None where there is none: a caller that needs a
    name it can stand behind (a chart cell beside a number) falls back to the column id rather than to a
    template that only restates the prefix."""
    return _EXACT.get(lang, _EXACT["en"]).get(feat)


def af_columns(df_raw: pd.DataFrame) -> set:
    """AL_ columns that carry allele-frequency VALUES in this batch (anything not purely 0/1).
    Heuristic from the batch itself: a column whose observed values are all 0/1 is labelled a
    population flag. The exact population behind each column is unrecoverable (transformed)."""
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
    """Value to display next to a contribution: the raw input when the column exists in the
    input row (conservation, categorical, amino acid, allele frequency), else the engineered /
    transformed value from the model matrix."""
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

FLAG_KEYS = ("cons_missing", "complex_aa", "region_flag", "borderline")


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
    return pd.DataFrame({"cons_missing": ek7.values, "complex_aa": complex_aa.values,
                         "region_flag": region.values, "borderline": border.values}, index=idx)


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
    assert family_of("EK_7") == "conservation" and family_of("AL_3_missing") == "missingness"
    assert family_of("ALfam_2") == "missingness" and family_of("ALn_fams") == "missingness"
    assert family_of("AL_12") == "allele_frequency" and family_of("CAT_6") == "population_flags"
    assert readable("EK_7") == "phyloP 100-way vertebrate"
    assert readable("AL_12", af_cols={"AL_12"}) == "Transformed AF · population column 12"
    assert verified_name("EK_7", "tr") == "phyloP 100 omurgalı" and verified_name("AL_12") is None
    assert readable("AL_12", af_cols=set()) == "Population flag column 12"
    assert readable("EK_6") == "phastCons 470-way mammalian"
    for tpr_, pi_, f1_ in [(0.8403, 0.167, 0.851), (0.8325, 0.286, 0.669), (0.7353, 0.143, 0.537)]:
        assert abs(f1_at_prior(tpr_, fpr_for_f1(tpr_, pi_, f1_), pi_)[2] - f1_) < 1e-9
    assert readable("EK_7_missing", "tr") == "Eksik: phyloP 100 omurgalı"
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
                                        "CAT_6": [None, "lcr"]}), [0.83, 0.2], 0.81)
    assert fl["cons_missing"].tolist() == [False, True] and fl["complex_aa"].tolist() == [False, True]
    assert fl["region_flag"].tolist() == [False, True] and fl["borderline"].tolist() == [True, False]
    print("explain.py self-check OK")

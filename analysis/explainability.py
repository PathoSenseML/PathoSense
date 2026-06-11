"""Explainability for the deployed models (PSR Comment 11: SHAP / feature-group analysis).

Each panel is driven by a different model:
  general            -> its own LightGBM (native-NaN, all features)
  cftr / pah / cancer-> the GENERAL-trained transfer scorer's LightGBM (what actually
                        scores those variants; cancer also has a 50% own component)

We compute SHAP via LightGBM's native pred_contrib (robust with native categoricals),
on the variants each model scores, then aggregate |SHAP| into interpretable FEATURE
GROUPS (conservation / allele-frequency / categorical / amino-acid / missingness) and
the top individual features (annotated with the verified column dictionary). Output:
docs/panel_research/explainability/ (EXPLAINABILITY.md + per-panel beeswarm PNGs).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import shap; HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False

from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from feature_engineering import (engineer_aa_features, engineer_acmg_features,
                                 engineer_interaction_features)
from preprocess_v2 import apply_keep_nan, apply_al
from predict import _engineer, _apply_own_preprocess
from utils import PANELS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "docs", "panel_research", "explainability")
os.makedirs(OUT, exist_ok=True)

# column-family dictionary for annotating top features
DICT = {
    "EK_1": "GERP++ NR", "EK_2": "GERP++ RS", "EK_3": "phyloP (sparse, 470-fam)",
    "EK_4": "phastCons", "EK_5": "phastCons", "EK_6": "phastCons 470way",
    "EK_7": "phyloP100way vertebrate", "EK_8": "phyloP17way primate",
    "EK_9": "phyloP470way mammalian",
    "CAT_1": "gnomAD popmax pop", "CAT_2": "All-of-Us popmax pop",
    "CAT_3": "archaic genotype", "CAT_4": "archaic genotype", "CAT_5": "archaic genotype",
    "CAT_6": "region flags (lcr/segdup/decoy)", "AL_185": "TOPMed allele number (AN)",
    "AA_1": "reference amino acid", "AA_2": "alternate amino acid",
}


def group_of(col):
    c = str(col)
    if c.endswith("_missing"):
        return "Missingness indicators"
    if c.startswith("EK_"):
        return "Conservation (EK)"
    if c.startswith(("AL_", "ALfam_", "ALn_", "ALmax", "ALmean", "ALpresent")):
        return "Allele frequency (AL)"
    if c.startswith("CAT_"):
        return "Categorical (population/region)"
    if c.startswith("AA_"):
        return "Amino acid (engineered)"
    if c.startswith(("FX_", "acmg_", "Q_")):
        return "Engineered meta (score-agree/ACMG/interactions)"
    return "Other"


def _project_transfer(panel, tart):
    """Rebuild the target matrix in the transfer scorer's feature space (mirrors
    transfer.transfer_score up to model input)."""
    df = pd.read_csv(PANELS[panel]); df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df); df = engineer_aa_features(df)
    X = df.reindex(columns=tart["raw_cols"])
    Xs = apply_keep_nan(X, tart["ppf"])
    if tart["alf"] is not None:
        Xs = pd.concat([Xs.reset_index(drop=True), apply_al(X, tart["alf"]).reset_index(drop=True)], axis=1)
        Xs = Xs.loc[:, ~Xs.columns.duplicated()]
    for col in tart["feats"]:
        if col not in Xs.columns:
            Xs[col] = np.nan
    Xs = Xs[tart["feats"]]
    for col, cats in tart["cat_cats"].items():
        Xs[col] = pd.Categorical(Xs[col].astype(object), categories=cats)
    return Xs, df["Label"].reset_index(drop=True)


def _own_matrix(panel, art):
    cfg = art["recipe"]
    df = pd.read_csv(PANELS[panel]); df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = _engineer(df, cfg)
    avail = [c for c in art["feature_cols"] if c in df.columns]
    X = df[avail].copy()
    return _apply_own_preprocess(X, art["own"]["prep_art"]), df["Label"].reset_index(drop=True)


def analyze(panel):
    art = joblib.load(os.path.join(BASE, "models", panel, "artifacts.joblib"))
    if art["transfer"]:
        model = art["transfer"]["art"]["model"]
        X, y = _project_transfer(panel, art["transfer"]["art"])
        driver = f"general-transfer scorer (w={art['transfer_w']})"
    else:
        model = art["own"]["models"][0]
        X, y = _own_matrix(panel, art)
        driver = "own model (native-NaN, all features)"
    feats = list(model.booster_.feature_name())
    X = X[[c for c in feats if c in X.columns]]
    # LightGBM native SHAP (handles categoricals); last col is base value
    contrib = model.booster_.predict(X, pred_contrib=True)
    sv = contrib[:, :-1]
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
    imp["group"] = imp["feature"].map(group_of)
    total = imp["mean_abs_shap"].sum() or 1.0
    grp = (imp.groupby("group")["mean_abs_shap"].sum() / total * 100).sort_values(ascending=False)
    top = imp.sort_values("mean_abs_shap", ascending=False).head(12).copy()
    top["name"] = top["feature"].map(lambda c: DICT.get(c, ""))
    # beeswarm plot
    plotted = False
    if HAVE_SHAP:
        try:
            Xn = X.copy()
            for c in Xn.columns:
                if isinstance(Xn[c].dtype, pd.CategoricalDtype):
                    Xn[c] = Xn[c].cat.codes
            plt.figure()
            shap.summary_plot(sv, Xn, max_display=15, show=False)
            plt.title(f"{panel.upper()} — SHAP (driver: {driver})")
            plt.tight_layout(); plt.savefig(os.path.join(OUT, f"shap_{panel}.png"), dpi=130)
            plt.close(); plotted = True
        except Exception as e:
            print(f"  [{panel}] beeswarm failed: {e}")
    return panel, driver, len(feats), grp, top, plotted


def main():
    sections = []
    for panel in ["general", "cancer", "pah", "cftr"]:
        panel, driver, nfeat, grp, top, plotted = analyze(panel)
        print(f"{panel}: driver={driver}, {nfeat} feats, plot={plotted}", flush=True)
        lines = [f"## {panel.upper()}  (driver: {driver}; {nfeat} features)\n",
                 "**Feature-group contribution (% of total |SHAP|):**\n",
                 "| group | % |", "|---|---|"]
        for g, v in grp.items():
            lines.append(f"| {g} | {v:.1f}% |")
        lines += ["", "**Top features:**\n", "| feature | dictionary | mean|SHAP| |", "|---|---|---|"]
        for _, r in top.iterrows():
            lines.append(f"| {r['feature']} | {r['name']} | {r['mean_abs_shap']:.4f} |")
        if plotted:
            lines += ["", f"![{panel} SHAP](shap_{panel}.png)"]
        sections.append("\n".join(lines))
    header = ("# Explainability — deployed models (SHAP feature-group analysis)\n\n"
              "SHAP via LightGBM `pred_contrib` on the variants each panel's driving model "
              "scores. general uses its own all-features model; cftr/pah/cancer are scored by "
              "the general-trained transfer scorer (cancer additionally blends a 50% own model). "
              "Aggregated into feature groups + top individual features (verified column dict).\n")
    with open(os.path.join(OUT, "EXPLAINABILITY.md"), "w") as f:
        f.write(header + "\n\n---\n\n".join(sections) + "\n")
    print(f"\nsaved -> {os.path.join(OUT, 'EXPLAINABILITY.md')}")


if __name__ == "__main__":
    main()

"""Render SHAP results as images: regenerate per-panel beeswarms on the CURRENT deployed models
and build a consolidated feature-group summary figure (SHAP % vs permutation-importance %).

Run:  .venv/bin/python -u analysis/make_shap_figure.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explainability import analyze, OUT

PANELS = ["general", "cancer", "pah", "cftr"]
SHORT = {
    "Allele frequency (AL)": "AL\n(allele freq)", "Conservation (EK)": "EK\n(conservation)",
    "Amino acid (engineered)": "AA\n(substitution)", "Categorical (population/region)": "CAT",
    "Missingness indicators": "Missing", "Engineered meta (score-agree/ACMG/interactions)": "Engineered",
    "Other": "Other",
}
ORDER = ["AL\n(allele freq)", "EK\n(conservation)", "AA\n(substitution)", "CAT", "Missing", "Engineered", "Other"]
COLORS = {"AL\n(allele freq)": "#d62728", "EK\n(conservation)": "#1f77b4", "AA\n(substitution)": "#2ca02c",
          "CAT": "#9467bd", "Missing": "#8c564b", "Engineered": "#ff7f0e", "Other": "#7f7f7f"}


def pct_map(series, as_pct_of_pos=False):
    d = {}
    for k, v in series.items():
        d[SHORT.get(str(k), str(k))] = v
    if as_pct_of_pos:
        tot = sum(max(x, 0) for x in d.values()) or 1.0
        d = {k: max(v, 0) / tot * 100 for k, v in d.items()}
    return d


def main():
    res = {}
    for p in PANELS:
        out = analyze(p)
        _, driver, nfeat, grp, top, plotted, base_auc, gpi = out
        res[p] = {"shap": pct_map(grp), "perm": pct_map(gpi, as_pct_of_pos=True),
                  "top": top, "driver": driver, "auc": base_auc, "nfeat": nfeat}
        print(f"{p}: AUC={base_auc:.3f}, top SHAP feats={list(top['feature'][:3])}", flush=True)

    groups = [g for g in ORDER if any(g in res[p]["shap"] for p in PANELS)]
    x = np.arange(len(PANELS)); width = 0.8 / len(groups)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, key, title in [(axes[0], "shap", "SHAP contribution (% of total |SHAP|)"),
                           (axes[1], "perm", "Permutation importance (% of total AUC drop)")]:
        for i, g in enumerate(groups):
            vals = [res[p][key].get(g, 0.0) for p in PANELS]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=g.replace("\n", " "),
                   color=COLORS.get(g, "#333"))
        ax.set_xticks(x); ax.set_xticklabels([p.upper() for p in PANELS], fontsize=11)
        ax.set_ylabel("% importance"); ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    axes[1].legend(title="feature group", fontsize=9, loc="upper right")
    fig.suptitle("PathoSense — feature-group importance per panel  (AL ≫ EK > AA ≫ CAT, "
                 "agreed by SHAP + permutation)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(OUT, "shap_group_summary.png")
    fig.savefig(out1, dpi=150); plt.close(fig)
    print(f"saved -> {out1}", flush=True)

    # top-features-per-panel figure (horizontal bars of mean|SHAP|, annotated with dictionary)
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    for ax, p in zip(axes2.ravel(), PANELS):
        top = res[p]["top"].head(10).iloc[::-1]
        labels = [f"{r.feature}" + (f"  ({r['name']})" if r["name"] else "")
                  for _, r in top.iterrows()]
        cols = [COLORS.get(SHORT.get(_grp(f), "Other"), "#333") for f in top["feature"]]
        ax.barh(range(len(top)), top["mean_abs_shap"], color=cols)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{p.upper()}  (driver AUC {res[p]['auc']:.3f})", fontsize=11, weight="bold")
        ax.set_xlabel("mean |SHAP|"); ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
    fig2.suptitle("PathoSense — top features by mean|SHAP| per panel", fontsize=14, weight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = os.path.join(OUT, "shap_top_features.png")
    fig2.savefig(out2, dpi=150); plt.close(fig2)
    print(f"saved -> {out2}", flush=True)


def _grp(c):
    from explainability import group_of
    return group_of(c)


if __name__ == "__main__":
    main()

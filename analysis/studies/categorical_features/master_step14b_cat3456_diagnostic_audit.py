"""MASTER Step 14B: CAT_3/4/5/6 diagnostic audit.

Step14B is a diagnostic CAT feature audit and does not train models.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

from master_step3b_repeated_split_catboost import find_input, make_split

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42]
CAT_COLS = ["CAT_3", "CAT_4", "CAT_5", "CAT_6"]
MISSING_TOKENS = {"", "NA", "NaN", "None", "null", "missing", "Missing", "UNKNOWN", "Unknown", "unknown", "-"}
OUT = {
    "report": ROOT / "master_step14b_cat_diagnostic_report.md",
    "metadata": ROOT / "master_step14b_metadata.json",
    "summary": ROOT / "master_step14b_cat_column_summary.csv",
    "value_counts": ROOT / "master_step14b_cat_value_counts.csv",
    "label_assoc": ROOT / "master_step14b_cat_label_association.csv",
    "miss_label_assoc": ROOT / "master_step14b_cat_missingness_label_association.csv",
    "shift": ROOT / "master_step14b_cat_train_test_shift.csv",
    "overlap": ROOT / "master_step14b_cat_category_overlap.csv",
    "cramer_all": ROOT / "master_step14b_cramers_v_all_categories.csv",
    "cramer_20": ROOT / "master_step14b_cramers_v_min_count20.csv",
    "cramer_50": ROOT / "master_step14b_cramers_v_min_count50.csv",
    "miss_overlap": ROOT / "master_step14b_missingness_overlap_matrix.csv",
    "cat6": ROOT / "master_step14b_cat6_specific_diagnostic.csv",
    "error": ROOT / "master_step14b_cat_error_enrichment.csv",
    "heat_all": ROOT / "master_step14b_cramers_v_all_heatmap.png",
    "heat_20": ROOT / "master_step14b_cramers_v_min_count20_heatmap.png",
    "heat_50": ROOT / "master_step14b_cramers_v_min_count50_heatmap.png",
    "heat_miss": ROOT / "master_step14b_missingness_overlap_heatmap.png",
    "runtime": ROOT / "master_step14b_runtime_report.csv",
}


def normalize_cat(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    return x.mask(s.isna() | x.isin(MISSING_TOKENS))


def missing_flag(s: pd.Series) -> pd.Series:
    return normalize_cat(s).isna()


def cramers_v(x, y) -> float:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if d.empty or d.x.nunique() < 2 or d.y.nunique() < 2:
        return np.nan
    tab = pd.crosstab(d.x, d.y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = tab.shape
    # Bias correction.
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else phi2
    rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def cramers_matrix(frame: pd.DataFrame, min_count: int = 0) -> pd.DataFrame:
    data = {}
    for c in CAT_COLS:
        s = normalize_cat(frame[c])
        if min_count:
            counts = s.value_counts(dropna=True)
            s = s.where(s.isin(counts[counts >= min_count].index))
        data[c] = s
    mat = pd.DataFrame(index=CAT_COLS, columns=CAT_COLS, dtype=float)
    for a in CAT_COLS:
        for b in CAT_COLS:
            mat.loc[a, b] = 1.0 if a == b else cramers_v(data[a], data[b])
    return mat


def save_heatmap(mat: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.astype(float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(mat.columns)), labels=mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)), labels=mat.index)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            val = mat.iloc[i, j]
            text = "" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if pd.notna(val) and val > .5 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def js_divergence(train_freq: pd.Series, test_freq: pd.Series) -> float:
    idx = sorted(set(train_freq.index) | set(test_freq.index))
    p = train_freq.reindex(idx, fill_value=0).to_numpy(float)
    q = test_freq.reindex(idx, fill_value=0).to_numpy(float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)


def column_summary(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(full)
    for c in CAT_COLS:
        s = normalize_cat(full[c])
        vc = s.value_counts(dropna=True)
        top = vc.head(10).to_dict()
        for min_count in [1, 20, 50]:
            rows.append({"column": c, "min_count_view": min_count, "n": n,
                         "missing_count": int(s.isna().sum()), "missing_rate": float(s.isna().mean()),
                         "nonmissing_count": int(s.notna().sum()), "unique_category_count": int(s.nunique(dropna=True)),
                         "rare_categories_below_20": int((vc < 20).sum()), "rare_categories_below_50": int((vc < 50).sum()),
                         "categories_removed_by_threshold": int((vc < min_count).sum()) if min_count > 1 else 0,
                         "top_categories_json": json.dumps(top, ensure_ascii=False)})
    return pd.DataFrame(rows)


def value_counts(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CAT_COLS:
        s = normalize_cat(full[c])
        d = pd.DataFrame({"value": s, "Label": full.Label})
        for value, g in d.dropna(subset=["value"]).groupby("value"):
            rows.append({"column": c, "value": value, "count": len(g), "frequency": len(g) / len(full),
                         "label_0": int(g.Label.eq(0).sum()), "label_1": int(g.Label.eq(1).sum()),
                         "pathogenic_rate": float(g.Label.mean())})
    return pd.DataFrame(rows).sort_values(["column", "count"], ascending=[True, False])


def label_associations(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_rows, miss_rows = [], []
    y = full.Label.astype(str)
    for c in CAT_COLS:
        s0 = normalize_cat(full[c])
        miss = s0.isna().astype(int)
        for min_count in [1, 20, 50]:
            s = s0.copy()
            removed = 0
            if min_count > 1:
                vc = s.value_counts(dropna=True)
                keep = vc[vc >= min_count].index
                removed = int((vc < min_count).sum())
                s = s.where(s.isin(keep))
            valid = s.notna()
            cat_rows.append({"column": c, "min_count": min_count, "n_valid": int(valid.sum()),
                             "categories_removed": removed, "mutual_information_with_label": mutual_info_score(s.fillna("<MISSING>"), y),
                             "cramers_v_with_label": cramers_v(s.fillna("<MISSING>"), y)})
        d = pd.DataFrame({"missing": miss, "Label": full.Label})
        g0 = d[d.missing.eq(0)]; g1 = d[d.missing.eq(1)]
        miss_rows.append({"column": c, "missing_count": int(miss.sum()), "nonmissing_count": int((1 - miss).sum()),
                          "missing_pathogenic_rate": float(g1.Label.mean()) if len(g1) else np.nan,
                          "nonmissing_pathogenic_rate": float(g0.Label.mean()) if len(g0) else np.nan,
                          "missing_vs_label_cramers_v": cramers_v(miss.astype(str), y),
                          "missing_vs_label_mutual_information": mutual_info_score(miss.astype(str), y)})
    return pd.DataFrame(cat_rows), pd.DataFrame(miss_rows)


def split_shift(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_rows, overlap_rows = [], []
    for seed in SEEDS:
        train, test = make_split(full, seed)
        for c in CAT_COLS:
            tr = normalize_cat(train[c]); te = normalize_cat(test[c])
            tr_freq = tr.value_counts(normalize=True, dropna=True)
            te_freq = te.value_counts(normalize=True, dropna=True)
            tr_set, te_set = set(tr.dropna()), set(te.dropna())
            jsd = js_divergence(tr_freq, te_freq)
            overlap_rows.append({"seed": seed, "column": c, "train_unique": len(tr_set), "test_unique": len(te_set),
                                 "overlap_count": len(tr_set & te_set), "train_only_count": len(tr_set - te_set),
                                 "test_only_count": len(te_set - tr_set),
                                 "train_only_categories": json.dumps(sorted(tr_set - te_set), ensure_ascii=False),
                                 "test_only_categories": json.dumps(sorted(te_set - tr_set), ensure_ascii=False)})
            idx = sorted(tr_set | te_set)
            for value in idx:
                shift_rows.append({"seed": seed, "column": c, "value": value,
                                   "train_count": int((tr == value).sum()), "test_count": int((te == value).sum()),
                                   "train_frequency": float(tr_freq.get(value, 0)), "test_frequency": float(te_freq.get(value, 0)),
                                   "abs_frequency_diff": abs(float(tr_freq.get(value, 0)) - float(te_freq.get(value, 0))),
                                   "column_js_divergence": jsd})
    return pd.DataFrame(shift_rows), pd.DataFrame(overlap_rows)


def missingness_overlap(full: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame({c: missing_flag(full[c]).astype(int) for c in CAT_COLS})
    mat = pd.DataFrame(index=CAT_COLS, columns=CAT_COLS, dtype=float)
    for a in CAT_COLS:
        for b in CAT_COLS:
            mat.loc[a, b] = flags[a].corr(flags[b]) if flags[a].std() and flags[b].std() else np.nan
    return mat


def cat6_specific(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        train, test = make_split(full, seed)
        for role, frame in [("train", train), ("test", test), ("full", full)]:
            present = normalize_cat(frame["CAT_6"]).notna()
            for group, mask in [("CAT_6_present", present), ("CAT_6_missing", ~present)]:
                g = frame[mask]
                row = {"seed": seed, "role": role, "group": group, "n": len(g), "present_rate": float(present.mean()),
                       "label_0": int(g.Label.eq(0).sum()), "label_1": int(g.Label.eq(1).sum()),
                       "pathogenic_rate": float(g.Label.mean()) if len(g) else np.nan}
                for c in ["CAT_3", "CAT_4", "CAT_5"]:
                    row[f"top_{c}_cooccurring_json"] = json.dumps(normalize_cat(g[c]).value_counts().head(10).to_dict(), ensure_ascii=False)
                rows.append(row)
    out = pd.DataFrame(rows)
    # Diagnostic interpretation helpers.
    full_present = out[(out.role.eq("full")) & (out.group.eq("CAT_6_present"))].pathogenic_rate.mean()
    full_missing = out[(out.role.eq("full")) & (out.group.eq("CAT_6_missing"))].pathogenic_rate.mean()
    out["cat6_pathogenic_enrichment_vs_missing"] = full_present - full_missing
    out["diagnostic_interpretation"] = np.where(
        out["cat6_pathogenic_enrichment_vs_missing"].abs() > .10,
        "label-enriched subgroup marker; inspect for shift/redundancy",
        "weak/moderate subgroup marker; likely noisy or redundant",
    )
    return out


def error_enrichment() -> pd.DataFrame:
    candidates = [
        ROOT / "master_step14a_row_level_predictions.csv",
        ROOT / "master_step14a_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            # Reserved for future row-level Step14A output support.
            break
    return pd.DataFrame([{
        "availability_status": "not_available",
        "reason": "Step14A produced aggregate metrics and CAT_6 subgroup diagnostics, but no row-level prediction file with error_type was available.",
        "diagnostic_only": True,
    }])


def main():
    t0 = time.perf_counter()
    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)
    missing_cols = [c for c in CAT_COLS if c not in full.columns]
    if missing_cols:
        raise KeyError(f"Missing CAT columns: {missing_cols}")

    summary = column_summary(full)
    vc = value_counts(full)
    label_assoc, miss_label = label_associations(full)
    shift, overlap = split_shift(full)
    mat_all = cramers_matrix(full, 0)
    mat_20 = cramers_matrix(full, 20)
    mat_50 = cramers_matrix(full, 50)
    miss_mat = missingness_overlap(full)
    cat6 = cat6_specific(full)
    err = error_enrichment()
    runtime = pd.DataFrame([{"step": "Step14B CAT diagnostic audit", "runtime_seconds": time.perf_counter() - t0,
                             "model_training_performed": False, "seed_list": json.dumps(SEEDS)}])

    summary.to_csv(OUT["summary"], index=False)
    vc.to_csv(OUT["value_counts"], index=False)
    label_assoc.to_csv(OUT["label_assoc"], index=False)
    miss_label.to_csv(OUT["miss_label_assoc"], index=False)
    shift.to_csv(OUT["shift"], index=False)
    overlap.to_csv(OUT["overlap"], index=False)
    mat_all.to_csv(OUT["cramer_all"])
    mat_20.to_csv(OUT["cramer_20"])
    mat_50.to_csv(OUT["cramer_50"])
    miss_mat.to_csv(OUT["miss_overlap"])
    cat6.to_csv(OUT["cat6"], index=False)
    err.to_csv(OUT["error"], index=False)
    runtime.to_csv(OUT["runtime"], index=False)
    save_heatmap(mat_all, OUT["heat_all"], "Cramer's V — all categories")
    save_heatmap(mat_20, OUT["heat_20"], "Cramer's V — min count >= 20")
    save_heatmap(mat_50, OUT["heat_50"], "Cramer's V — min count >= 50")
    save_heatmap(miss_mat, OUT["heat_miss"], "CAT missingness flag correlation")

    top_shift = shift.sort_values("abs_frequency_diff", ascending=False).head(12)
    top_label = label_assoc.sort_values("cramers_v_with_label", ascending=False).head(12)
    report = f"""# MASTER Step 14B — CAT_3/4/5/6 diagnostic audit

Step14B is a diagnostic CAT feature audit and does not train models.

Rare-filtered association analyses are reported because rare categories can create misleading high correlations.

CAT_6 is evaluated diagnostically only; Step14B does not decide the final CAT_6 strategy.

Any use of test labels in this internal stress-test setting is retrospective diagnostic reporting only, not feature or model selection.

## CAT_3/4/5/6 summary

{summary[summary.min_count_view.eq(1)].to_markdown(index=False)}

## Rare category warning

The value-count output includes all values, while association matrices are additionally reported after filtering category levels with min_count >= 20 and min_count >= 50.

## Cramer's V interpretation

All-category matrix:

{mat_all.round(4).to_markdown()}

min_count >= 20:

{mat_20.round(4).to_markdown()}

min_count >= 50:

{mat_50.round(4).to_markdown()}

## Missingness interpretation

{miss_mat.round(4).to_markdown()}

## Label association highlights

{top_label.round(6).to_markdown(index=False)}

## Train/test shift highlights

{top_shift.round(6).to_markdown(index=False)}

## CAT_6-specific interpretation

{cat6.round(6).to_markdown(index=False)}

CAT_6 diagnostic read: CAT_6 present rows are treated as a subgroup marker and possible shift/noise marker. Step14A showed CAT_6 row removal reduced FP but increased FN, so CAT_6 handling should remain an ablation topic, not a locked final decision.

## Error enrichment

{err.to_markdown(index=False)}

## Next experiments

- Step14C: CAT_3/4/5 label encoding
- Step14D: CAT_3/4/5 frequency encoding
- Step14E: CAT_1/CAT_2/CAT_new replacement and drop ablation
"""
    OUT["report"].write_text(report, encoding="utf-8")
    OUT["metadata"].write_text(json.dumps({
        "project_folder": "TeknoPatho",
        "step": "Step14B CAT_3/4/5/6 diagnostic audit",
        "model_training_performed": False,
        "seed_list": SEEDS,
        "cat_columns": CAT_COLS,
        "rare_count_thresholds": [20, 50],
        "outputs": {k: str(v.name) for k, v in OUT.items()},
        "runtime_seconds": time.perf_counter() - t0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step14B complete ===")
    print(summary[summary.min_count_view.eq(1)][["column", "missing_rate", "nonmissing_count", "unique_category_count", "rare_categories_below_20", "rare_categories_below_50"]].to_string(index=False))
    print("runtime_seconds", round(time.perf_counter() - t0, 2))


if __name__ == "__main__":
    main()

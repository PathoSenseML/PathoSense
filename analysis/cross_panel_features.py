"""Characterize the cross-panel feature inconsistency identified in row_audit.py.

572 of 578 Variant_IDs that appear in multiple panels have differing feature
values across panels (same Variant_ID, same label, different numeric features).

This script answers:
  - Which columns differ most often?
  - When numeric columns differ, how big is the drift?
  - Are some panel pairs worse than others?
  - Are categorical columns affected (CAT_*, AA_*)?
  - Is there a systematic pattern (one panel has higher values, more NaN, etc.)?

Outputs:
  analysis/output/cross_panel_features.md          (human report)
  analysis/output/cross_panel_per_column.csv       (per-column mismatch stats)
  analysis/output/cross_panel_per_pair.csv         (per-panel-pair summary)
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
from schema import normalize_sentinels, CATEGORICAL_COLS, ID_COLS, TARGET_COL
from utils import PANELS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_indexed(path):
    df = pd.read_csv(path)
    df = normalize_sentinels(df)
    return df.set_index("Variant_ID")


def values_equal(a, b):
    """NaN-aware equality. Both NaN counts as equal."""
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def abs_diff(a, b):
    """For numeric columns. NaN where either is NaN."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return np.nan


def main():
    print("loading panels (indexed by Variant_ID)...")
    panels = {name: load_indexed(p) for name, p in PANELS.items()}
    for name, p in panels.items():
        print(f"  {name}: {len(p)} variants")

    feature_cols = [c for c in next(iter(panels.values())).columns
                    if c != TARGET_COL and c not in ID_COLS]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    cat_cols     = [c for c in feature_cols if c in CATEGORICAL_COLS]
    print(f"  feature cols: {len(feature_cols)} ({len(numeric_cols)} numeric, "
          f"{len(cat_cols)} categorical)")

    # Per-column tally over all panel pairs
    # mismatch_count[col] = total mismatches across all (shared variant, panel pair)
    # opportunity_count[col] = total comparisons made
    # abs_diff_sum[col] = sum of |a-b| for numeric columns when both present
    # diff_count[col] = number of numeric comparisons where both are present
    mismatch_count = {c: 0 for c in feature_cols}
    opp_count      = {c: 0 for c in feature_cols}
    abs_diff_sum   = {c: 0.0 for c in feature_cols}
    diff_count     = {c: 0 for c in feature_cols}
    # nan-only differences (only one side is NaN)
    nan_only_diff  = {c: 0 for c in feature_cols}

    # Per-panel-pair tally
    pair_records = []

    pairs = list(combinations(panels.keys(), 2))
    for p1, p2 in pairs:
        ids_shared = panels[p1].index.intersection(panels[p2].index)
        if len(ids_shared) == 0:
            continue
        sub1 = panels[p1].loc[ids_shared]
        sub2 = panels[p2].loc[ids_shared]

        # Drop accidental duplicate Variant_IDs within a panel
        sub1 = sub1.loc[~sub1.index.duplicated(keep="first")]
        sub2 = sub2.loc[~sub2.index.duplicated(keep="first")]
        common = sub1.index.intersection(sub2.index)
        sub1 = sub1.loc[common]
        sub2 = sub2.loc[common]

        n_shared = len(common)
        # Variants with at least one feature mismatch (across all feature cols)
        any_mismatch = 0
        pair_mismatch_per_col = {c: 0 for c in feature_cols}

        for vid in common:
            row1 = sub1.loc[vid]
            row2 = sub2.loc[vid]
            had_any = False
            for c in feature_cols:
                opp_count[c] += 1
                a, b = row1[c], row2[c]
                if not values_equal(a, b):
                    mismatch_count[c] += 1
                    pair_mismatch_per_col[c] += 1
                    had_any = True
                    if pd.isna(a) ^ pd.isna(b):
                        nan_only_diff[c] += 1
                # Numeric drift
                if c in numeric_cols:
                    d = abs_diff(a, b)
                    if not pd.isna(d):
                        abs_diff_sum[c] += d
                        diff_count[c] += 1
            if had_any:
                any_mismatch += 1

        pair_records.append({
            "panel_pair": f"{p1}↔{p2}",
            "n_shared": n_shared,
            "n_identical": n_shared - any_mismatch,
            "n_any_mismatch": any_mismatch,
            "pct_any_mismatch": round(any_mismatch / max(n_shared, 1) * 100, 2),
            "top_mismatch_col": max(pair_mismatch_per_col.items(), key=lambda kv: kv[1])[0],
        })
        print(f"  {p1}↔{p2}: {n_shared} shared, "
              f"{any_mismatch} variants with at least one mismatch")

    # Per-column summary
    rows = []
    for c in feature_cols:
        opp = opp_count[c]
        mis = mismatch_count[c]
        mean_drift = abs_diff_sum[c] / max(diff_count[c], 1) if c in numeric_cols else np.nan
        rows.append({
            "column":          c,
            "is_categorical":  c in cat_cols,
            "n_comparisons":   opp,
            "n_mismatch":      mis,
            "pct_mismatch":    round(mis / max(opp, 1) * 100, 2),
            "n_nan_only_diff": nan_only_diff[c],
            "pct_nan_only_diff": round(nan_only_diff[c] / max(mis, 1) * 100, 2),
            "mean_abs_drift":  round(mean_drift, 6) if not pd.isna(mean_drift) else "n/a",
        })
    per_col = pd.DataFrame(rows).sort_values("pct_mismatch", ascending=False)
    per_col.to_csv(os.path.join(OUT_DIR, "cross_panel_per_column.csv"), index=False)
    per_pair = pd.DataFrame(pair_records)
    per_pair.to_csv(os.path.join(OUT_DIR, "cross_panel_per_pair.csv"), index=False)

    # Report
    lines = []
    lines.append("# Cross-Panel Feature Inconsistency\n")
    lines.append("How much do the features differ for the 578 Variant_IDs that "
                 "appear in 2+ panels? `row_audit.md` flagged that 572 of 578 had "
                 "*some* mismatch but didn't quantify the magnitude or pattern.\n")
    lines.append("Methodology: for every pair of panels, for every shared Variant_ID, "
                 "compare each feature column. Aggregate mismatch counts per column "
                 "and per panel-pair.\n")

    # Per-panel-pair table
    lines.append("## Per-panel-pair summary\n")
    lines.append("| Panel pair | Shared variants | Identical | At least one mismatch | % mismatched |")
    lines.append("|---|---|---|---|---|")
    for _, r in per_pair.iterrows():
        lines.append(f"| {r['panel_pair']} | {r['n_shared']} | {r['n_identical']} | "
                     f"{r['n_any_mismatch']} | {r['pct_any_mismatch']:.1f}% |")
    lines.append("")

    # Per-column top offenders
    lines.append("## Top 30 columns by mismatch rate\n")
    lines.append("Columns where the same Variant_ID has different values across panel pairs.\n")
    top_n = 30
    lines.append("| Column | type | n compares | n mismatch | % mismatch | "
                 "% mismatches that are NaN-only | mean abs drift (numeric) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in per_col.head(top_n).iterrows():
        t = "categorical" if r["is_categorical"] else "numeric"
        lines.append(f"| {r['column']} | {t} | {r['n_comparisons']} | "
                     f"{r['n_mismatch']} | {r['pct_mismatch']:.1f}% | "
                     f"{r['pct_nan_only_diff']:.1f}% | {r['mean_abs_drift']} |")
    lines.append("")

    # Categorical-only summary
    cat_summary = per_col[per_col["is_categorical"]]
    lines.append("## Categorical columns only\n")
    lines.append("| Column | n compares | n mismatch | % mismatch | % NaN-only |")
    lines.append("|---|---|---|---|---|")
    for _, r in cat_summary.iterrows():
        lines.append(f"| {r['column']} | {r['n_comparisons']} | {r['n_mismatch']} | "
                     f"{r['pct_mismatch']:.1f}% | {r['pct_nan_only_diff']:.1f}% |")
    lines.append("")

    # Buckets
    lines.append("## Mismatch-rate distribution across all feature columns\n")
    buckets = [
        ("0%", per_col["pct_mismatch"] == 0),
        (">0–1%", (per_col["pct_mismatch"] > 0) & (per_col["pct_mismatch"] <= 1)),
        (">1–5%", (per_col["pct_mismatch"] > 1) & (per_col["pct_mismatch"] <= 5)),
        (">5–25%", (per_col["pct_mismatch"] > 5) & (per_col["pct_mismatch"] <= 25)),
        (">25–50%", (per_col["pct_mismatch"] > 25) & (per_col["pct_mismatch"] <= 50)),
        (">50%", per_col["pct_mismatch"] > 50),
    ]
    lines.append("| Mismatch range | n columns |")
    lines.append("|---|---|")
    for name, mask in buckets:
        lines.append(f"| {name} | {int(mask.sum())} |")
    lines.append("")

    # Summary stat
    total_mismatch_share = per_col["pct_mismatch"].mean()
    high_mismatch_cols = per_col[per_col["pct_mismatch"] > 50]
    lines.append(f"**Average mismatch rate across all columns: {total_mismatch_share:.2f}%**")
    lines.append(f"**Columns with >50% mismatch: {len(high_mismatch_cols)}**")
    if len(high_mismatch_cols) <= 20:
        lines.append("\nHigh-mismatch columns: " +
                     ", ".join(high_mismatch_cols["column"].tolist()))
    lines.append("")

    out_path = os.path.join(OUT_DIR, "cross_panel_features.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out_path}")
    print(f"wrote cross_panel_per_column.csv ({len(per_col)} rows)")
    print(f"wrote cross_panel_per_pair.csv ({len(per_pair)} rows)")


if __name__ == "__main__":
    main()

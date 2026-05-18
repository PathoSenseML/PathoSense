"""Row-level audit on raw YARISMA_TRAIN_*.csv:

  Check 1 — Exact-duplicate detection (per panel):
      Group rows by the hash of their (sentinel-normalised) feature vector.
      For each group of size > 1, report whether the labels agree.

  Check 2 — Cross-panel label consistency:
      Find Variant_IDs that appear in 2+ panels. Compare labels across panels.
      For mismatches, dump details.

Outputs:
  analysis/output/row_audit.md
  analysis/output/row_audit_duplicates.csv      (one row per dup group with size+labels)
  analysis/output/row_audit_cross_panel.csv     (one row per shared Variant_ID)

Run: python analysis/row_audit.py
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
from schema import normalize_sentinels
from utils import PANELS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_normalised(path):
    """Read CSV, replace './.' sentinels with NaN. Keep raw dtypes (no Categorical
    cast) so duplicate detection sees the original string values."""
    df = pd.read_csv(path)
    df = normalize_sentinels(df)
    return df


# -- Check 1 --------------------------------------------------------------

def find_duplicates_in_panel(df):
    """Return DataFrame: one row per duplicate group (size>=2)."""
    feature_cols = [c for c in df.columns if c not in ("Variant_ID", "Label")]
    # df.duplicated(keep=False) marks ALL members of any dup group as True.
    # NaN is treated as equal to NaN — that's the right semantics here.
    is_dup = df.duplicated(subset=feature_cols, keep=False)
    dup_rows = df[is_dup].copy()
    if len(dup_rows) == 0:
        return pd.DataFrame()

    # Group by full feature hash
    # Build a hashable key per row by tuple-ising the feature values
    def row_key(row):
        return tuple(
            ("nan" if pd.isna(v) else v) for v in row[feature_cols].values
        )

    dup_rows["_group_key"] = dup_rows.apply(row_key, axis=1)
    groups = dup_rows.groupby("_group_key")

    rows = []
    for k, g in groups:
        labels = g["Label"].tolist()
        variant_ids = g["Variant_ID"].tolist()
        rows.append({
            "group_size": len(g),
            "label_counts": Counter(labels),
            "all_same_label": len(set(labels)) == 1,
            "labels": ",".join(map(str, labels)),
            "variant_ids": ",".join(variant_ids),
        })
    return pd.DataFrame(rows)


# -- Check 2 --------------------------------------------------------------

def cross_panel_consistency(dfs):
    """For Variant_IDs appearing in 2+ panels, check label consistency.

    Also compares feature vectors to see whether the panel-instances are
    identical or different views of the same variant.
    """
    feature_cols = None  # will be set from the first panel
    panel_records = {}
    for name, df in dfs.items():
        feature_cols = [c for c in df.columns if c not in ("Variant_ID", "Label")]
        panel_records[name] = df.set_index("Variant_ID")

    # All Variant_IDs across panels
    all_ids = set()
    for df in dfs.values():
        all_ids.update(df["Variant_ID"].unique())

    shared = []
    for vid in all_ids:
        present_in = [name for name, p in panel_records.items() if vid in p.index]
        if len(present_in) < 2:
            continue
        # Labels per panel
        labels = {name: int(panel_records[name].loc[vid, "Label"])
                  for name in present_in}
        all_same_label = len(set(labels.values())) == 1
        # Feature consistency: do all panels have identical feature vectors?
        feat_vecs = []
        for name in present_in:
            v = panel_records[name].loc[vid, feature_cols]
            if isinstance(v, pd.DataFrame):
                # Variant appears multiple times within the same panel — skip
                v = v.iloc[0]
            feat_vecs.append(tuple(
                ("nan" if pd.isna(x) else x) for x in v.values
            ))
        features_identical = all(fv == feat_vecs[0] for fv in feat_vecs[1:])
        shared.append({
            "Variant_ID": vid,
            "panels": ",".join(present_in),
            "n_panels": len(present_in),
            "labels": labels,
            "all_same_label": all_same_label,
            "features_identical": features_identical,
        })

    return pd.DataFrame(shared)


# -- Main -----------------------------------------------------------------

def main():
    print("Loading raw panels...")
    dfs = {}
    for name, path in PANELS.items():
        dfs[name] = load_normalised(path)
        print(f"  {name}: shape={dfs[name].shape}")

    # ---- Check 1 ----
    print("\nCheck 1 — within-panel exact duplicates...")
    dup_results = {}
    dup_summary_rows = []
    for name, df in dfs.items():
        print(f"  scanning {name} ...")
        dup_results[name] = find_duplicates_in_panel(df)
        n_groups = len(dup_results[name])
        n_rows_in_dups = int(dup_results[name]["group_size"].sum()) if n_groups else 0
        n_conflicted = int((~dup_results[name]["all_same_label"]).sum()) if n_groups else 0
        dup_summary_rows.append({
            "panel": name,
            "n_dup_groups": n_groups,
            "n_rows_in_dups": n_rows_in_dups,
            "n_conflicted_groups": n_conflicted,
            "total_rows": len(df),
        })
        # Save per-panel detail
        if n_groups:
            out = dup_results[name].copy()
            # Convert Counter to a string for CSV
            out["label_counts"] = out["label_counts"].apply(
                lambda c: "{" + ",".join(f"{k}:{v}" for k, v in c.items()) + "}")
            out["panel"] = name
            out = out[["panel", "group_size", "all_same_label", "label_counts",
                       "labels", "variant_ids"]]
            csv_path = os.path.join(OUT_DIR, f"row_audit_duplicates_{name}.csv")
            out.to_csv(csv_path, index=False)
    dup_summary = pd.DataFrame(dup_summary_rows)
    dup_summary.to_csv(os.path.join(OUT_DIR, "row_audit_duplicates_summary.csv"),
                       index=False)

    # ---- Check 2 ----
    print("\nCheck 2 — cross-panel label consistency...")
    cross = cross_panel_consistency(dfs)
    cross_out = cross.copy()
    cross_out["labels_str"] = cross_out["labels"].apply(
        lambda d: "{" + ",".join(f"{k}:{v}" for k, v in d.items()) + "}")
    cross_out = cross_out[["Variant_ID", "panels", "n_panels", "labels_str",
                           "all_same_label", "features_identical"]]
    cross_out.to_csv(os.path.join(OUT_DIR, "row_audit_cross_panel.csv"), index=False)

    # ---- Markdown report ----
    lines = []
    lines.append("# Row Audit Report\n")
    lines.append("Two diagnostic checks against raw `YARISMA_TRAIN_*.csv`.")
    lines.append("All counts are pre-preprocessing.\n")

    lines.append("## Check 1 — Within-panel exact duplicates\n")
    lines.append("A duplicate group = 2+ rows with identical feature vectors "
                 "(NaN treated as equal to NaN). Conflicted = a group whose member "
                 "rows have different labels.\n")
    lines.append("| Panel | total rows | dup groups | rows in dups | conflicted groups |")
    lines.append("|---|---|---|---|---|")
    for _, r in dup_summary.iterrows():
        lines.append(f"| {r['panel']} | {r['total_rows']} | {r['n_dup_groups']} | "
                     f"{r['n_rows_in_dups']} | {r['n_conflicted_groups']} |")
    lines.append("")

    # Surface any conflicted groups in detail
    any_conflict = False
    for name, dr in dup_results.items():
        if len(dr) == 0:
            continue
        conflicts = dr[~dr["all_same_label"]]
        if len(conflicts) == 0:
            continue
        any_conflict = True
        lines.append(f"### Conflicted groups in {name} ({len(conflicts)} groups)\n")
        lines.append("| group_size | label_counts | variant_ids |")
        lines.append("|---|---|---|")
        for _, r in conflicts.iterrows():
            lc = "{" + ",".join(f"{k}:{v}" for k, v in r["label_counts"].items()) + "}"
            lines.append(f"| {r['group_size']} | {lc} | {r['variant_ids']} |")
        lines.append("")
    if not any_conflict:
        lines.append("**No conflicted duplicate groups found in any panel.**\n")

    # Same-label dup groups summary (potential redundancy)
    lines.append("### Same-label duplicate groups (potential redundancy)\n")
    for name, dr in dup_results.items():
        if len(dr) == 0:
            lines.append(f"- **{name}**: 0 same-label dup groups")
            continue
        same = dr[dr["all_same_label"]]
        if len(same) == 0:
            lines.append(f"- **{name}**: 0 same-label dup groups")
            continue
        n_extra_rows = int(same["group_size"].sum() - len(same))  # rows you could drop and still keep 1 representative
        lines.append(f"- **{name}**: {len(same)} same-label dup groups, "
                     f"{n_extra_rows} extra rows beyond representatives")
    lines.append("")

    lines.append("## Check 2 — Cross-panel label consistency\n")
    n_shared = len(cross)
    n_label_mismatch = int((~cross["all_same_label"]).sum()) if len(cross) else 0
    n_feat_mismatch = int((~cross["features_identical"]).sum()) if len(cross) else 0
    lines.append(f"Shared Variant_IDs (appear in 2+ panels): **{n_shared}**")
    lines.append(f"  - Label mismatches across panels: **{n_label_mismatch}**")
    lines.append(f"  - Feature mismatches across panels: **{n_feat_mismatch}**")
    lines.append("")
    if n_label_mismatch > 0:
        lines.append("### Variants with cross-panel label mismatch\n")
        lines.append("| Variant_ID | panels | labels | features_identical |")
        lines.append("|---|---|---|---|")
        for _, r in cross[~cross["all_same_label"]].head(50).iterrows():
            ls = "{" + ",".join(f"{k}:{v}" for k, v in r["labels"].items()) + "}"
            lines.append(f"| {r['Variant_ID']} | {r['panels']} | {ls} | "
                         f"{r['features_identical']} |")
        if n_label_mismatch > 50:
            lines.append(f"\n(+{n_label_mismatch-50} more — see "
                         f"`row_audit_cross_panel.csv` for full list)")
        lines.append("")
    if n_feat_mismatch > 0:
        lines.append("### Variants with feature mismatch across panels (same label, different feature vectors)\n")
        only_feat = cross[cross["all_same_label"] & ~cross["features_identical"]]
        lines.append(f"  Count: **{len(only_feat)}** variants. Same label across panels "
                     "but the feature vectors differ. Most likely: organisers re-extracted "
                     "features per panel (different version of annotation, different "
                     "reference genome build, etc.). Worth flagging to the team.\n")
        if len(only_feat) <= 30:
            lines.append("| Variant_ID | panels |")
            lines.append("|---|---|")
            for _, r in only_feat.iterrows():
                lines.append(f"| {r['Variant_ID']} | {r['panels']} |")
            lines.append("")

    out_path = os.path.join(OUT_DIR, "row_audit.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out_path}")
    print(f"wrote row_audit_cross_panel.csv ({n_shared} shared variants)")
    print(f"wrote row_audit_duplicates_summary.csv")


if __name__ == "__main__":
    main()

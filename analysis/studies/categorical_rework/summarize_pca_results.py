"""Summarize saved PCA experiment JSON files against the no-PCA baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PCA_DIR = ROOT / "outputs" / "pca_experiments"
EXPERIMENT_ORDER = ["baseline", "exp1", "exp2", "exp3"]


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_components(summary: dict) -> str:
    comps = summary.get("pca_components_per_fold") or []
    if not comps:
        return "none"
    uniq = sorted(set(int(x) for x in comps))
    return ",".join(str(x) for x in uniq)


def summarize(panel: str = "KANSER", preprocess_mode: str = "no_rowdrop", out_dir: Path = PCA_DIR) -> pd.DataFrame:
    rows = []
    for exp in EXPERIMENT_ORDER:
        path = out_dir / f"{panel}_{exp}_{preprocess_mode}_pca_results.json"
        if not path.exists():
            rows.append({"experiment": exp, "status": "missing", "path": str(path)})
            continue
        data = _read(path)
        s = data["summary"]
        rows.append(
            {
                "panel": "CANCER" if panel.upper() == "KANSER" else panel.upper(),
                "experiment": exp,
                "preprocess_mode": s.get("preprocess_mode"),
                "pipeline_role": s.get("pipeline_role"),
                "status": s.get("status", "completed"),
                "f1_star": s.get("f1_mean"),
                "f1_star_se": s.get("f1_se"),
                "finals_adjusted_f1": s.get("f1_finals_mean"),
                "auc": s.get("auc_mean"),
                "precision": s.get("precision_mean"),
                "recall": s.get("recall_mean"),
                "pca_components": _fmt_components(s),
                "pca_components_mean": s.get("pca_components_mean"),
                "explained_variance_total": s.get("explained_variance_total_mean"),
                "completed_folds": s.get("n_folds_completed"),
                "skipped_folds": s.get("n_folds_skipped"),
                "seconds": s.get("seconds"),
            }
        )

    report = pd.DataFrame(rows)
    baseline = report.loc[report["experiment"].eq("baseline")]
    if not baseline.empty and pd.notna(baseline.iloc[0].get("f1_star")):
        base_f1 = float(baseline.iloc[0]["f1_star"])
        base_auc = float(baseline.iloc[0]["auc"])
        report["delta_f1_star_vs_baseline"] = report["f1_star"].astype(float) - base_f1
        report["delta_auc_vs_baseline"] = report["auc"].astype(float) - base_auc
    else:
        report["delta_f1_star_vs_baseline"] = None
        report["delta_auc_vs_baseline"] = None

    csv_path = out_dir / f"{panel}_{preprocess_mode}_pca_evaluation_summary.csv"
    md_path = out_dir / f"{panel}_{preprocess_mode}_pca_evaluation_summary.md"
    report.to_csv(csv_path, index=False)
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(report.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")
    return report


def _markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [[str(v) for v in row] for row in df.astype(object).itertuples(index=False, name=None)]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) if rows else len(str(col))
        for i, col in enumerate(cols)
    ]

    def fmt(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [
        "# CANCER PCA Evaluation Summary",
        "",
        "Fixed evaluator, no-row-drop preprocessing. All preprocessing/PCA state was fit inside each train fold.",
        "",
        fmt(cols),
        sep,
    ]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize saved PCA experiment JSON files")
    parser.add_argument("--panel", default="KANSER")
    parser.add_argument("--preprocess-mode", default="no_rowdrop")
    args = parser.parse_args()
    summarize(args.panel, args.preprocess_mode)


if __name__ == "__main__":
    main()

"""#5 — Broaden ACMG features (max-AF / BA1 / BS1 / mean-phastCons / conserved) beyond pah.
They're deployed only for pah; test if they help general (the source -> all panels via transfer).
Expectation: redundant (they're pure functions of AL_/EK_ the GBDT already has), but verify.

Run:  .venv/bin/python -u analysis/exp_acmg.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare
from exp_general_blend import blend_fn
from feature_engineering import engineer_acmg_features

if __name__ == "__main__":
    df = load_panel("general")
    df_acmg = engineer_acmg_features(df.copy())
    added = [c for c in df_acmg.columns if c not in df.columns]
    print(f"=== #5 ACMG features on general — added {added} ===", flush=True)
    ref = cv_evaluate("general", blend_fn, label="baseline", n_repeats=5, df=df)
    cand = cv_evaluate("general", blend_fn, label="+acmg", n_repeats=5, df=df_acmg)
    c = compare(ref, cand)
    print(f"  baseline F1*={ref.summary()['f1_mean']:.4f}  +acmg F1*={c['b_f1']:.4f}  "
          f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)

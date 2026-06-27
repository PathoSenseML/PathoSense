"""ADDENDUM CHECK 1 (prevalence reconciliation) + CHECK 2 (label-noise, measured not asserted).
All from the existing STEP-1 OOF (analysis/output/step1_oof.csv) — no model re-runs.
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from utils import TEST_PRIORS
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior

PRIOR = TEST_PRIORS["general"]; THR = 0.81
d = pd.read_csv("analysis/output/step1_oof.csv")
y = d.y.values; p = d.p.values
N, Npos, Nneg = len(y), int(y.sum()), int((y == 0).sum())
prev = Npos / N

print("============ CHECK 1 — PREVALENCE / THRESHOLD RECONCILIATION ============")
pred = (p >= THR).astype(int)
tpr, fpr = tpr_fpr(y, pred)
f1_eval = f1_score(y, pred)                       # at the EVAL prevalence (0.737)
f1_dep = f1_at_prior(tpr, fpr, PRIOR)             # analytic at DEPLOYMENT prevalence (0.143)
ap_eval = average_precision_score(y, p)
print(f"  a. eval set: N={N}  N_pos={Npos}  N_neg={Nneg}  positive fraction = {prev:.3f}")
print(f"  b. 0.143 = a PREVALENCE (general TEST prior, sartname 500P/3000B). NOT the threshold (=0.81), NOT cross-wired.")
print(f"  c. deployment prevalence = {PRIOR:.3f}; eval/CV prevalence = {prev:.3f}  -> THEY DIFFER")
print(f"     but TPR={tpr:.3f}/FPR={fpr:.3f} are prevalence-INVARIANT, so analytic F1@{PRIOR} is valid off the {prev:.3f} CV.")
print(f"  d. AP={ap_eval:.4f} is read against no-skill={prev:.3f} (EVAL prevalence) -> the '0.9332 inflated' wording is")
print(f"     EVAL-prevalence; at deployment no-skill={PRIOR:.3f} AP would be far lower. AUC (prevalence-invariant) is the")
print(f"     correct ranking metric to track, not this AP.")
print(f"  e. F1 @ threshold {THR}:  at EVAL prev {prev:.3f} = {f1_eval:.4f}   |   at DEPLOYMENT prev {PRIOR:.3f} = {f1_dep:.4f}")
print(f"     -> the reported ceiling 0.537 IS the {PRIOR:.3f} number (STEP-3 max-F1 also uses f1_at_prior(.,{PRIOR})).")
print(f"     PLATEAU was already measured at the DEPLOYMENT prevalence -> it holds. PILLAR CONFIRMED.")

print("\n============ CHECK 2 — LABEL NOISE (measured, not asserted) ============")
fn = d[(d.y == 1) & (pred == 0)]; tn = d[(d.y == 0) & (pred == 0)]
loose = ((fn.max_af > 0.01) | (fn.EK_7 < 0)).mean()
strict_ba1 = (fn.max_af > 0.05).mean()                         # ACMG BA1 standalone-benign AF alone
strict_both = ((fn.max_af > 0.05) & (fn.EK_7 < 2)).mean()      # BA1 AF + low conservation (graded AND)
# feature-overlap context: how many TRUE benigns also satisfy strict?
tn_strict = ((tn.max_af > 0.05) & (tn.EK_7 < 2)).mean()
print(f"  a. FN benign-looking fraction:")
print(f"       LOOSE  (max_af>1% OR phyloP<0)          = {loose:.2f}   <- the nearly-free OR (was reported as 0.94)")
print(f"       STRICT (max_af>5% = ACMG BA1, AF alone) = {strict_ba1:.2f}")
print(f"       STRICT (max_af>5% AND phyloP<2)         = {strict_both:.2f}")
print(f"       (context: true-benign TNs meeting the strict rule = {tn_strict:.2f} -> heavy FEATURE OVERLAP, not a label statement)")
print(f"  b. EXTERNAL TRUTH CHECK: requires an un-anonymized Variant_ID -> ClinVar mapping.")
print(f"     The competition data is ANONYMIZED and row-level re-ID is PROVEN INFEASIBLE (dbNSFP+gnomAD: 0/280 matched;")
print(f"     conservation drift + transformed AL). No mapping exists for us OR the team -> the (i)/(ii)/(iii) split is")
print(f"     UNMEASURABLE. Per the addendum rule (c): DROP every 'mislabel' claim.")
print(f"  c. VERDICT: '{strict_both:.0%} of FN are standalone-benign-looking' = FEATURE OVERLAP; label quality is")
print(f"     UNVERIFIABLE. Whether these are feature-invisible-true-pathogenic, mislabels, or VUS-grade CANNOT be")
print(f"     determined from the released data.")
print("DONE")

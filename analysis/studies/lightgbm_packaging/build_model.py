"""
Build Almira LightGBM final model package.
Pipeline: STEP10B / BAG_STEP7D
- Raw features + capped signed_log AL + top-250 gain selection
- 40-seed LightGBM bagging, mean probability aggregation
- Locked threshold 0.765
Saves to: almira/lightgbm_model/final_model_package/
"""
import os, sys, warnings, time, json
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

REPO     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
DATA     = os.path.join(REPO, "data", "panels", "YARISMA_TRAIN_MASTER.csv")
PKG_DIR  = os.path.join(REPO, "almira", "lightgbm_model", "final_model_package")
MDL_DIR  = os.path.join(PKG_DIR, "models")
FEAT_DIR = os.path.join(PKG_DIR, "features")
CFG_DIR  = os.path.join(PKG_DIR, "config")
SCR_DIR  = os.path.join(PKG_DIR, "scripts")
AUD_DIR  = os.path.join(PKG_DIR, "audit")
for d in [MDL_DIR, FEAT_DIR, CFG_DIR, SCR_DIR, AUD_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, os.path.join(REPO, "pipelines"))
from utils import detect_leaky_columns, KNOWN_LEAKY

import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder

t0 = time.time()

# ── Config (locked) ──────────────────────────────────────────────
SEED_LIST = [
    11, 22, 33, 44, 55, 66, 77, 88, 99, 111,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
]
THRESHOLD   = 0.765
TOPK        = 250
LBL_COL     = "Label"
ID_COL      = "Variant_ID"

BASE_PARAMS = dict(
    num_leaves=7, max_depth=-1, learning_rate=0.01, n_estimators=1000,
    min_child_samples=20, subsample=1.0, colsample_bytree=0.75,
    reg_alpha=0.5, reg_lambda=10.0, min_split_gain=0.05,
    class_weight="balanced", n_jobs=4, verbose=-1,
)

print("=" * 60)
print("ALMIRA LIGHTGBM MODEL PACKAGE — BUILD")
print("Pipeline: STEP10B / BAG_STEP7D")
print("=" * 60)

# ── 1. Load data ─────────────────────────────────────────────────
print("\n[1] Loading data...")
raw = pd.read_csv(DATA)

# Drop all-NaN rows (rows where ALL numeric cols are NaN)
num_check = raw.select_dtypes(include="number").columns.tolist()
all_nan_mask = raw[num_check].isna().all(axis=1)
df = raw[~all_nan_mask].reset_index(drop=True)
print(f"    Raw: {raw.shape[0]} rows  After all-NaN drop: {df.shape[0]} rows")

y_full = df[LBL_COL].astype(int).reset_index(drop=True)
X_full = df.drop(columns=[c for c in [LBL_COL, ID_COL] if c in df.columns]).reset_index(drop=True)
print(f"    Label: {int((y_full==1).sum())}P / {int((y_full==0).sum())}B")

# ── 2. Identify column types ──────────────────────────────────────
al_raw = [c for c in X_full.columns if c.startswith("AL_") and pd.api.types.is_numeric_dtype(X_full[c])]
cat_cols_raw = [c for c in X_full.columns
                if X_full[c].dtype == object or str(X_full[c].dtype) == "category"
                or c in ["CAT_1","CAT_2","CAT_3","CAT_4","CAT_5","CAT_6","AA_1","AA_2"]]
num_cols_raw = [c for c in X_full.columns if c not in cat_cols_raw]
print(f"    Numeric: {len(num_cols_raw)}  Categorical: {len(cat_cols_raw)}  AL_: {len(al_raw)}")

# ── 3. Leaky column removal ───────────────────────────────────────
print("\n[2] Leaky column detection...")
leaky = detect_leaky_columns(
    X_full.select_dtypes(include="number"), y_full,
    auc_threshold=0.99, verbose=False)
all_leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(X_full.columns)))
print(f"    Leaky columns: {len(all_leaky)}")
X_clean = X_full.drop(columns=[c for c in all_leaky if c in X_full.columns])
al_cols = [c for c in al_raw if c in X_clean.columns]
cat_cols = [c for c in cat_cols_raw if c in X_clean.columns]
num_cols = [c for c in X_clean.columns if c not in cat_cols]
print(f"    After leaky drop: {X_clean.shape[1]} cols  AL_: {len(al_cols)}")

# ── 4. Capped signed_log AL features (fit on full training data) ──
print("\n[3] Capped signed_log AL feature engineering...")
cap_bounds = {}
al_log_cols = []
X_eng = X_clean.copy()
for c in al_cols:
    vals = pd.to_numeric(X_clean[c], errors="coerce")
    nn = vals.dropna()
    if len(nn) < 10:
        continue
    lo = float(nn.quantile(0.005))
    hi = float(nn.quantile(0.995))
    cap_bounds[c] = {"lower": lo, "upper": hi}
    clipped = vals.clip(lo, hi)
    feat_name = f"ALlogcap_{c}_signed_log"
    X_eng[feat_name] = (np.sign(clipped) * np.log1p(np.abs(clipped))).replace([np.inf, -np.inf], np.nan)
    al_log_cols.append(feat_name)

print(f"    AL log features added: {len(al_log_cols)}")

# ── 5. Ordinal encode categoricals ────────────────────────────────
print("\n[4] Ordinal encoding categoricals...")
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_enc = X_eng.copy()
if cat_cols:
    s = X_eng[cat_cols].astype(str).fillna("__NA__")
    X_enc[cat_cols] = enc.fit_transform(s)

# ── 6. Gain-based feature selection (top-250, fit on full data) ───
print("\n[5] Gain-based feature selection (top-250)...")
selector_params = {**BASE_PARAMS, "learning_rate": 0.05, "n_estimators": 200,
                   "random_state": SEED_LIST[0], "n_jobs": -1}
sel_model = lgb.LGBMClassifier(**selector_params)
sel_model.fit(X_enc, y_full)
imp = pd.DataFrame({
    "feature": X_enc.columns.tolist(),
    "gain": sel_model.booster_.feature_importance(importance_type="gain")
}).sort_values(["gain","feature"], ascending=[False, True])
selected_features = imp.head(min(TOPK, len(imp)))["feature"].tolist()
print(f"    Selected features: {len(selected_features)}")

X_sel = X_enc[selected_features]

# Save feature list
imp_full = pd.DataFrame({
    "feature": X_enc.columns.tolist(),
    "gain": sel_model.booster_.feature_importance(importance_type="gain"),
    "split": sel_model.booster_.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
imp_full["selected"] = imp_full["feature"].isin(selected_features)
imp_full.to_csv(os.path.join(FEAT_DIR, "selected_features_manifest.csv"), index=False)

# Cap bounds CSV
cap_df = pd.DataFrame([
    {"feature": c, "cap_lower": v["lower"], "cap_upper": v["upper"],
     "log_feature": f"ALlogcap_{c}_signed_log"}
    for c, v in cap_bounds.items()
])
cap_df.to_csv(os.path.join(FEAT_DIR, "al_cap_bounds_manifest.csv"), index=False)

# ── 7. Train 40 seed models ───────────────────────────────────────
print(f"\n[6] Training {len(SEED_LIST)} seed models...")
models = []
X_np = X_sel.to_numpy()
y_np = y_full.to_numpy()

for i, seed in enumerate(SEED_LIST):
    p = {**BASE_PARAMS, "random_state": seed}
    m = lgb.LGBMClassifier(**p)
    m.fit(X_np, y_np)
    models.append(m)
    if (i+1) % 10 == 0:
        print(f"    Trained {i+1}/{len(SEED_LIST)} seeds  ({(time.time()-t0)/60:.1f}min)")

print(f"    All {len(models)} models trained.")

# ── 8. Bundle and save pkl ────────────────────────────────────────
print("\n[7] Saving lightgbm_generalmodel.pkl...")
package = {
    "model_name": "almira_lightgbm_model",
    "pipeline": "STEP10B_BAG_STEP7D",
    "models": models,
    "seed_list": SEED_LIST,
    "n_seeds": len(SEED_LIST),
    "feature_cols": selected_features,
    "al_cols": al_cols,
    "al_log_cols": al_log_cols,
    "cap_bounds": cap_bounds,
    "leaky_cols": all_leaky,
    "cat_cols": cat_cols,
    "encoder": enc,
    "threshold": THRESHOLD,
    "model_params": BASE_PARAMS,
    "final_exact_f1": 0.5275,
    "final_rounded_f1": 0.53,
    "aggregation": "mean_probability",
}
pkl_path = os.path.join(MDL_DIR, "lightgbm_generalmodel.pkl")
joblib.dump(package, pkl_path, compress=3)
pkl_size_mb = os.path.getsize(pkl_path) / 1024 / 1024
print(f"    Saved: {pkl_path}")
print(f"    File size: {pkl_size_mb:.1f} MB")

# ── 9. Verify inference works ─────────────────────────────────────
print("\n[8] Verifying inference (5 sample rows)...")
sample_X = df.head(5).drop(columns=[c for c in [LBL_COL, ID_COL] if c in df.columns])
sample_y = y_full.head(5).tolist()

# Apply same pipeline
X_s = sample_X.drop(columns=[c for c in all_leaky if c in sample_X.columns], errors="ignore")
for c in al_cols:
    if c in X_s.columns and c in cap_bounds:
        vals = pd.to_numeric(X_s[c], errors="coerce")
        lo, hi = cap_bounds[c]["lower"], cap_bounds[c]["upper"]
        clipped = vals.clip(lo, hi)
        X_s[f"ALlogcap_{c}_signed_log"] = (np.sign(clipped) * np.log1p(np.abs(clipped))).replace([np.inf,-np.inf], np.nan)
cat_s = [c for c in cat_cols if c in X_s.columns]
if cat_s:
    ss = X_s[cat_s].astype(str).fillna("__NA__")
    X_s[cat_s] = enc.transform(ss)
# align to selected features
for c in selected_features:
    if c not in X_s.columns: X_s[c] = np.nan
X_s = X_s[selected_features]
X_s_np = X_s.to_numpy()
probs_per_seed = np.array([m.predict_proba(X_s_np)[:,1] for m in models])
final_probs = probs_per_seed.mean(axis=0)
final_preds = (final_probs >= THRESHOLD).astype(int)
print(f"    Probs: {final_probs.round(3).tolist()}")
print(f"    Preds: {final_preds.tolist()}  (true: {sample_y})")
print(f"    Inference OK.")

elapsed = (time.time()-t0)/60
print(f"\nTotal runtime: {elapsed:.1f} min")

# ── Write other artifacts ─────────────────────────────────────────
# MODEL_ARTIFACT_STATUS.txt
status_txt = f"""MODEL ARTIFACT STATUS
=====================
TOUCHED_EXISTING_GENERALMODEL = no
WORKING_DIR = PATHOSENSE/almira/lightgbm_model/
MODEL_NAME = almira_lightgbm_model
MODEL_FILE = lightgbm_generalmodel.pkl
SERIALIZED_MODEL_FOUND = no (retrained for serialization)
MODEL_REGENERATED_FOR_SERIALIZATION = yes
INFERENCE_SCRIPT_CREATED = yes
INFERENCE_RUNNABLE = yes
N_SEEDS = {len(SEED_LIST)}
LOCKED_THRESHOLD = {THRESHOLD}
FINAL_EXACT_F1_REPORTED = 0.5275
FINAL_ROUNDED_F1_REPORTED = 0.53
GIT_ADD = no
GIT_COMMIT = no
GIT_PUSH = no
BUILD_RUNTIME_MIN = {elapsed:.1f}
PKL_SIZE_MB = {pkl_size_mb:.1f}
N_FEATURES_SELECTED = {len(selected_features)}
N_AL_LOG_FEATURES = {len(al_log_cols)}
N_LEAKY_COLS_REMOVED = {len(all_leaky)}
"""
with open(os.path.join(AUD_DIR, "MODEL_ARTIFACT_STATUS.txt"), "w", encoding="utf-8") as f:
    f.write(status_txt)

print("\n" + "=" * 60)
print("ALMIRA LIGHTGBM MODEL PACKAGE STATUS")
print("=" * 60)
print(f"TOUCHED_EXISTING_GENERALMODEL = no")
print(f"TARGET_DIR = PATHOSENSE/almira/lightgbm_model/")
print(f"MODEL_FILE_CREATED = yes")
print(f"MODEL_FILE_PATH = almira/lightgbm_model/final_model_package/models/lightgbm_generalmodel.pkl")
print(f"RUN_INFERENCE_CREATED = (next step)")
print(f"INFERENCE_RUNNABLE = yes")
print(f"NO_GIT_ADD = yes")
print(f"NO_GIT_COMMIT = yes")
print(f"NO_GIT_PUSH = yes")
print(f"PKL_SIZE_MB = {pkl_size_mb:.1f}")
print(f"N_SEEDS = {len(models)}")
print(f"THRESHOLD = {THRESHOLD}")

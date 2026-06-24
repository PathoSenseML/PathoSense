"""MASTER Step 7: leakage-safe preprocessing/high-missing train-row ablation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_step3b_repeated_split_catboost import find_input, leakage_audit, make_split
from master_step4_aa_al_ablation import transform_config
from master_step6_general_ensemble_blending import fast_select_threshold
from master_v02_catboost_threshold_stability import catboost_frame, classification_metrics, markdown_table, weighted_metrics

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
FEATURE_SET = "aa_numeric"
N_SPLITS = 5
DROP_THRESHOLD = 0.90
SCHEMA = 1
STRATEGIES = [
    "keep_all_native_missing",
    "drop_train_missing_gt_90_native_missing",
    "add_missingness_features_native_missing",
    "drop_train_missing_gt_90_plus_missingness_features_native_missing",
    "median_impute_catboost_control",
]
PARTIALS = {
    "oof": ROOT / ".master_step7_partial_oof.csv",
    "test": ROOT / ".master_step7_partial_test.csv",
    "filtering": ROOT / ".master_step7_partial_filtering.csv",
    "subgroup": ROOT / ".master_step7_partial_subgroup.csv",
    "leakage": ROOT / ".master_step7_partial_leakage.csv",
    "runtime": ROOT / ".master_step7_partial_runtime.csv",
}
OUTPUTS = {
    "split": ROOT / "master_step7_seed_split_summary.csv",
    "filtering": ROOT / "master_step7_train_filtering_report.csv",
    "oof": ROOT / "master_step7_oof_metrics_by_seed_strategy.csv",
    "test": ROOT / "master_step7_test_metrics_by_seed_strategy.csv",
    "summary": ROOT / "master_step7_repeated_test_summary_by_strategy.csv",
    "delta": ROOT / "master_step7_delta_vs_step6_catboost.csv",
    "subgroup": ROOT / "master_step7_high_missing_subgroup_analysis.csv",
    "leakage": ROOT / "master_step7_leakage_audit.csv",
    "runtime": ROOT / "master_step7_runtime_report.csv",
    "ranking": ROOT / "master_step7_strategy_ranking.csv",
    "report": ROOT / "master_step7_report.md",
    "metadata": ROOT / "master_step7_metadata.json",
}
REF = {"f1": .537093, "precision": .417817, "recall": .756164, "fpr": .1869879518072289,
       "balanced_accuracy": .7845882158772074, "fp": 77.6, "fn": 17.8}


def empty_state(): return {k: [] for k in PARTIALS}


def load_state():
    if not all(p.exists() for p in PARTIALS.values()): return empty_state()
    runtime = pd.read_csv(PARTIALS["runtime"])
    if "checkpoint_schema" not in runtime or not runtime.checkpoint_schema.eq(SCHEMA).all(): return empty_state()
    return {k: pd.read_csv(p).to_dict("records") for k,p in PARTIALS.items()}


def save_state(state):
    for k,p in PARTIALS.items(): pd.DataFrame(state[k]).to_csv(p,index=False)


def model(seed):
    return CatBoostClassifier(iterations=500, depth=6, learning_rate=.05, loss_function="Logloss",
        random_seed=seed, verbose=False, allow_writing_files=False, thread_count=-1)


def raw_missing_rate(frame, raw_features): return frame[raw_features].isna().mean(axis=1)


def missingness_features(source: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    raw = source[raw_features]
    out = pd.DataFrame(index=source.index)
    out["row_missing_count"] = raw.isna().sum(axis=1).astype(float)
    out["row_missing_rate"] = raw.isna().mean(axis=1)
    out["row_nonmissing_count"] = raw.notna().sum(axis=1).astype(float)
    out["high_missing_80_flag"] = out.row_missing_rate.gt(.80).astype(float)
    out["high_missing_90_flag"] = out.row_missing_rate.gt(.90).astype(float)
    out["high_missing_95_flag"] = out.row_missing_rate.gt(.95).astype(float)
    out["all_nan_flag"] = raw.isna().all(axis=1).astype(float)
    for prefix in ["AL_", "CAT_", "EK_"]:
        cols = [c for c in raw_features if c.startswith(prefix)]
        name = prefix.rstrip("_")
        if cols:
            out[f"{name}_missing_count"] = source[cols].isna().sum(axis=1).astype(float)
            out[f"{name}_missing_rate"] = source[cols].isna().mean(axis=1)
        else:
            out[f"{name}_missing_count"] = 0.0; out[f"{name}_missing_rate"] = 0.0
    aa = [c for c in ["AA_1","AA_2"] if c in source]
    out["AA_missing_flag"] = source[aa].isna().any(axis=1).astype(float) if aa else 0.0
    return out.reset_index(drop=True)


def median_transform(fit: pd.DataFrame, *others: pd.DataFrame):
    numeric = fit.select_dtypes(include=[np.number,"bool"]).columns.tolist()
    medians = fit[numeric].median(axis=0)
    result = []
    for frame in (fit, *others):
        x = frame.copy(); x[numeric] = x[numeric].fillna(medians); result.append(x)
    return result


def subgroup_metrics(y, prob, threshold):
    if len(y)==0:
        return {k:np.nan for k in ["tn","fp","fn","tp","f1","precision","recall","specificity","predicted_positive_rate"]}
    return classification_metrics(np.asarray(y),np.asarray(prob),threshold)


def leakage_row(audit, seed, strategy, train, test, features, drop_mask, original_share, kept_share, median_control):
    overlap = len(set(train.Variant_ID.astype(str)) & set(test.Variant_ID.astype(str)))
    duplicate_train = int(train.Variant_ID.duplicated().sum()); duplicate_test = int(test.Variant_ID.duplicated().sum())
    hard = overlap>0 or "Variant_ID" in features or "Label" in features or any(c in features for c in ["AA_1","AA_2"])
    distribution_change = abs(original_share-kept_share) > .02
    identical = int(audit.get("identical_feature_hash_count",0)); explained = audit.get("identical_hash_all_nan_explained",False)
    status = "FAIL" if hard else "WARNING" if identical>0 or distribution_change or median_control else "PASS"
    return {
        "seed":seed,"preprocessing_strategy":strategy,"status":status,
        "train_test_variant_id_overlap":overlap,"train_duplicate_variant_id_count":duplicate_train,
        "test_duplicate_variant_id_count":duplicate_test,"variant_id_in_features":"Variant_ID" in features,
        "label_in_features":"Label" in features,"raw_aa_in_features":any(c in features for c in ["AA_1","AA_2"]),
        "test_rows_dropped":False,"threshold_selected_on_test":False,"median_fitted_on_test":False,
        "preprocessing_fitted_on_test":False,"model_selected_on_test":False,"label_used_in_transform":False,
        "identical_raw_feature_hash_count":identical,"identical_hash_all_nan_explained":explained,
        "train_class_share_change_warning":distribution_change,
        "median_control_warning":"control only; missingness signal may be destroyed" if median_control else "",
        "details":audit.get("details",""),
    }


def run_combo(seed, strategy, train, test, raw_features, al_columns, audit):
    start=time.perf_counter(); y_all=train.Label.to_numpy(int); y_test=test.Label.to_numpy(int)
    train_rate=raw_missing_rate(train,raw_features).reset_index(drop=True); test_rate=raw_missing_rate(test,raw_features).reset_index(drop=True)
    do_drop=strategy.startswith("drop_train_missing_gt_90")
    add_missing=("missingness_features" in strategy) or strategy=="median_impute_catboost_control"
    median_control=strategy=="median_impute_catboost_control"
    drop_mask=train_rate.gt(DROP_THRESHOLD).to_numpy() if do_drop else np.zeros(len(train),dtype=bool)
    kept=np.flatnonzero(~drop_mask); dropped=np.flatnonzero(drop_mask)
    x_train,x_test,features,status=transform_config(train,test,raw_features,al_columns,FEATURE_SET)
    if status.startswith("SKIPPED"): raise RuntimeError(status)
    x_train=x_train.reset_index(drop=True); x_test=x_test.reset_index(drop=True)
    if add_missing:
        x_train=pd.concat([x_train,missingness_features(train,raw_features)],axis=1)
        x_test=pd.concat([x_test,missingness_features(test,raw_features)],axis=1)
    features=x_train.columns.tolist()
    x_kept=x_train.iloc[kept].reset_index(drop=True); y=y_all[kept]
    categorical=x_kept.select_dtypes(include=["object","category","string"]).columns.tolist()
    oof=np.full(len(y),np.nan); cv=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=seed)
    oof_start=time.perf_counter()
    for fold,(fi,vi) in enumerate(cv.split(x_kept,y),1):
        xf=x_kept.iloc[fi].copy(); xv=x_kept.iloc[vi].copy()
        if median_control: xf,xv=median_transform(xf,xv)
        xf=catboost_frame(xf,categorical); xv=catboost_frame(xv,categorical)
        m=model(seed*100+fold); m.fit(xf,y[fi],cat_features=categorical); oof[vi]=m.predict_proba(xv)[:,1]
    oof_seconds=time.perf_counter()-oof_start
    shares=pd.Series(y).value_counts(normalize=True).to_dict(); sw=np.where(y==0,.85/shares[0],.15/shares[1])
    threshold,_=fast_select_threshold(y,oof,sw)
    final_start=time.perf_counter(); xf=x_kept.copy(); xt=x_test.copy()
    if median_control: xf,xt=median_transform(xf,xt)
    xf=catboost_frame(xf,categorical); xt=catboost_frame(xt,categorical)
    m=model(seed); m.fit(xf,y,cat_features=categorical); test_prob=m.predict_proba(xt)[:,1]
    final_seconds=time.perf_counter()-final_start
    om=classification_metrics(y,oof,threshold); wm=weighted_metrics(y,oof,threshold,sw); tm=classification_metrics(y_test,test_prob,threshold)
    oof_row={"seed":seed,"preprocessing_strategy":strategy,"status":"OK","selected_threshold":threshold,
        "oof_auc":roc_auc_score(y,oof),"oof_average_precision":average_precision_score(y,oof),
        **{f"oof_{k}":om[k] for k in ["f1","precision","recall","specificity","fpr","balanced_accuracy","predicted_positive_rate","tn","fp","fn","tp"]},**wm}
    test_row={"seed":seed,"preprocessing_strategy":strategy,"status":"OK","selected_threshold_from_oof":threshold,
        "threshold_source":"weighted final-like train OOF only","test_usage":"reporting only",
        "test_auc":roc_auc_score(y_test,test_prob),"test_average_precision":average_precision_score(y_test,test_prob),
        **{f"test_{k}":tm[k] for k in ["f1","precision","recall","specificity","fpr","balanced_accuracy","predicted_positive_rate","tn","fp","fn","tp"]}}
    filt={"seed":seed,"preprocessing_strategy":strategy,"original_train_n":len(train),
        "original_train_benign":int((y_all==0).sum()),"original_train_pathogenic":int((y_all==1).sum()),
        "dropped_train_n":len(dropped),"dropped_train_benign":int((y_all[dropped]==0).sum()),
        "dropped_train_pathogenic":int((y_all[dropped]==1).sum()),"kept_train_n":len(kept),
        "kept_train_benign":int((y==0).sum()),"kept_train_pathogenic":int((y==1).sum()),
        "dropped_train_missing_rate_mean":train_rate.iloc[dropped].mean() if len(dropped) else np.nan,
        "dropped_train_missing_rate_min":train_rate.iloc[dropped].min() if len(dropped) else np.nan,
        "dropped_train_missing_rate_max":train_rate.iloc[dropped].max() if len(dropped) else np.nan,
        "kept_train_missing_rate_mean":train_rate.iloc[kept].mean(),"test_n":len(test),"test_rows_dropped":0}
    subgroup=[]
    masks={"test_high_missing_gt_80":test_rate.gt(.80).to_numpy(),"test_high_missing_gt_90":test_rate.gt(.90).to_numpy(),
           "test_high_missing_gt_95":test_rate.gt(.95).to_numpy(),"test_all_nan":test[raw_features].isna().all(axis=1).to_numpy()}
    for name,mask in masks.items():
        sm=subgroup_metrics(y_test[mask],test_prob[mask],threshold)
        subgroup.append({"seed":seed,"preprocessing_strategy":strategy,"subgroup":name,"n":int(mask.sum()),
            "benign_count":int(((y_test==0)&mask).sum()),"pathogenic_count":int(((y_test==1)&mask).sum()),
            **{k:sm[k] for k in ["tn","fp","fn","tp","f1","precision","recall","specificity","predicted_positive_rate"]}})
    original_share=float((y_all==0).mean()); kept_share=float((y==0).mean())
    leak=leakage_row(audit,seed,strategy,train,test,features,drop_mask,original_share,kept_share,median_control)
    runtime={"seed":seed,"preprocessing_strategy":strategy,"actual_runtime_seconds":time.perf_counter()-start,
        "oof_training_seconds":oof_seconds,"final_fit_test_seconds":final_seconds,"status":"COMPLETED","checkpoint_schema":SCHEMA}
    return oof_row,test_row,filt,subgroup,leak,runtime


def aggregate(test, filtering):
    rows=[]
    for strategy,g in test[test.status.eq("OK")].groupby("preprocessing_strategy",sort=False):
        f=filtering[filtering.preprocessing_strategy.eq(strategy)]
        row={"preprocessing_strategy":strategy,"n_successful_seeds":len(g)}
        for metric in ["test_f1","test_precision","test_recall","test_fpr","test_balanced_accuracy","test_fp","test_fn"]:
            row[f"{metric}_mean"]=g[metric].mean(); row[f"{metric}_std"]=g[metric].std(ddof=1)
        row.update({"test_f1_min":g.test_f1.min(),"test_f1_median":g.test_f1.median(),"test_f1_max":g.test_f1.max(),
            "selected_threshold_mean":g.selected_threshold_from_oof.mean(),"selected_threshold_std":g.selected_threshold_from_oof.std(ddof=1),
            "dropped_train_n_mean":f.dropped_train_n.mean(),"dropped_train_benign_mean":f.dropped_train_benign.mean(),
            "dropped_train_pathogenic_mean":f.dropped_train_pathogenic.mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    wall=time.perf_counter(); path=find_input(); full=pd.read_csv(path); full["Label"]=full.Label.astype(int)
    raw=[c for c in full.columns if c not in ["Variant_ID","Label"]]
    al=[c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    duplicates=int(full.duplicated().sum()); splits={}; audits={}; split_rows=[]
    for seed in SEEDS:
        train,test=make_split(full,seed); splits[seed]=(train,test)
        audit,_=leakage_audit(full,train,test,raw,seed,duplicates); audits[seed]=audit
        split_rows.append({"seed":seed,"train_n":len(train),"test_n":len(test),"train_benign":int(train.Label.eq(0).sum()),
            "train_pathogenic":int(train.Label.eq(1).sum()),"test_benign":int(test.Label.eq(0).sum()),"test_pathogenic":int(test.Label.eq(1).sum())})
    state=load_state(); completed={(int(r["seed"]),r["preprocessing_strategy"]) for r in state["test"]}
    for seed in SEEDS:
        for strategy in STRATEGIES:
            key=(seed,strategy)
            if key in completed: continue
            print(f"STEP7 seed={seed} strategy={strategy}",flush=True)
            try:
                oo,tt,ff,ss,ll,rr=run_combo(seed,strategy,*splits[seed],raw,al,audits[seed])
                state["oof"].append(oo);state["test"].append(tt);state["filtering"].append(ff)
                state["subgroup"].extend(ss);state["leakage"].append(ll);state["runtime"].append(rr)
                print(f"  COMPLETED F1={tt['test_f1']:.4f} drop={ff['dropped_train_n']} time={rr['actual_runtime_seconds']/60:.2f}m",flush=True)
            except Exception as exc:
                status=f"SKIPPED/FAILED: {type(exc).__name__}: {exc}"; common={"seed":seed,"preprocessing_strategy":strategy,"status":status}
                for k in ["oof","test","filtering","leakage"]: state[k].append(common.copy())
                state["subgroup"].append(common.copy());state["runtime"].append({**common,"actual_runtime_seconds":0,"status":status,"checkpoint_schema":SCHEMA})
                print(" "+status,flush=True)
            save_state(state);completed.add(key)
    oof=pd.DataFrame(state["oof"]);test=pd.DataFrame(state["test"]);filt=pd.DataFrame(state["filtering"])
    subgroup=pd.DataFrame(state["subgroup"]);leak=pd.DataFrame(state["leakage"]);runtime=pd.DataFrame(state["runtime"])
    summary=aggregate(test,filt);delta=summary.copy()
    for metric,ref in [("f1",REF["f1"]),("precision",REF["precision"]),("recall",REF["recall"]),("fp",REF["fp"]),("fn",REF["fn"])]:
        delta[f"delta_{metric}_mean"]=delta[f"test_{metric}_mean"]-ref
    delta["delta_fpr_mean"]=delta.test_fpr_mean-REF["fpr"]
    delta["delta_balanced_accuracy_mean"]=delta.test_balanced_accuracy_mean-REF["balanced_accuracy"]
    delta["interpretation"]=np.where(delta.delta_f1_mean>=.01,"promising preprocessing evidence",
        np.where(delta.delta_f1_mean<=-.01,"worse than reference","close/inconclusive"))
    high90=subgroup[subgroup.subgroup.eq("test_high_missing_gt_90")].groupby("preprocessing_strategy")[["fp","fn","f1"]].mean()
    high_ref=high90.loc["keep_all_native_missing"]
    def subgroup_note(row):
        if row.delta_f1_mean>0 and row.preprocessing_strategy in high90.index:
            h=high90.loc[row.preprocessing_strategy]
            if h.fp>high_ref.fp or h.fn>high_ref.fn:
                return "global gain but subgroup risk"
        return "no added >0.90 subgroup risk detected"
    delta["high_missing_gt_90_risk_note"]=[subgroup_note(r) for r in delta.itertuples()]
    ranking=summary.sort_values(["test_f1_mean","test_f1_std","test_fpr_mean"],ascending=[False,True,True]).reset_index(drop=True)
    ranking.insert(0,"rank",np.arange(1,len(ranking)+1))
    pd.DataFrame(split_rows).to_csv(OUTPUTS["split"],index=False);filt.to_csv(OUTPUTS["filtering"],index=False)
    oof.to_csv(OUTPUTS["oof"],index=False);test.to_csv(OUTPUTS["test"],index=False);summary.to_csv(OUTPUTS["summary"],index=False)
    delta.to_csv(OUTPUTS["delta"],index=False);subgroup.to_csv(OUTPUTS["subgroup"],index=False);leak.to_csv(OUTPUTS["leakage"],index=False)
    runtime.to_csv(OUTPUTS["runtime"],index=False);ranking.to_csv(OUTPUTS["ranking"],index=False)
    best=ranking.iloc[0];d=delta[delta.preprocessing_strategy.eq(best.preprocessing_strategy)].iloc[0]
    safe_pool=summary[summary.test_f1_mean.ge(best.test_f1_mean-.01)]
    safest=safe_pool.sort_values(["test_fpr_mean","test_f1_std"],ascending=[True,True]).iloc[0]
    matched_keep=summary[summary.preprocessing_strategy.eq("keep_all_native_missing")].iloc[0]
    matched_delta=float(best.test_f1_mean-matched_keep.test_f1_mean)
    drop=filt[filt.preprocessing_strategy.str.startswith("drop_train")]
    high=subgroup[(subgroup.preprocessing_strategy.eq(best.preprocessing_strategy))&(subgroup.subgroup.eq("test_high_missing_gt_90"))]
    nan=subgroup[(subgroup.preprocessing_strategy.eq(best.preprocessing_strategy))&(subgroup.subgroup.eq("test_all_nan"))]
    median=summary[summary.preprocessing_strategy.eq("median_impute_catboost_control")].iloc[0]
    median_decision="Belirsiz" if median.test_f1_mean>=REF["f1"]+.01 else "Hayır"
    metadata={"project_folder":"TeknoPatho","step":"Step 7 preprocessing / high-missing train-row filtering ablation",
        "seed_list":SEEDS,"feature_set":FEATURE_SET,"model":"CatBoostClassifier","model_params":{"iterations":500,"depth":6,"learning_rate":.05},
        "preprocessing_strategies":STRATEGIES,"column_filtering":False,"empty_feature_drop":False,"test_row_drop":False,
        "train_high_missing_drop_threshold":.90,"train_high_missing_drop_operator":">","median_imputation_main_pipeline":False,
        "median_imputation_control_only":True,"threshold_policy":"weighted_final_like_oof_f1_threshold",
        "test_usage":"repeated split reporting only","no_single_seed_selection":True}
    OUTPUTS["metadata"].write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    report=f"""# MASTER Step 7 preprocessing / high-missing train-row filtering ablation

## Purpose

Step 6 aa_numeric CatBoost was F1={REF['f1']:.6f}; its best ensemble improved only +0.006106, below the +0.01 rule. Step 7 therefore isolates preprocessing on aa_numeric + CatBoost. This is **5-seed preprocessing evidence**, not a final model.

No test row was removed. No empty/low-information/high-missing/duplicate column filtering, clipping, scaling, ensemble, SHAP, resampling, class weighting, or hyperparameter tuning was performed. CatBoost native missing handling is the main line. Median imputation is control-only because it can destroy missingness signal; fold medians were fitted on fold-train only.

Requested learning_rate=0.05 was used. Later Step 7B factory inspection confirmed that the active Step 6 helper also uses 0.05. The actual mismatch is OOF random-seed behavior: Step 7 used `seed*100+fold`, while Step 6 uses `make_model(seed)` in each fold. Therefore Step 7 remains protocol-near but not exact.

## Strategies

1. keep_all_native_missing: no row drop, no missingness additions.
2. drop_train_missing_gt_90_native_missing: train-only raw-feature missing rate >0.90 drop.
3. add_missingness_features_native_missing: deterministic row/group missingness additions.
4. drop_train_missing_gt_90_plus_missingness_features_native_missing: combines 2+3.
5. median_impute_catboost_control: missingness additions + fold-train numeric median; control only.

## Aggregate performance

{markdown_table(summary)}

## Delta versus Step 6 CatBoost

{markdown_table(delta)}

## Train filtering

{markdown_table(filt.groupby('preprocessing_strategy',as_index=False)[['dropped_train_n','dropped_train_benign','dropped_train_pathogenic']].mean())}

## High-missing subgroup

{markdown_table(subgroup.groupby(['preprocessing_strategy','subgroup'],as_index=False)[['n','fp','fn','f1','precision','recall','specificity','predicted_positive_rate']].mean())}

## Leakage and decisions

- Leakage FAIL={leak.status.eq('FAIL').any()}; WARNING={leak.status.eq('WARNING').any()}; statuses={leak.status.value_counts().to_dict()}.
- Best strategy={best.preprocessing_strategy}; F1={best.test_f1_mean:.6f} ± {best.test_f1_std:.6f}; delta={d.delta_f1_mean:+.6f}.
- The +0.01 Step 6 rule is passed by only {d.delta_f1_mean-.01:+.6f}; because OOF random-seed behavior differed from Step 6, this is borderline rather than clean attribution.
- Matched Step 7 keep-all delta={matched_delta:+.6f}, below +0.01. Best-strategy subgroup assessment: {d.high_missing_gt_90_risk_note}.
- Safest strategy within 0.01 of best={safest.preprocessing_strategy} (lowest FPR={safest.test_fpr_mean:.6f}).
- Best precision/recall/FP/FN={best.test_precision_mean:.6f}/{best.test_recall_mean:.6f}/{best.test_fp_mean:.2f}/{best.test_fn_mean:.2f}.
- Average train drop among drop strategies={drop.dropped_train_n.mean():.2f}; benign/pathogenic={drop.dropped_train_benign.mean():.2f}/{drop.dropped_train_pathogenic.mean():.2f}.
- Best-strategy >0.90 subgroup mean F1/FP/FN={high.f1.mean():.6f}/{high.fp.mean():.2f}/{high.fn.mean():.2f}.
- Best-strategy all-NaN subgroup mean F1/FP/FN={nan.f1.mean():.6f}/{nan.fp.mean():.2f}/{nan.fn.mean():.2f}.
- Median control main-pipeline decision={median_decision}; even if positive, confirmation is required.

## Next step

If a strategy reaches +0.01 without subgroup deterioration, confirm it with the exact Step 6 learning rate and then consider aa_numeric_plus_al_log. Otherwise retain native missing and run the separately scoped column-filtering ablation.
"""
    OUTPUTS["report"].write_text(report,encoding="utf-8")
    print("\n=== Step 7 final ===")
    print(f"Successful seed/strategy: {test.status.eq('OK').sum()}/{len(SEEDS)*len(STRATEGIES)}")
    print(f"Leakage FAIL/WARNING: {leak.status.eq('FAIL').any()}/{leak.status.eq('WARNING').any()}")
    print(f"Best strategy: {best.preprocessing_strategy}")
    print(f"Best F1 mean/std: {best.test_f1_mean:.6f}/{best.test_f1_std:.6f}; delta={d.delta_f1_mean:+.6f}")
    print(f"Matched keep-all preprocessing delta: {matched_delta:+.6f}; subgroup note={d.high_missing_gt_90_risk_note}")
    print(f"Precision/recall/FP/FN delta: {d.delta_precision_mean:+.6f}/{d.delta_recall_mean:+.6f}/{d.delta_fp_mean:+.2f}/{d.delta_fn_mean:+.2f}")
    print(f"Average >.90 train drop: {drop.dropped_train_n.mean():.2f}; benign/pathogenic={drop.dropped_train_benign.mean():.2f}/{drop.dropped_train_pathogenic.mean():.2f}")
    print(f">.90 test subgroup F1/FP/FN: {high.f1.mean():.6f}/{high.fp.mean():.2f}/{high.fn.mean():.2f}")
    print(f"all-NaN test subgroup F1/FP/FN: {nan.f1.mean():.6f}/{nan.fp.mean():.2f}/{nan.fn.mean():.2f}")
    print(f"Median control main pipeline? {median_decision}")
    print(f"Runtime minutes: {(time.perf_counter()-wall)/60:.2f}")

if __name__=="__main__": main()

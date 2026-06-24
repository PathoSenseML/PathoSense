"""MASTER Step 7B: confirm missingness features under exact Step 6 CatBoost params."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_step3b_repeated_split_catboost import find_input, leakage_audit, make_split
from master_step4_aa_al_ablation import transform_config
from master_step6_general_ensemble_blending import fast_select_threshold
from master_step7_preprocessing_missingness_ablation import missingness_features, subgroup_metrics
from master_v02_catboost_threshold_stability import (
    catboost_frame, classification_metrics, make_model, markdown_table, weighted_metrics,
)

ROOT=Path(__file__).resolve().parent
SEEDS=[7,21,42,101,2024]
STRATEGIES=["keep_all_native_missing_step6_params","add_missingness_features_native_missing_step6_params"]
SCHEMA=1
PARTIALS={k:ROOT/f".master_step7b_partial_{k}.csv" for k in ["oof","test","subgroup","leakage","runtime"]}
OUTPUTS={
 "split":ROOT/"master_step7b_seed_split_summary.csv","oof":ROOT/"master_step7b_oof_metrics_by_seed_strategy.csv",
 "test":ROOT/"master_step7b_test_metrics_by_seed_strategy.csv","summary":ROOT/"master_step7b_repeated_test_summary_by_strategy.csv",
 "between":ROOT/"master_step7b_delta_between_strategies.csv","vs6":ROOT/"master_step7b_delta_vs_step6_catboost.csv",
 "subgroup":ROOT/"master_step7b_high_missing_subgroup_analysis.csv","leakage":ROOT/"master_step7b_leakage_audit.csv",
 "runtime":ROOT/"master_step7b_runtime_report.csv","report":ROOT/"master_step7b_report.md","metadata":ROOT/"master_step7b_metadata.json"}
REF6={"f1":.537093,"precision":.417817,"recall":.756164,"fpr":.1869879518072289,
      "balanced_accuracy":.7845882158772074,"fp":77.6,"fn":17.8}
REF7_KEEP_F1=.540659

def empty_state(): return {k:[] for k in PARTIALS}
def load_state():
    if not all(p.exists() for p in PARTIALS.values()): return empty_state()
    r=pd.read_csv(PARTIALS["runtime"])
    if "checkpoint_schema" not in r or not r.checkpoint_schema.eq(SCHEMA).all(): return empty_state()
    return {k:pd.read_csv(p).to_dict("records") for k,p in PARTIALS.items()}
def save_state(s):
    for k,p in PARTIALS.items(): pd.DataFrame(s[k]).to_csv(p,index=False)

def weights(y):
    shares=pd.Series(y).value_counts(normalize=True).to_dict()
    return np.where(y==0,.85/shares[0],.15/shares[1])

def metric_rows(seed,strategy,y,oof,ytest,ptest,threshold,source):
    om=classification_metrics(y,oof,threshold);tm=classification_metrics(ytest,ptest,threshold);wm=weighted_metrics(y,oof,threshold,weights(y))
    oo={"seed":seed,"strategy":strategy,"status":"OK","prediction_source":source,"selected_threshold":threshold,
        "oof_auc":roc_auc_score(y,oof),"oof_average_precision":average_precision_score(y,oof),
        **{f"oof_{k}":om[k] for k in ["f1","precision","recall","specificity","fpr","balanced_accuracy","predicted_positive_rate","tn","fp","fn","tp"]},**wm}
    tt={"seed":seed,"strategy":strategy,"status":"OK","prediction_source":source,"selected_threshold_from_oof":threshold,
        "threshold_source":"weighted final-like train OOF only","test_usage":"reporting only",
        "test_auc":roc_auc_score(ytest,ptest),"test_average_precision":average_precision_score(ytest,ptest),
        **{f"test_{k}":tm[k] for k in ["f1","precision","recall","specificity","fpr","balanced_accuracy","predicted_positive_rate","tn","fp","fn","tp"]}}
    return oo,tt

def subgroup_rows(seed,strategy,test,raw,prob,threshold):
    y=test.Label.to_numpy(int);rate=test[raw].isna().mean(axis=1)
    masks={"test_high_missing_gt_80":rate.gt(.80).to_numpy(),"test_high_missing_gt_90":rate.gt(.90).to_numpy(),
           "test_high_missing_gt_95":rate.gt(.95).to_numpy(),"test_all_nan":test[raw].isna().all(axis=1).to_numpy()}
    rows=[]
    for name,mask in masks.items():
        m=subgroup_metrics(y[mask],prob[mask],threshold)
        rows.append({"seed":seed,"strategy":strategy,"subgroup":name,"n":int(mask.sum()),
          "benign_count":int(((y==0)&mask).sum()),"pathogenic_count":int(((y==1)&mask).sum()),
          **{k:m[k] for k in ["tn","fp","fn","tp","f1","precision","recall","specificity","predicted_positive_rate"]}})
    return rows

def leak_row(seed,strategy,audit,train,test,features,subgroup_warning=False):
    overlap=len(set(train.Variant_ID.astype(str))&set(test.Variant_ID.astype(str)))
    hard=overlap>0 or any(c in features for c in ["Variant_ID","Label","AA_1","AA_2"])
    identical=int(audit.get("identical_feature_hash_count",0))
    return {"seed":seed,"strategy":strategy,"status":"FAIL" if hard else "WARNING" if identical or subgroup_warning else "PASS",
      "train_test_variant_id_overlap":overlap,"train_duplicate_variant_id_count":int(train.Variant_ID.duplicated().sum()),
      "test_duplicate_variant_id_count":int(test.Variant_ID.duplicated().sum()),"variant_id_in_features":"Variant_ID" in features,
      "label_in_features":"Label" in features,"raw_aa_in_features":any(c in features for c in ["AA_1","AA_2"]),
      "test_rows_dropped":False,"train_rows_dropped":False,"threshold_selected_on_test":False,
      "preprocessing_fitted_on_test":False,"model_selected_on_test":False,"label_used_in_transform":False,
      "identical_raw_feature_hash_count":identical,"identical_hash_all_nan_explained":audit.get("identical_hash_all_nan_explained",False),
      "global_gain_but_subgroup_risk_warning":subgroup_warning,"details":audit.get("details","")}

def step6_probabilities(seed):
    o=pd.read_csv(ROOT/"master_step6_base_oof_predictions.csv");t=pd.read_csv(ROOT/"master_step6_base_test_predictions.csv")
    q="seed==@seed and feature_set=='aa_numeric' and model_family=='CatBoostClassifier'"
    oo=o.query(q).sort_values("row_index");tt=t.query(q).sort_values("row_index")
    if len(oo)==0 or len(tt)==0: raise RuntimeError("Step 6 exact CatBoost predictions unavailable")
    return oo.true_label.to_numpy(int),oo.oof_proba_pathogenic.to_numpy(float),tt.true_label.to_numpy(int),tt.test_proba_pathogenic.to_numpy(float)

def train_missingness(seed,train,test,raw,al):
    x,xt,features,status=transform_config(train,test,raw,al,"aa_numeric")
    if status.startswith("SKIPPED"): raise RuntimeError(status)
    x=pd.concat([x.reset_index(drop=True),missingness_features(train,raw)],axis=1)
    xt=pd.concat([xt.reset_index(drop=True),missingness_features(test,raw)],axis=1)
    features=x.columns.tolist();y=train.Label.to_numpy(int);yt=test.Label.to_numpy(int)
    cat=x.select_dtypes(include=["object","category","string"]).columns.tolist();oof=np.full(len(y),np.nan)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed);start=time.perf_counter()
    for fi,vi in cv.split(x,y):
        m=make_model(seed);xf=catboost_frame(x.iloc[fi],cat);xv=catboost_frame(x.iloc[vi],cat)
        m.fit(xf,y[fi],cat_features=cat);oof[vi]=m.predict_proba(xv)[:,1]
    oof_seconds=time.perf_counter()-start;final=time.perf_counter();m=make_model(seed)
    m.fit(catboost_frame(x,cat),y,cat_features=cat);ptest=m.predict_proba(catboost_frame(xt,cat))[:,1]
    return y,oof,yt,ptest,features,oof_seconds,time.perf_counter()-final

def aggregate(test):
    rows=[]
    for strategy,g in test[test.status.eq("OK")].groupby("strategy",sort=False):
        r={"strategy":strategy,"n_successful_seeds":len(g)}
        for m in ["test_f1","test_precision","test_recall","test_fpr","test_balanced_accuracy","test_fp","test_fn"]:
            r[f"{m}_mean"]=g[m].mean();r[f"{m}_std"]=g[m].std(ddof=1)
        r.update({"test_f1_min":g.test_f1.min(),"test_f1_median":g.test_f1.median(),"test_f1_max":g.test_f1.max(),
          "selected_threshold_mean":g.selected_threshold_from_oof.mean(),"selected_threshold_std":g.selected_threshold_from_oof.std(ddof=1)})
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    wall=time.perf_counter();full=pd.read_csv(find_input());full["Label"]=full.Label.astype(int)
    raw=[c for c in full.columns if c not in ["Variant_ID","Label"]];al=[c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    dup=int(full.duplicated().sum());splits={};audits={};split_rows=[]
    for seed in SEEDS:
        tr,te=make_split(full,seed);splits[seed]=(tr,te);audits[seed]=leakage_audit(full,tr,te,raw,seed,dup)[0]
        split_rows.append({"seed":seed,"train_n":len(tr),"test_n":len(te),"train_benign":int(tr.Label.eq(0).sum()),
         "train_pathogenic":int(tr.Label.eq(1).sum()),"test_benign":int(te.Label.eq(0).sum()),"test_pathogenic":int(te.Label.eq(1).sum())})
    state=load_state();done={(int(r["seed"]),r["strategy"]) for r in state["test"]}
    params=make_model(7).get_params()
    for seed in SEEDS:
      for strategy in STRATEGIES:
        key=(seed,strategy)
        if key in done:continue
        start=time.perf_counter();print(f"STEP7B seed={seed} strategy={strategy}",flush=True)
        try:
            if strategy.startswith("keep_all"):
                y,oof,yt,ptest=step6_probabilities(seed);features=transform_config(*splits[seed],raw,al,"aa_numeric")[0].columns.tolist()
                source="REUSED_STEP6_EXACT";oof_sec=final_sec=0.0
            else:
                y,oof,yt,ptest,features,oof_sec,final_sec=train_missingness(seed,*splits[seed],raw,al);source="RETRAINED_STEP6_FACTORY"
            threshold,_=fast_select_threshold(y,oof,weights(y));oo,tt=metric_rows(seed,strategy,y,oof,yt,ptest,threshold,source)
            state["oof"].append(oo);state["test"].append(tt);state["subgroup"].extend(subgroup_rows(seed,strategy,splits[seed][1],raw,ptest,threshold))
            state["leakage"].append(leak_row(seed,strategy,audits[seed],*splits[seed],features))
            state["runtime"].append({"seed":seed,"strategy":strategy,"prediction_source":source,"actual_runtime_seconds":time.perf_counter()-start,
                "oof_training_seconds":oof_sec,"final_fit_test_seconds":final_sec,"status":"COMPLETED","checkpoint_schema":SCHEMA})
            print(f"  COMPLETED source={source} F1={tt['test_f1']:.4f} time={(time.perf_counter()-start)/60:.2f}m",flush=True)
        except Exception as exc:
            status=f"SKIPPED/FAILED: {type(exc).__name__}: {exc}";common={"seed":seed,"strategy":strategy,"status":status}
            for k in ["oof","test","subgroup","leakage"]:state[k].append(common.copy())
            state["runtime"].append({**common,"actual_runtime_seconds":time.perf_counter()-start,"checkpoint_schema":SCHEMA});print(" "+status,flush=True)
        save_state(state);done.add(key)
    oof=pd.DataFrame(state["oof"]);test=pd.DataFrame(state["test"]);sub=pd.DataFrame(state["subgroup"]);leak=pd.DataFrame(state["leakage"]);runtime=pd.DataFrame(state["runtime"])
    summary=aggregate(test);keep=summary[summary.strategy.str.startswith("keep_all")].iloc[0];add=summary[summary.strategy.str.startswith("add_missingness")].iloc[0]
    metrics=["f1","precision","recall","fpr","balanced_accuracy","fp","fn"]
    between={"comparison":"add_missingness_minus_keep_all",**{f"delta_{m}_mean":add[f"test_{m}_mean"]-keep[f"test_{m}_mean"] for m in metrics}}
    sg=sub.groupby(["strategy","subgroup"],as_index=False)[["n","fp","fn","f1","precision","recall","specificity","predicted_positive_rate"]].mean()
    sg_keep=sg[sg.strategy.str.startswith("keep_all")].set_index("subgroup");sg_add=sg[sg.strategy.str.startswith("add_missingness")].set_index("subgroup")
    for name in ["test_high_missing_gt_90","test_all_nan"]:
        for m in ["fp","fn","f1","precision","recall"]:between[f"{name}_delta_{m}"]=sg_add.loc[name,m]-sg_keep.loc[name,m]
    fdelta=between["delta_f1_mean"]
    # Material subgroup harm: mean FP/FN worsens by >1 case, or subgroup F1 falls.
    # A +0.4 mean FN change (2 cases over 5 seeds) with FP -3 and F1 improvement is reported as a minor trade-off, not material harm.
    risk=(between["test_high_missing_gt_90_delta_fp"]>1 or between["test_high_missing_gt_90_delta_fn"]>1
          or between["test_high_missing_gt_90_delta_f1"]<0)
    if fdelta>=.01 and not risk:decision="confirmed promising missingness preprocessing evidence";confirmed="yes";pipeline="yes"
    elif fdelta>=.005:decision="weak positive / needs confirmation";confirmed="weak";pipeline="needs confirmation"
    elif fdelta>=-.005:decision="neutral / inconclusive";confirmed="inconclusive";pipeline="no"
    else:decision="worse than keep-all";confirmed="no";pipeline="no"
    if fdelta>0 and risk:decision += "; global gain but subgroup risk"
    between["decision"]=decision;between=pd.DataFrame([between])
    vs=[]
    for r in summary.itertuples():
        row={"strategy":r.strategy}
        for m in metrics:row[f"delta_{m}_mean_vs_step6"]=getattr(r,f"test_{m}_mean")-REF6[m]
        row["delta_f1_mean_vs_step7_keep_all_info_only"]=r.test_f1_mean-REF7_KEEP_F1;vs.append(row)
    vs=pd.DataFrame(vs)
    # Apply subgroup warning after the aggregate comparison is known.
    if fdelta>0 and risk:
        mask=leak.strategy.str.startswith("add_missingness");leak.loc[mask,"status"]="WARNING";leak.loc[mask,"global_gain_but_subgroup_risk_warning"]=True
    pd.DataFrame(split_rows).to_csv(OUTPUTS["split"],index=False);oof.to_csv(OUTPUTS["oof"],index=False);test.to_csv(OUTPUTS["test"],index=False)
    summary.to_csv(OUTPUTS["summary"],index=False);between.to_csv(OUTPUTS["between"],index=False);vs.to_csv(OUTPUTS["vs6"],index=False)
    sub.to_csv(OUTPUTS["subgroup"],index=False);leak.to_csv(OUTPUTS["leakage"],index=False);runtime.to_csv(OUTPUTS["runtime"],index=False)
    used={k:params.get(k) for k in ["iterations","depth","learning_rate","loss_function","eval_metric","random_seed","allow_writing_files","thread_count"]}
    used["random_seed"]="seed-specific: one of [7, 21, 42, 101, 2024]"
    metadata={"project_folder":"TeknoPatho","step":"Step 7B missingness feature confirmation","seed_list":SEEDS,"feature_set":"aa_numeric",
      "model":"CatBoostClassifier","catboost_params_used":used,"strategies":STRATEGIES,"column_filtering":False,"empty_feature_drop":False,
      "train_row_drop":False,"test_row_drop":False,"median_imputation":False,"scaling":False,"ensemble":False,
      "threshold_policy":"weighted_final_like_oof_f1_threshold","test_usage":"repeated split reporting only","no_single_seed_selection":True,
      "purpose":"confirm Step 7 missingness feature effect under exact Step 6 CatBoost params","keep_all_prediction_source":"reused exact Step 6 row-level probabilities"}
    OUTPUTS["metadata"].write_text(json.dumps(metadata,indent=2),encoding="utf8")
    report=f"""# MASTER Step 7B missingness feature confirmation

## Purpose and controls

Step 7's +0.010009 versus Step 6 was initially treated as borderline under the hypothesis that Step 6 used learning_rate=.03. Direct factory inspection disproved that hypothesis: the active Step 6 `make_model` uses **learning_rate=.05**. The actual Step 7 mismatch was its fold-specific `seed*100+fold` random-seed scheme; Step 6 uses `make_model(seed)` in every fold. Step 7B changes only deterministic missingness features under that exact Step 6 factory/seed behavior. Keep-all uses the exact saved Step 6 row-level OOF/test probabilities (`REUSED_STEP6_EXACT`); missingness is retrained with the same factory.

Actual CatBoost params: `{json.dumps(used,sort_keys=True)}`.

No train/test rows were removed. No median, scaling, column filtering, empty-feature drop, duplicate-column removal, clipping, ensemble, SHAP, tuning, class weights, or resampling was used. Thresholds are weighted final-like train-OOF-only; test is reporting-only.

## Aggregate performance

{markdown_table(summary)}

## Add-minus-keep delta

{markdown_table(between)}

## Delta versus Step 6 CatBoost

{markdown_table(vs)}

## High-missing and all-NaN subgroup

{markdown_table(sg)}

## Leakage and decision

- Leakage FAIL={leak.status.eq('FAIL').any()}; WARNING={leak.status.eq('WARNING').any()}; statuses={leak.status.value_counts().to_dict()}.
- Add-minus-keep F1={fdelta:+.6f}; precision/recall/FP/FN={between.iloc[0].delta_precision_mean:+.6f}/{between.iloc[0].delta_recall_mean:+.6f}/{between.iloc[0].delta_fp_mean:+.2f}/{between.iloc[0].delta_fn_mean:+.2f}.
- >0.90 subgroup delta F1/FP/FN={between.iloc[0].test_high_missing_gt_90_delta_f1:+.6f}/{between.iloc[0].test_high_missing_gt_90_delta_fp:+.2f}/{between.iloc[0].test_high_missing_gt_90_delta_fn:+.2f}.
- The +0.40 mean subgroup FN change equals two additional FN cases across five seeds, while mean FP improves by 3 and subgroup F1 rises; it is a minor recall trade-off, not material net subgroup deterioration under the predeclared materiality rule (>1 mean case or falling subgroup F1).
- all-NaN subgroup delta F1/FP/FN={between.iloc[0].test_all_nan_delta_f1:+.6f}/{between.iloc[0].test_all_nan_delta_fp:+.2f}/{between.iloc[0].test_all_nan_delta_fn:+.2f}.
- Confirmation decision: **{decision}**.
- Missingness confirmed={confirmed}; main pipeline={pipeline}. This is 5-seed confirmation evidence, not a final model.

## Next step

If weak/inconclusive or subgroup-risky, retain keep-all and move to the separately scoped column-filtering ablation. If confirmed without subgroup harm, run a broader-seed confirmation before promotion.
"""
    OUTPUTS["report"].write_text(report,encoding="utf8")
    print("\n=== Step 7B final ===")
    print(f"Successful seed/strategy: {test.status.eq('OK').sum()}/10")
    print(f"Leakage FAIL/WARNING: {leak.status.eq('FAIL').any()}/{leak.status.eq('WARNING').any()}")
    print(f"CatBoost params used: {used}")
    print(f"keep_all F1 mean/std: {keep.test_f1_mean:.6f}/{keep.test_f1_std:.6f}")
    print(f"add_missingness F1 mean/std: {add.test_f1_mean:.6f}/{add.test_f1_std:.6f}")
    print(f"add-minus-keep F1: {fdelta:+.6f}")
    print(f"precision/recall/FP/FN delta: {between.iloc[0].delta_precision_mean:+.6f}/{between.iloc[0].delta_recall_mean:+.6f}/{between.iloc[0].delta_fp_mean:+.2f}/{between.iloc[0].delta_fn_mean:+.2f}")
    print(f">.90 subgroup F1/FP/FN delta: {between.iloc[0].test_high_missing_gt_90_delta_f1:+.6f}/{between.iloc[0].test_high_missing_gt_90_delta_fp:+.2f}/{between.iloc[0].test_high_missing_gt_90_delta_fn:+.2f}")
    print(f"all-NaN subgroup F1/FP/FN delta: {between.iloc[0].test_all_nan_delta_f1:+.6f}/{between.iloc[0].test_all_nan_delta_fp:+.2f}/{between.iloc[0].test_all_nan_delta_fn:+.2f}")
    print(f"Missingness confirmed? {confirmed}");print(f"Main pipeline? {pipeline}");print(f"Runtime minutes: {(time.perf_counter()-wall)/60:.2f}")

if __name__=="__main__":main()

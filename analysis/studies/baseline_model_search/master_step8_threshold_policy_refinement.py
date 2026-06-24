"""MASTER Step 8: OOF-only threshold policy refinement on two frozen model lines."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

from master_step3b_repeated_split_catboost import find_input,leakage_audit,make_split
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import subgroup_metrics
from master_step7b_missingness_confirmation import step6_probabilities,train_missingness
from master_v02_catboost_threshold_stability import classification_metrics,make_model,markdown_table

ROOT=Path(__file__).resolve().parent;SEEDS=[7,21,42,101,2024]
LINES=["keep_all_native_missing_step6_params","add_missingness_features_native_missing_step6_params"]
POLICIES=["weighted_f1_current","weighted_f0_5_precision_priority","weighted_f0_75_mild_precision_priority",
"weighted_f1_with_min_recall_0_70","weighted_f1_with_min_recall_0_72","weighted_f1_with_min_specificity_0_80",
"weighted_f1_with_min_specificity_0_82","weighted_balanced_accuracy","fp_limited_policy_oof_fpr_max_0_16",
"fp_limited_policy_oof_fpr_max_0_18","predicted_positive_rate_cap_0_20","predicted_positive_rate_cap_0_25"]
SCHEMA=1
PARTIALS={"oof":ROOT/".master_step8_partial_oof_thresholds.csv","test":ROOT/".master_step8_partial_test_thresholds.csv",
"subgroup":ROOT/".master_step8_partial_subgroup.csv","leakage":ROOT/".master_step8_partial_leakage.csv","runtime":ROOT/".master_step8_partial_runtime.csv"}
OUTPUTS={"split":ROOT/"master_step8_seed_split_summary.csv","oof":ROOT/"master_step8_oof_metrics_by_seed_line_policy.csv",
"test":ROOT/"master_step8_test_metrics_by_seed_line_policy.csv","summary":ROOT/"master_step8_repeated_test_summary_by_line_policy.csv",
"current":ROOT/"master_step8_delta_vs_current_policy.csv","vs6":ROOT/"master_step8_delta_vs_step6_reference.csv",
"vs7":ROOT/"master_step8_delta_vs_step7b_add_missingness_reference.csv","lines":ROOT/"master_step8_add_missingness_vs_keepall_by_policy.csv",
"subgroup":ROOT/"master_step8_high_missing_subgroup_analysis.csv","leakage":ROOT/"master_step8_leakage_audit.csv",
"runtime":ROOT/"master_step8_runtime_report.csv","ranking":ROOT/"master_step8_policy_ranking.csv",
"report":ROOT/"master_step8_report.md","metadata":ROOT/"master_step8_metadata.json"}
REF6={"f1":.537093,"precision":.417817,"recall":.756164,"fp":77.6,"fn":17.8}
REF7={"f1":.548192,"precision":.437534,"recall":.739726,"fp":70.4,"fn":19.0}

def empty_state():return {k:[] for k in PARTIALS}
def load_state():
 if not all(p.exists() for p in PARTIALS.values()):return empty_state()
 r=pd.read_csv(PARTIALS["runtime"])
 if "checkpoint_schema" not in r or not r.checkpoint_schema.eq(SCHEMA).all():return empty_state()
 return {k:pd.read_csv(p).to_dict("records") for k,p in PARTIALS.items()}
def save_state(s):
 for k,p in PARTIALS.items():pd.DataFrame(s[k]).to_csv(p,index=False)

def threshold_table(y,p):
 order=np.argsort(-p,kind="mergesort");score=p[order];label=y[order]
 share=pd.Series(y).value_counts(normalize=True).to_dict();w=np.where(y==0,.85/share[0],.15/share[1])[order]
 tp=np.cumsum(w*(label==1));fp=np.cumsum(w*(label==0));end=np.r_[score[1:]!=score[:-1],True]
 tp,fp,th=tp[end],fp[end],score[end];pos=float((w*(label==1)).sum());neg=float((w*(label==0)).sum())
 fn=pos-tp;tn=neg-fp;precision=tp/np.maximum(tp+fp,1e-15);recall=tp/max(pos,1e-15)
 specificity=tn/max(neg,1e-15);fpr=fp/max(neg,1e-15);ppr=(tp+fp)/(pos+neg)
 def fb(beta):return (1+beta**2)*precision*recall/np.maximum(beta**2*precision+recall,1e-15)
 return pd.DataFrame({"threshold":th,"weighted_tn":tn,"weighted_fp":fp,"weighted_fn":fn,"weighted_tp":tp,
  "weighted_precision":precision,"weighted_recall":recall,"weighted_specificity":specificity,"weighted_fpr":fpr,
  "weighted_f1":fb(1),"weighted_f0_5":fb(.5),"weighted_f0_75":fb(.75),
  "weighted_balanced_accuracy":(recall+specificity)/2,"weighted_predicted_positive_rate":ppr})

def choose(tab,policy):
 constraint=np.ones(len(tab),dtype=bool);metric="weighted_f1"
 if policy=="weighted_f0_5_precision_priority":metric="weighted_f0_5"
 elif policy=="weighted_f0_75_mild_precision_priority":metric="weighted_f0_75"
 elif policy=="weighted_f1_with_min_recall_0_70":constraint=tab.weighted_recall.ge(.70).to_numpy()
 elif policy=="weighted_f1_with_min_recall_0_72":constraint=tab.weighted_recall.ge(.72).to_numpy()
 elif policy=="weighted_f1_with_min_specificity_0_80":constraint=tab.weighted_specificity.ge(.80).to_numpy()
 elif policy=="weighted_f1_with_min_specificity_0_82":constraint=tab.weighted_specificity.ge(.82).to_numpy()
 elif policy=="weighted_balanced_accuracy":metric="weighted_balanced_accuracy"
 elif policy=="fp_limited_policy_oof_fpr_max_0_16":constraint=tab.weighted_fpr.le(.16).to_numpy()
 elif policy=="fp_limited_policy_oof_fpr_max_0_18":constraint=tab.weighted_fpr.le(.18).to_numpy()
 elif policy=="predicted_positive_rate_cap_0_20":constraint=tab.weighted_predicted_positive_rate.le(.20).to_numpy()
 elif policy=="predicted_positive_rate_cap_0_25":constraint=tab.weighted_predicted_positive_rate.le(.25).to_numpy()
 valid=np.flatnonzero(constraint)
 if len(valid)==0:return None,metric
 vals=tab[metric].to_numpy();best=np.nanmax(vals[valid]);ties=valid[np.isclose(vals[valid],best,rtol=0,atol=1e-15)]
 # thresholds are descending; last tie favors recall, matching the current policy.
 return tab.iloc[int(ties[-1])],metric

def rows(seed,line,policy,y,oof,yt,pt,selected,score_name,source):
 th=float(selected.threshold);om=classification_metrics(y,oof,th);tm=classification_metrics(yt,pt,th)
 common={"seed":seed,"model_line":line,"threshold_policy":policy,"policy_status":"OK","probability_source":source}
 oo={**common,"selected_threshold":th,"selected_oof_policy_score":float(selected[score_name]),
 "oof_auc":roc_auc_score(y,oof),"oof_average_precision":average_precision_score(y,oof),
 **{f"oof_{k}":om[k] for k in ["f1","precision","recall","specificity","fpr","balanced_accuracy","predicted_positive_rate","tn","fp","fn","tp"]},
 **{f"weighted_oof_{k.removeprefix('weighted_')}":float(selected[k]) for k in ["weighted_f1","weighted_f0_5","weighted_f0_75","weighted_precision","weighted_recall","weighted_specificity","weighted_fpr","weighted_balanced_accuracy","weighted_predicted_positive_rate"]}}
 tt={**common,"selected_threshold_from_oof":th,"threshold_source":"train OOF only","test_usage":"reporting only",
 "test_auc":roc_auc_score(yt,pt),"test_average_precision":average_precision_score(yt,pt),
 **{f"test_{k}":tm[k] for k in ["f1","precision","recall","specificity","fpr","balanced_accuracy","predicted_positive_rate","tn","fp","fn","tp"]}}
 return oo,tt

def subgroup_rows(seed,line,policy,test,raw,pt,threshold):
 y=test.Label.to_numpy(int);rate=test[raw].isna().mean(axis=1);masks={"test_high_missing_gt_80":rate.gt(.80).to_numpy(),
 "test_high_missing_gt_90":rate.gt(.90).to_numpy(),"test_high_missing_gt_95":rate.gt(.95).to_numpy(),"test_all_nan":test[raw].isna().all(axis=1).to_numpy()};out=[]
 for name,mask in masks.items():
  sm=subgroup_metrics(y[mask],pt[mask],threshold);out.append({"seed":seed,"model_line":line,"threshold_policy":policy,"subgroup":name,
  "n":int(mask.sum()),"benign_count":int(((y==0)&mask).sum()),"pathogenic_count":int(((y==1)&mask).sum()),
  **{k:sm[k] for k in ["tn","fp","fn","tp","f1","precision","recall","specificity","predicted_positive_rate"]}})
 return out

def leak(seed,line,audit,train,test,features,label_match,retrained):
 overlap=len(set(train.Variant_ID.astype(str))&set(test.Variant_ID.astype(str)));hard=overlap>0 or not label_match or any(c in features for c in ["Variant_ID","Label","AA_1","AA_2"])
 identical=int(audit.get("identical_feature_hash_count",0))
 return {"seed":seed,"model_line":line,"status":"FAIL" if hard else "WARNING" if identical else "PASS","train_test_variant_id_overlap":overlap,
 "variant_id_in_features":"Variant_ID" in features,"label_in_features":"Label" in features,"raw_aa_in_features":any(c in features for c in ["AA_1","AA_2"]),
 "threshold_selected_on_test":False,"threshold_policy_selected_on_test":False,"test_rows_dropped":False,"train_rows_dropped":False,
 "preprocessing_fitted_on_test":False,"model_selected_on_test":False,"label_used_in_transform":False,"saved_probability_row_order_checked":True,
 "saved_probability_label_mismatch":not label_match,"identical_raw_feature_hash_count":identical,
 "identical_hash_all_nan_explained":audit.get("identical_hash_all_nan_explained",False),"probabilities_retrained_info":retrained,"details":audit.get("details","")}

def aggregate(test):
 out=[]
 for (line,policy),g in test[test.policy_status.eq("OK")].groupby(["model_line","threshold_policy"],sort=False):
  r={"model_line":line,"threshold_policy":policy,"n_successful_seeds":len(g)}
  for m in ["test_f1","test_precision","test_recall","test_specificity","test_fpr","test_balanced_accuracy","test_predicted_positive_rate","test_fp","test_fn"]:
   r[f"{m}_mean"]=g[m].mean();r[f"{m}_std"]=g[m].std(ddof=1)
  r.update({"test_f1_min":g.test_f1.min(),"test_f1_median":g.test_f1.median(),"test_f1_max":g.test_f1.max(),
  "selected_threshold_mean":g.selected_threshold_from_oof.mean(),"selected_threshold_std":g.selected_threshold_from_oof.std(ddof=1)});out.append(r)
 return pd.DataFrame(out)

def delta_labels(d):
 if d>=.01:return "strong threshold evidence"
 if d>=.005:return "weak positive threshold evidence"
 if d>=-.003:return "neutral / trade-off"
 return "worse unless FP reduction is required"

def main():
 wall=time.perf_counter();full=pd.read_csv(find_input());full["Label"]=full.Label.astype(int);raw=[c for c in full if c not in ["Variant_ID","Label"]]
 al=[c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])];dup=int(full.duplicated().sum());splits={};audits={};sr=[]
 for seed in SEEDS:
  tr,te=make_split(full,seed);splits[seed]=(tr,te);audits[seed]=leakage_audit(full,tr,te,raw,seed,dup)[0]
  sr.append({"seed":seed,"train_n":len(tr),"test_n":len(te),"train_benign":int(tr.Label.eq(0).sum()),"train_pathogenic":int(tr.Label.eq(1).sum()),"test_benign":int(te.Label.eq(0).sum()),"test_pathogenic":int(te.Label.eq(1).sum())})
 state=load_state();done={(int(r["seed"]),r["model_line"],r["threshold_policy"]) for r in state["test"]}
 leak_done={(int(r["seed"]),r["model_line"]) for r in state["leakage"]}
 for seed in SEEDS:
  for line in LINES:
   pending=[p for p in POLICIES if (seed,line,p) not in done]
   if not pending:continue
   train,test=splits[seed];train_start=time.perf_counter()
   if line.startswith("keep_all"):
    y,oof,yt,pt=step6_probabilities(seed);features=transform_config(train,test,raw,al,"aa_numeric")[0].columns.tolist();source="REUSED_STEP6_EXACT";train_seconds=0.;retrained=False
   else:
    y,oof,yt,pt,features,oof_s,final_s=train_missingness(seed,train,test,raw,al);source="RETRAINED_EXACT_STEP6_FACTORY";train_seconds=oof_s+final_s;retrained=True
   label_match=np.array_equal(y,train.Label.to_numpy(int)) and np.array_equal(yt,test.Label.to_numpy(int))
   if not label_match:raise RuntimeError(f"saved probability label/order mismatch seed={seed} line={line}")
   if (seed,line) not in leak_done:state["leakage"].append(leak(seed,line,audits[seed],train,test,features,label_match,retrained));leak_done.add((seed,line))
   tab=threshold_table(y,oof)
   for i,policy in enumerate(pending):
    start=time.perf_counter();sel,metric=choose(tab,policy)
    if sel is None:
     common={"seed":seed,"model_line":line,"threshold_policy":policy,"policy_status":"SKIPPED: no feasible OOF threshold"};state["oof"].append(common);state["test"].append(common);state["subgroup"].append(common)
    else:
     oo,tt=rows(seed,line,policy,y,oof,yt,pt,sel,metric,source);state["oof"].append(oo);state["test"].append(tt);state["subgroup"].extend(subgroup_rows(seed,line,policy,test,raw,pt,float(sel.threshold)))
    state["runtime"].append({"seed":seed,"model_line":line,"threshold_policy":policy,"probability_source":source,
    "model_probability_seconds":train_seconds if i==0 else 0.,"threshold_policy_seconds":time.perf_counter()-start,"status":"COMPLETED","checkpoint_schema":SCHEMA})
    save_state(state);done.add((seed,line,policy))
   print(f"STEP8 seed={seed} line={line} policies={len(pending)} source={source} minutes={(time.perf_counter()-train_start)/60:.2f}",flush=True)
 oof=pd.DataFrame(state["oof"]);test=pd.DataFrame(state["test"]);sub=pd.DataFrame(state["subgroup"]);leakdf=pd.DataFrame(state["leakage"]);runtime=pd.DataFrame(state["runtime"]);summary=aggregate(test)
 current=summary[summary.threshold_policy.eq("weighted_f1_current")].set_index("model_line");dcur=[]
 for r in summary.itertuples():
  ref=current.loc[r.model_line];row={"model_line":r.model_line,"threshold_policy":r.threshold_policy}
  for m in ["f1","precision","recall","specificity","fpr","balanced_accuracy","predicted_positive_rate","fp","fn"]:row[f"delta_{m}_mean"]=getattr(r,f"test_{m}_mean")-ref[f"test_{m}_mean"]
  row["f1_decision"]=delta_labels(row["delta_f1_mean"]);row["fp_reduction_meaningful"]=row["delta_fp_mean"]<=-5;row["fn_risk_high"]=row["delta_fn_mean"]>3;row["recall_drop_high"]=row["delta_recall_mean"]<-.04;dcur.append(row)
 dcur=pd.DataFrame(dcur)
 def vsref(ref,name):
  out=[]
  for r in summary.itertuples():
   row={"model_line":r.model_line,"threshold_policy":r.threshold_policy,"reference":name}
   for m in ["f1","precision","recall","fp","fn"]:row[f"delta_{m}_mean"]=getattr(r,f"test_{m}_mean")-ref[m]
   out.append(row)
  return pd.DataFrame(out)
 vs6=vsref(REF6,"Step 6 CatBoost keep_all");vs7=vsref(REF7,"Step 7B add_missingness current")
 keep=summary[summary.model_line.str.startswith("keep_all")].set_index("threshold_policy");add=summary[summary.model_line.str.startswith("add_missingness")].set_index("threshold_policy");line_rows=[]
 for p in POLICIES:
  if p not in keep.index or p not in add.index:continue
  row={"threshold_policy":p}
  for m in ["f1","precision","recall","specificity","fpr","balanced_accuracy","predicted_positive_rate","fp","fn"]:row[f"add_minus_keep_{m}_mean"]=add.loc[p,f"test_{m}_mean"]-keep.loc[p,f"test_{m}_mean"]
  line_rows.append(row)
 line_delta=pd.DataFrame(line_rows)
 sg=sub[sub.get("subgroup",pd.Series(dtype=str)).notna()].groupby(["model_line","threshold_policy","subgroup"],as_index=False)[["n","fp","fn","f1","precision","recall","specificity","predicted_positive_rate"]].mean()
 # Add same-line/current subgroup risk assessment to delta table.
 risk=[]
 for r in dcur.itertuples():
  cur=sg[(sg.model_line.eq(r.model_line))&(sg.threshold_policy.eq("weighted_f1_current"))&(sg.subgroup.eq("test_high_missing_gt_90"))]
  alt=sg[(sg.model_line.eq(r.model_line))&(sg.threshold_policy.eq(r.threshold_policy))&(sg.subgroup.eq("test_high_missing_gt_90"))]
  flag=False
  if len(cur) and len(alt) and r.delta_f1_mean>0:
   c,a=cur.iloc[0],alt.iloc[0];flag=bool(a.fp>c.fp or a.fn>c.fn+1 or a.f1<c.f1)
  risk.append(flag)
 dcur["global_gain_but_high_missing_subgroup_risk"]=risk
 if any(risk):
  risky=set(zip(dcur.loc[dcur.global_gain_but_high_missing_subgroup_risk,"model_line"],dcur.loc[dcur.global_gain_but_high_missing_subgroup_risk,"threshold_policy"]))
  # leakage is line-level; record whether any exploratory policy on the line has subgroup risk.
  for line in {x[0] for x in risky}:leakdf.loc[leakdf.model_line.eq(line),"line_has_high_missing_policy_warning"]=True;leakdf.loc[leakdf.model_line.eq(line),"status"]="WARNING"
 ranking=summary.sort_values(["test_f1_mean","test_f1_std","test_fp_mean"],ascending=[False,True,True]).reset_index(drop=True);ranking.insert(0,"reporting_rank",np.arange(1,len(ranking)+1));ranking["test_usage"]="reporting only; not policy selection"
 pd.DataFrame(sr).to_csv(OUTPUTS["split"],index=False);oof.to_csv(OUTPUTS["oof"],index=False);test.to_csv(OUTPUTS["test"],index=False);summary.to_csv(OUTPUTS["summary"],index=False)
 dcur.to_csv(OUTPUTS["current"],index=False);vs6.to_csv(OUTPUTS["vs6"],index=False);vs7.to_csv(OUTPUTS["vs7"],index=False);line_delta.to_csv(OUTPUTS["lines"],index=False)
 sub.to_csv(OUTPUTS["subgroup"],index=False);leakdf.to_csv(OUTPUTS["leakage"],index=False);runtime.to_csv(OUTPUTS["runtime"],index=False);ranking.to_csv(OUTPUTS["ranking"],index=False)
 best=ranking.iloc[0];lowest=summary.sort_values(["test_fp_mean","test_f1_mean"],ascending=[True,False]).iloc[0]
 balanced=summary.sort_values(["test_balanced_accuracy_mean","test_f1_mean"],ascending=[False,False]).iloc[0]
 bestd=dcur[(dcur.model_line.eq(best.model_line))&(dcur.threshold_policy.eq(best.threshold_policy))].iloc[0]
 add_better=line_delta.set_index("threshold_policy").loc["weighted_f1_current","add_minus_keep_f1_mean"]>0
 allnan=sg[sg.subgroup.eq("test_all_nan")];allnan_change=allnan.groupby(["model_line","threshold_policy"])[["fp","fn","f1"]].mean().nunique().max()>1
 # Governance: test is reporting-only, so do not promote an exploratory policy from this ranking.
 recommendation="weighted_f1_current (retain; exploratory alternatives need independent confirmation)"
 params=make_model(7).get_params();used={k:params.get(k) for k in ["iterations","depth","learning_rate","loss_function","eval_metric","allow_writing_files","thread_count"]};used["random_seed"]="seed-specific"
 metadata={"project_folder":"TeknoPatho","step":"Step 8 threshold policy refinement","seed_list":SEEDS,"model_lines":LINES,"feature_set":"aa_numeric","model":"CatBoostClassifier","catboost_params_used":used,
 "threshold_policies":POLICIES,"threshold_candidate_source":"unique train OOF probabilities","model_training":True,"model_training_detail":"keep_all reused; add_missingness retrained because row probabilities absent",
 "feature_engineering_changed":False,"column_filtering":False,"empty_feature_drop":False,"train_row_drop":False,"test_row_drop":False,"median_imputation":False,"scaling":False,"ensemble":False,
 "threshold_selection_source":"train_oof_only","test_usage":"reporting_only","no_single_seed_selection":True}
 OUTPUTS["metadata"].write_text(json.dumps(metadata,indent=2),encoding="utf8")
 report=f"""# MASTER Step 8 threshold policy refinement

## Protocol

Step 7B confirmed deterministic missingness features under the exact Step 6 factory. Step 8 freezes both model lines and changes only threshold policy. Keep-all probabilities are reused exactly from Step 6. Step 7B did not save row-level add-missingness probabilities, so that line was retrained with the exact factory; no tuning occurred. Threshold candidates are **unique train-OOF probabilities**. Test is reporting-only and was never used to choose a threshold value or policy.

No model/feature changes, AL-log, AA variation, median, scaling, row/column filtering, clipping, ensemble, SHAP, class weighting, or resampling occurred.

Policies: `{POLICIES}`.

- `weighted_f1_current`: final-like weighted F1 reference.
- `weighted_f0_5_precision_priority`: stronger precision/FP priority.
- `weighted_f0_75_mild_precision_priority`: mild precision priority.
- `weighted_f1_with_min_recall_0_70/0_72`: weighted F1 with recall floor.
- `weighted_f1_with_min_specificity_0_80/0_82`: weighted F1 with specificity floor.
- `weighted_balanced_accuracy`: balances recall and specificity.
- `fp_limited_policy_oof_fpr_max_0_16/0_18`: weighted F1 under OOF FPR cap.
- `predicted_positive_rate_cap_0_20/0_25`: weighted F1 under positive-rate cap.

## Aggregate reporting results

{markdown_table(summary)}

## Delta versus same-line current policy

{markdown_table(dcur)}

## Delta versus Step 6 and Step 7B references

{markdown_table(vs6)}

{markdown_table(vs7)}

## Add-missingness minus keep-all by policy

{markdown_table(line_delta)}

## High-missing / all-NaN analysis

{markdown_table(sg)}

All-NaN subgroup changed across policies/lines: **{allnan_change}**. Persistent all-NaN pathogenic FN should be read from the subgroup table; threshold-only changes cannot create missing information.

## Risk, ranking, and recommendation

- Leakage FAIL={leakdf.status.eq('FAIL').any()}; WARNING={leakdf.status.eq('WARNING').any()}; statuses={leakdf.status.value_counts().to_dict()}.
- Reporting-only best F1={best.model_line}/{best.threshold_policy}: {best.test_f1_mean:.6f} ± {best.test_f1_std:.6f}; same-line current delta={bestd.delta_f1_mean:+.6f}; FP/FN delta={bestd.delta_fp_mean:+.2f}/{bestd.delta_fn_mean:+.2f}; recall delta={bestd.delta_recall_mean:+.6f}.
- Lowest-FP reporting policy={lowest.model_line}/{lowest.threshold_policy}: FP={lowest.test_fp_mean:.2f}, F1={lowest.test_f1_mean:.6f}.
- Highest balanced-accuracy reporting policy={balanced.model_line}/{balanced.threshold_policy}: BA={balanced.test_balanced_accuracy_mean:.6f}.
- Add-missingness remains better than keep-all under current policy={add_better}.
- Any high-missing subgroup-risk flag={dcur.global_gain_but_high_missing_subgroup_risk.any()}.
- Main recommendation: **{recommendation}**. This is governance-preserving 5-seed threshold evidence, not a final model or test-selected promotion.

## Next step

If an FP-control trade-off is operationally desired, pre-register one candidate from OOF behavior and confirm on new seeds/locked evaluation. Otherwise retain weighted F1 current and proceed to the separately scoped column-filtering study.
"""
 OUTPUTS["report"].write_text(report,encoding="utf8")
 print("\n=== Step 8 final ===");print(f"Successful seed/line/policy: {test.policy_status.eq('OK').sum()}/{len(SEEDS)*len(LINES)*len(POLICIES)}")
 print(f"Leakage FAIL/WARNING: {leakdf.status.eq('FAIL').any()}/{leakdf.status.eq('WARNING').any()}")
 print(f"Best reporting F1: {best.model_line}/{best.threshold_policy} {best.test_f1_mean:.6f}±{best.test_f1_std:.6f}; delta={bestd.delta_f1_mean:+.6f}")
 print(f"FP/FN/recall delta: {bestd.delta_fp_mean:+.2f}/{bestd.delta_fn_mean:+.2f}/{bestd.delta_recall_mean:+.6f}")
 print(f"Lowest FP: {lowest.model_line}/{lowest.threshold_policy} FP={lowest.test_fp_mean:.2f}")
 print(f"Balanced: {balanced.model_line}/{balanced.threshold_policy} BA={balanced.test_balanced_accuracy_mean:.6f}")
 print(f"Add missingness better under current? {add_better}");print(f">.90 subgroup risk? {dcur.global_gain_but_high_missing_subgroup_risk.any()}");print(f"all-NaN changed? {allnan_change}")
 print(f"Recommendation: {recommendation}");print(f"Runtime minutes: {(time.perf_counter()-wall)/60:.2f}")

if __name__=="__main__":main()

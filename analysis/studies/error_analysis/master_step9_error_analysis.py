"""MASTER Step 9: frozen-line FP/FN and subgroup descriptive error analysis."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd

from master_step3b_repeated_split_catboost import find_input,leakage_audit,make_split
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step7b_missingness_confirmation import step6_probabilities,train_missingness
from master_step8_threshold_policy_refinement import threshold_table,choose
from master_v02_catboost_threshold_stability import classification_metrics,markdown_table

ROOT=Path(__file__).resolve().parent;SEEDS=[7,21,42,101,2024]
PRIMARY="add_missingness_features_native_missing_step6_params";CONTROL="keep_all_native_missing_step6_params"
CURRENT="weighted_f1_current";WATCH="weighted_f0_75_mild_precision_priority";PARTIAL=ROOT/".master_step9_partial_row_predictions.csv"
FILES=["row_level_predictions","primary_error_rows","variant_error_frequency","top_persistent_false_positives","top_persistent_false_negatives",
"top_high_confidence_false_positives","top_high_confidence_false_negatives","near_threshold_errors","subgroup_metrics","all_nan_analysis",
"missingness_effect_control_vs_primary","fixed_fp_by_missingness","introduced_fn_by_missingness","aa_pair_error_summary","aa_numeric_bin_analysis",
"feature_group_error_summary","feature_shift_fp_vs_tn","feature_shift_fn_vs_tp","threshold_watchlist_tradeoff","leakage_audit","runtime_report"]
OUT={k:ROOT/f"master_step9_{k}.csv" for k in FILES};OUT.update({"report":ROOT/"master_step9_report.md","metadata":ROOT/"master_step9_metadata.json"})

def err(y,p):return np.where(y==0,np.where(p==0,"TN","FP"),np.where(p==0,"FN","TP"))
def confidence(a):return pd.cut(a,[-np.inf,.02,.05,.15,np.inf],labels=["very_low_margin","low_margin","medium_margin","high_margin"],right=False).astype(str)
def findcol(cols,terms,exclude=()):
 for c in cols:
  z=c.lower()
  if all(t.lower() in z for t in terms) and not any(t.lower() in z for t in exclude):return c
 return None
def aa_map(cols):
 return {"blosum62_score":findcol(cols,["blosum"]),"grantham_distance":findcol(cols,["grantham"]),
 "hydrophobicity_before":findcol(cols,["hydrophobicity","_1"]),"hydrophobicity_after":findcol(cols,["hydrophobicity","_2"]),
 "hydrophobicity_delta":findcol(cols,["hydrophobicity","diff"],["absdiff"]),"molecular_weight_delta":findcol(cols,["molecular_weight","diff"],["absdiff"]),
 "charge_delta":findcol(cols,["charge","diff"]),"polarity_delta":findcol(cols,["polarity","diff"],["absdiff"])}

def context(frame,raw,x,aa):
 m=missingness_features(frame,raw);out=pd.DataFrame({"Variant_ID":frame.Variant_ID.astype(str).to_numpy(),"true_label":frame.Label.to_numpy(int)})
 for c in ["row_missing_count","row_missing_rate","high_missing_80_flag","high_missing_90_flag","high_missing_95_flag","all_nan_flag","AL_missing_rate","CAT_missing_rate","EK_missing_rate","AA_missing_flag"]:out["raw_missing_count" if c=="row_missing_count" else ("raw_missing_rate" if c=="row_missing_rate" else c)]=m[c].to_numpy()
 out["AA_1_raw"]=frame.AA_1.astype("string").fillna("<NA>").to_numpy();out["AA_2_raw"]=frame.AA_2.astype("string").fillna("<NA>").to_numpy();out["AA_pair"]=out.AA_1_raw+">"+out.AA_2_raw
 for target,source in aa.items():out[target]=pd.to_numeric(x[source],errors="coerce").to_numpy() if source else np.nan
 return out

def row_block(seed,role,line,policy,ctx,prob,threshold):
 pred=(prob>=threshold).astype(int);margin=prob-threshold;out=ctx.copy();out.insert(0,"seed",seed);out.insert(1,"split_role",role);out.insert(2,"model_line",line);out.insert(3,"threshold_policy",policy)
 out["pred_proba_pathogenic"]=prob;out["selected_threshold_from_oof"]=threshold;out["pred_label"]=pred;out["error_type"]=err(out.true_label.to_numpy(),pred);out["probability_margin"]=margin;out["abs_margin"]=np.abs(margin);out["confidence_bin"]=confidence(np.abs(margin));return out

def produce_seed(seed,train,test,raw,al):
 x,xt,features,status=transform_config(train,test,raw,al,"aa_numeric")
 if status.startswith("SKIPPED"):raise RuntimeError(status)
 amap=aa_map(x.columns.tolist());ctr=context(train,raw,x.reset_index(drop=True),amap);cte=context(test,raw,xt.reset_index(drop=True),amap)
 cy,coof,cyt,cpt=step6_probabilities(seed);py,poof,pyt,ppt,_,_,_=train_missingness(seed,train,test,raw,al)
 if not(np.array_equal(cy,py) and np.array_equal(cyt,pyt) and np.array_equal(py,train.Label.to_numpy(int)) and np.array_equal(pyt,test.Label.to_numpy(int))):raise RuntimeError("prediction label/order mismatch")
 cth=float(choose(threshold_table(cy,coof),CURRENT)[0].threshold);pth=float(choose(threshold_table(py,poof),CURRENT)[0].threshold);wth=float(choose(threshold_table(py,poof),WATCH)[0].threshold)
 blocks=[]
 for role,ctx,cprob,pprob in [("train_oof",ctr,coof,poof),("test_reporting",cte,cpt,ppt)]:
  blocks.extend([row_block(seed,role,CONTROL,CURRENT,ctx,cprob,cth),row_block(seed,role,PRIMARY,CURRENT,ctx,pprob,pth),row_block(seed,role,PRIMARY,WATCH,ctx,pprob,wth)])
 return pd.concat(blocks,ignore_index=True),features,amap

def subgroup_table(primary):
 masks={"all_rows":lambda d:np.ones(len(d),bool),"high_missing_gt_80":lambda d:d.raw_missing_rate.gt(.80),"high_missing_gt_90":lambda d:d.raw_missing_rate.gt(.90),
 "high_missing_gt_95":lambda d:d.raw_missing_rate.gt(.95),"all_nan":lambda d:d.all_nan_flag.eq(1),"aa_missing":lambda d:d.AA_missing_flag.eq(1),
 "non_aa_missing":lambda d:d.AA_missing_flag.eq(0),"AL_missing_rate_gt_80":lambda d:d.AL_missing_rate.gt(.80),"CAT_missing_rate_gt_80":lambda d:d.CAT_missing_rate.gt(.80),"EK_missing_rate_gt_80":lambda d:d.EK_missing_rate.gt(.80)}
 rows=[]
 for role,d in primary.groupby("split_role"):
  for name,fun in masks.items():
   g=d[fun(d)];y=g.true_label.to_numpy(int);p=g.pred_label.to_numpy(int);tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tp=int(((y==1)&(p==1)).sum())
   precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);specificity=tn/max(tn+fp,1);f1=2*precision*recall/max(precision+recall,1e-15)
   rows.append({"split_role":role,"subgroup":name,"n":len(g),"benign_count":int((y==0).sum()),"pathogenic_count":int((y==1).sum()),"tn":tn,"fp":fp,"fn":fn,"tp":tp,"f1":f1,"precision":precision,"recall":recall,"specificity":specificity,"fpr":fp/max(fp+tn,1),"predicted_positive_rate":p.mean() if len(p) else np.nan,"mean_probability":g.pred_proba_pathogenic.mean(),"mean_margin":g.probability_margin.mean(),"small_n_warning":len(g)<20})
 return pd.DataFrame(rows)

def variant_frequency(primary):
 rows=[]
 for vid,g in primary.groupby("Variant_ID"):
  lab=int(g.true_label.mode().iloc[0]);errors=g.error_type.isin(["FP","FN"])
  fp=int(g.error_type.eq("FP").sum());fn=int(g.error_type.eq("FN").sum());n0=int(g.true_label.eq(0).sum());n1=int(g.true_label.eq(1).sum())
  rows.append({"Variant_ID":vid,"true_label":lab,"n_test_exposures":int(g.split_role.eq("test_reporting").sum()),"n_train_oof_exposures":int(g.split_role.eq("train_oof").sum()),"n_total_exposures":len(g),"n_fp":fp,"n_fn":fn,"n_tp":int(g.error_type.eq("TP").sum()),"n_tn":int(g.error_type.eq("TN").sum()),"error_rate":errors.mean(),"fp_rate_when_label_0":fp/max(n0,1) if lab==0 else np.nan,"fn_rate_when_label_1":fn/max(n1,1) if lab==1 else np.nan,"mean_pred_proba":g.pred_proba_pathogenic.mean(),"std_pred_proba":g.pred_proba_pathogenic.std(),"mean_margin":g.probability_margin.mean(),"std_margin":g.probability_margin.std(),"persistent_error_flag":errors.mean()>=.60 and len(g)>=3,"persistent_fp_flag":lab==0 and fp/max(n0,1)>=.60,"persistent_fn_flag":lab==1 and fn/max(n1,1)>=.60})
 return pd.DataFrame(rows)

def effect_table(rows):
 c=rows[(rows.model_line.eq(CONTROL))&(rows.threshold_policy.eq(CURRENT))];p=rows[(rows.model_line.eq(PRIMARY))&(rows.threshold_policy.eq(CURRENT))]
 keys=["seed","split_role","Variant_ID"];cols=keys+["true_label","pred_label","error_type","pred_proba_pathogenic"]
 z=c[cols].merge(p[cols+[
 "raw_missing_rate","AL_missing_rate","CAT_missing_rate","EK_missing_rate","all_nan_flag","high_missing_90_flag","AA_pair"]],on=keys,suffixes=("_control","_primary"),validate="one_to_one")
 z["proba_delta"]=z.pred_proba_pathogenic_primary-z.pred_proba_pathogenic_control
 conditions=[z.error_type_control.eq("FP")&z.error_type_primary.eq("TN"),z.error_type_control.eq("TN")&z.error_type_primary.eq("FP"),z.error_type_control.eq("FN")&z.error_type_primary.eq("TP"),z.error_type_control.eq("TP")&z.error_type_primary.eq("FN")]
 z["change_group"]=np.select(conditions,["fixed_fp_by_missingness","introduced_fp_by_missingness","fixed_fn_by_missingness","introduced_fn_by_missingness"],default="unchanged_or_other");return z

def aa_summary(primary):
 rows=[]
 for (role,pair),g in primary.groupby(["split_role","AA_pair"],dropna=False):
  counts=g.error_type.value_counts();n=len(g);errors=int(counts.get("FP",0)+counts.get("FN",0));tp=int(counts.get("TP",0));fp=int(counts.get("FP",0));fn=int(counts.get("FN",0));precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
  rows.append({"split_role":role,"AA_pair":pair,"n":n,"benign_count":int(g.true_label.eq(0).sum()),"pathogenic_count":int(g.true_label.eq(1).sum()),"fp":fp,"fn":fn,"tp":tp,"tn":int(counts.get("TN",0)),"error_rate":errors/n,"f1":2*precision*recall/max(precision+recall,1e-15),"mean_BLOSUM62":g.blosum62_score.mean(),"mean_Grantham":g.grantham_distance.mean(),"mean_hydrophobicity_delta":g.hydrophobicity_delta.mean(),"high_confidence_error_count":int((g.error_type.isin(["FP","FN"])&g.confidence_bin.eq("high_margin")).sum()),"support":"low_support" if n<3 else "supported"})
 return pd.DataFrame(rows)

def bin_analysis(primary):
 specs={"grantham_distance":([-np.inf,0,50,100,150,np.inf],["missing_or_negative","0-50","50-100","100-150",">150"]),
 "blosum62_score":([-np.inf,-2.5,-.5,.5,2.5,np.inf],["<=-3","-2 to -1","0","1 to 2",">=3"]),
 "hydrophobicity_delta_abs":([-np.inf,1,3,np.inf],["0-1","1-3",">3"])};d=primary.copy();d["hydrophobicity_delta_abs"]=d.hydrophobicity_delta.abs();rows=[]
 for feat,(bins,labels) in specs.items():
  vals=d[feat];cats=pd.cut(vals,bins=bins,labels=labels,include_lowest=True);cats=cats.astype("object").where(vals.notna(),"missing")
  for (role,b),g in d.assign(_bin=cats).groupby(["split_role","_bin"]):
   y=g.true_label.to_numpy();p=g.pred_label.to_numpy();tp=((y==1)&(p==1)).sum();fp=((y==0)&(p==1)).sum();fn=((y==1)&(p==0)).sum();pr=tp/max(tp+fp,1);re=tp/max(tp+fn,1)
   rows.append({"feature":feat,"split_role":role,"bin":b,"n":len(g),"benign_count":int((y==0).sum()),"pathogenic_count":int((y==1).sum()),"fp":fp,"fn":fn,"f1":2*pr*re/max(pr+re,1e-15),"predicted_positive_rate":p.mean()})
 return pd.DataFrame(rows)

def feature_descriptives(primary,feature_frames):
 # feature_frames holds primary aa_numeric+missingness matrices aligned to primary row order by seed/role.
 rows=[];fp_rows=[];fn_rows=[]
 for role,d in primary.groupby("split_role",sort=False):
  X=pd.concat([feature_frames[(s,role)] for s in SEEDS],ignore_index=True);d=d.reset_index(drop=True)
  num=X.select_dtypes(include=[np.number,"bool"]);groups={"AA_derived":[c for c in num if any(k in c.lower() for k in ["aa1_","aa2_","delta_","blosum","grantham","same_group","changed_group"])],"AL":[c for c in num if c.startswith("AL_")],"CAT":[c for c in num if c.startswith("CAT_")],"EK":[c for c in num if c.startswith("EK_")],"missingness":[c for c in num if c in ["row_missing_count","row_missing_rate","row_nonmissing_count","high_missing_80_flag","high_missing_90_flag","high_missing_95_flag","all_nan_flag","AL_missing_count","AL_missing_rate","CAT_missing_count","CAT_missing_rate","EK_missing_count","EK_missing_rate","AA_missing_flag"]]}
  for name,cols in groups.items():
   if not cols:continue
   miss=X[cols].isna().mean(axis=1);mean=num[cols].mean(axis=1);std=num[cols].std(axis=1)
   for e in ["FP","FN","TN","TP"]:
    mask=d.error_type.eq(e).to_numpy();rows.append({"split_role":role,"feature_group":name,"error_type":e,"n":int(mask.sum()),"group_missing_rate_mean":miss[mask].mean(),"group_value_mean_of_numeric_features":mean[mask].mean(),"group_value_std_of_numeric_features":std[mask].mean()})
  for feature in num:
   vals=pd.to_numeric(num[feature],errors="coerce")
   for a,b,target in [("FP","TN",fp_rows),("FN","TP",fn_rows)]:
    va=vals[d.error_type.eq(a).to_numpy()];vb=vals[d.error_type.eq(b).to_numpy()];pooled=np.sqrt((va.var()+vb.var())/2);smd=(va.mean()-vb.mean())/pooled if pooled and np.isfinite(pooled) else np.nan
    target.append({"split_role":role,"feature":feature,"comparison":f"{a}_vs_{b}","standardized_mean_difference":smd,"error_mean":va.mean(),"correct_mean":vb.mean(),"error_missing_rate":va.isna().mean(),"correct_missing_rate":vb.isna().mean(),"missing_rate_difference":va.isna().mean()-vb.isna().mean(),"descriptive_only":True})
 return pd.DataFrame(rows),pd.DataFrame(fp_rows),pd.DataFrame(fn_rows)

def main():
 wall=time.perf_counter();full=pd.read_csv(find_input());full["Label"]=full.Label.astype(int);raw=[c for c in full if c not in ["Variant_ID","Label"]];al=[c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])];dup=int(full.duplicated().sum())
 existing=pd.read_csv(PARTIAL) if PARTIAL.exists() else pd.DataFrame();done=set(existing.seed.unique()) if len(existing) else set();blocks=[existing] if len(existing) else [];runtime=pd.read_csv(OUT["runtime_report"]).to_dict("records") if OUT["runtime_report"].exists() else [];leaks=[];features_by_seed={};aa_found={}
 for seed in SEEDS:
  train,test=make_split(full,seed);audit=leakage_audit(full,train,test,raw,seed,dup)[0];x,xt,features,_=transform_config(train,test,raw,al,"aa_numeric");features_by_seed[seed]=(x.reset_index(drop=True),xt.reset_index(drop=True));aa_found=aa_map(x.columns.tolist())
  overlap=len(set(train.Variant_ID.astype(str))&set(test.Variant_ID.astype(str)));hard=overlap>0 or any(c in features for c in ["Variant_ID","Label","AA_1","AA_2"])
  for line in [CONTROL,PRIMARY]:leaks.append({"seed":seed,"model_line":line,"status":"FAIL" if hard else "WARNING" if audit.get("identical_feature_hash_count",0)>0 else "PASS","train_test_variant_id_overlap":overlap,"variant_id_in_features":"Variant_ID" in features,"label_in_features":"Label" in features,"raw_aa_in_features":any(c in features for c in ["AA_1","AA_2"]),"threshold_selected_on_test":False,"threshold_policy_selected_on_test":False,"test_rows_dropped":False,"train_rows_dropped":False,"preprocessing_fitted_on_test":False,"model_selected_on_test":False,"label_used_in_transform":False,"saved_probability_label_mismatch":False,"identical_raw_feature_hash_count":audit.get("identical_feature_hash_count",0),"identical_hash_all_nan_explained":audit.get("identical_hash_all_nan_explained",False)})
  if seed in done:continue
  start=time.perf_counter();b,_,_=produce_seed(seed,train,test,raw,al);blocks.append(b);pd.concat(blocks,ignore_index=True).to_csv(PARTIAL,index=False);runtime.append({"seed":seed,"action":"produce primary/control/watchlist row predictions","runtime_seconds":time.perf_counter()-start,"status":"COMPLETED"});print(f"STEP9 seed={seed} rows={len(b)} minutes={(time.perf_counter()-start)/60:.2f}",flush=True)
 rows=pd.concat(blocks,ignore_index=True)
 # Refresh deterministic AA-derived columns even when an older prediction partial is reused.
 full_x=transform_config(full,full,raw,al,"aa_numeric")[0].reset_index(drop=True);aa_found=aa_map(full_x.columns.tolist())
 for target,source in aa_found.items():
  if source:
   lookup=pd.Series(pd.to_numeric(full_x[source],errors="coerce").to_numpy(),index=full.Variant_ID.astype(str))
   rows[target]=rows.Variant_ID.astype(str).map(lookup)
 rows.to_csv(OUT["row_level_predictions"],index=False)
 primary=rows[(rows.model_line.eq(PRIMARY))&(rows.threshold_policy.eq(CURRENT))].copy();errors=primary[primary.error_type.isin(["FP","FN"])].copy();errors.to_csv(OUT["primary_error_rows"],index=False)
 vf=variant_frequency(primary);vf.to_csv(OUT["variant_error_frequency"],index=False);vf[vf.persistent_fp_flag].sort_values(["fp_rate_when_label_0","n_fp"],ascending=False).to_csv(OUT["top_persistent_false_positives"],index=False);vf[vf.persistent_fn_flag].sort_values(["fn_rate_when_label_1","n_fn"],ascending=False).to_csv(OUT["top_persistent_false_negatives"],index=False)
 errors[(errors.error_type.eq("FP"))&(errors.confidence_bin.eq("high_margin"))].sort_values("abs_margin",ascending=False).to_csv(OUT["top_high_confidence_false_positives"],index=False);errors[(errors.error_type.eq("FN"))&(errors.confidence_bin.eq("high_margin"))].sort_values("abs_margin",ascending=False).to_csv(OUT["top_high_confidence_false_negatives"],index=False);errors[errors.abs_margin.lt(.05)].sort_values("abs_margin").to_csv(OUT["near_threshold_errors"],index=False)
 sg=subgroup_table(primary);sg.to_csv(OUT["subgroup_metrics"],index=False)
 an_seed=primary[primary.all_nan_flag.eq(1)].groupby(["seed","split_role"]).agg(all_nan_n=("Variant_ID","size"),all_nan_benign_count=("true_label",lambda x:(x==0).sum()),all_nan_pathogenic_count=("true_label",lambda x:(x==1).sum()),all_nan_FP=("error_type",lambda x:(x=="FP").sum()),all_nan_FN=("error_type",lambda x:(x=="FN").sum()),all_nan_TN=("error_type",lambda x:(x=="TN").sum()),all_nan_TP=("error_type",lambda x:(x=="TP").sum()),all_nan_mean_proba=("pred_proba_pathogenic","mean"),all_nan_threshold=("selected_threshold_from_oof","mean"),all_nan_predicted_positive_rate=("pred_label","mean")).reset_index();an_var=primary[primary.all_nan_flag.eq(1)].groupby(["Variant_ID","true_label"]).agg(exposure_count=("seed","size"),n_fn=("error_type",lambda x:(x=="FN").sum()),n_fp=("error_type",lambda x:(x=="FP").sum()),mean_pred_proba=("pred_proba_pathogenic","mean"),mean_margin=("probability_margin","mean")).reset_index();an_seed["record_type"]="seed_split_summary";an_var["record_type"]="variant_summary";pd.concat([an_seed,an_var],ignore_index=True,sort=False).to_csv(OUT["all_nan_analysis"],index=False)
 effect=effect_table(rows)
 effect["group_missing_pattern"]="AL>80="+effect.AL_missing_rate.gt(.8).astype(int).astype(str)+"|CAT>80="+effect.CAT_missing_rate.gt(.8).astype(int).astype(str)+"|EK>80="+effect.EK_missing_rate.gt(.8).astype(int).astype(str)+"|all_nan="+effect.all_nan_flag.astype(int).astype(str)
 effect.to_csv(OUT["missingness_effect_control_vs_primary"],index=False);effect[effect.change_group.eq("fixed_fp_by_missingness")].to_csv(OUT["fixed_fp_by_missingness"],index=False);effect[effect.change_group.eq("introduced_fn_by_missingness")].to_csv(OUT["introduced_fn_by_missingness"],index=False)
 effect_summary=[]
 for (role,change_name),g in effect[~effect.change_group.eq("unchanged_or_other")].groupby(["split_role","change_group"]):
  vids=g.Variant_ID.astype(str).tolist();source=full.set_index(full.Variant_ID.astype(str)).reindex(vids)[raw]
  top_missing=source.isna().sum().sort_values(ascending=False).head(10)
  effect_summary.append({"split_role":role,"change_group":change_name,"n":len(g),"mean_raw_missing_rate":g.raw_missing_rate.mean(),"mean_AL_missing_rate":g.AL_missing_rate.mean(),"mean_CAT_missing_rate":g.CAT_missing_rate.mean(),"mean_EK_missing_rate":g.EK_missing_rate.mean(),"all_nan_rate":g.all_nan_flag.mean(),"high_missing_90_rate":g.high_missing_90_flag.mean(),"top_AA_pairs":"|".join(g.AA_pair.value_counts().head(5).index.astype(str)),"top_raw_missing_features":"|".join(f"{k}:{int(v)}" for k,v in top_missing.items()),"top_group_missing_patterns":"|".join(g.group_missing_pattern.value_counts().head(5).index.astype(str)),"mean_proba_delta":g.proba_delta.mean()})
 effect_summary=pd.DataFrame(effect_summary)
 aa=aa_summary(primary);aa.to_csv(OUT["aa_pair_error_summary"],index=False);bins=bin_analysis(primary);bins.to_csv(OUT["aa_numeric_bin_analysis"],index=False)
 # Build aligned primary feature matrices without refitting models.
 feature_frames={}
 for seed in SEEDS:
  train,test=make_split(full,seed);x,xt=features_by_seed[seed];mx=missingness_features(train,raw);mxt=missingness_features(test,raw);feature_frames[(seed,"train_oof")]=pd.concat([x,mx],axis=1);feature_frames[(seed,"test_reporting")]=pd.concat([xt,mxt],axis=1)
 group,fp_shift,fn_shift=feature_descriptives(primary,feature_frames)
 group["difference_vs_correct_same_label"]=np.nan
 for role in group.split_role.unique():
  for fg in group.feature_group.unique():
   mask=(group.split_role.eq(role)&group.feature_group.eq(fg));lookup=group[mask].set_index("error_type")
   for e,c in [("FP","TN"),("FN","TP")]:
    ix=group.index[mask&group.error_type.eq(e)]
    if len(ix) and e in lookup.index and c in lookup.index:group.loc[ix,"difference_vs_correct_same_label"]=lookup.loc[e,"group_value_mean_of_numeric_features"]-lookup.loc[c,"group_value_mean_of_numeric_features"]
 group.to_csv(OUT["feature_group_error_summary"],index=False);fp_shift.to_csv(OUT["feature_shift_fp_vs_tn"],index=False);fn_shift.to_csv(OUT["feature_shift_fn_vs_tp"],index=False)
 watch=rows[(rows.model_line.eq(PRIMARY))&(rows.threshold_policy.eq(WATCH))];cur=primary;keys=["seed","split_role","Variant_ID"];w=cur[keys+["true_label","error_type","raw_missing_rate","high_missing_90_flag","all_nan_flag"]].merge(watch[keys+["error_type"]],on=keys,suffixes=("_current","_watch"));w["fixed_fp"]=(w.error_type_current.eq("FP")&w.error_type_watch.eq("TN"));w["introduced_fn"]=(w.error_type_current.eq("TP")&w.error_type_watch.eq("FN"));trade=w.groupby("split_role").agg(n=("Variant_ID","size"),fixed_fp=("fixed_fp","sum"),introduced_fn=("introduced_fn","sum"),fixed_fp_high_missing_90=("high_missing_90_flag",lambda x:int((x.astype(bool)&w.loc[x.index,"fixed_fp"]).sum())),introduced_fn_high_missing_90=("high_missing_90_flag",lambda x:int((x.astype(bool)&w.loc[x.index,"introduced_fn"]).sum())),fixed_fp_all_nan=("all_nan_flag",lambda x:int((x.astype(bool)&w.loc[x.index,"fixed_fp"]).sum())),introduced_fn_all_nan=("all_nan_flag",lambda x:int((x.astype(bool)&w.loc[x.index,"introduced_fn"]).sum()))).reset_index();trade["status"]="FP-control alternative requires confirmation; not primary";trade.to_csv(OUT["threshold_watchlist_tradeoff"],index=False)
 pd.DataFrame(leaks).to_csv(OUT["leakage_audit"],index=False);pd.DataFrame(runtime).to_csv(OUT["runtime_report"],index=False)
 test_primary=primary[primary.split_role.eq("test_reporting")];oof_primary=primary[primary.split_role.eq("train_oof")]
 def aggmetric(d):
  per=[]
  for seed,g in d.groupby("seed"):m=classification_metrics(g.true_label.to_numpy(),g.pred_proba_pathogenic.to_numpy(),g.selected_threshold_from_oof.iloc[0]);per.append(m)
  return pd.DataFrame(per).mean()
 tm=aggmetric(test_primary);om=aggmetric(oof_primary);persist_fp=int(vf.persistent_fp_flag.sum());persist_fn=int(vf.persistent_fn_flag.sum());hfp=int(((errors.error_type.eq("FP"))&errors.confidence_bin.eq("high_margin")).sum());hfn=int(((errors.error_type.eq("FN"))&errors.confidence_bin.eq("high_margin")).sum());allnan_fn=bool(((primary.all_nan_flag.eq(1))&(primary.true_label.eq(1))&(primary.error_type.eq("FN"))).any())
 change=effect.change_group.value_counts();fixed_fp=int(change.get("fixed_fp_by_missingness",0));introduced_fn=int(change.get("introduced_fn_by_missingness",0));risk_aa=aa[(aa.support.eq("supported"))].sort_values(["error_rate","n"],ascending=[False,False]).head(10);risk_sg=sg.sort_values(["f1","n"],ascending=[True,False]).head(10)
 metadata={"project_folder":"TeknoPatho","step":"Step 9 FP/FN + subgroup error analysis","primary_line":PRIMARY,"control_line":CONTROL,"watchlist_line":f"{PRIMARY}/{WATCH}","seed_list":SEEDS,"feature_set":"aa_numeric","model":"CatBoostClassifier","threshold_policy_primary":CURRENT,"model_training_changed":False,"threshold_policy_changed":False,"feature_engineering_changed":False,"test_usage":"reporting_only","oof_analysis_primary":True,"no_model_selection_from_test":True,"no_threshold_selection_from_test":True,"aa_fields_found":aa_found,"aa_fields_not_found":[k for k,v in aa_found.items() if v is None]};OUT["metadata"].write_text(json.dumps(metadata,indent=2),encoding="utf8")
 report=f"""# MASTER Step 9 FP/FN + subgroup error analysis

## Scope and governance

Step 8 retained weighted_f1_current because alternatives did not materially improve F1 and increased FN/recall risk. Primary is `{PRIMARY}/{CURRENT}`; control is `{CONTROL}/{CURRENT}`; `{WATCH}` is reporting-only. This is descriptive **error-analysis evidence**, not model/threshold selection. OOF drives hypotheses; test patterns are reporting-only.

Primary probabilities were regenerated with the exact frozen factory solely because prior row-level add-missingness probabilities were absent. No model, feature, or threshold policy changed. No tuning, AL-log, AA expansion, one-hot, median, scaling, row/column filtering, clipping, class weight, resampling, ensemble, SHAP, or permutation importance was used.

## OOF and test overview

- Primary OOF mean F1/FP/FN={om.f1:.6f}/{om.fp:.2f}/{om.fn:.2f}.
- Primary test mean F1/FP/FN={tm.f1:.6f}/{tm.fp:.2f}/{tm.fn:.2f}.
- Persistent FP/FN variants={persist_fp}/{persist_fn}; high-confidence FP/FN exposures={hfp}/{hfn}; near-threshold errors={len(errors[errors.abs_margin.lt(.05)])}.

## Missingness and all-NaN

- Missingness fixed FP exposures={fixed_fp}; introduced FN exposures={introduced_fn}. Full change types are in the effect table. OOF and test must be read separately because their class priors differ.
- All-NaN pathogenic FN persists={allnan_fn}. All-NaN benign/pathogenic behavior is reported by seed/split and Variant_ID; it is a missing-information limitation, not threshold evidence.

{markdown_table(sg)}

Missingness change-group profiles (including top AA pairs and raw missing columns):

{markdown_table(effect_summary)}

## AA patterns

Detected AA-derived fields: `{json.dumps(aa_found)}`. Not found: `{[k for k,v in aa_found.items() if v is None]}`.

Highest supported error-rate AA pairs (hypothesis-only):

{markdown_table(risk_aa)}

BLOSUM/Grantham/hydrophobicity bin details are in `master_step9_aa_numeric_bin_analysis.csv`. Low-support pairs are explicitly flagged and cannot support decisions.

## Feature-group descriptive patterns

{markdown_table(group)}

Feature-level standardized shifts are descriptive only; they are hypothesis sources for Step 11/12, not feature-selection evidence.

## FP-control watchlist

{markdown_table(trade)}

The watchlist remains **FP-control alternative requires confirmation** and is not promoted.

## Leakage

- FAIL={pd.DataFrame(leaks).status.eq('FAIL').any()}; WARNING={pd.DataFrame(leaks).status.eq('WARNING').any()}; statuses={pd.DataFrame(leaks).status.value_counts().to_dict()}.

## Next-step hypotheses

1. **Column filtering:** inspect large, OOF-consistent FP-vs-TN and FN-vs-TP missing-rate/value shifts; pre-register columns before any ablation.
2. **AA expansion:** prioritize supported AA pairs and BLOSUM/Grantham bins that are risky in both OOF and test; ignore low-support/test-only patterns.
3. **AL-log / AL:** test whether OOF-consistent AL missing/value shifts explain persistent FP without worsening pathogenic recall.
4. **Score-fold / stability:** persistent variants and seed-varying near-threshold errors require broader-seed confirmation; preserve reporting-only test governance.
5. **Explainability:** if later authorized, focus on AA-derived, AL, CAT, EK, and missingness groups highlighted here; this step itself used no explainability method.

Patterns found only in test must be called **test-only observation**; aligned OOF/test direction may be called a **consistent pattern** after support checks.
""";OUT["report"].write_text(report,encoding="utf8")
 print("\n=== Step 9 final ===");print(f"Leakage FAIL/WARNING: {pd.DataFrame(leaks).status.eq('FAIL').any()}/{pd.DataFrame(leaks).status.eq('WARNING').any()}");print(f"Primary test F1/FP/FN mean: {tm.f1:.6f}/{tm.fp:.2f}/{tm.fn:.2f}");print(f"Primary OOF F1/FP/FN mean: {om.f1:.6f}/{om.fp:.2f}/{om.fn:.2f}");print(f"Persistent FP/FN: {persist_fp}/{persist_fn}");print(f"High-confidence FP/FN: {hfp}/{hfn}");print(f"all-NaN pathogenic FN persists? {allnan_fn}");print(f"Missingness fixed FP/introduced FN exposures: {fixed_fp}/{introduced_fn}");print("Risky AA pairs:",risk_aa.AA_pair.head(5).tolist());print("Riskiest subgroup:",risk_sg[["split_role","subgroup"]].head(1).to_dict("records"));print("Next: pre-registered column/AA/AL ablations with broader-seed stability confirmation");print(f"Runtime minutes: {(time.perf_counter()-wall)/60:.2f}")

if __name__=="__main__":main()

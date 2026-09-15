"""Frozen P0/P1/P2 finalization comparison; research-only, no production writes.

P0 is current ES. P1 refits final models at current ES-selected counts while
keeping own OOF calibration basis. P2 uses median count across five frozen ES
splits, recomputed inside every outer/inner train partition, then full-row refit.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path[:0]=[str(HERE),str(ROOT/'pipelines')]
from calibration import fit_platt, apply_platt
from config import SEED
from feature_engineering import engineer_aa_features
from general_blend import _cat_general,_cat_model,_cat_repr,_cats,_rf_general,_rf_repr,_xgb_model
from harness import f1_at_prior,mcc_at_prior_analytic,tpr_fpr
from models import get_model_params
from predict import LGBM_PARAM_OVERRIDE,REDUN_PRUNE_LGBM,_apply_own_preprocess,_engineer,_fit_own_preprocess,_lgbm_redun_keep
from preprocess_v2 import fit_keep_nan,fit_al,apply_al
from recipe import PANEL_RECIPE
from schema import drop_all_nan_rows,normalize_sentinels,partition_columns,to_categorical
from transfer import TRANSFER_AL_FEATURES,TRANSFER_BLEND,TRANSFER_BLEND_WEIGHTS,TRANSFER_TARGETRANK,fit_target_pop_grids,project_target,transfer_score_matrix
from utils import KNOWN_LEAKY,PANELS as PANEL_PATHS,TEST_PRIORS,detect_leaky_columns,optimize_threshold_f1_balanced

POLICIES=('P0','P1','P2'); PANEL_NAMES=('general','cancer','pah','cftr')
OWN_SPLITS=(42,43,44,45,46); SOURCE_LGB_SPLITS=(49,50,51,52,53); SOURCE_CAT_SPLITS=(47,48,49,50,51); GENERAL_CAT_SPLITS=(141,142,143,144,145)
FROZEN={'general':.81,'cancer':.80}

def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashes():
 p=[Path(v) for v in PANEL_PATHS.values()]+list((ROOT/'models').rglob('artifacts.joblib'))+[HERE/'exp_finalization_compare.py',HERE/'harness.py']+list((ROOT/'pipelines').glob('*.py'))
 return {str(x.relative_to(ROOT)):h(x) for x in p if x.is_file()}
def sh(a): return hashlib.sha256(np.asarray(a,dtype=np.int64).tobytes()).hexdigest()[:16]
def raw(panel):
 d=drop_all_nan_rows(normalize_sentinels(pd.read_csv(PANEL_PATHS[panel])),verbose=False).reset_index(drop=True)
 if d.Variant_ID.duplicated().any(): raise AssertionError(f'duplicate usable ID: {panel}')
 return d
def prep_own(X,y,Xapply,cfg):
 X=X.copy()
 for c in X:
  if isinstance(X[c].dtype,pd.CategoricalDtype): X[c]=X[c].cat.remove_unused_categories()
 numeric=[c for c in X if not isinstance(X[c].dtype,pd.CategoricalDtype)]
 leak=sorted(set(detect_leaky_columns(X[numeric],y,.99,verbose=False))|(KNOWN_LEAKY&set(X)))
 fit,art=_fit_own_preprocess(X.drop(columns=leak,errors='ignore'),y,cfg)
 return fit,_apply_own_preprocess(Xapply.drop(columns=leak,errors='ignore'),art).reindex(columns=fit.columns)
def make_lgb(params,seed): return lgb.LGBMClassifier(**{**params,'random_state':seed,'n_jobs':4})
def record(rec,**kw): rec.append(kw)
def lgb_policy(X,y,params,model_seed,base_split,robust_splits,holdout,patience,rec,tag,selections=None):
 """Return P0 current, P1 same-count full refit, P2 robust-count full refit."""
 def select(split,policy):
  ii=np.arange(len(y)); a,b=train_test_split(ii,test_size=holdout,random_state=split,stratify=y)
  m=make_lgb(params,model_seed); t=time.time(); m.fit(X.iloc[a],y.iloc[a],eval_set=[(X.iloc[b],y.iloc[b])],callbacks=[lgb.early_stopping(patience,verbose=False),lgb.log_evaluation(0)])
  best=max(int(m.best_iteration_ or params['n_estimators']),1); got=int(m.booster_.num_trees())
  if got<1: raise AssertionError('LGB produced no trees')
  # best_iteration can exceed physical trees after no-split iterations. Match the
  # effective tree count actually used by the current predictor, not empty rounds.
  n=min(best,got)
  record(rec,tag=tag,kind='lgb',policy=policy,phase='select',model_seed=model_seed,split_seed=split,eligible_rows=len(y),gradient_rows=len(a),es_rows=len(b),tree_count=n,reported_best_iteration=best,observed_trees=got,structural_stop=got<best,seconds=time.time()-t)
  return m,n
 p0,n0=select(base_split,'P0'); selected={}
 for split in robust_splits:
  selected[split]=select(split,'P2')
 if selections is not None:
  selections.update({split:model for split,(model,count) in selected.items()})
 n2=int(np.median([count for model,count in selected.values()]))
 def refit(n,policy):
  t=time.time(); m=make_lgb({**params,'n_estimators':n},model_seed); m.fit(X,y); got=int(m.booster_.num_trees())
  # LightGBM can stop naturally before the requested count when no valid split
  # remains. This is structural termination, not validation-based early stopping.
  if not 1 <= got <= n: raise AssertionError(f'LGB invalid tree count: observed={got}, requested={n}')
  record(rec,tag=tag,kind='lgb',policy=policy,phase='refit',model_seed=model_seed,split_seed=base_split,eligible_rows=len(y),gradient_rows=len(y),es_rows=0,tree_count=n,observed_trees=got,structural_stop=got<n,seconds=time.time()-t)
  return m
 return p0,refit(n0,'P1'),refit(n2,'P2')
def cat_policy(X,y,prototype,base_split,robust_splits,rec,tag,selections=None):
 def fresh(**over):
  p=prototype.get_params();p.update(over);return CatBoostClassifier(**p)
 def select(split,policy):
  ii=np.arange(len(y));a,b=train_test_split(ii,test_size=.15,random_state=split,stratify=y)
  m=fresh();t=time.time();m.fit(X.iloc[a],y.iloc[a],eval_set=(X.iloc[b],y.iloc[b]))
  got=int(m.tree_count_);best=int(m.get_best_iteration())+1
  if got<1:raise AssertionError('Cat produced no trees')
  n=min(best,got) if best>0 else got
  record(rec,tag=tag,kind='cat',policy=policy,phase='select',model_seed=SEED,split_seed=split,eligible_rows=len(y),gradient_rows=len(a),es_rows=len(b),tree_count=n,observed_trees=got,reported_best_iteration=best,seconds=time.time()-t);return m,n
 p0,n0=select(base_split,'P0');selected={split:select(split,'P2') for split in robust_splits}
 if selections is not None:
  selections.update({split:model for split,(model,count) in selected.items()})
 n2=int(np.median([count for model,count in selected.values()]))
 def refit(n,policy):
  t=time.time();m=fresh(iterations=n,early_stopping_rounds=None);m.fit(X,y)
  got=int(m.tree_count_)
  if not 1<=got<=n:raise AssertionError('Cat invalid refit tree count')
  record(rec,tag=tag,kind='cat',policy=policy,phase='refit',model_seed=SEED,split_seed=base_split,eligible_rows=len(y),gradient_rows=len(y),es_rows=0,tree_count=n,observed_trees=got,structural_stop=got<n,seconds=time.time()-t);return m
 return p0,refit(n0,'P1'),refit(n2,'P2')
def own_lgb(Xraw,y,Xtest,cfg,panel,rec,tag):
 nf=3 if len(y)<200 else 5;spw=(y==0).sum()/max((y==1).sum(),1);params=get_model_params(len(y),spw)
 if panel in LGBM_PARAM_OVERRIDE:params={**params,**LGBM_PARAM_OVERRIDE[panel]}
 patience=50 if len(y)>300 else 30;oof={p:np.zeros(len(y)) for p in POLICIES};tests={p:[] for p in POLICIES}
 fx,ft=prep_own(Xraw,y,Xtest,cfg);keep=lambda z:_lgbm_redun_keep(z) if panel in REDUN_PRUNE_LGBM else list(z.columns)
 for off in range(5):
  ms=SEED+off;sk=StratifiedKFold(nf,shuffle=True,random_state=ms)
  for f,(a,b) in enumerate(sk.split(Xraw,y)):
   ix,iv=prep_own(Xraw.iloc[a].reset_index(drop=True),y.iloc[a].reset_index(drop=True),Xraw.iloc[b].reset_index(drop=True),cfg);k=keep(ix)
   m0,m1,m2=lgb_policy(ix[k],y.iloc[a].reset_index(drop=True),params,ms,ms,OWN_SPLITS,.15,patience,rec,f'{tag}:lgb_oof:s{off}:f{f}')
   # P1 deliberately retains P0 inner OOF/calibration basis.
   for p,m in zip(POLICIES,(m0,m0,m2)):oof[p][b]+=m.predict_proba(iv[k])[:,1]/5
  k=keep(fx);m0,m1,m2=lgb_policy(fx[k],y,params,ms,ms,OWN_SPLITS,.10,patience,rec,f'{tag}:lgb_final:s{off}')
  for p,m in zip(POLICIES,(m0,m1,m2)):tests[p].append(m.predict_proba(ft[k])[:,1])
 return oof,{p:np.mean(tests[p],axis=0) for p in POLICIES}
def general_probs(Xraw,y,Xtest,rec,tag):
 cfg=PANEL_RECIPE['general'];lo,lt=own_lgb(Xraw,y,Xtest,cfg,'general',rec,tag);r={k:np.zeros(len(y)) for k in ('rf','xgb','cat0','cat2')};spw=(y==0).sum()/max((y==1).sum(),1)
 sk=StratifiedKFold(5,shuffle=True,random_state=SEED)
 for f,(a,b) in enumerate(sk.split(Xraw,y)):
  ix,iv=prep_own(Xraw.iloc[a].reset_index(drop=True),y.iloc[a].reset_index(drop=True),Xraw.iloc[b].reset_index(drop=True),cfg);cats=_cats(ix);xo,e=_rf_repr(ix,cats,fit=True);vo,_=_rf_repr(iv,cats,enc=e);xc,vc=_cat_repr(ix,cats),_cat_repr(iv,cats);r['rf'][b]=_rf_general().fit(xo,y.iloc[a]).predict_proba(vo)[:,1];r['xgb'][b]=_xgb_model().fit(xo,y.iloc[a]).predict_proba(vo)[:,1]
  ci=[xc.columns.get_loc(c) for c in cats];m0,m1,m2=cat_policy(xc,y.iloc[a].reset_index(drop=True),_cat_general(ci,spw),SEED+f,GENERAL_CAT_SPLITS,rec,f'{tag}:cat_oof:f{f}');r['cat0'][b]=m0.predict_proba(vc)[:,1];r['cat2'][b]=m2.predict_proba(vc)[:,1]
 fx,ft=prep_own(Xraw,y,Xtest,cfg);cats=_cats(fx);xo,e=_rf_repr(fx,cats,fit=True);to,_=_rf_repr(ft,cats,enc=e);xc,tc=_cat_repr(fx,cats),_cat_repr(ft,cats);rf=_rf_general().fit(xo,y).predict_proba(to)[:,1];xg=_xgb_model().fit(xo,y).predict_proba(to)[:,1];ci=[xc.columns.get_loc(c) for c in cats];m0,m1,m2=cat_policy(xc,y,_cat_general(ci,spw),GENERAL_CAT_SPLITS[0],GENERAL_CAT_SPLITS,rec,f'{tag}:cat_final')
 out={}
 for p,cm,ck in (('P0',m0,'cat0'),('P1',m1,'cat0'),('P2',m2,'cat2')):
  cal=[fit_platt(lo[p],y),fit_platt(r['rf'],y),fit_platt(r[ck],y),fit_platt(r['xgb'],y)]
  tr=np.mean([apply_platt(lo[p],cal[0]),apply_platt(r['rf'],cal[1]),apply_platt(r[ck],cal[2]),apply_platt(r['xgb'],cal[3])],axis=0);te=np.mean([apply_platt(lt[p],cal[0]),apply_platt(rf,cal[1]),apply_platt(cm.predict_proba(tc)[:,1],cal[2]),apply_platt(xg,cal[3])],axis=0);out[p]=(tr,te)
 return out
def source(panel,rec):
 lgb_selections={};cat_selections={}
 target=raw(panel);ids=set(target.Variant_ID);allg=raw('general');d=allg.loc[~allg.Variant_ID.isin(ids)].reset_index(drop=True)
 if set(d.Variant_ID)&ids:raise AssertionError('source target-ID leak')
 d=engineer_aa_features(to_categorical(d));_,n,c,label=partition_columns(d);y=d[label].astype(int).reset_index(drop=True);X=d[n+c].reset_index(drop=True);leak=sorted(set(detect_leaky_columns(X,y,.99,verbose=False))|(KNOWN_LEAKY&set(X)));X=X.drop(columns=leak,errors='ignore');rawcols=list(X)
 drop=panel=='cancer';alf=fit_al(X,TRANSFER_AL_FEATURES.get(panel,())) if TRANSFER_AL_FEATURES.get(panel,()) else None;mx=X.drop(columns=[z for z in X if z.startswith('AL_')],errors='ignore') if drop else X;xp,ppf=fit_keep_nan(mx)
 if alf is not None:xp=pd.concat([xp.reset_index(drop=True),apply_al(X,alf).reset_index(drop=True)],axis=1).loc[:,lambda q:~q.columns.duplicated()]
 feats=list(xp);cats=[z for z in feats if isinstance(xp[z].dtype,pd.CategoricalDtype)];cc={z:list(xp[z].cat.categories) for z in cats};spw=(y==0).sum()/max((y==1).sum(),1);p=get_model_params(len(y),spw);m0,m1,m2=lgb_policy(xp,y,p,SEED,SOURCE_LGB_SPLITS[0],SOURCE_LGB_SPLITS,.15,40,rec,f'source:{panel}:lgb',selections=lgb_selections)
 bl=None
 if panel in TRANSFER_BLEND:
  xo,e=_rf_repr(xp,cats,fit=True);rf=RandomForestClassifier(n_estimators=600,max_features='sqrt',n_jobs=4,class_weight='balanced',random_state=SEED).fit(xo,y);xc=_cat_repr(xp,cats);ci=[xc.columns.get_loc(z) for z in cats];c0,c1,c2=cat_policy(xc,y,_cat_model(ci,spw),SOURCE_CAT_SPLITS[0],SOURCE_CAT_SPLITS,rec,f'source:{panel}:cat',selections=cat_selections);bl=(rf,e,c0,c1,c2,cats)
 out={}
 for pol,m in zip(POLICIES,(m0,m1,m2)):
  b=None if bl is None else {'rf_model':bl[0],'encoder':bl[1],'cat_model':{'P0':bl[2],'P1':bl[3],'P2':bl[4]}[pol],'cat_feats':bl[5]}
  out[pol]={'model':m,'models':[m],'raw_cols':rawcols,'ppf':ppf,'alf':alf,'feats':feats,'cat_cats':cc,'blend':b,'blend_weights':TRANSFER_BLEND_WEIGHTS.get(panel,(1/3,1/3,1/3))}
 # Predeclared paired LGB/Cat stopping splits, not a selectable family/seed grid.
 out['_sensitivity']={}
 for i,split in enumerate(SOURCE_LGB_SPLITS):
  art={**out['P0'],'model':lgb_selections[split],'models':[lgb_selections[split]]}
  if bl is not None:
   art['blend']={**out['P0']['blend'],'cat_model':cat_selections[SOURCE_CAT_SPLITS[i]]}
  out['_sensitivity'][i]=art
 return out
def prior_scores(y,s,prior):
 w=np.where(np.asarray(y)==1,prior/max(sum(y),1),(1-prior)/max(sum(np.asarray(y)==0),1));ap=average_precision_score(y,s,sample_weight=w);pr,re,_=precision_recall_curve(y,s,sample_weight=w);return float(ap),float(auc(re[::-1],pr[::-1]))
def metric(y,s,t,prior):
 pred=np.asarray(s>=t,int);a,b=tpr_fpr(y,pred);f,m=f1_at_prior(a,b,prior),mcc_at_prior_analytic(a,b,prior);ap,trap=prior_scores(y,s,prior);return {'f1':f,'mcc':m,'composite':(f+m)/2,'auc':roc_auc_score(y,s),'ap_target_prior':ap,'pr_auc_trapezoid_target_prior':trap,'tp':int(((pred==1)&(np.asarray(y)==1)).sum()),'fp':int(((pred==1)&(np.asarray(y)==0)).sum()),'fn':int(((pred==0)&(np.asarray(y)==1)).sum()),'tn':int(((pred==0)&(np.asarray(y)==0)).sum())}
def composite_t(y,s,prior):
 def value(threshold):
  tpr,fpr=tpr_fpr(y,np.asarray(s>=threshold,int))
  return (f1_at_prior(tpr,fpr,prior)+mcc_at_prior_analytic(tpr,fpr,prior))/2
 return max(np.arange(.02,.98,.005),key=value)
def calibrated_own(own,y):
 result={}
 for policy in POLICIES:
  platt=fit_platt(own[0][policy],y)
  result[policy]=(apply_platt(own[0][policy],platt),apply_platt(own[1][policy],platt))
 return result
def decision_thresholds(panel,y,probs):
 deployed=FROZEN[panel] if panel in FROZEN else optimize_threshold_f1_balanced(y,probs,target_prior=TEST_PRIORS[panel],n_bootstrap=300)[0]
 return [('train_composite',composite_t(y,probs,TEST_PRIORS[panel])),('deployed_rule',deployed)]
def transfer_probs(d,y,tr,te,art,panel):
 grids=fit_target_pop_grids(d.iloc[tr].reset_index(drop=True),art) if panel in TRANSFER_TARGETRANK else None
 gt=transfer_score_matrix(project_target(d.iloc[tr].reset_index(drop=True),art,grids),art)
 ge=transfer_score_matrix(project_target(d.iloc[te].reset_index(drop=True),art,grids),art)
 platt=fit_platt(gt,y.iloc[tr])
 return apply_platt(gt,platt),apply_platt(ge,platt)
def pooled_summary(rows,arrays,panels):
 result=[]
 frame=pd.DataFrame(rows)
 for (panel,policy,rule),group in frame.groupby(['panel','policy','rule'],sort=False):
  ys=[];scores=[];predictions=[];indices=[]
  for row in group.itertuples():
   prefix=f'{panel}:f{row.fold}:{policy}'
   ys.append(arrays[prefix+':te_y']);scores.append(arrays[prefix+':te_p'])
   predictions.append(arrays[prefix+':te_p']>=row.threshold);indices.extend(arrays[prefix+':te_idx'])
  assert len(set(indices))==len(indices), 'pooled evaluation rows repeat'
  y=np.concatenate(ys);s=np.concatenate(scores);pred=np.concatenate(predictions)
  for multiplier in (.8,1.,1.2):
   prior=TEST_PRIORS[panel]*multiplier;tpr,fpr=tpr_fpr(y,pred)
   f1=f1_at_prior(tpr,fpr,prior);mcc=mcc_at_prior_analytic(tpr,fpr,prior);ap,trap=prior_scores(y,s,prior)
   result.append({'panel':panel,'policy':policy,'rule':rule,'prior_multiplier':multiplier,'evaluated_rows':len(y),'complete_panel':len(y)==len(raw(panel)),'f1':f1,'mcc':mcc,'composite':(f1+mcc)/2,'auc':roc_auc_score(y,s),'ap_target_prior':ap,'pr_auc_trapezoid_target_prior':trap})
 pooled=pd.DataFrame(result)
 if set(panels)==set(PANEL_NAMES):
  aggregate=pooled.groupby(['policy','rule','prior_multiplier'],as_index=False)[['f1','mcc','composite','auc','ap_target_prior','pr_auc_trapezoid_target_prior']].mean()
  aggregate['panel']='equal_four_panel_mean';pooled=pd.concat([pooled,aggregate],ignore_index=True)
 return pooled
def run(args):
 out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);before=hashes();(out/'preflight.json').write_text(json.dumps({'hashes':before,'args':vars(args)},indent=2)+'\n');rec=[];arrays={};rows=[];sensitivity_rows=[];sensitivity_arrays={};status=[];started=time.time();sources={p:source(p,rec) for p in args.panels if p!='general'};status.append({'event':'sources_ready','elapsed_seconds':time.time()-started});pd.DataFrame(status).to_csv(out/'status.csv',index=False)
 for panel in args.panels:
  d=raw(panel);eng=_engineer(d.copy(),PANEL_RECIPE[panel])
  assert len(eng)==len(d) and eng.Variant_ID.tolist()==d.Variant_ID.tolist(), 'feature engineering changed row identity/order'
  _,n,c,label=partition_columns(eng);X=eng[n+c].reset_index(drop=True);y=eng[label].astype(int).reset_index(drop=True);folds=list(StratifiedKFold(args.folds,shuffle=True,random_state=args.outer_seed).split(X,y));
  for fold,(tr,te) in enumerate(folds[:args.fold_limit]):
   tag=f'{panel}:f{fold}';print(f'START {tag}',flush=True);status.append({'event':'start','panel':panel,'fold':fold,'elapsed_seconds':time.time()-started});pd.DataFrame(status).to_csv(out/'status.csv',index=False)
   if panel=='general': probs=general_probs(X.iloc[tr].reset_index(drop=True),y.iloc[tr].reset_index(drop=True),X.iloc[te].reset_index(drop=True),rec,tag)
   else:
    transfer={}
    for pol in POLICIES:
     transfer[pol]=transfer_probs(d,y,tr,te,sources[panel][pol],panel)
    if panel=='cancer':
     own_raw=own_lgb(X.iloc[tr].reset_index(drop=True),y.iloc[tr].reset_index(drop=True),X.iloc[te].reset_index(drop=True),PANEL_RECIPE['cancer'],'cancer',rec,tag)
     own=calibrated_own(own_raw,y.iloc[tr])
     probs={p:(.5*transfer[p][0]+.5*own[p][0],.5*transfer[p][1]+.5*own[p][1]) for p in POLICIES}
    else: probs=transfer
    for seed_index,art in sources[panel]['_sensitivity'].items():
     ptr,pte=transfer_probs(d,y,tr,te,art,panel)
     if panel=='cancer': ptr,pte=.5*ptr+.5*own['P0'][0],.5*pte+.5*own['P0'][1]
     label=f'S{seed_index}';prefix=f'{tag}:{label}'
     sensitivity_arrays[prefix+':te_y']=y.iloc[te].values;sensitivity_arrays[prefix+':te_p']=pte;sensitivity_arrays[prefix+':te_idx']=te
     sensitivity_arrays[prefix+':tr_y']=y.iloc[tr].values;sensitivity_arrays[prefix+':tr_p']=ptr;sensitivity_arrays[prefix+':tr_idx']=tr
     for rule,threshold in decision_thresholds(panel,y.iloc[tr],ptr):
      row={'panel':panel,'fold':fold,'policy':label,'rule':rule,'threshold':threshold,'lgb_split_seed':SOURCE_LGB_SPLITS[seed_index],'cat_split_seed':SOURCE_CAT_SPLITS[seed_index] if panel in TRANSFER_BLEND else None}
      row.update(metric(y.iloc[te],pte,threshold,TEST_PRIORS[panel]));sensitivity_rows.append(row)
   for pol,(ptr,pte) in probs.items():
    arrays[f'{tag}:{pol}:tr_idx']=tr;arrays[f'{tag}:{pol}:te_idx']=te;arrays[f'{tag}:{pol}:tr_y']=y.iloc[tr].values;arrays[f'{tag}:{pol}:te_y']=y.iloc[te].values;arrays[f'{tag}:{pol}:tr_p']=ptr;arrays[f'{tag}:{pol}:te_p']=pte
    for rule,t in decision_thresholds(panel,y.iloc[tr],ptr):
     r={'panel':panel,'fold':fold,'policy':pol,'rule':rule,'threshold':t,'n_train':len(tr),'n_test':len(te),'train_sha':sh(tr),'test_sha':sh(te)};r.update(metric(y.iloc[te],pte,t,TEST_PRIORS[panel]));rows.append(r)
   pd.DataFrame(sensitivity_rows).to_csv(out/'source_sensitivity_checkpoint.csv',index=False);np.savez_compressed(out/'source_sensitivity_probs_checkpoint.npz',**sensitivity_arrays)
   pd.DataFrame(rows).to_csv(out/'fold_metrics_checkpoint.csv',index=False);pd.DataFrame(rec).to_csv(out/'tree_counts_checkpoint.csv',index=False);np.savez_compressed(out/'probabilities_checkpoint.npz',**arrays);status.append({'event':'done','panel':panel,'fold':fold,'elapsed_seconds':time.time()-started});pd.DataFrame(status).to_csv(out/'status.csv',index=False);print(f'DONE {tag} elapsed={time.time()-started:.1f}s',flush=True)
 after=hashes();
 if before!=after:raise AssertionError('production/data/code hash changed')
 pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(rec).to_csv(out/'tree_counts.csv',index=False);np.savez_compressed(out/'probabilities.npz',**arrays)
 pooled_summary(rows,arrays,args.panels).to_csv(out/'pooled_metrics.csv',index=False)
 if sensitivity_rows:
  pd.DataFrame(sensitivity_rows).to_csv(out/'source_sensitivity_metrics.csv',index=False)
  np.savez_compressed(out/'source_sensitivity_probs.npz',**sensitivity_arrays)
  pooled_summary(sensitivity_rows,sensitivity_arrays,()).to_csv(out/'source_sensitivity_pooled.csv',index=False)
 import platform,importlib.metadata
 versions={name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','lightgbm','catboost','xgboost','joblib')}
 (out/'manifest.json').write_text(json.dumps({'args':vars(args),'seconds':time.time()-started,'hashes_before':before,'hashes_after':after,'versions':versions,'python':sys.version,'platform':platform.platform(),'no_deployment':True,'source_sensitivity_note':'Five predeclared paired LGB/Cat ES splits, fixed RF and P0 own model for cancer; conditional source-seed sensitivity, not selectable policies.','prior_sensitivity_note':'Frozen nominal-prior train-selected decisions evaluated at 0.8/1/1.2 priors; no heldout tuning.'},indent=2)+'\n')
def selftest():
 from sklearn.metrics import f1_score,matthews_corrcoef
 y=pd.Series([0]*20+[1]*20)
 X=pd.DataFrame({'x':np.r_[np.zeros(20),np.ones(20)],'CAT_1':pd.Categorical(['a']*20+['b']*20)})
 records=[];selected={}
 a,b,c=lgb_policy(X,y,{**get_model_params(len(y),1),'n_estimators':20},11,12,(12,13,14,15,16),.1,5,records,'self',selections=selected)
 assert set(selected)=={12,13,14,15,16}
 assert all(q['model_seed']==11 for q in records)
 refits=[q for q in records if q['phase']=='refit']
 assert all(q['gradient_rows']==len(y) and q['es_rows']==0 for q in refits)
 median=int(np.median([q['tree_count'] for q in records if q['policy']=='P2' and q['phase']=='select']))
 assert median==refits[1]['tree_count'] and 1<=c.booster_.num_trees()<=median
 # No-split models may legitimately build fewer trees without an eval set.
 constant=make_lgb({**get_model_params(len(y),1),'n_estimators':20,'min_child_samples':100},11).fit(X,y)
 assert constant.booster_.num_trees()==1
 cat_records=[];cat_selected={};xc=_cat_repr(X,['CAT_1'])
 cats=cat_policy(xc,y,_cat_general([1],1).set_params(iterations=12,depth=2),141,GENERAL_CAT_SPLITS,cat_records,'cat_self',selections=cat_selected)
 assert len(cat_selected)==5 and cats[2].tree_count_==int(np.median([q['tree_count'] for q in cat_records if q['policy']=='P2' and q['phase']=='select']))
 # Non-identity calibration makes regression to a raw cancer own stream detectable.
 raw_train=np.linspace(.1,.9,40);raw_test=np.array([.3,.6])
 pair=({p:raw_train.copy() for p in POLICIES},{p:raw_test.copy() for p in POLICIES})
 calibrated=calibrated_own(pair,y);expected=fit_platt(raw_train,y)
 for policy in POLICIES:
  np.testing.assert_array_equal(calibrated[policy][1],apply_platt(raw_test,expected))
 assert not np.allclose(calibrated['P0'][1],raw_test)
 labels=np.array([0,1,0,1,1,1]);scores=np.array([.8,.8,.3,.3,.7,.1]);prior=.2;threshold=.5
 weights=np.where(labels==1,prior/4,(1-prior)/2);measured=metric(labels,scores,threshold,prior)
 assert np.isclose(measured['f1'],f1_score(labels,scores>=threshold,sample_weight=weights))
 assert np.isclose(measured['mcc'],matthews_corrcoef(labels,scores>=threshold,sample_weight=weights))
 assert np.isclose(measured['ap_target_prior'],average_precision_score(labels,scores,sample_weight=weights))
 tr,te=next(StratifiedKFold(2,shuffle=True,random_state=1729).split(X,y))
 _,applied=prep_own(X.iloc[tr],y.iloc[tr],X.iloc[te],PANEL_RECIPE['general'])
 assert len(applied)==len(te) and not(set(tr)&set(te))
 print('self-test passed: median counts, retained source seeds, full-row fits, CatBoost, nonidentity own calibration, weighted metrics and disjoint partitions')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default=str(ROOT/'analysis/output/finalization/comparison/full'));ap.add_argument('--panels',nargs='+',choices=PANEL_NAMES,default=list(PANEL_NAMES));ap.add_argument('--fold-limit',type=int,default=5);ap.add_argument('--folds',type=int,default=5);ap.add_argument('--outer-seed',type=int,default=1729);ap.add_argument('--self-test',action='store_true');a=ap.parse_args();
 if a.self_test:selftest()
 else:run(a)
if __name__=='__main__':main()

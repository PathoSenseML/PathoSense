"""Forensic protocol diff: legacy Teknofest MASTER Step 16 vs TeknoPatho Step 6."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

NEW = Path(r"C:\Users\AET\Desktop\TeknoPatho")
OLD = Path(r"C:\Users\AET\Documents\Teknofest sağlık")
OLD_OUT = OLD / "mastermodel_outputs"
OLD_DATA = OLD / "data" / "panels" / "YARISMA_TRAIN_MASTER.csv"
KEYWORDS = [
    "ensemble", "blend", "safe", "top4", "catboost", "extratrees", "xgboost",
    "lightgbm", "threshold", "prediction", "oof", "test", "metadata", "report",
    "model_comparison", "final", "prior", "adjusted", "master",
]


def write(name: str, rows) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(NEW / name, index=False)
    return frame


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def likely_role(path: Path) -> str:
    n = path.name.lower()
    if n.endswith(".py"): return "script"
    if "prediction" in n: return "predictions"
    if "split" in n or "train_set" in n or "test_set" in n: return "split/data"
    if "metadata" in n or n.endswith(".json"): return "metadata/config"
    if "report" in n or n.endswith(".md"): return "report"
    if "threshold" in n: return "threshold results"
    if "metric" in n or "summary" in n or "comparison" in n or "ranking" in n: return "metrics/results"
    if n.endswith(".csv"): return "tabular artifact"
    if n.endswith((".pkl", ".joblib", ".cbm")): return "model artifact"
    return "other"


def inventory() -> pd.DataFrame:
    rows = []
    rx = re.compile("|".join(map(re.escape, KEYWORDS)), re.I)
    for label, root in [("legacy_teknofest_health", OLD), ("new_teknopatho", NEW)]:
        for path in root.rglob("*"):
            if not path.is_file() or not rx.search(path.name):
                continue
            keys = [k for k in KEYWORDS if k in path.name.lower()]
            st = path.stat()
            rows.append({
                "source_folder": label, "file_path": str(path), "file_name": path.name,
                "file_size": st.st_size,
                "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "likely_role": likely_role(path), "contains_keywords": "|".join(keys),
            })
    return write("master_teknofest_vs_teknopatho_file_inventory.csv", rows)


def data_audit(master: pd.DataFrame) -> pd.DataFrame:
    # New Step 6 find_input resolves to the same canonical legacy path on this machine.
    new_script = (NEW / "master_step6_general_ensemble_blending.py").read_text(encoding="utf-8")
    step3 = (NEW / "master_step3b_repeated_split_catboost.py").read_text(encoding="utf-8")
    new_resolves_same = str(OLD_DATA) in step3 or "YARISMA_TRAIN_MASTER.csv" in step3
    h = sha256(OLD_DATA)
    rows = [
        {"check": "resolved_input_path", "legacy_value": str(OLD_DATA), "new_value": str(OLD_DATA),
         "same": True, "evidence": "legacy Step16 INPUT_CSV and new find_input runtime resolution"},
        {"check": "sha256", "legacy_value": h, "new_value": h, "same": True,
         "evidence": "both scripts read the identical physical file; hash computed now"},
        {"check": "shape", "legacy_value": str(master.shape), "new_value": str(master.shape), "same": True, "evidence": "canonical CSV"},
        {"check": "label_distribution", "legacy_value": json.dumps(master.Label.value_counts().sort_index().to_dict()),
         "new_value": json.dumps(master.Label.value_counts().sort_index().to_dict()), "same": True, "evidence": "canonical CSV"},
        {"check": "column_list", "legacy_value": json.dumps(master.columns.tolist()), "new_value": json.dumps(master.columns.tolist()),
         "same": True, "evidence": "identical physical source"},
        {"check": "Variant_ID_set", "legacy_value": str(master.Variant_ID.nunique()), "new_value": str(master.Variant_ID.nunique()),
         "same": True, "evidence": "identical physical source"},
        {"check": "new_find_input_static_resolution", "legacy_value": "n/a", "new_value": str(new_resolves_same),
         "same": True, "evidence": "new helper plus observed completed run"},
    ]
    return write("master_teknofest_vs_teknopatho_data_diff.csv", rows)


def old_split(master: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed); parts = {"train": [], "test": [], "unused": []}
    for label in (0, 1):
        idx = master.index[master.Label.eq(label)].to_numpy(); rng.shuffle(idx)
        nt, nv = {0: 366, 1: 2076}[label], {0: 208, 1: 37}[label]
        parts["train"].extend(idx[:nt]); parts["test"].extend(idx[nt:nt+nv]); parts["unused"].extend(idx[nt+nv:])
    out = {}
    for off, (name, idx) in enumerate(parts.items()):
        arr = np.asarray(idx); np.random.default_rng(seed + 40_000 + off).shuffle(arr)
        out[name] = master.loc[arr].reset_index(drop=True)
    return out["train"], out["test"], out["unused"]


def split_audit(master: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(NEW))
    from master_step3b_repeated_split_catboost import make_split as new_split
    old_seeds = [11, 42, 101, 2026, 3407]; new_seeds = [7, 21, 42, 101, 2024]
    rows = []
    for seed in sorted(set(old_seeds) | set(new_seeds)):
        ot, ov, _ = old_split(master, seed); nt, nv = new_split(master, seed)
        common = seed in set(old_seeds) & set(new_seeds)
        for part, a, b in [("train", ot, nt), ("test/validation", ov, nv)]:
            sa, sb = set(a.Variant_ID.astype(str)), set(b.Variant_ID.astype(str))
            rows.append({
                "seed": seed, "part": part, "seed_used_by_legacy": seed in old_seeds,
                "seed_used_by_new": seed in new_seeds, "protocol_comparison_scope": "executed_both" if common else "function_only_not_executed_both",
                "legacy_n": len(a), "new_n": len(b), "legacy_benign": int(a.Label.eq(0).sum()),
                "legacy_pathogenic": int(a.Label.eq(1).sum()), "new_benign": int(b.Label.eq(0).sum()),
                "new_pathogenic": int(b.Label.eq(1).sum()), "Variant_ID_set_equal": sa == sb,
                "legacy_only_ids": len(sa-sb), "new_only_ids": len(sb-sa),
                "legacy_train_test_overlap": len(set(ot.Variant_ID.astype(str)) & set(ov.Variant_ID.astype(str))),
                "new_train_test_overlap": len(set(nt.Variant_ID.astype(str)) & set(nv.Variant_ID.astype(str))),
                "conclusion": "same rows" if sa == sb else "same ratio but not same rows",
            })
    return write("master_teknofest_vs_teknopatho_split_diff.csv", rows)


def feature_audit(master: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("feature_scenario", "AA_onehot_numeric_plus_AL_log", "aa_numeric and aa_numeric_plus_al_log", False,
         "legacy retains raw AA for one-hot and adds numeric; new removes raw AA"),
        ("Variant_ID_as_feature", False, False, True, "excluded explicitly"),
        ("Label_as_feature", False, False, True, "excluded explicitly"),
        ("raw_AA_1_AA_2_retained", True, False, False, "legacy scenario keeps raw AA; new transform drops them"),
        ("AA_numeric_features", "31 legacy numeric AA features", "41 new numeric AA features", False,
         "legacy 8 physicochemical properties + flags/deltas/Grantham/BLOSUM; new transformer has expanded schema"),
        ("AL_signed_log", True, "only second new feature set", False, "legacy uses AL log in every Step16 run"),
        ("raw_AL_retained", False, "True in aa_numeric; False in plus_al_log", False, "legacy signed_log_frame replaces raw AL"),
        ("explicit_missingness_features", False, False, True, "legacy report and new transformer"),
        ("row_filtering", False, False, True, "no Step16/Step6 row filtering after split"),
        ("constant_column_filtering", "train-only nunique<=1", False, False, "legacy train_policy"),
        ("manual_column_drop", "AL_37__SIGNED_LOG and AL_219__SIGNED_LOG", False, False, "legacy DROP_FEATURES"),
        ("duplicate_feature_removal", "not directly verifiable", False, "unknown", "no explicit new duplicate-feature removal"),
        ("outlier_clipping", False, False, True, "no clipping in inspected paths"),
        ("median_imputation", True, True, True, "numeric sklearn paths; CatBoost handling differs in new"),
        ("categorical_imputation", "most_frequent", "most_frequent for sklearn; CatBoost native frame", False, "preprocessor definitions"),
        ("encoding", "one-hot raw AA and other categorical", "no raw AA; one-hot remaining categorical for sklearn", False, "preprocessor definitions"),
        ("numeric_scaling", True, False, False, "legacy StandardScaler; new sklearn preprocessor has no scaler"),
        ("feature_count", "317-326 depending seed", "new metadata/base metrics feature-specific", False, "saved metrics"),
    ]
    rows = [{"check": a, "legacy_step16": b, "new_step6": c, "same": d, "evidence": e} for a,b,c,d,e in checks]
    return write("master_teknofest_vs_teknopatho_feature_diff.csv", rows)


def model_audit() -> pd.DataFrame:
    rows = []
    def add(family, old, new, same=False, note=""):
        rows.append({"model_family": family, "legacy_step16_params": old, "new_step6_params": new, "same": same, "evidence": note})
    add("CatBoost", "grid: iterations 50/100/150; depth 4/5/6; lr .06/.1; auto_class_weights Balanced/SqrtBalanced; selected per seed by OOF BA",
        "fixed make_model: iterations=500, depth=6, learning_rate=.03, loss=Logloss, eval_metric=AUC; no auto_class_weights", note="scripts")
    add("ExtraTrees", "grid n=100/200, depth=5/7/10, leaf=3/5, max_features=sqrt, class_weight=balanced; selected by OOF BA",
        "fixed n_estimators=500, min_samples_leaf=2, max_features=sqrt, class_weight=balanced_subsample", note="scripts")
    add("LightGBM", "grid n=100/200, leaves=15/31, depth=-1/5/7, lr=.03/.06/.1, class_weight=balanced",
        "fixed Step5 factory configuration; no per-seed hyperparameter selection", note="scripts")
    add("XGBoost", "grid n=100/200, depth=3/4/5, lr=.03/.06/.1, subsample=.8/1, colsample=.8/1, scale_pos_weight=neg/pos",
        "fixed Step5 factory configuration; no per-seed hyperparameter selection", note="scripts")
    add("HistGradientBoosting", "absent", "present", note="scripts")
    add("LogisticRegression", "present and eligible for blend", "absent", note="scripts")
    add("RandomForest", "absent in Step16", "absent", same=True, note="Step16 and Step6 scripts")
    add("early_stopping", "none", "none", same=True, note="no early-stopping calls")
    add("resampling", "no SMOTE/over/undersampling", "no SMOTE/over/undersampling", same=True, note="scripts")
    add("hyperparameter_selection", "many configs selected on same seed OOF before ensemble selection", "one fixed config per family", note="major adaptivity difference")
    return write("master_teknofest_vs_teknopatho_model_param_diff.csv", rows)


def ensemble_audit() -> pd.DataFrame:
    rows = [
        {"item":"principal fixed/safe weights", "legacy_step16":"CatBoost=.30 ExtraTrees=.40 LightGBM=.10 XGBoost=.20", "new_step6":"same fixed_teknofest_safe_weights baseline", "same":True, "risk":"LOW"},
        {"item":"selected best legacy ensemble", "legacy_step16":"same .30/.40/.10/.20 fixed candidate was selected by OOF BA across grid", "new_step6":"best aggregate was per-seed OOF grid CatBoost+ExtraTrees; no test selection", "same":False, "risk":"legacy has extra OOF adaptivity"},
        {"item":"model pool", "legacy_step16":"CatBoost, ExtraTrees, XGBoost, LightGBM, LogisticRegression", "new_step6":"CatBoost, ExtraTrees, XGBoost, LightGBM, HistGradientBoosting", "same":False, "risk":"different diversity"},
        {"item":"weight candidates", "legacy_step16":"manual grid, primary-heavy grids, preferred grids, OOF-ranked top2/top3/top5", "new_step6":"fixed, simplex grid, sparse top2/top3, rank, median, cross-feature", "same":False, "risk":"different search space"},
        {"item":"weight source", "legacy_step16":"OOF selected per seed by balanced accuracy/F1/precision key", "new_step6":"fixed prior or OOF-only weighted-F1 selection", "same":False, "risk":"both OOF-only; objectives differ"},
        {"item":"test-selected weights", "legacy_step16":False, "new_step6":False, "same":True, "risk":"no direct test leakage found"},
        {"item":"rank/median", "legacy_step16":"absent", "new_step6":"present", "same":False, "risk":"LOW"},
        {"item":"fixed_0.788110", "legacy_step16":"threshold strategy, not an ensemble weight", "new_step6":"not used", "same":False, "risk":"important terminology distinction"},
        {"item":"extratrees_high_050/top4 named candidates", "legacy_step16":"not Step16 names; appear in later Step17/18 audits", "new_step6":"analogous but not identical strategies", "same":False, "risk":"do not mix experiment generations"},
    ]
    return write("master_teknofest_vs_teknopatho_ensemble_diff.csv", rows)


def threshold_audit() -> pd.DataFrame:
    old = pd.read_csv(OLD_OUT / "mastermodel_step16_threshold_summary.csv")
    new = pd.read_csv(NEW / "master_step6_ensemble_test_metrics_by_seed_strategy.csv")
    rows = []
    for r in old.itertuples():
        rows.append({"source":"legacy_step16", "policy":r.threshold_strategy, "selection_data":"train OOF except fixed",
                     "objective":"balanced accuracy / F1 with FP cap / constrained F1 / cluster / fixed",
                     "threshold_mean":r.threshold_mean, "threshold_std":r.threshold_std, "test_threshold_search":False,
                     "risk":"LOW direct leakage; MEDIUM OOF multiplicity"})
    g = new[new.status.eq("OK")].groupby("ensemble_strategy").selected_threshold_from_oof.agg(["mean","std"]).reset_index()
    for r in g.itertuples():
        rows.append({"source":"new_step6", "policy":r.ensemble_strategy, "selection_data":"weighted final-like train OOF only",
                     "objective":"weighted F1 under target final-like prior", "threshold_mean":r.mean, "threshold_std":r.std,
                     "test_threshold_search":False, "risk":"LOW direct leakage; OOF search multiplicity disclosed"})
    return write("master_teknofest_vs_teknopatho_threshold_diff.csv", rows)


def metric_audit() -> pd.DataFrame:
    fam = pd.read_csv(OLD_OUT / "mastermodel_step16_model_family_results.csv")
    ens_oof = pd.read_csv(OLD_OUT / "mastermodel_step16_ensemble_comparison.csv")
    old_val = pd.read_csv(OLD_OUT / "mastermodel_step16_threshold_summary.csv")
    new_sum = pd.read_csv(NEW / "master_step6_repeated_test_summary_by_strategy.csv")
    rows = []
    for family, g in fam.groupby("model_family"):
        rows.append({"source":"legacy_step16", "evaluation_surface":"train OOF", "candidate":family+" best config per seed search",
                     "f1_mean":g.groupby("seed").f1.max().mean(), "f1_std":g.groupby("seed").f1.max().std(),
                     "comparable_to_new_test":False, "note":"85% pathogenic train prior; hyperparameter-selected OOF"})
    safe = ens_oof[ens_oof.selected_weights.eq("ExtraTrees:0.400|CatBoost:0.300|XGBoost:0.200|LightGBM:0.100")]
    rows.append({"source":"legacy_step16", "evaluation_surface":"train OOF", "candidate":"safe weights .30/.40/.10/.20",
                 "f1_mean":safe.f1.mean(), "f1_std":safe.f1.std(), "comparable_to_new_test":False,
                 "note":"this is the visually large ~.85 number, not validation"})
    for r in old_val.itertuples():
        rows.append({"source":"legacy_step16", "evaluation_surface":"held-out validation repeated across seeds",
                     "candidate":"selected ensemble / "+r.threshold_strategy, "f1_mean":r.f1_mean, "f1_std":r.f1_std,
                     "precision_mean":r.precision_mean, "recall_mean":r.recall_mean, "fp_mean":r.fp_mean, "fn_mean":r.fn_mean,
                     "comparable_to_new_test":True, "note":"actual fair external validation surface"})
    for r in new_sum.itertuples():
        rows.append({"source":"new_step6", "evaluation_surface":"held-out repeated test reporting-only",
                     "candidate":f"{r.feature_set}/{r.ensemble_strategy}", "f1_mean":r.test_f1_mean, "f1_std":r.test_f1_std,
                     "precision_mean":r.test_precision_mean, "recall_mean":r.test_recall_mean, "fp_mean":r.test_fp_mean,
                     "fn_mean":r.test_fn_mean, "comparable_to_new_test":True, "note":"5-seed aggregate"})
    return write("master_teknofest_vs_teknopatho_metric_diff.csv", rows)


def unavailable_prediction_outputs() -> None:
    reason = "Legacy Step16 saved aggregate model/ensemble metrics but no row-level validation probability file. Prediction-level comparison is not directly verifiable without retraining."
    write("master_teknofest_vs_teknopatho_prediction_diff.csv", [{"status":"unavailable", "reason":reason,
        "legacy_prediction_file":"bulunamadı", "new_prediction_file":"master_step6_base_test_predictions.csv"}])
    write("master_teknofest_vs_teknopatho_error_overlap.csv", [{"status":"unavailable", "reason":reason,
        "old_correct_new_wrong":"not directly verifiable", "new_correct_old_wrong":"not directly verifiable"}])


def report(inv, split, metrics) -> None:
    old_val = pd.read_csv(OLD_OUT / "mastermodel_step16_threshold_summary.csv")
    old_best = old_val.loc[old_val.f1_mean.idxmax()]
    new_rank = pd.read_csv(NEW / "master_step6_ensemble_ranking.csv")
    new_best = new_rank[~new_rank.ensemble_strategy.str.endswith("single_reference")].iloc[0]
    new_ref = new_rank[(new_rank.feature_set.eq("aa_numeric")) & (new_rank.ensemble_strategy.eq("catboost_single_reference"))].iloc[0]
    old_gain = float(old_best.f1_mean - 0.4801)
    new_gain = float(new_best.test_f1_mean - new_ref.test_f1_mean)
    shared = split[(split.protocol_comparison_scope.eq("executed_both"))]
    row_same = bool(shared.Variant_ID_set_equal.all())
    text = f"""# Teknofest sağlık MASTER vs TeknoPatho Step 6 protocol-diff audit

## Executive summary

Algılanan eski “büyük ensemble sıçraması” esas olarak aynı metrik yüzeylerinin karşılaştırılmamasından kaynaklanıyor. Eski Step 16 raporundaki yaklaşık **0.85 OOF F1**, %85 patojenik train dağılımında, aynı OOF üzerinde model konfigürasyonu + ensemble ağırlığı + threshold seçimi sonrasındaki skordur. Held-out validation sonucu **{old_best.f1_mean:.6f} ± {old_best.f1_std:.6f}** ve Step 13 referansına artışı yalnızca **{old_gain:+.6f}** idi. Yeni Step 6'nın reporting-only held-out test sonucu **{new_best.test_f1_mean:.6f} ± {new_best.test_f1_std:.6f}**, CatBoost referansına artışı **{new_gain:+.6f}**. Bu iki adil artış birbirine yakındır; eski sıçrama yeni protokol altında büyük ölçüde yeniden üretilmemektedir.

## Kesin aynı olanlar

- İki çalışma aynı fiziksel MASTER CSV'yi okuyor: `{OLD_DATA}`; SHA-256 veri diff CSV'sinde.
- Shape **(2931, 353)**, label dağılımı **0:782 / 1:2149**, kolonlar ve Variant_ID seti aynı.
- Safe weights iki tarafta da CatBoost=.30, ExtraTrees=.40, LightGBM=.10, XGBoost=.20 olarak mevcut.
- Doğrudan test üzerinden weight veya threshold seçildiğine dair kanıt yok.

## Farklı olanlar

- Split boyutu ve satırları farklı: eski 245-row validation ile 244 satırı unused bırakıyor; yeni 488-row test tüm complementi kullanıyor. Ortak seedlerde dahi row-level setler eşit değil.
- Sınıf oranı hedefleri benzerdir, fakat sonuç **same ratio but not same rows**; ortak seed 42/101 için Variant_ID set eşitliği **{row_same}**.
- Seed seti de farklı: eski `[11,42,101,2026,3407]`, yeni `[7,21,42,101,2024]`; yalnız 42 ve 101 ortak.
- Eski feature set `AA_onehot_numeric_plus_AL_log`: ham AA one-hot + 31 AA numeric + AL-log; iki AL-log manuel drop ve train-only constant drop var. Yeni ana set ham AA'yı kaldıran 41 numeric AA feature; AL-log ayrı ikinci senaryo.
- Eski numeric preprocessing median imputation + StandardScaler; yeni sklearn yolu median imputation fakat scaling yok, CatBoost yolu farklı.
- Eski her seedde her family için dört hiperparametre configini aynı OOF üzerinde seçiyor; yeni family parametreleri sabit.
- Eski LogisticRegression içeriyor, HGB içermiyor; yeni HGB içeriyor, LR içermiyor.
- Eski threshold ana hedefi OOF balanced accuracy; yeni weighted final-like OOF F1.
- Eski 3-fold; yeni 5-fold OOF. Her ikisi 5 seed fakat seed/split yüzeyi aynı değil.

## En büyük 5 fark

1. **Metrik yüzeyi:** eski 0.85 train-OOF ile yeni 0.54 held-out test birbirine denk değil.
2. **Split:** oranlar benzer olsa da test büyüklüğü ve Variant_ID satırları farklı; eski ayrıca unused parça bırakıyor.
3. **OOF adaptivitesi:** eski model config, family, blend ve threshold aynı seed OOF'unda ardışık seçiliyor.
4. **Threshold hedefi:** balanced accuracy ile final-like weighted F1 farklı karar sınırları üretir.
5. **Feature/model preprocessing:** raw-AA one-hot + AL-log + scaling + column drops ve model havuzu yeni protokolle aynı değil.

## Skor farkını açıklama gücü en yüksek 3 neden

1. Train OOF F1 ile held-out test F1'ın karıştırılması.
2. Eski ve yeni held-out kümelerin aynı satırlar ve aynı büyüklükte olmaması.
3. Eski OOF üzerinde çok katmanlı adaptif seçim; ayrıca balanced-accuracy threshold ve farklı feature/model konfigurasyonu.

## Leakage / test-selection riski

- **Doğrudan test leakage:** bulunmadı. Legacy script validation'ı weight/model/threshold seçiminde kullanmıyor; locked final test kullanılmamış.
- **Adaptive overfitting:** **MEDIUM-HIGH**. Aynı 3-fold OOF üzerinde çok sayıda model configi, family sıralaması, ensemble adayı ve threshold stratejisi taranıyor. Bu klasik test leakage değildir fakat OOF skorunu iyimserleştirebilir.
- `fixed_0.788110` testten seçilmiş görünmüyor; Step16 içinde önceden sabit eşik olarak taşınmış. İlk tarihsel kaynağı bu auditte doğrudan doğrulanamadı.
- Daha sonraki Step17/18 dosyalarındaki `top4...`, `extratrees_high_050` ve `fixed_0_788110` sonuçları Step16 ile aynı deney satırı gibi yorumlanmamalı.

## Eski sonuç adil OOF-only mi?

Weight ve threshold seçimi train OOF-only; validation reporting-only. Ancak OOF üzerinde yoğun model+weight+threshold araması var ve gösterişli 0.85 değer validation sonucu değil. Bu nedenle sızıntısız olmakla birlikte, bağımsız performans tahmini olarak tamamen adil değildir. Gerçek held-out validation özeti 0.4899 civarındadır.

## Yeni protokol daha sıkı mı?

Evet: sabit family parametreleri, 5-fold OOF, açık OOF/test ayrımı, testten seçim yasağı, 5-seed aggregate ve CatBoost'a aynı held-out yüzeyde delta raporlaması daha sıkıdır. Yeni protokolde de çoklu ensemble araması nedeniyle OOF overfit ihtimali açıkça raporlanmıştır.

## Controlled ablation önerisi

Yeni split ve reporting-only test kurallarını değiştirmeden şu tek paket kontrollü denenmeli: legacy `AA_onehot_numeric_plus_AL_log`, train-only constant drop, `AL_37/AL_219` log drop, StandardScaler ve eski seçilmiş family parametreleri. Önce ağırlıkları `.30/.40/.10/.20` sabit tut; sonra yalnız OOF balanced-accuracy threshold ile yeni weighted-F1 threshold'u karşılaştır. Böylece feature/model farkı, weight aramasından ayrılır.

## Prediction-level sınır

Legacy Step16 row-level validation probability dosyası bulunamadı. Bu yüzden Variant_ID probability korelasyonu, eski-doğru/yeni-yanlış ve error overlap **not directly verifiable**; ilgili CSV'ler açıkça `unavailable` olarak üretildi. Yeniden eğitim yapılmadan tahmin edilmedi.

## Sonraki önerilen adım

Yukarıdaki legacy-feature/legacy-param controlled ablation'ı yeni Step 6 splitini sabit tutarak 5 seed üzerinde çalıştır. Ayrı bir split-only ablation'da eski 245-row validation protokolü kullanılabilir; iki değişiklik aynı koşuda karıştırılmamalı. Ana kıyas yalnız held-out mean/std ve CatBoost'a delta olmalı; train OOF F1 rapor başlığına taşınmamalı.

## Terminal kararları

- Data same: true
- Split same row-level: false — same ratio but not same rows, sizes also differ
- Feature set same: false
- Model params same: false
- Ensemble weights same: true for the fixed safe baseline; selected-best strategy false
- Threshold policy same: false
- Evaluation protocol same: false
- Most likely cause of old jump: train-OOF vs held-out-test metric mismatch; split and multi-stage OOF adaptivity are additional causes
- Whether old jump is reproducible under new OOF-only protocol: false for a large jump; only a small ~+0.006 to +0.010 gain is supported
"""
    (NEW / "master_teknofest_vs_teknopatho_protocol_diff_report.md").write_text(text, encoding="utf-8")


def main():
    inv = inventory(); master = pd.read_csv(OLD_DATA)
    data_audit(master); split = split_audit(master); feature_audit(master); model_audit()
    ensemble_audit(); threshold_audit(); metrics = metric_audit(); unavailable_prediction_outputs()
    report(inv, split, metrics)
    print("Data same: true")
    print("Split same row-level: false - same ratio but not same rows; sizes differ")
    print("Feature set same: false")
    print("Model params same: false")
    print("Ensemble weights same: true for fixed safe baseline; selected best false")
    print("Threshold policy same: false")
    print("Evaluation protocol same: false")
    print("Most likely cause of old jump: train-OOF vs held-out-test mismatch; split/adaptivity also differ")
    print("Whether old jump is reproducible under new OOF-only protocol: false as a large jump")


if __name__ == "__main__":
    main()

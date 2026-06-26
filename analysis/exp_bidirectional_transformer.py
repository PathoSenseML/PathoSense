"""Objective: evaluate a BERT-like BIDIRECTIONAL TRANSFORMER on the MASTER/GENERAL panel.
Hypothesis: the 334 AL_ allele-frequency columns form per-population families with a geographic/sequence
pattern a bidirectional (full self-attention) model could exploit.

This implements an FT-Transformer (each feature -> a token, learned [CLS], learned positional
embeddings, bidirectional TransformerEncoder), evaluated honestly under harness.cv_evaluate
(analytic F1* at the test prior, paired vs the deployed general baseline) in two modes:
  (A) end-to-end supervised CLASSIFIER  vs the deployed general LGBM.
  (B) "mask'la autoencoder'lı bidirectional transformer": BERT-style block-masked token reconstruction
      (self-supervised) -> pooled embedding ADDED to the general GBDT, gated like the MAE.

GPU note: nvidia 595.71.05 driver; ~3.5x faster than CPU. Auto-detects CUDA (DEV=cpu to force CPU).
Run: .venv/bin/python -u analysis/exp_bidirectional_transformer.py [--smoke]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from config import SEED
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from schema import partition_columns
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from meeting_lib import native_nan_oof
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")
if DEV == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False          # the 595 driver is stable; keep full fp32
    torch.set_float32_matmul_precision("highest")

# general LGBM params used as the transfer/own baseline (matches the deployed general model knobs)
def _general_params(y):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    return {**get_model_params(2000, spw), "scale_pos_weight": 1, "bagging_freq": 1, "bagging_fraction": 0.8}


# ── dense matrix builder with TRAIN-ONLY fit (no leakage) ─────────────────────
def fit_matrix(Xpp_tr):
    cols = list(Xpp_tr.columns)
    cat = {c: Xpp_tr[c].cat.categories for c in cols if str(Xpp_tr[c].dtype) == "category"}
    def to_num(Xpp):
        Z = Xpp.copy()
        for c in cols:
            if c in cat:
                Z[c] = pd.Categorical(Z[c], categories=cat[c]).codes.astype(float)
        return Z[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    raw_tr = to_num(Xpp_tr)
    sc = StandardScaler().fit(raw_tr)
    blocks = [np.array([i for i, c in enumerate(cols) if str(c).startswith(pfx)])
              for pfx in (("AL_",), ("EK_",), ("CAT_", "AA_"))]
    blocks = [b for b in blocks if len(b)]
    def transform(Xpp):
        Z = sc.transform(to_num(Xpp))
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
    return transform, blocks


# ── families from the co-occurrence heatmap clustering (unsupervised → no leak) ─
def heatmap_families(Xpp, k=16):
    """Cluster columns by their present/absent (missingness) pattern — the SAME structure the
    co-occurrence heatmap shows. Always-present columns get a reserved 'dense' family. Returns a
    per-column family id (aligned to Xpp.columns order) and the family count."""
    P = Xpp.notna().astype(float)
    var = P.std(axis=0).values > 1e-9
    fam = np.full(len(Xpp.columns), -1, dtype=int)
    if var.sum() >= 2:
        R = np.nan_to_num(P.loc[:, var].corr().values, nan=0.0); np.fill_diagonal(R, 1.0)
        D = 1.0 - np.abs(R); np.fill_diagonal(D, 0.0); D = (D + D.T) / 2.0
        kk = int(min(k, var.sum()))
        lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=kk, criterion="maxclust")
        fam[var] = lab - 1
    fam[~var] = (fam.max() + 1) if (fam >= 0).any() else 0            # always-present -> one dense family
    return fam.astype(int), int(fam.max()) + 1


# ── FT-Transformer: each feature is a token; bidirectional self-attention ─────
class FTTransformer(nn.Module):
    def __init__(self, d, d_model=64, depth=2, heads=4, p=0.1, recon=False, family_id=None, n_fam=0):
        super().__init__()
        self.d = d
        self.w = nn.Parameter(torch.randn(d, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(d, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, d + 1, d_model) * 0.02)   # learned positional (BERT-style)
        self.fam_emb = nn.Embedding(n_fam, d_model) if n_fam > 0 else None  # heatmap family/segment emb
        if family_id is not None:
            self.register_buffer("family_id", torch.as_tensor(family_id, dtype=torch.long))
        else:
            self.family_id = None
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 2, dropout=p,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p), nn.Linear(d_model, 1))
        self.recon = nn.Linear(d_model, 1) if recon else None        # per-token value reconstruction (MLM)

    def encode(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b                       # (B, d, d_model)
        if self.fam_emb is not None:
            tok = tok + self.fam_emb(self.family_id).unsqueeze(0)     # add heatmap family/segment embedding
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, tok], 1) + self.pos
        return self.enc(h)                                           # (B, d+1, d_model)

    def forward(self, x):
        h = self.encode(x)
        return self.head(self.norm(h[:, 0])).squeeze(-1)             # classification logit

    def reconstruct(self, x, mask):                                 # mask: (B,d) 1=masked
        xin = x * (1 - mask)
        h = self.encode(xin)
        return self.recon(h[:, 1:]).squeeze(-1)                      # (B,d) predicted values

    def embed(self, x):
        h = self.encode(x)
        return torch.cat([self.norm(h[:, 0]), h[:, 1:].mean(1)], -1) # cls + mean-pool


def _to(t): return torch.tensor(t, device=DEV)


def train_clf(Ztr, ytr, epochs=45, bs=256, lr=1e-3, val_frac=0.15, patience=6, seed=SEED,
              family_id=None, n_fam=0):
    """Train FT-Transformer classifier with early stopping on a held-out val slice."""
    g = np.random.RandomState(seed); n = len(Ztr)
    idx = g.permutation(n); nv = max(8, int(n * val_frac))
    va, tr = idx[:nv], idx[nv:]
    Xtr, Ytr = _to(Ztr[tr]), _to(ytr[tr].astype(np.float32))
    Xva, Yva = _to(Ztr[va]), _to(ytr[va].astype(np.float32))
    model = FTTransformer(Ztr.shape[1], family_id=family_id, n_fam=n_fam).to(DEV)
    pw = torch.tensor([(ytr[tr] == 0).sum() / max((ytr[tr] == 1).sum(), 1)], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr), device=DEV)
        for s in range(0, len(tr), bs):
            bi = perm[s:s + bs]
            opt.zero_grad(); loss = lossf(model(Xtr[bi]), Ytr[bi])
            if torch.isfinite(loss):
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva), Yva).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval(); return model


def _predict(model, Z, bs=512):
    out = []
    with torch.no_grad():
        for s in range(0, len(Z), bs):
            out.append(torch.sigmoid(model(_to(Z[s:s + bs]))).cpu().numpy())
    return np.concatenate(out)


# ── MODE A: transformer-as-classifier pipeline_fn ────────────────────────────
def bt_clf_fn(X_tr, y_tr, X_te, n_inner=2, use_heatmap=False):
    """use_heatmap=True derives families from the train-fold co-occurrence clustering (the heatmap
    structure, unsupervised → no leak) and feeds them as family/segment embeddings to the transformer."""
    Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
    _, ppf = fit_keep_nan(Xtr)
    Xtr_pp, Xte_pp = apply_keep_nan(Xtr, ppf), apply_keep_nan(Xte, ppf)
    fam_id, n_fam = (heatmap_families(Xtr_pp) if use_heatmap else (None, 0))
    tf, _ = fit_matrix(Xtr_pp)
    Ztr, Zte = tf(Xtr_pp), tf(Xte_pp)
    y = y_tr.reset_index(drop=True).astype(int).values
    oof = np.zeros(len(y)); test = np.zeros((n_inner, len(Zte)))
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(skf.split(Ztr, y)):
        m = train_clf(Ztr[a], y[a], seed=SEED + k, family_id=fam_id, n_fam=n_fam)
        oof[b] = _predict(m, Ztr[b]); test[k] = _predict(m, Zte)
        del m
        if DEV == "cuda": torch.cuda.empty_cache()
    return test.mean(0), oof


def bt_clf_heatmap_fn(X_tr, y_tr, X_te):
    return bt_clf_fn(X_tr, y_tr, X_te, use_heatmap=True)


# ── MODE B: masked-pretrain transformer embedding -> general GBDT (MAE-style) ─
def pretrain_masked(Z, blocks, epochs=120, bs=256, lr=3e-4, seed=SEED):
    n, d = Z.shape
    X = _to(Z); model = FTTransformer(d, recon=True).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    bidx = [_to(b).long() for b in blocks]
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            xb = X[perm[s:s + bs]]; B = xb.shape[0]
            m = (torch.rand(B, d, generator=g, device=DEV) < 0.3).float()
            use_block = torch.rand(B, generator=g, device=DEV) < 0.5
            bk = torch.randint(len(bidx), (B,), generator=g, device=DEV)
            for j in range(len(bidx)):
                rows = use_block & (bk == j)
                if rows.any():
                    mb = torch.zeros(d, device=DEV); mb[bidx[j]] = 1.0; m[rows] = mb
            rec = model.reconstruct(xb, m)
            loss = (((rec - xb) ** 2) * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()
    with torch.no_grad():
        emb = torch.cat([model.embed(_to(Z[s:s + 512])) for s in range(0, n, 512)]).cpu().numpy()
    return emb


def make_bt_mae_fn(emb_df, P):
    def fn(X_tr, y_tr, X_te):
        Xtr2 = pd.concat([X_tr.reset_index(drop=True), emb_df.iloc[X_tr.index].reset_index(drop=True)], axis=1)
        Xte2 = pd.concat([X_te.reset_index(drop=True), emb_df.iloc[X_te.index].reset_index(drop=True)], axis=1)
        return native_nan_oof(Xtr2, y_tr, Xte2, params=P)
    return fn


def main(smoke=False):
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    P = _general_params(y)
    print(f"=== Bidirectional Transformer on GENERAL (dev={DEV}, smoke={smoke}) ===", flush=True)

    if smoke:
        from sklearn.metrics import roc_auc_score
        Xd, _ = _drop_leaky(X.iloc[:2000], y.iloc[:2000], X.iloc[:2000]); Xpp, _ = fit_keep_nan(Xd)
        fam, nf = heatmap_families(Xpp)
        print(f"  families from heatmap: {nf} (sizes {np.bincount(fam)[:8]}...)", flush=True)
        t0 = time.time()
        tp, oof = bt_clf_heatmap_fn(X.iloc[:2000], y.iloc[:2000], X.iloc[2000:2400])
        print(f"  smoke heatmap-clf: {time.time()-t0:.1f}s | test AUC {roc_auc_score(y.iloc[2000:2400], tp):.4f} "
              f"| oof AUC {roc_auc_score(y.iloc[:2000], oof):.4f}", flush=True)
        return

    nrep = 3
    base = cv_evaluate("general", lambda a, b, c: native_nan_oof(a, b, c, params=P),
                       label="base", n_repeats=nrep, df=df, n_jobs=6)
    print(f"  base        : F1*={base.f1.mean():.4f} AUC={base.auc.mean():.4f}", flush=True)

    # plain transformer classifier (no family structure) — GPU, folds sequential
    clf = cv_evaluate("general", bt_clf_fn, label="bt_clf", n_repeats=nrep, df=df, n_jobs=1)
    cA = compare(base, clf)
    print(f"  bt_clf      : F1*={clf.f1.mean():.4f} AUC={clf.auc.mean():.4f} "
          f"Δvs_base={cA['delta_f1']:+.4f} ({cA['sigma']:+.2f}σ) [{cA['verdict']}]", flush=True)

    # HEATMAP-structured transformer: co-occurrence families -> family/segment embeddings
    clfh = cv_evaluate("general", bt_clf_heatmap_fn, label="bt_clf_heatmap", n_repeats=nrep, df=df, n_jobs=1)
    cH = compare(base, clfh); cHvsClf = compare(clf, clfh)
    print(f"  bt_clf_heat : F1*={clfh.f1.mean():.4f} AUC={clfh.auc.mean():.4f} "
          f"Δvs_base={cH['delta_f1']:+.4f} ({cH['sigma']:+.2f}σ) [{cH['verdict']}] | "
          f"Δvs_plain_BT={cHvsClf['delta_f1']:+.4f} ({cHvsClf['sigma']:+.2f}σ)", flush=True)

    rows = [{"model": "bt_classifier_plain", **cA, "auc": round(clf.auc.mean(), 4)},
            {"model": "bt_classifier_heatmap", **cH, "auc": round(clfh.auc.mean(), 4),
             "delta_f1_vs_plain": cHvsClf["delta_f1"], "sigma_vs_plain": cHvsClf["sigma"]}]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bidirectional_transformer_heatmap.csv"), index=False)
    print("\nsaved -> bidirectional_transformer_heatmap.csv", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)

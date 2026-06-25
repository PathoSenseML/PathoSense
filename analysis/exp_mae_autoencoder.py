"""BLOCK-MASKED AUTO-ENCODER (MAE) on GPU. Mask whole feature blocks (AL_ together, EK_,
CAT/AA), reconstruct the masked positions (canonical MAE objective) → self-supervised representation.
Extract the bottleneck embedding, ADD it to the general GBDT, gate F1*/AUC honestly vs deployed.
AE is UNSUPERVISED (no labels) so the embedding carries no label leak; GBDT gate uses honest label-CV.
Run: .venv/bin/python -u analysis/exp_mae_autoencoder.py [n_embed]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from config import SEED
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from schema import partition_columns
from preprocess_v2 import fit_keep_nan
from meeting_lib import native_nan_oof
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
torch.manual_seed(SEED); np.random.seed(SEED)
# GPU note (2026-06-25): the unofficial nvidia-open-dkms 610 / CUDA-13.3 driver faulted under sustained
# CUDA load (NVRM Xid 13/31/43/109 — even a bare cublasSgemm loop died). FIXED by downgrading to the
# Arch-official nvidia 595.71.05; GPU now runs the full 400ep train in ~6.5s vs ~23s on CPU (~3.5x).
# Auto-detect CUDA; force DEV=cpu via env to override. tf32-off + non-finite-skip hardening kept below.
DEV = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")
if DEV == "cuda":
    # Force full fp32 precision: TF32 matmul makes this MAE produce NaN losses on Ada (RTX 4060).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def build_matrix(Xpp):
    Z = Xpp.copy()
    for c in Z.columns:
        if str(Z[c].dtype) == "category":
            Z[c] = Z[c].cat.codes.astype(float)
    Z = Z.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    Z = StandardScaler().fit_transform(Z).astype(np.float32)
    Z = np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
    cols = list(Xpp.columns)
    blocks = [np.array([i for i, c in enumerate(cols) if str(c).startswith(pfx)])
              for pfx in (("AL_",), ("EK_",), ("CAT_", "AA_"))]
    blocks = [b for b in blocks if len(b)]
    return Z, blocks


class MAE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, k))
        self.dec = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, d))

    def forward(self, x):
        z = self.enc(x); return self.dec(z), z


def train_mae(Z, blocks, k, epochs=400):
    n, d = Z.shape
    X = torch.tensor(Z, device=DEV)
    model = MAE(d, k).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    bidx = [torch.tensor(b, device=DEV) for b in blocks]
    g = torch.Generator(device=DEV); g.manual_seed(SEED)
    last_loss = float("nan"); n_skipped = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, 256):
            bi = perm[s:s + 256]; xb = X[bi]; B = xb.shape[0]
            m = (torch.rand(B, d, generator=g, device=DEV) < 0.3).float()      # default: random 30%
            use_block = torch.rand(B, generator=g, device=DEV) < 0.5           # half the rows: block-mask
            blk_choice = torch.randint(len(bidx), (B,), generator=g, device=DEV)
            for bk in range(len(bidx)):
                rows = use_block & (blk_choice == bk)
                if rows.any():
                    mb = torch.zeros(d, device=DEV); mb[bidx[bk]] = 1.0
                    m[rows] = mb                                               # whole block masked together
            xin = xb * (1 - m)                                                 # zero masked inputs
            rec, _ = model(xin)
            loss = (((rec - xb) ** 2) * m).sum() / m.sum().clamp(min=1.0)      # loss on masked positions
            opt.zero_grad()
            if not torch.isfinite(loss):                                       # rare GPU fp32 glitch: skip the
                n_skipped += 1; continue                                       # step so it can't poison Adam
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if torch.isfinite(gn):
                opt.step(); last_loss = float(loss.item())
        if ep == 0 and not (last_loss == last_loss):                          # all of epoch 0 was non-finite
            raise RuntimeError("MAE diverged at epoch 0 — check inputs")
    if n_skipped:
        print(f"  (skipped {n_skipped} non-finite steps)", flush=True)
    model.eval()
    with torch.no_grad():
        emb = model.enc(X).cpu().numpy()
    return emb, last_loss


def main(k):
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    P = {**get_model_params(2000, spw), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
    Xd, _ = _drop_leaky(X, y, X); Xpp, _ = fit_keep_nan(Xd)
    Z, blocks = build_matrix(Xpp)
    print(f"=== GPU block-masked MAE (k={k}, dev={DEV}); d={Z.shape[1]} blocks={[len(b) for b in blocks]} ===", flush=True)
    emb, final_loss = train_mae(Z, blocks, k)
    emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
    emb_df = pd.DataFrame(emb, columns=[f"MAE_{i}" for i in range(emb.shape[1])])
    print(f"  trained (final masked-recon loss {final_loss:.4f}); active dims {int((emb.std(0)>1e-3).sum())}/{k}", flush=True)

    def base_fn(Xtr, ytr, Xte):
        return native_nan_oof(Xtr, ytr, Xte, params=P)

    def mae_fn(Xtr, ytr, Xte):
        Xtr2 = pd.concat([Xtr.reset_index(drop=True), emb_df.iloc[Xtr.index].reset_index(drop=True)], axis=1)
        Xte2 = pd.concat([Xte.reset_index(drop=True), emb_df.iloc[Xte.index].reset_index(drop=True)], axis=1)
        return native_nan_oof(Xtr2, ytr, Xte2, params=P)

    rows = []
    for seed in (SEED, SEED + 1000):
        b = cv_evaluate("general", base_fn, label="base", n_repeats=6, df=df, seed=seed)
        m = cv_evaluate("general", mae_fn, label="mae", n_repeats=6, df=df, seed=seed)
        c = compare(b, m); da = m.auc.mean() - b.auc.mean()
        print(f"  seed{seed}: base F1*={b.f1.mean():.4f} AUC={b.auc.mean():.4f} | +MAE F1*={m.f1.mean():.4f} "
              f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) AUC={m.auc.mean():.4f} ΔAUC={da:+.4f}", flush=True)
        rows.append({"seed": seed, "k": k, "base_f1": b.f1.mean(), "mae_f1": m.f1.mean(),
                     "df1": c["delta_f1"], "sigma": c["sigma"], "dauc": da})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mae_autoencoder.csv"), index=False)
    print(f"\nsaved -> mae_autoencoder.csv", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48)

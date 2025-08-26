# analysis/deep_ops.py
# Deep representation learning on per-window trajectory features
# - Builds a feature matrix from outputs/run-traj-*/per_window.csv
# - Optional temporal context (stack features from t-k .. t .. t+k)
# - Trains a small PyTorch autoencoder
# - Saves latent vectors and reconstruction error as CSV + quick plots

from __future__ import annotations
import os
import json
import math
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ---- torch (print a helpful message if missing) ----
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise RuntimeError(
        "PyTorch is required for deep_ops. Try:\n"
        "  conda install -n mdcapstone -c pytorch pytorch\n"
        "or\n"
        "  pip install torch --upgrade"
    ) from e

# plotting is optional
try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

from sklearn.preprocessing import StandardScaler

# ------------------------
# Config / model
# ------------------------

@dataclass
class AEConfig:
    latent_dim: int = 8
    hidden: Tuple[int, ...] = (128, 64)
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    patience: int = 10
    device: str = "auto"  # "cpu" | "cuda" | "mps" | "auto"
    context: int = 0      # temporal context radius k; 0 = no stacking

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        enc_layers = []
        last = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.encoder = nn.Sequential(*enc_layers, nn.Linear(last, latent_dim))

        dec_layers = []
        last = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        dec_layers += [nn.Linear(last, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

# ------------------------
# Data utilities
# ------------------------

DEFAULT_FEATURES = [
    # We'll use any of these that exist in per_window.csv
    "mean_phi", "mean_psi",
    "var_phi", "var_psi",
    "pct_disallowed",
    "mean_rama_distance",
    "sasa_mean", "sasa_std",
    "chi_rotamer_dev_mean", "chi_rotamer_dev_std",
    "ss_coil_pct", "ss_helix_pct", "ss_sheet_pct",
]

META_COLS = {"start", "end", "n_frames"}  # kept for merge / plotting

def _select_features(df: pd.DataFrame, prefer: List[str] = DEFAULT_FEATURES) -> List[str]:
    cols = [c for c in prefer if c in df.columns]
    # fallback: all numeric except obvious meta
    if not cols:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in num if c not in META_COLS]
    return cols

def _build_temporal_matrix(
    X: np.ndarray, k: int
) -> np.ndarray:
    """Stack context frames [t-k .. t .. t+k] for each t. Uses edge padding."""
    if k <= 0:
        return X
    T, D = X.shape
    out = np.zeros((T, (2 * k + 1) * D), dtype=np.float32)
    for t in range(T):
        pieces = []
        for dt in range(-k, k + 1):
            tt = min(max(t + dt, 0), T - 1)
            pieces.append(X[tt])
        out[t] = np.concatenate(pieces, axis=0)
    return out

def load_per_window(run_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    f = os.path.join(run_dir, "per_window.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f}. Run analysis/run_trajectory_ops.py first.")
    df = pd.read_csv(f)
    # sort by time
    if "start" in df.columns:
        df = df.sort_values("start").reset_index(drop=True)
    # drop all-null columns
    df = df.dropna(axis=1, how="all")
    feats = _select_features(df)
    return df, feats

def make_dataset(
    df: pd.DataFrame, feat_cols: List[str], context_k: int
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    X = df[feat_cols].astype(np.float32).fillna(0.0).values
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X).astype(np.float32)
    Xc = _build_temporal_matrix(Xn, k=context_k)
    return Xc, Xn, scaler

# ------------------------
# Training / scoring
# ------------------------

def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Metal
    if pref in ("auto", "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    # CUDA (auto)
    if pref == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_autoencoder(
    X: np.ndarray, cfg: AEConfig, seed: int = 0
) -> Tuple[AutoEncoder, List[float], List[float]]:
    torch.manual_seed(seed)
    device = pick_device(cfg.device)
    N, D = X.shape

    # small val split (unsupervised)
    n_val = max(1, int(0.1 * N))
    idx = np.arange(N)
    np.random.shuffle(idx)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr = torch.from_numpy(X[tr_idx]).to(device)
    Xva = torch.from_numpy(X[val_idx]).to(device)

    model = AutoEncoder(D, cfg.latent_dim, cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.MSELoss()

    train_ds = TensorDataset(Xtr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    best_val = math.inf
    bad = 0
    tr_hist, va_hist = [], []

    for epoch in range(cfg.epochs):
        model.train()
        tl = 0.0
        for (xb,) in train_loader:
            opt.zero_grad()
            xhat, _ = model(xb)
            loss = crit(xhat, xb)
            loss.backward()
            opt.step()
            tl += float(loss.item()) * xb.size(0)
        tl /= len(train_ds)
        tr_hist.append(tl)

        model.eval()
        with torch.no_grad():
            xhat, _ = model(Xva)
            vl = float(crit(xhat, Xva).item())
        va_hist.append(vl)

        if vl < best_val - 1e-6:
            best_val = vl
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        # print concise progress
        print(f"[AE] epoch {epoch+1:03d}/{cfg.epochs}  train={tl:.5f}  val={vl:.5f}  (device={device})")
        if bad >= cfg.patience:
            print("[AE] Early stopping")
            break

    if best_val < math.inf:
        model.load_state_dict(best_state)
    return model, tr_hist, va_hist

def encode_and_score(model: AutoEncoder, X: np.ndarray, device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    xt = torch.from_numpy(X).to(device)
    model.eval()
    with torch.no_grad():
        xhat, z = model(xt)
        # per-sample MSE
        err = torch.mean((xhat - xt) ** 2, dim=1).cpu().numpy()
        Z = z.cpu().numpy()
    return Z, err

# ------------------------
# Orchestration
# ------------------------

def run_deep(
    run_dir: str,
    cfg: AEConfig = AEConfig(),
) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    outd = os.path.join(run_dir, "deep")
    os.makedirs(outd, exist_ok=True)

    df, feat_cols = load_per_window(run_dir)
    Xc, Xn, scaler = make_dataset(df, feat_cols, cfg.context)

    model, tr_hist, va_hist = train_autoencoder(Xc, cfg, seed=0)
    Z, err = encode_and_score(model, Xc)

    # Save artifacts
    # 1) scaler + cfg + feature list + model
    with open(os.path.join(outd, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(outd, "config.json"), "w") as f:
        json.dump({
            "latent_dim": cfg.latent_dim,
            "hidden": list(cfg.hidden),
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "device": str(pick_device(cfg.device)),
            "context": cfg.context,
            "feature_columns": feat_cols,
        }, f, indent=2)
    torch.save(model.state_dict(), os.path.join(outd, "ae.pt"))

    # 2) CSV outputs
    lat_df = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(Z.shape[1])])
    lat_df["t"] = np.arange(len(Z))
    lat_df.to_csv(os.path.join(outd, "latent.csv"), index=False)

    err_df = pd.DataFrame({"t": np.arange(len(err)), "deep_recon_error": err})
    err_df.to_csv(os.path.join(outd, "recon_error.csv"), index=False)

    # 3) Merge back into per_window for convenient downstream use
    merged = df.copy()
    merged["deep_recon_error"] = err
    for i in range(Z.shape[1]):
        merged[f"z{i+1}"] = Z[:, i]
    merged.to_csv(os.path.join(run_dir, "per_window_deep.csv"), index=False)

    # 4) Small plots
    if _HAVE_PLT:
        try:
            # loss curves
            plt.figure()
            plt.plot(tr_hist, label="train")
            plt.plot(va_hist, label="val")
            plt.xlabel("epoch"); plt.ylabel("MSE")
            plt.title("Autoencoder loss")
            plt.legend()
            plt.savefig(os.path.join(outd, "loss_curves.png"), bbox_inches="tight")
            plt.close()

            # recon error over time
            if "start" in df.columns:
                xs = df["start"].values
            else:
                xs = np.arange(len(err))
            plt.figure()
            plt.plot(xs, err)
            plt.xlabel("window index" if "start" not in df.columns else "start frame")
            plt.ylabel("reconstruction MSE")
            plt.title("Deep anomaly score over time")
            plt.savefig(os.path.join(outd, "recon_over_time.png"), bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    print(f"[deep_ops] Wrote:\n  - {os.path.join(outd, 'latent.csv')}\n  - {os.path.join(outd, 'recon_error.csv')}\n  - {os.path.join(run_dir, 'per_window_deep.csv')}")
    return {
        "run_dir": run_dir,
        "out_dir": outd,
        "n_samples": len(err),
        "features": feat_cols,
    }

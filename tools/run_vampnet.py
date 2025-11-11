# tools/run_vampnet.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import deeptime as dt
from deeptime.util.torch import MLP
from deeptime.util.data import TrajectoryDataset
from deeptime.decomposition.deep import VAMPNet

# Import optimized utilities
from utils import (
    minmax_normalize,
    map_to_active_set,
    compute_transition_surprise,
    load_features_cached,
    compute_frame_scores,
    ProgressBar
)


def compute_vampnet_scores(chi, hard, msm, lag, T):
    """
    Optimized computation of VAMPNet anomaly scores.
    
    Args:
        chi: Soft assignments (T x n_states)
        hard: Hard state assignments (T,)
        msm: Fitted MSM model
        lag: Lag time
        T: Total number of frames
    
    Returns:
        Combined frame scores
    """
    # Map to active set
    count_model = msm.count_model
    active_states = np.arange(msm.n_states)
    active_symbols = count_model.states_to_symbols(active_states)
    sym2active = {int(sym): int(i) for i, sym in enumerate(active_symbols)}
    mapped = np.array([sym2active.get(int(s), -1) for s in hard], dtype=np.int64)
    mask = mapped >= 0
    
    # Rarity scores
    pi = msm.stationary_distribution
    rarity = np.ones(T, dtype=np.float32)
    rarity[mask] = 1.0 - pi[mapped[mask]]
    
    # Transition surprise (optimized)
    trans = compute_transition_surprise(mapped, msm.transition_matrix, lag, mask)
    
    # Combine scores
    return compute_frame_scores(rarity, trans, np.zeros_like(rarity))


def main(feat_path, out_dir, lag, n_states, batch_size, epochs, device_str, use_cache=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----- Load features (T x F)
    print(f"[1/6] Loading features from {feat_path}")
    cache_dir = out / ".cache" if use_cache else None
    X = load_features_cached(feat_path, cache_dir=cache_dir)
    X = X.astype(np.float32)
    T, F = X.shape
    print(f"  Loaded features: shape=({T}, {F})")

    # ----- Time-lagged dataset (positional args per deeptime API)
    print(f"[2/6] Creating time-lagged dataset with lag={lag}")
    dataset = TrajectoryDataset(lag, X)
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    n_train = max(1, n - n_val)

    # Deterministic split
    gen = torch.Generator().manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=gen
    )

    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_val = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    print(f"  Train pairs: {n_train}, Validation pairs: {n_val}")

    # ----- VAMPNet lobe (MLP -> n_states)
    print(f"[3/6] Building VAMPNet with n_states={n_states}")
    lobe = MLP(units=[F, 128, 128, n_states], nonlinearity=nn.ReLU)
    device = torch.device(device_str)
    vampnet = VAMPNet(lobe=lobe, learning_rate=1e-3, device=device)
    print(f"  Device: {device}, Architecture: {F} -> 128 -> 128 -> {n_states}")

    # ----- Train
    print(f"[4/6] Training VAMPNet for {epochs} epochs")
    model = vampnet.fit(
        loader_train, n_epochs=epochs, validation_loader=loader_val
    ).fetch_model()
    print("  Training complete")

    # ----- Soft assignments χ (T x n_states)
    print("[5/6] Computing soft assignments and MSM")
    with torch.no_grad():
        chi = model.transform(X)  # numpy float array

    # ----- Discretize to hard labels (integers) for MSM
    hard = chi.argmax(1).astype(np.int64)

    # ----- Estimate MSM on largest connected set
    msm_est = dt.markov.msm.MaximumLikelihoodMSM(reversible=True, lagtime=lag)
    msm = msm_est.fit([hard]).fetch_model()
    print(f"  MSM: {msm.n_states} active states, timescale={msm.timescales()[0]:.2f}")

    # ----- Frame-level scores (optimized computation)
    print("[6/6] Computing anomaly scores")
    score100 = compute_vampnet_scores(chi, hard, msm, lag, T)

    # ----- Write outputs
    pd.DataFrame({
        "frame": np.arange(T),
        "score": score100
    }).to_csv(out / "frame_scores.csv", index=False)
    np.save(out / "soft_assignments.npy", chi)

    print(f"[ok] Pipeline complete! Results written to {out}")
    print(f"  - frame_scores.csv: {T} frames")
    print(f"  - soft_assignments.npy: shape={chi.shape}")
    print(f"  - Score statistics: mean={score100.mean():.2f}, std={score100.std():.2f}, max={score100.max():.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optimized VAMPNet anomaly detection pipeline")
    ap.add_argument("--features", required=True, help="features.npy or CSV (T x F)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--lag", type=int, default=20, help="Lag time (try 10–30 for ~200 frames)")
    ap.add_argument("--n_states", type=int, default=4, help="Number of states (keep small for short trajectories)")
    ap.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    ap.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    ap.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    ap.add_argument("--no_cache", action="store_true", help="Disable feature caching")
    args = ap.parse_args()
    main(args.features, args.out_dir, args.lag, args.n_states, 
         args.batch_size, args.epochs, args.device, use_cache=not args.no_cache)

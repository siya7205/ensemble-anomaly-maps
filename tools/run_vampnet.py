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

def _01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mn, mx = x.min(), x.max()
    return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn)

def main(feat_path, out_dir, lag, n_states, batch_size, epochs, device_str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----- Load features (T x F)
    X = np.load(feat_path) if feat_path.endswith(".npy") \
        else np.loadtxt(feat_path, delimiter=",", skiprows=1)
    X = X.astype(np.float32)
    T, F = X.shape

    # ----- Time-lagged dataset (positional args per deeptime API)
    # Creates (x_t, x_{t+lag}) pairs internally.
    dataset = TrajectoryDataset(lag, X)  # NOTE: no "data=" kwarg
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    n_train = max(1, n - n_val)

    # Deterministic split
    gen = torch.Generator().manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)

    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_val   = DataLoader(val_set,   batch_size=len(val_set), shuffle=False)

    # ----- VAMPNet lobe (MLP -> n_states)
    lobe = MLP(units=[F, 128, 128, n_states], nonlinearity=nn.ReLU)
    device = torch.device(device_str)

    # Constructor takes lobe / device / lr; batching is via DataLoader in .fit()
    vampnet = VAMPNet(lobe=lobe, learning_rate=1e-3, device=device)  #  [oai_citation:1‡DeepTime ML](https://deeptime-ml.github.io/latest/notebooks/examples/ala2-example.html?utm_source=chatgpt.com)

    # ----- Train
    model = vampnet.fit(loader_train, n_epochs=epochs, validation_loader=loader_val).fetch_model()

    # ----- Soft assignments χ (T x n_states)
    with torch.no_grad():
        chi = model.transform(X)  # numpy float array

    # ----- Discretize to hard labels (integers) for MSM
    hard = chi.argmax(1).astype(np.int64)

    # ----- Estimate MSM on largest connected set
    msm_est = dt.markov.msm.MaximumLikelihoodMSM(reversible=True, lagtime=lag)  #  [oai_citation:2‡DeepTime ML](https://deeptime-ml.github.io/latest/api/generated/deeptime.markov.msm.MaximumLikelihoodMSM.html?utm_source=chatgpt.com)
    msm = msm_est.fit([hard]).fetch_model()

    # ----- Map original labels -> MSM active-set indices
    count_model = msm.count_model
    active_states = np.arange(msm.n_states)  # 0..n_active-1
    # original symbols (labels) present in the active set:
    active_symbols = count_model.states_to_symbols(active_states)       #  [oai_citation:3‡DeepTime ML](https://deeptime-ml.github.io/latest/api/generated/deeptime.markov.msm.MarkovStateModel.html?utm_source=chatgpt.com)
    sym2active = {int(sym): int(i) for i, sym in enumerate(active_symbols)}
    mapped = np.array([sym2active.get(int(s), -1) for s in hard], dtype=np.int64)
    mask = mapped >= 0

    # ----- Frame-level scores: rarity + transition surprise (smeared over lag)
    pi = msm.stationary_distribution                         # length = n_active
    rarity = np.ones(len(hard), dtype=np.float32)
    rarity[mask] = 1.0 - pi[mapped[mask]]

    Tmat = msm.transition_matrix
    trans = np.zeros(len(hard), dtype=np.float32)
    if T > lag:
        valid_pair = mask[:-lag] & mask[lag:]
        i = mapped[:-lag][valid_pair]
        j = mapped[lag:][valid_pair]
        jump = -np.log(np.maximum(Tmat[i, j], 1e-12))
        for t0, cost in zip(np.nonzero(valid_pair)[0], jump):
            trans[t0:t0 + lag] += cost / lag

    score100 = 100.0 * np.median(np.vstack([_01(rarity), _01(trans)]), axis=0)

    # ----- Write outputs for your coloring/viewer
    pd.DataFrame({"frame": np.arange(T), "score": score100}).to_csv(out / "frame_scores.csv", index=False)
    np.save(out / "soft_assignments.npy", chi)

    print(f"[ok] wrote {out/'frame_scores.csv'} and {out/'soft_assignments.npy'}")
    print(f"[info] T={T}, F={F}, lag={lag}, n_states={n_states}, train_pairs={n_train}, val_pairs={n_val}, active_states={msm.n_states}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.npy or CSV (T x F)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lag", type=int, default=20, help="try 10–30 for ~200 frames")
    ap.add_argument("--n_states", type=int, default=4, help="keep small on short trajectories")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = ap.parse_args()
    main(args.features, args.out_dir, args.lag, args.n_states, args.batch_size, args.epochs, args.device)

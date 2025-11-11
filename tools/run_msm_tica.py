# tools/run_msm_tica.py
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pyemma
from sklearn.cluster import KMeans
from deeptime.decomposition import TICA  # modern TICA

# Import optimized utilities
from utils import (
    minmax_normalize,
    map_to_active_set,
    compute_transition_surprise,
    load_features_cached,
    compute_local_density,
    compute_frame_scores,
    ProgressBar
)

def main(feat_path, out_dir, lag_tica, lag_msm, n_clusters, use_cache=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/7] Loading features from {feat_path}")
    # Use cached loading for faster repeated runs
    cache_dir = out / ".cache" if use_cache else None
    X = load_features_cached(feat_path, cache_dir=cache_dir)
    print(f"  Loaded features: shape={X.shape}, dtype={X.dtype}")

    # ---- TICA (deeptime)
    print(f"[2/7] Running TICA with lag={lag_tica}")
    tica = TICA(lagtime=lag_tica).fit(X).fetch_model()
    Y = tica.transform(X)  # [T, d]
    np.save(out / "tica_coords.npy", Y)
    print(f"  TICA output: shape={Y.shape}")

    # ---- KMeans clustering in TICA space
    print(f"[3/7] KMeans clustering with n_clusters={n_clusters}")
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    dtraj = km.fit_predict(Y).astype(np.int64)  # labels 0..n_clusters-1
    np.save(out / "dtraj.npy", dtraj)
    print(f"  Clustering complete: unique_labels={len(np.unique(dtraj))}")

    # ---- MSM on largest connected set
    print(f"[4/7] Estimating MSM with lag={lag_msm}")
    msm = pyemma.msm.estimate_markov_model(
        [dtraj], lag=lag_msm, reversible=True, connectivity='largest'
    )
    np.save(out / "P.npy", msm.P)
    np.save(out / "pi.npy", msm.pi)
    print(f"  MSM complete: n_states={msm.n_states}, largest_eigenvalue={msm.timescales()[0]:.2f}")

    # ---- Validation (best-effort plots)
    print("[5/7] Generating validation plots")
    lags = np.unique(np.linspace(
        max(1, lag_msm // 2),
        min(len(dtraj) // 2, lag_msm * 5),
        6,
        dtype=int
    ))
    try:
        its = pyemma.msm.its([dtraj], lags=lags, errors='bayes')
        plt.figure()
        pyemma.plots.plot_implied_timescales(its)
        plt.tight_layout()
        plt.savefig(out / "its.png", dpi=180)
        plt.close()
        print("  ITS plot saved")
    except Exception as e:
        print(f"  [warn] ITS plotting failed: {e}")

    try:
        ck = msm.cktest(2)
        pyemma.plots.plot_cktest(ck)
        plt.legend(loc='upper center', ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(out / "cktest.png", dpi=180)
        plt.close()
        print("  CK-test plot saved")
    except Exception as e:
        print(f"  [warn] CK plotting failed: {e}")

    # ---- Map labels to active-set indices (critical!)
    print("[6/7] Computing anomaly scores")
    dmap = map_to_active_set(dtraj, msm.active_set, n_clusters)
    mask = dmap >= 0  # frames that live in the active set

    # ---- Scores: rarity
    rarity = np.ones(len(dtraj), dtype=np.float64)
    rarity[mask] = 1.0 - msm.pi[dmap[mask]]

    # ---- Transition surprise (optimized vectorized version)
    trans = compute_transition_surprise(dmap, msm.P, lag_msm, mask)

    # ---- Local density (optimized with parallel k-NN)
    dens = compute_local_density(Y, n_neighbors=20)

    # ---- Combine scores using optimized function
    score100 = compute_frame_scores(rarity, trans, dens)

    # ---- Save results
    print("[7/7] Saving results")
    import pandas as pd
    pd.DataFrame({
        'frame': np.arange(len(score100)),
        'score': score100
    }).to_csv(out / 'frame_scores.csv', index=False)
    
    print(f"[ok] Pipeline complete! Results written to {out}")
    print(f"  - frame_scores.csv: {len(score100)} frames")
    print(f"  - Score statistics: mean={score100.mean():.2f}, std={score100.std():.2f}, max={score100.max():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optimized MSM+TICA anomaly detection pipeline")
    ap.add_argument("--features", required=True, help="Path to features file (.npy or .csv)")
    ap.add_argument("--out_dir", required=True, help="Output directory for results")
    ap.add_argument("--lag_tica", type=int, default=10, help="TICA lag time")
    ap.add_argument("--lag_msm", type=int, default=30, help="MSM lag time")
    ap.add_argument("--n_clusters", type=int, default=30, help="Number of KMeans clusters")
    ap.add_argument("--no_cache", action="store_true", help="Disable feature caching")
    args = ap.parse_args()
    main(args.features, args.out_dir, args.lag_tica, args.lag_msm, args.n_clusters, 
         use_cache=not args.no_cache)

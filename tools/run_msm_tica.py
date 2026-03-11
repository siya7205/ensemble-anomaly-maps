# tools/run_msm_tica.py
import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from deeptime.decomposition import TICA  # modern TICA
from deeptime.markov.msm import MaximumLikelihoodMSM  # modern MSM

# Suppress deeptime's per-state-set verbose warnings (logged at WARNING level
# via logging.getLogger(__file__) inside deeptime's MLMSM estimator).  We
# target only that specific logger so other logging is not affected.
import deeptime.markov.msm._maximum_likelihood_msm as _deeptime_mlmsm
_deeptime_msm_log = logging.getLogger(_deeptime_mlmsm.__file__)
_deeptime_msm_log.setLevel(logging.ERROR)

# Import optimized utilities
from utils import (
    minmax_normalize,
    compute_transition_surprise,
    load_features_cached,
    compute_local_density,
    compute_frame_scores,
    ProgressBar
)


def estimate_transition_matrix(dtraj, lag, n_states=None, reversible=True):
    """
    Estimate transition matrix from discrete trajectory using deeptime.
    
    Args:
        dtraj: Discrete trajectory (array of state indices)
        lag: Lag time for MSM estimation
        n_states: Number of states (if None, inferred from dtraj)
        reversible: Whether to enforce detailed balance
    
    Returns:
        P: Transition matrix
        pi: Stationary distribution
        active_set: Active states (in original cluster indices)
        msm: The fitted MSM model
    """
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM
    from deeptime.markov.tools.estimation import largest_connected_set
    
    # Count transitions
    count_model = TransitionCountEstimator(lagtime=lag, count_mode='sliding').fit_fetch([dtraj])
    
    # Find the largest connected set
    lcs = largest_connected_set(count_model.count_matrix)
    
    # Map back to original state indices
    original_active_states = count_model.states[lcs]
    
    # Estimate MSM (it will use the connected submatrix internally).
    # (_deeptime_msm_log is already set to ERROR at module level to suppress
    # the per-state-set "Skipping state set" warnings.)
    msm = MaximumLikelihoodMSM(reversible=reversible).fit_fetch(count_model)
    
    return msm.transition_matrix, msm.stationary_distribution, original_active_states, msm


def map_to_active_set(dtraj, active_set, n_clusters):
    """
    Vectorized mapping of cluster labels to active set indices.
    
    Args:
        dtraj: Original discrete trajectory (cluster labels)
        active_set: Active state indices from MSM
        n_clusters: Total number of clusters
    
    Returns:
        Mapped trajectory with -1 for inactive states
    """
    mapping = -np.ones(n_clusters, dtype=np.int64)
    active_set = np.asarray(active_set, dtype=np.int64)
    mapping[active_set] = np.arange(len(active_set))
    return mapping[np.asarray(dtraj, dtype=np.int64)]


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
    # Save eigenvectors (right singular vectors) for downstream metric computation
    np.savez(out / "tica_model.npz", eigenvectors=tica.singular_vectors_right)
    print(f"  TICA output: shape={Y.shape}")

    # ---- KMeans clustering in TICA space
    print(f"[3/7] KMeans clustering with n_clusters={n_clusters}")
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    dtraj = km.fit_predict(Y).astype(np.int64)  # labels 0..n_clusters-1
    np.save(out / "dtraj.npy", dtraj)
    print(f"  Clustering complete: unique_labels={len(np.unique(dtraj))}")

    # ---- MSM on largest connected set (using deeptime instead of pyemma)
    print(f"[4/7] Estimating MSM with lag={lag_msm}")
    P, pi, active_set, msm = estimate_transition_matrix(dtraj, lag_msm, n_clusters, reversible=True)
    np.save(out / "P.npy", P)
    np.save(out / "pi.npy", pi)
    print(f"  MSM complete: n_states={len(active_set)}, timescales={msm.timescales()[:3]}")

    # ---- Validation (simplified plots without pyemma)
    print("[5/7] Generating validation plots")
    try:
        # Plot implied timescales
        from deeptime.markov import TransitionCountEstimator
        lags = np.unique(np.linspace(
            max(1, lag_msm // 2),
            min(len(dtraj) // 2, lag_msm * 5),
            6,
            dtype=int
        ))
        
        timescales_list = []
        for lag_test in lags:
            try:
                counts = TransitionCountEstimator(lagtime=lag_test, count_mode='sliding').fit_fetch([dtraj])
                msm_test = MaximumLikelihoodMSM(reversible=True).fit_fetch(counts)
                ts = msm_test.timescales()
                timescales_list.append(ts[:min(5, len(ts))])
            except Exception:
                continue
        
        if timescales_list:
            plt.figure(figsize=(8, 5))
            timescales_arr = np.array([list(t) + [np.nan]*(5-len(t)) for t in timescales_list])
            valid_lags = lags[:len(timescales_list)]
            for i in range(min(5, timescales_arr.shape[1])):
                plt.plot(valid_lags, timescales_arr[:, i], 'o-', label=f'ITS {i+1}')
            plt.xlabel('Lag time')
            plt.ylabel('Implied timescale')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "its.png", dpi=180)
            plt.close()
            print("  ITS plot saved")
    except Exception as e:
        print(f"  [warn] ITS plotting failed: {e}")

    # ---- Map labels to active-set indices (critical!)
    print("[6/7] Computing anomaly scores")
    dmap = map_to_active_set(dtraj, active_set, n_clusters)
    mask = dmap >= 0  # frames that live in the active set

    # ---- Scores: rarity
    rarity = np.ones(len(dtraj), dtype=np.float64)
    rarity[mask] = 1.0 - pi[dmap[mask]]

    # ---- Transition surprise (optimized vectorized version)
    trans = compute_transition_surprise(dmap, P, lag_msm, mask)

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

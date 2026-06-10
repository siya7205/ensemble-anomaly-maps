"""
===============================================================
Module: msm/pipeline.py
Description: tICA generation, clustering, and MSM artifact construction.
Author: Siya Jethliya
Project: Ensemble Anomaly Maps
===============================================================
"""

from pathlib import Path

import numpy as np
from deeptime.clustering import KMeans
from deeptime.decomposition import TICA
from deeptime.markov.msm import MaximumLikelihoodMSM

MIN_FRAMES_FOR_MSM = 4


# ------------------------------
# Run TICA Projection
# ------------------------------
def run_tica(feature_matrix, lag=10, dim=5):
    if len(feature_matrix) < MIN_FRAMES_FOR_MSM:
        raise ValueError(f"At least {MIN_FRAMES_FOR_MSM} frames are required for stable tICA estimation.")
    # Keep lag below one quarter of trajectory length for stable covariance estimates.
    lag = min(lag, len(feature_matrix) // 4)
    lag = max(lag, 1)
    tica_model = TICA(lagtime=lag, dim=dim).fit(feature_matrix).fetch_model()
    tica_coords = tica_model.transform(feature_matrix)
    return tica_coords, tica_model


# ------------------------------
# Cluster TICA States
# ------------------------------
def cluster_states(tica_coords, n_clusters=20, seed=42):
    if len(tica_coords) < MIN_FRAMES_FOR_MSM:
        raise ValueError(f"At least {MIN_FRAMES_FOR_MSM} frames are required for meaningful clustering.")
    # Prevent over-clustering by capping clusters to half the number of frames.
    n_clusters = min(n_clusters, len(tica_coords) // 2)
    n_clusters = max(n_clusters, 2)
    kmeans_model = (
        KMeans(n_clusters=n_clusters, max_iter=100, n_jobs=1, fixed_seed=seed)
        .fit(tica_coords)
        .fetch_model()
    )
    dtraj = kmeans_model.transform(tica_coords).astype(np.int64)
    return dtraj, kmeans_model


# ------------------------------
# Build Markov State Model
# ------------------------------
def build_msm(dtraj, lag=10):
    lag = min(lag, len(dtraj) // 4)
    lag = max(lag, 1)
    msm = MaximumLikelihoodMSM(lagtime=lag, reversible=True).fit(dtraj).fetch_model()
    return msm, msm.transition_matrix, msm.stationary_distribution


# ------------------------------
# Save MSM Artifacts
# ------------------------------
def save_msm_artifacts(artifacts_dir, tica_coords, dtraj, transition_matrix, stationary_distribution):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "tica_coords.npy", tica_coords)
    np.save(artifacts_dir / "dtraj.npy", dtraj)
    np.save(artifacts_dir / "P.npy", transition_matrix)
    np.save(artifacts_dir / "pi.npy", stationary_distribution)

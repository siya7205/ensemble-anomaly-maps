"""
===============================================================
Module: scoring/anomaly_v2.py
Description: Multi-signal anomaly scoring using kinetic and density signals.
Author: Siya Jethliya
Project: Ensemble Anomaly Maps
===============================================================
"""

import numpy as np
from scipy.ndimage import median_filter
from sklearn.neighbors import NearestNeighbors


# ------------------------------
# Rank Normalize Signal
# ------------------------------
def rank_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    if len(values) == 1:
        return np.zeros_like(values)
    if np.all(values == values[0]):
        return np.zeros_like(values)
    ranks = np.argsort(np.argsort(values))
    return ranks / (len(values) - 1)


# ------------------------------
# Quantile Normalize Signal
# ------------------------------
def quantile_normalize(values, lower=0.01, upper=0.99):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    q_low = np.quantile(values, lower)
    q_high = np.quantile(values, upper)
    if q_high <= q_low:
        return np.zeros_like(values)
    return np.clip((values - q_low) / (q_high - q_low), 0, 1)


# ------------------------------
# Smooth Time Series
# ------------------------------
def moving_median(values, window=5):
    if window < 2:
        return values
    if window % 2 == 0:
        window += 1
    return median_filter(values, size=window, mode="nearest")


# ------------------------------
# Compute Kinetic Signals
# ------------------------------
def compute_kinetic_signals(msm, dtraj, lag_msm):
    n_frames = len(dtraj)
    pi = msm.stationary_distribution
    P = msm.transition_matrix
    n_states = msm.n_states

    rarity = np.ones(n_frames, dtype=np.float64)
    for t in range(n_frames):
        state = dtraj[t]
        if 0 <= state < n_states:
            rarity[t] = 1.0 - pi[state]

    surprise = np.zeros(n_frames, dtype=np.float64)
    epsilon = 1e-12
    for t in range(n_frames - lag_msm):
        state_from, state_to = dtraj[t], dtraj[t + lag_msm]
        if 0 <= state_from < n_states and 0 <= state_to < n_states:
            surprise[t] = -np.log(max(P[state_from, state_to], epsilon))

    return rarity, surprise


# ------------------------------
# Compute Local Density Signal
# ------------------------------
def compute_local_density_signal(tica_coords, k=20):
    k = min(k, len(tica_coords) - 1)
    if k < 1:
        return np.zeros(len(tica_coords))
    model = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(tica_coords)
    distances, _ = model.kneighbors(tica_coords)
    return -distances.mean(axis=1)


# ------------------------------
# Fuse Signals
# ------------------------------
def fuse_signals(signals, method="median", normalize_method="rank"):
    normalized = {}
    for name, signal in signals.items():
        if normalize_method == "quantile":
            normalized[name] = quantile_normalize(signal)
        else:
            normalized[name] = rank_normalize(signal)

    stacked = np.column_stack([normalized[name] for name in signals.keys()])
    if method == "mean":
        fused = np.mean(stacked, axis=1)
    else:
        fused = np.median(stacked, axis=1)

    return fused, normalized

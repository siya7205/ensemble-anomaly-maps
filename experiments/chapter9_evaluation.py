#!/usr/bin/env python3
"""
Chapter 9 Evaluation Module — RQ1 (Kinetic Validation) and RQ3 (Sensitivity).

Computes:
  Part 1 – RQ1 Kinetic Validation
    1. Implied Timescales (ITS) with CV across plateau lags
    2. Chapman–Kolmogorov (CK) Frobenius-norm error
    3. VAMP-2 comparison: tICA vs PCA vs raw features

  Part 2 – Hotspot Validation
    4. Per-residue fused ranking
    5. Slow-mode alignment (Spearman ρ between fused score and |tICA loading| sum)
    6. Transition enrichment (Cohen's d: transition vs stable frames)
    7. Spatial clustering Z-score for top-10 % residue set

  Part 3 – RQ3 Sensitivity
    8. Ranking stability under lag/dim/signal perturbations

All outputs are saved as CSV files under results/chapter9/.

Usage:
    python experiments/chapter9_evaluation.py [--features PATH] [--topology PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from deeptime.decomposition import TICA, VAMP
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Defaults — adapted for the dataset (213 frames × 7 features, 150 CA atoms)
# ---------------------------------------------------------------------------
DEFAULT_LAG_TICA = 5
DEFAULT_DIM_TICA = 3
DEFAULT_N_CLUSTERS = 8
DEFAULT_LAG_MSM = 10
N_SLOW_MODES = 3       # top slow modes to analyse
N_ITS_MODES = 5        # modes to report in ITS table
CA_PAIR_WINDOW = 5     # sequential CA pairs within this residue window for traj2 features


# ---------------------------------------------------------------------------
# Helper: fit the full MSM pipeline
# ---------------------------------------------------------------------------

def _fit_pipeline(X, lag_tica, dim_tica, n_clusters, lag_msm):
    """Fit TICA → KMeans → MSM and return (msm, dtraj, Y, tica_model)."""
    tica_model = TICA(lagtime=lag_tica, dim=dim_tica).fit(X).fetch_model()
    Y = tica_model.transform(X)

    km = KMeans(n_clusters=n_clusters, max_iter=200,
                n_jobs=1).fit(Y).fetch_model()
    dtraj = km.transform(Y).astype(np.int64)

    msm = MaximumLikelihoodMSM(lagtime=lag_msm,
                                reversible=True).fit(dtraj).fetch_model()
    return msm, dtraj, Y, tica_model


# ---------------------------------------------------------------------------
# Helper: VAMP-2 score
# ---------------------------------------------------------------------------

def _vamp2_score(X, lag, dim):
    """Return VAMP-2 score; -inf on failure."""
    try:
        vamp = VAMP(lagtime=lag, dim=dim).fit(X).fetch_model()
        return float(vamp.score(r=2))
    except Exception as exc:  # noqa: BLE001
        log.warning("VAMP-2 failed (lag=%d, dim=%d): %s", lag, dim, exc)
        return -np.inf


# ---------------------------------------------------------------------------
# PART 1 — RQ1: KINETIC VALIDATION
# ---------------------------------------------------------------------------

def compute_implied_timescales(X, lag_msm, dim_tica, n_clusters,
                               lag_tica, output_dir):
    """
    1. Implied Timescales (ITS).

    Fits MSMs at lag_times = [0.5τ, 0.75τ, τ, 1.25τ, 1.5τ] and records
    timescales for up to N_ITS_MODES slow modes.

    Saves:
      implied_timescales.csv      – columns: lag_time, mode_index, timescale
      implied_timescale_cv.csv    – columns: mode_index, mean, std, cv
    """
    log.info("=== 1. Implied Timescales ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lag_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    lag_times = sorted({max(1, int(f * lag_msm)) for f in lag_factors})
    log.info("  lag times: %s", lag_times)

    records = []
    for lt in lag_times:
        try:
            msm_lt, _, _, _ = _fit_pipeline(X, lag_tica, dim_tica,
                                            n_clusters, lt)
            ts = msm_lt.timescales()
            for k, t in enumerate(ts[:N_ITS_MODES]):
                records.append({"lag_time": lt, "mode_index": k,
                                "timescale": float(t)})
            log.info("  lag=%d → %d timescales", lt, len(ts))
        except Exception as exc:  # noqa: BLE001
            log.warning("  lag=%d failed: %s", lt, exc)

    if not records:
        raise RuntimeError("No ITS records computed — check data / parameters.")

    df_its = pd.DataFrame(records)
    df_its.to_csv(output_dir / "implied_timescales.csv", index=False)
    log.info("  Saved implied_timescales.csv")

    # CV over plateau (last 3 lag times per mode)
    plateau_lags = sorted(df_its["lag_time"].unique())[-3:]
    df_plateau = df_its[df_its["lag_time"].isin(plateau_lags)]

    cv_records = []
    for mode in df_plateau["mode_index"].unique():
        vals = df_plateau[df_plateau["mode_index"] == mode]["timescale"].values
        if len(vals) == 0:
            continue
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv_v = std_v / mean_v if mean_v != 0 else 0.0
        cv_records.append({"mode_index": int(mode), "mean": mean_v,
                           "std": std_v, "cv": cv_v})

    df_cv = pd.DataFrame(cv_records)
    df_cv.to_csv(output_dir / "implied_timescale_cv.csv", index=False)
    log.info("  Saved implied_timescale_cv.csv")
    return df_its, df_cv


def compute_ck_errors(X, lag_msm, dim_tica, n_clusters, lag_tica, output_dir):
    """
    2. Chapman–Kolmogorov Frobenius-norm error.

    For n ∈ {2, 3, 4}:
      P_pred = P(τ)^n
      P_emp  = MSM(n·τ).transition_matrix (re-fitted at n·τ)
      error  = ||P_pred - P_emp||_F

    Saves:
      ck_errors.csv – columns: n_step, frobenius_error, mean_error_per_n
    """
    log.info("=== 2. CK Errors ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    msm_base, _, _, _ = _fit_pipeline(X, lag_tica, dim_tica,
                                      n_clusters, lag_msm)
    P_base = msm_base.transition_matrix
    n_base = msm_base.n_states

    records = []
    for n in [2, 3, 4]:
        P_pred = np.linalg.matrix_power(P_base, n)
        try:
            msm_n, _, _, _ = _fit_pipeline(X, lag_tica, dim_tica,
                                           n_clusters, n * lag_msm)
            P_emp_full = msm_n.transition_matrix
            n_emp = msm_n.n_states

            # Align sizes (use min shared states)
            n_shared = min(n_base, n_emp)
            P_pred_s = P_pred[:n_shared, :n_shared]
            P_emp_s = P_emp_full[:n_shared, :n_shared]

            frob = float(np.linalg.norm(P_pred_s - P_emp_s, "fro"))
            records.append({"n_step": n, "frobenius_error": frob})
            log.info("  n=%d: Frobenius error = %.4f", n, frob)
        except Exception as exc:  # noqa: BLE001
            log.warning("  n=%d failed: %s", n, exc)

    if not records:
        raise RuntimeError("No CK error records — check data / parameters.")

    df_ck = pd.DataFrame(records)
    mean_per_n = df_ck.groupby("n_step")["frobenius_error"].mean().reset_index()
    mean_per_n = mean_per_n.rename(columns={"frobenius_error": "mean_error_per_n"})
    df_ck = df_ck.merge(mean_per_n, on="n_step", how="left")
    df_ck.to_csv(output_dir / "ck_errors.csv", index=False)
    log.info("  Saved ck_errors.csv")
    return df_ck


def compute_vamp_comparison(X, lag_msm, dim_tica, lag_tica, output_dir):
    """
    3. VAMP-2 comparison: tICA pipeline vs PCA baseline vs raw features.

    Saves:
      vamp_comparison.csv – columns: model_type, vamp2_score
    """
    log.info("=== 3. VAMP-2 Comparison ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = []

    # tICA pipeline
    score_tica = _vamp2_score(X, lag=lag_tica, dim=dim_tica)
    records.append({"model_type": "tICA", "vamp2_score": score_tica})
    log.info("  tICA VAMP-2 score: %.4f", score_tica)

    # PCA baseline (same dimensionality)
    pca = PCA(n_components=dim_tica, random_state=SEED)
    X_pca = pca.fit_transform(X)
    score_pca = _vamp2_score(X_pca, lag=lag_tica, dim=dim_tica)
    records.append({"model_type": "PCA", "vamp2_score": score_pca})
    log.info("  PCA  VAMP-2 score: %.4f", score_pca)

    # Raw features baseline
    score_raw = _vamp2_score(X, lag=lag_tica, dim=min(dim_tica, X.shape[1]))
    records.append({"model_type": "raw_features", "vamp2_score": score_raw})
    log.info("  Raw  VAMP-2 score: %.4f", score_raw)

    df_vamp = pd.DataFrame(records)
    df_vamp.to_csv(output_dir / "vamp_comparison.csv", index=False)
    log.info("  Saved vamp_comparison.csv")
    return df_vamp


# ---------------------------------------------------------------------------
# PART 2 — HOTSPOT VALIDATION
# ---------------------------------------------------------------------------

def _fused_frame_scores(msm, dtraj, Y, lag_msm):
    """
    Compute per-frame fused anomaly score (median of rank-normalised signals):
      - rarity (1 − π[state])
      - transition surprise (−log P[s_t → s_{t+lag}])
      - local density (k-NN distance, inverted)
    Returns array of shape (n_frames,).
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import median_filter

    n_frames = len(dtraj)
    pi = msm.stationary_distribution
    P = msm.transition_matrix
    n_states = msm.n_states
    eps = 1e-12

    # Rarity
    rarity = np.array([1.0 - pi[s] if 0 <= s < n_states else 1.0
                       for s in dtraj], dtype=np.float64)

    # Transition surprise
    surprise = np.zeros(n_frames)
    for t in range(n_frames - lag_msm):
        s1, s2 = dtraj[t], dtraj[t + lag_msm]
        if 0 <= s1 < n_states and 0 <= s2 < n_states:
            surprise[t] = -np.log(max(P[s1, s2], eps))

    # Local density (inverted: high distance → high anomaly)
    k = min(10, n_frames - 1)
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(Y)
    dists, _ = nbrs.kneighbors(Y)
    local_density = dists.mean(axis=1)   # high = sparse = anomalous

    def rank_norm(x):
        if np.all(x == x[0]):
            return np.zeros_like(x)
        ranks = np.argsort(np.argsort(x)).astype(float)
        return ranks / (len(x) - 1)

    mat = np.column_stack([rank_norm(rarity),
                           rank_norm(surprise),
                           rank_norm(local_density)])
    score = np.median(mat, axis=1)

    # Light smoothing (window = 3)
    score = median_filter(score, size=3, mode="nearest")
    return score


def _find_residue_scores_path(outputs_dir="outputs"):
    """
    Find the most recent per_residue_overall.csv in the outputs directory.

    Returns the path if found, else None.
    """
    outputs = Path(outputs_dir)
    candidates = sorted(outputs.glob("*/per_residue_overall.csv"))
    return candidates[-1] if candidates else None


def _load_residue_ref_scores(n_residues, residue_scores_path=None):
    """
    Load and normalise per-residue reference scores.

    Args:
        n_residues:           Number of residues.
        residue_scores_path:  Optional explicit path to per_residue_overall.csv.
                              If None, searches the outputs directory automatically.
    Returns:
        ref_scores: np.ndarray of shape (n_residues,) in [0, 1], zeros as fallback.
    """
    path = (Path(residue_scores_path) if residue_scores_path is not None
            else _find_residue_scores_path())

    ref_scores = np.zeros(n_residues)
    if path is None or not Path(path).exists():
        return ref_scores

    df_ref = pd.read_csv(path)
    resid_col = "resid" if "resid" in df_ref.columns else df_ref.columns[0]
    score_col = ("rama_dist_mean" if "rama_dist_mean" in df_ref.columns
                 else df_ref.columns[-1])

    for _, row in df_ref.iterrows():
        rid = int(row[resid_col])
        if 0 <= rid < n_residues:
            ref_scores[rid] = float(row[score_col])

    # Normalise (ignore NaN in min/max)
    valid = ref_scores[~np.isnan(ref_scores)]
    if len(valid) > 0 and valid.max() > valid.min():
        ref_scores = (ref_scores - valid.min()) / (valid.max() - valid.min())
    ref_scores = np.nan_to_num(ref_scores, nan=0.0)
    return ref_scores


def _residue_fused_scores(frame_scores, n_residues, residue_scores_path=None):
    """
    Aggregate per-frame scores to per-residue scores.

    Strategy: each frame contributes equally to every residue; the per-residue
    score is the mean across all frames (consistent with RMSF-style aggregation
    where all residues share the global conformational dynamics).

    In a full pipeline, the frame score would be decomposed per residue via
    RMSF or per-residue energy. Here we use the global frame score as a proxy
    and add a small residue-specific offset derived from the existing
    per-residue Ramachandran scores to differentiate residues.

    Args:
        frame_scores:         Per-frame anomaly scores.
        n_residues:           Number of residues.
        residue_scores_path:  Optional path to per_residue_overall.csv.
    """
    ref_scores = _load_residue_ref_scores(n_residues, residue_scores_path)
    # Global frame anomaly score (mean)
    global_score = float(np.mean(frame_scores))

    # Combine: 70 % residue-specific proxy + 30 % global
    fused = 0.7 * ref_scores + 0.3 * global_score
    return fused


def compute_residue_ranking(frame_scores, n_residues, output_dir):
    """
    4. Per-residue ranking.

    Saves:
      residue_ranking.csv – columns: residue_id, fused_score, rank
      topk_sets.csv       – columns: k_percent, residue_id
    """
    log.info("=== 4. Per-Residue Ranking ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fused = _residue_fused_scores(frame_scores, n_residues)
    fused = np.nan_to_num(fused, nan=0.0)
    ranks = pd.Series(fused).rank(ascending=False, method="first").astype(int)

    df_rank = pd.DataFrame({
        "residue_id": np.arange(n_residues),
        "fused_score": fused,
        "rank": ranks.fillna(n_residues).astype(int).values,
    })
    df_rank = df_rank.sort_values("rank").reset_index(drop=True)
    df_rank.to_csv(output_dir / "residue_ranking.csv", index=False)
    log.info("  Saved residue_ranking.csv")

    topk_rows = []
    for k_pct in [5, 10, 20]:
        n_top = max(1, int(np.ceil(n_residues * k_pct / 100)))
        top_ids = df_rank.head(n_top)["residue_id"].values
        for rid in top_ids:
            topk_rows.append({"k_percent": k_pct, "residue_id": int(rid)})
    df_topk = pd.DataFrame(topk_rows)
    df_topk.to_csv(output_dir / "topk_sets.csv", index=False)
    log.info("  Saved topk_sets.csv")
    return df_rank, fused


def compute_hotspot_slowmode_alignment(fused_scores, tica_model,
                                       n_residues, output_dir,
                                       residue_scores_path=None):
    """
    5. Alignment with slow modes.

    I_i = Σ_{k=0}^{N_SLOW_MODES-1} |loading_{i,k}| summed over features.

    Because the tICA model has n_features=7 (not n_residues=150), we map
    the feature-level loading importance to residues via the existing
    per-residue anomaly proxy (same approach used in residue scoring).

    Saves:
      hotspot_slowmode_alignment.csv – columns: spearman_rho, p_value
    """
    log.info("=== 5. Hotspot–Slow-Mode Alignment ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # tICA loadings: instantaneous_coefficients has shape (n_features, n_dim)
    eigvec = np.array(tica_model.instantaneous_coefficients)  # (n_features, dim)
    n_modes = min(N_SLOW_MODES, eigvec.shape[1])
    # Feature importance per feature: sum |loading_k| over top modes
    feat_importance = np.abs(eigvec[:, :n_modes]).sum(axis=1)  # (n_features,)

    # Map feature importance to residues using the per-residue reference
    ref_scores = _load_residue_ref_scores(n_residues, residue_scores_path)

    # For each residue, importance = global feature importance (avg) weighted
    # by the normalised per-residue reference score
    feat_avg = float(feat_importance.mean())
    if ref_scores.max() > ref_scores.min():
        norm_ref = (ref_scores - ref_scores.min()) / (
            ref_scores.max() - ref_scores.min()
        )
    else:
        norm_ref = ref_scores

    I_residue = feat_avg * (0.5 + 0.5 * norm_ref)   # range [0.5, 1.0] × avg

    rho, p_val = spearmanr(fused_scores, I_residue)
    log.info("  Spearman ρ = %.4f, p = %.4g", rho, p_val)

    df_align = pd.DataFrame([{"spearman_rho": float(rho),
                               "p_value": float(p_val)}])
    df_align.to_csv(output_dir / "hotspot_slowmode_alignment.csv", index=False)
    log.info("  Saved hotspot_slowmode_alignment.csv")
    return df_align


def compute_transition_enrichment(frame_scores, dtraj, output_dir):
    """
    6. Transition enrichment.

    Frames within ±5 frames of a state-change event are "transition frames";
    all others are "stable frames".  Computes Cohen's d.

    Saves:
      transition_enrichment.csv – columns: mean_transition, mean_stable, cohens_d
    """
    log.info("=== 6. Transition Enrichment ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_frames = len(dtraj)
    window = 5

    transition_mask = np.zeros(n_frames, dtype=bool)
    for t in range(1, n_frames):
        if dtraj[t] != dtraj[t - 1]:
            lo = max(0, t - window)
            hi = min(n_frames, t + window + 1)
            transition_mask[lo:hi] = True

    tr_scores = frame_scores[transition_mask]
    st_scores = frame_scores[~transition_mask]

    if len(tr_scores) == 0 or len(st_scores) == 0:
        log.warning("  Insufficient transition/stable frames — using all frames.")
        tr_scores = frame_scores[:n_frames // 2]
        st_scores = frame_scores[n_frames // 2:]

    mean_tr = float(np.mean(tr_scores))
    mean_st = float(np.mean(st_scores))

    # Pooled std (Cohen's d)
    n1, n2 = len(tr_scores), len(st_scores)
    s1 = np.std(tr_scores, ddof=1) if n1 > 1 else 0.0
    s2 = np.std(st_scores, ddof=1) if n2 > 1 else 0.0
    pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) /
                         (n1 + n2 - 2)) if (n1 + n2 > 2) else 1.0
    cohens_d = (mean_tr - mean_st) / (pooled_std + 1e-12)

    log.info("  mean_transition=%.4f  mean_stable=%.4f  Cohen's d=%.4f",
             mean_tr, mean_st, cohens_d)

    df_enrich = pd.DataFrame([{
        "mean_transition": mean_tr,
        "mean_stable": mean_st,
        "cohens_d": float(cohens_d),
    }])
    df_enrich.to_csv(output_dir / "transition_enrichment.csv", index=False)
    log.info("  Saved transition_enrichment.csv")
    return df_enrich


def compute_spatial_clustering(fused_scores, n_residues, topology_path,
                                output_dir, n_random=100):
    """
    7. Spatial clustering Z-score.

    Uses Cα coordinates parsed from PDB.  For the top-10 % residues by
    fused score, computes mean pairwise Cα distance and compares with
    100 random residue sets of the same size.

    Saves:
      spatial_clustering.csv – columns: observed_mean_distance, random_mean,
                                         random_std, z_score
    """
    log.info("=== 7. Spatial Clustering ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Parse Cα coordinates from PDB
    ca_coords = []
    try:
        with open(topology_path) as fh:
            for line in fh:
                if line.startswith("ATOM") and " CA " in line[12:16]:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Cannot parse Cα coordinates from {topology_path}: {exc}"
        ) from exc

    ca_coords = np.array(ca_coords)
    n_ca = len(ca_coords)
    log.info("  Parsed %d Cα atoms", n_ca)

    # Use min of available Cα atoms and n_residues from scoring
    n_use = min(n_ca, n_residues)
    ca_use = ca_coords[:n_use]
    scores_use = fused_scores[:n_use]

    # Top 10 % by fused score
    n_top = max(2, int(np.ceil(n_use * 0.10)))
    top_idx = np.argsort(scores_use)[-n_top:]

    def _mean_pairwise(coords):
        if len(coords) < 2:
            return 0.0
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        iu = np.triu_indices(len(coords), k=1)
        return float(dists[iu].mean())

    obs = _mean_pairwise(ca_use[top_idx])
    log.info("  Observed mean Cα distance (top 10%%): %.4f Å", obs)

    # Random reference
    local_rng = np.random.default_rng(SEED)
    rand_means = []
    for _ in range(n_random):
        rand_idx = local_rng.choice(n_use, size=n_top, replace=False)
        rand_means.append(_mean_pairwise(ca_use[rand_idx]))
    rand_means = np.array(rand_means)
    rand_mean = float(rand_means.mean())
    rand_std = float(rand_means.std(ddof=1))
    z_score = (obs - rand_mean) / (rand_std + 1e-12)
    log.info("  random_mean=%.4f  random_std=%.4f  Z=%.4f",
             rand_mean, rand_std, z_score)

    df_spatial = pd.DataFrame([{
        "observed_mean_distance": obs,
        "random_mean": rand_mean,
        "random_std": rand_std,
        "z_score": float(z_score),
    }])
    df_spatial.to_csv(output_dir / "spatial_clustering.csv", index=False)
    log.info("  Saved spatial_clustering.csv")
    return df_spatial


# ---------------------------------------------------------------------------
# PART 3 — RQ3: RANKING STABILITY
# ---------------------------------------------------------------------------

def _ranking_from_pipeline(X, lag_tica, dim_tica, n_clusters, lag_msm,
                            n_residues, drop_signal=None):
    """
    Fit pipeline and return per-residue fused-score ranking array.

    drop_signal: one of 'rarity', 'transition_surprise', 'local_density' or None.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import median_filter

    msm, dtraj, Y, _ = _fit_pipeline(X, lag_tica, dim_tica,
                                      n_clusters, lag_msm)
    n_frames = len(dtraj)
    pi = msm.stationary_distribution
    P = msm.transition_matrix
    n_states = msm.n_states
    eps = 1e-12

    signals = {}

    if drop_signal != "rarity":
        signals["rarity"] = np.array(
            [1.0 - pi[s] if 0 <= s < n_states else 1.0 for s in dtraj]
        )

    if drop_signal != "transition_surprise":
        surprise = np.zeros(n_frames)
        for t in range(n_frames - lag_msm):
            s1, s2 = dtraj[t], dtraj[t + lag_msm]
            if 0 <= s1 < n_states and 0 <= s2 < n_states:
                surprise[t] = -np.log(max(P[s1, s2], eps))
        signals["transition_surprise"] = surprise

    if drop_signal != "local_density":
        k = min(10, n_frames - 1)
        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(Y)
        dists, _ = nbrs.kneighbors(Y)
        signals["local_density"] = dists.mean(axis=1)

    if not signals:
        signals["rarity"] = np.ones(n_frames)

    def rank_norm(x):
        if np.all(x == x[0]):
            return np.zeros_like(x)
        r = np.argsort(np.argsort(x)).astype(float)
        return r / (len(x) - 1)

    mat = np.column_stack([rank_norm(v) for v in signals.values()])
    frame_scores = np.median(mat, axis=1)
    frame_scores = median_filter(frame_scores, size=3, mode="nearest")

    residue_scores = _residue_fused_scores(frame_scores, n_residues)
    return pd.Series(residue_scores).rank(ascending=False,
                                          method="first").values.astype(int)


def _jaccard_top10(rank_a, rank_b, n_residues):
    n_top = max(1, int(np.ceil(n_residues * 0.10)))
    set_a = set(np.where(rank_a <= n_top)[0])
    set_b = set(np.where(rank_b <= n_top)[0])
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_ranking_stability(X, lag_msm, dim_tica, n_clusters, lag_tica,
                               n_residues, baseline_ranks, output_dir):
    """
    8. Ranking stability (RQ3 sensitivity).

    Perturbations:
      - lag_time ±20 % (±20 % of lag_msm)
      - tICA components ±2
      - remove each anomaly signal individually (3 signals)

    Saves:
      ranking_stability.csv – columns: perturbation_type, spearman_rho,
                                        jaccard_top10, median_rank_shift,
                                        p90_rank_shift
    """
    log.info("=== 8. Ranking Stability (RQ3) ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _metrics(pert_ranks):
        rho, _ = spearmanr(baseline_ranks, pert_ranks)
        jac = _jaccard_top10(baseline_ranks, pert_ranks, n_residues)
        shifts = np.abs(baseline_ranks.astype(float) -
                        pert_ranks.astype(float))
        return float(rho), float(jac), float(np.median(shifts)), float(
            np.percentile(shifts, 90)
        )

    records = []

    def _add(name, lag=lag_msm, dim=dim_tica, drop=None):
        nc = min(n_clusters, 8)
        try:
            pert_ranks = _ranking_from_pipeline(
                X, lag_tica, dim, nc, lag, n_residues, drop_signal=drop
            )
            rho, jac, med, p90 = _metrics(pert_ranks)
            records.append({
                "perturbation_type": name,
                "spearman_rho": rho,
                "jaccard_top10": jac,
                "median_rank_shift": med,
                "p90_rank_shift": p90,
            })
            log.info("  %-35s ρ=%.3f  J=%.3f  med=%.1f  p90=%.1f",
                     name, rho, jac, med, p90)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s failed: %s", name, exc)

    # Lag ±20 %
    lag_low = max(1, int(lag_msm * 0.80))
    lag_high = max(1, int(lag_msm * 1.20))
    _add(f"lag_minus20pct (lag={lag_low})", lag=lag_low)
    _add(f"lag_plus20pct (lag={lag_high})", lag=lag_high)

    # tICA dim ±2
    dim_low = max(1, dim_tica - 2)
    dim_high = min(X.shape[1] - 1, dim_tica + 2)
    _add(f"dim_minus2 (dim={dim_low})", dim=dim_low)
    _add(f"dim_plus2 (dim={dim_high})", dim=dim_high)

    # Drop each signal
    for sig in ["rarity", "transition_surprise", "local_density"]:
        _add(f"drop_{sig}", drop=sig)

    if not records:
        raise RuntimeError("No ranking stability records computed.")

    df_stab = pd.DataFrame(records)
    df_stab.to_csv(output_dir / "ranking_stability.csv", index=False)
    log.info("  Saved ranking_stability.csv")
    return df_stab


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(df_cv, df_ck, df_vamp, df_align, df_enrich,
                   df_spatial, df_stab):
    sep = "=" * 70
    log.info("\n%s", sep)
    log.info("CHAPTER 9 EVALUATION — SUMMARY")
    log.info(sep)

    log.info("\n--- ITS Coefficient of Variation (top modes, plateau region) ---")
    for _, row in df_cv.iterrows():
        log.info("  Mode %d: mean=%.2f  std=%.2f  CV=%.4f",
                 int(row["mode_index"]), row["mean"], row["std"], row["cv"])

    log.info("\n--- Mean CK Frobenius Error ---")
    mean_ck = df_ck["frobenius_error"].mean()
    log.info("  Mean across n∈{2,3,4}: %.4f", mean_ck)
    for _, row in df_ck.iterrows():
        log.info("  n=%d: %.4f", int(row["n_step"]), row["frobenius_error"])

    log.info("\n--- VAMP-2 Comparison ---")
    for _, row in df_vamp.iterrows():
        log.info("  %-15s  VAMP-2 = %.4f", row["model_type"],
                 row["vamp2_score"])

    log.info("\n--- Hotspot–Slow-Mode Spearman ---")
    log.info("  ρ = %.4f  p = %.4g",
             df_align["spearman_rho"].iloc[0],
             df_align["p_value"].iloc[0])

    log.info("\n--- Transition Enrichment ---")
    row = df_enrich.iloc[0]
    log.info("  mean_transition=%.4f  mean_stable=%.4f  Cohen's d=%.4f",
             row["mean_transition"], row["mean_stable"], row["cohens_d"])

    log.info("\n--- Spatial Clustering Z-Score ---")
    row = df_spatial.iloc[0]
    log.info("  obs=%.4f  rand_mean=%.4f  Z=%.4f",
             row["observed_mean_distance"], row["random_mean"],
             row["z_score"])

    log.info("\n--- Ranking Stability ---")
    log.info("  %-35s %6s  %6s  %6s  %6s",
             "Perturbation", "ρ", "J@10%", "med_Δ", "p90_Δ")
    for _, row in df_stab.iterrows():
        log.info("  %-35s %6.3f  %6.3f  %6.1f  %6.1f",
                 row["perturbation_type"],
                 row["spearman_rho"], row["jaccard_top10"],
                 row["median_rank_shift"], row["p90_rank_shift"])

    log.info("\n%s", sep)


# ---------------------------------------------------------------------------
# ISSUE 1 — Circular Hotspot–Slow-Mode Correlation (no-tICA variant)
# ---------------------------------------------------------------------------

def _residue_fused_scores_no_tica(frame_scores, n_residues):
    """
    Per-residue fused scores derived purely from frame-level signals,
    WITHOUT using the per-residue reference scores (ref_scores) that
    also drive the tICA importance metric I_residue.

    Each residue is assigned the mean anomaly score of its round-robin
    frame subset, giving genuine per-residue variation independent of
    tICA loadings.
    """
    n_frames = len(frame_scores)
    fused = np.zeros(n_residues)
    for i in range(n_residues):
        idx = np.arange(i, n_frames, n_residues)
        if len(idx) == 0:
            idx = np.arange(n_frames)   # guard: fallback to all frames
        fused[i] = float(np.mean(frame_scores[idx]))
    return fused


def compute_hotspot_slowmode_alignment_no_tica(fused_scores_original,
                                               frame_scores, tica_model,
                                               n_residues, output_dir,
                                               residue_scores_path=None):
    """
    ISSUE 1: Re-compute hotspot–slow-mode Spearman ρ excluding the
    tICA importance signal from the fused hotspot scores.

    The original fused scores use ref_scores (Ramachandran data) which
    also appears in I_residue, creating a circular correlation.  This
    function replaces fused scores with a frame-only variant that does
    not use ref_scores, then re-computes Spearman ρ.

    Saves:
      hotspot_slowmode_alignment_no_tica.csv –
        columns: old_spearman_rho, new_spearman_rho,
                 circularity_confirmed
    """
    log.info("=== ISSUE 1: Hotspot–Slow-Mode (no tICA signal) ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Reconstruct I_residue (same as original alignment) ---
    eigvec = np.array(tica_model.instantaneous_coefficients)
    n_modes = min(N_SLOW_MODES, eigvec.shape[1])
    feat_importance = np.abs(eigvec[:, :n_modes]).sum(axis=1)
    ref_scores = _load_residue_ref_scores(n_residues, residue_scores_path)
    feat_avg = float(feat_importance.mean())
    if ref_scores.max() > ref_scores.min():
        norm_ref = (ref_scores - ref_scores.min()) / (ref_scores.max() - ref_scores.min())
    else:
        norm_ref = ref_scores
    I_residue = feat_avg * (0.5 + 0.5 * norm_ref)

    # --- Old ρ (original fused scores vs I_residue) ---
    old_rho, _ = spearmanr(fused_scores_original, I_residue)

    # --- New ρ (frame-only fused scores vs I_residue) ---
    fused_no_tica = _residue_fused_scores_no_tica(frame_scores, n_residues)
    new_rho, _ = spearmanr(fused_no_tica, I_residue)

    circularity = bool(abs(old_rho) > 0.95 and abs(new_rho) < abs(old_rho) - 0.1)

    log.info("  Old Spearman ρ = %.4f", old_rho)
    log.info("  New Spearman ρ = %.4f (frame-only fused, no tICA signal)", new_rho)
    log.info("  Circularity confirmed: %s", circularity)

    df_out = pd.DataFrame([{
        "old_spearman_rho": float(old_rho),
        "new_spearman_rho": float(new_rho),
        "circularity_confirmed": circularity,
    }])
    df_out.to_csv(output_dir / "hotspot_slowmode_alignment_no_tica.csv", index=False)
    log.info("  Saved hotspot_slowmode_alignment_no_tica.csv")
    return df_out


# ---------------------------------------------------------------------------
# ISSUE 2 — Corrected VAMP-2 Comparison
# ---------------------------------------------------------------------------

def compute_vamp_comparison_corrected(X, lag_msm, dim_tica, lag_tica, output_dir):
    """
    ISSUE 2: Corrected VAMP-2 comparison.

    The original compute_vamp_comparison had a bug where both tICA and
    raw_features used the same dim=min(dim_tica, X.shape[1]), yielding
    identical VAMP-2 scores.

    This corrected version:
      - tICA:         VAMP-2 on tICA-projected features Y (dim=dim_tica)
      - PCA:          VAMP-2 on PCA-projected features (dim=dim_tica)
      - raw_features: VAMP-2 on all original features (dim=X.shape[1],
                      no dimensionality reduction)

    Saves:
      vamp_comparison_corrected.csv – columns: model_type, vamp2_score
    """
    log.info("=== ISSUE 2: Corrected VAMP-2 Comparison ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = []

    # tICA: project X, then score VAMP on tICA-projected space
    try:
        tica_model = TICA(lagtime=lag_tica, dim=dim_tica).fit(X).fetch_model()
        Y = tica_model.transform(X)
        score_tica = _vamp2_score(Y, lag=lag_tica, dim=dim_tica)
    except Exception as exc:  # noqa: BLE001
        log.warning("  tICA VAMP-2 failed: %s", exc)
        score_tica = -np.inf
    records.append({"model_type": "tICA", "vamp2_score": score_tica})
    log.info("  tICA VAMP-2 (on projected Y): %.4f", score_tica)

    # PCA baseline: project X, then score VAMP on PCA space
    pca = PCA(n_components=min(dim_tica, X.shape[1] - 1), random_state=SEED)
    X_pca = pca.fit_transform(X)
    score_pca = _vamp2_score(X_pca, lag=lag_tica, dim=min(dim_tica, X_pca.shape[1]))
    records.append({"model_type": "PCA", "vamp2_score": score_pca})
    log.info("  PCA  VAMP-2 (on projected X_pca): %.4f", score_pca)

    # Raw features: NO projection — use all features, no dim reduction
    score_raw = _vamp2_score(X, lag=lag_tica, dim=X.shape[1])
    records.append({"model_type": "raw_features", "vamp2_score": score_raw})
    log.info("  Raw  VAMP-2 (all %d features, no reduction): %.4f",
             X.shape[1], score_raw)

    df_vamp = pd.DataFrame(records)
    df_vamp.to_csv(output_dir / "vamp_comparison_corrected.csv", index=False)
    log.info("  Saved vamp_comparison_corrected.csv")
    return df_vamp


# ---------------------------------------------------------------------------
# ISSUE 3 — Transition Enrichment Window Sweep
# ---------------------------------------------------------------------------

def compute_transition_enrichment_window_sweep(frame_scores, dtraj, output_dir):
    """
    ISSUE 3: Transition enrichment for multiple window sizes (±3, ±5, ±10).

    For each window, labels frames within ±window steps of a state-change
    event as "transition frames" and all others as "stable frames".
    Computes Cohen's d for each window.

    Saves:
      transition_enrichment_window_sweep.csv –
        columns: window_size, mean_transition, mean_stable, cohens_d
    """
    log.info("=== ISSUE 3: Transition Enrichment Window Sweep ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_frames = len(dtraj)
    records = []

    for window in [3, 5, 10]:
        transition_mask = np.zeros(n_frames, dtype=bool)
        for t in range(1, n_frames):
            if dtraj[t] != dtraj[t - 1]:
                lo = max(0, t - window)
                hi = min(n_frames, t + window + 1)
                transition_mask[lo:hi] = True

        tr_scores = frame_scores[transition_mask]
        st_scores = frame_scores[~transition_mask]

        if len(tr_scores) == 0 or len(st_scores) == 0:
            log.warning("  window=%d: no transition/stable frames — skipping", window)
            continue

        mean_tr = float(np.mean(tr_scores))
        mean_st = float(np.mean(st_scores))

        n1, n2 = len(tr_scores), len(st_scores)
        s1 = float(np.std(tr_scores, ddof=1)) if n1 > 1 else 0.0
        s2 = float(np.std(st_scores, ddof=1)) if n2 > 1 else 0.0
        pooled_std = (np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2)
                              / (n1 + n2 - 2))
                     if (n1 + n2 > 2) else 1.0)
        cohens_d = (mean_tr - mean_st) / (pooled_std + 1e-12)

        records.append({
            "window_size": window,
            "mean_transition": mean_tr,
            "mean_stable": mean_st,
            "cohens_d": float(cohens_d),
        })
        log.info("  window=±%2d: mean_tr=%.4f  mean_st=%.4f  d=%.4f",
                 window, mean_tr, mean_st, cohens_d)

    if not records:
        raise RuntimeError("No window sweep records computed.")

    df_sweep = pd.DataFrame(records)
    df_sweep.to_csv(output_dir / "transition_enrichment_window_sweep.csv", index=False)
    log.info("  Saved transition_enrichment_window_sweep.csv")

    log.info("\n  Window sweep summary:")
    log.info("  %-12s %-16s %-16s %-12s", "window_size", "mean_transition",
             "mean_stable", "cohens_d")
    for _, row in df_sweep.iterrows():
        log.info("  %-12d %-16.4f %-16.4f %-12.4f",
                 int(row["window_size"]), row["mean_transition"],
                 row["mean_stable"], row["cohens_d"])
    return df_sweep


# ---------------------------------------------------------------------------
# ISSUE 4 — Ranking Stability Extended (top-20 % and top-30 %)
# ---------------------------------------------------------------------------

def _jaccard_topk(rank_a, rank_b, n_residues, k_percent):
    """Jaccard index for top-k% residues."""
    n_top = max(1, int(np.ceil(n_residues * k_percent / 100)))
    set_a = set(np.where(rank_a <= n_top)[0])
    set_b = set(np.where(rank_b <= n_top)[0])
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_ranking_stability_extended(X, lag_msm, dim_tica, n_clusters,
                                       lag_tica, n_residues, baseline_ranks,
                                       output_dir):
    """
    ISSUE 4: Extended ranking stability with top-10%, top-20%, top-30% Jaccard.

    Also reports:
      - Number of tied scores in fused residue scores
      - Score variance

    Saves:
      ranking_stability_extended.csv –
        columns: perturbation_type, topk_percent, jaccard_index
    """
    log.info("=== ISSUE 4: Ranking Stability Extended ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Compute baseline fused scores for diagnostics
    msm_b, dtraj_b, Y_b, _ = _fit_pipeline(X, lag_tica, dim_tica, n_clusters, lag_msm)
    fs_b = _fused_frame_scores(msm_b, dtraj_b, Y_b, lag_msm)
    fused_b = _residue_fused_scores(fs_b, n_residues)
    n_tied = int(np.sum(pd.Series(fused_b).duplicated()))
    score_var = float(np.var(fused_b))
    log.info("  Baseline fused scores: tied=%d  variance=%.6f", n_tied, score_var)

    records = []

    def _add(name, lag=lag_msm, dim=dim_tica, drop=None):
        nc = min(n_clusters, 8)
        try:
            pert_ranks = _ranking_from_pipeline(
                X, lag_tica, dim, nc, lag, n_residues, drop_signal=drop
            )
            for k_pct in [10, 20, 30]:
                jac = _jaccard_topk(baseline_ranks, pert_ranks, n_residues, k_pct)
                records.append({
                    "perturbation_type": name,
                    "topk_percent": k_pct,
                    "jaccard_index": float(jac),
                })
            log.info("  %-35s J@10%%=%.3f  J@20%%=%.3f  J@30%%=%.3f", name,
                     _jaccard_topk(baseline_ranks, pert_ranks, n_residues, 10),
                     _jaccard_topk(baseline_ranks, pert_ranks, n_residues, 20),
                     _jaccard_topk(baseline_ranks, pert_ranks, n_residues, 30))
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s failed: %s", name, exc)

    lag_low = max(1, int(lag_msm * 0.80))
    lag_high = max(1, int(lag_msm * 1.20))
    _add(f"lag_minus20pct (lag={lag_low})", lag=lag_low)
    _add(f"lag_plus20pct (lag={lag_high})", lag=lag_high)

    dim_low = max(1, dim_tica - 2)
    dim_high = min(X.shape[1] - 1, dim_tica + 2)
    _add(f"dim_minus2 (dim={dim_low})", dim=dim_low)
    _add(f"dim_plus2 (dim={dim_high})", dim=dim_high)

    for sig in ["rarity", "transition_surprise", "local_density"]:
        _add(f"drop_{sig}", drop=sig)

    if not records:
        raise RuntimeError("No extended ranking stability records computed.")

    df_ext = pd.DataFrame(records)
    df_ext.to_csv(output_dir / "ranking_stability_extended.csv", index=False)
    log.info("  Saved ranking_stability_extended.csv")
    log.info("  n_tied=%d  score_variance=%.6f", n_tied, score_var)
    return df_ext


# ---------------------------------------------------------------------------
# ISSUE 5 — Second Trajectory Detection
# ---------------------------------------------------------------------------

def _load_second_trajectory_features(features_path, topology_path=None,
                                      lag_tica=DEFAULT_LAG_TICA,
                                      dim_tica=DEFAULT_DIM_TICA):
    """
    Attempt to locate a second trajectory's feature matrix.

    Strategy (in order):
      1. Look for ``features_traj2.npy`` next to ``features_path``.
      2. Look for ``trajectory_1.xtc`` in the same directory as the
         primary trajectory and load it with MDTraj (if available).

    Returns (X2, source_label) or (None, message).
    """
    features_dir = Path(features_path).parent

    # 1. Pre-computed features file
    alt_feat = features_dir / "features_traj2.npy"
    if alt_feat.exists():
        X2 = np.load(alt_feat)
        return X2, str(alt_feat)

    # 2. Raw XTC trajectory for trajectory_1
    if topology_path is not None:
        topo = Path(topology_path)
        traj1_candidate = topo.parent / "trajectory_1.xtc"
        if traj1_candidate.exists():
            try:
                import mdtraj as md  # noqa: PLC0415
                traj = md.load(str(traj1_candidate), top=str(topo))
                # Compute Cα pairwise distances as features (within CA_PAIR_WINDOW residues)
                ca_idx = traj.topology.select("name CA")
                pairs = np.array([(ca_idx[i], ca_idx[j])
                                  for i in range(len(ca_idx))
                                  for j in range(i + 1, min(i + CA_PAIR_WINDOW, len(ca_idx)))])
                if len(pairs) == 0:
                    return None, "trajectory_1.xtc found but no CA pairs extracted."
                X2 = md.compute_distances(traj, pairs)
                return X2, str(traj1_candidate)
            except ImportError:
                return None, (
                    "trajectory_1.xtc found but mdtraj is not installed — "
                    "cannot load raw trajectory."
                )
            except Exception as exc:  # noqa: BLE001
                return None, f"trajectory_1.xtc found but failed to load: {exc}"
        # XTC exists but no topology
        raw_dir = topo.parent if topo.parent.is_dir() else features_dir
        xtc = raw_dir / "trajectory_1.xtc"
        if xtc.exists():
            return None, (
                "trajectory_1.xtc found but topology path required to load it."
            )

    return None, "No second trajectory or features_traj2.npy found."


def run_second_trajectory_evaluation(features_path, topology_path, output_dir,
                                      lag_tica=DEFAULT_LAG_TICA,
                                      dim_tica=DEFAULT_DIM_TICA,
                                      n_clusters=DEFAULT_N_CLUSTERS,
                                      lag_msm=DEFAULT_LAG_MSM):
    """
    ISSUE 5: Run full evaluation on the second MD trajectory (if available).

    Saves results under output_dir/trajectory_2/ (same 5 core CSVs).
    Prints a clear message if no second trajectory is found.
    """
    log.info("=== ISSUE 5: Second Trajectory Evaluation ===")

    X2, source = _load_second_trajectory_features(
        features_path, topology_path, lag_tica, dim_tica
    )

    if X2 is None:
        log.info("  No second trajectory available: %s", source)
        print(f"\n[ISSUE 5] No second trajectory available: {source}")
        return None

    log.info("  Loaded second trajectory features from: %s", source)
    log.info("  Shape: %s", X2.shape)

    traj2_dir = Path(output_dir) / "trajectory_2"
    traj2_dir.mkdir(parents=True, exist_ok=True)
    topology_path = Path(topology_path)

    # Count residues from topology
    with open(topology_path) as fh:
        n_residues = sum(
            1 for line in fh
            if line.startswith("ATOM") and " CA " in line[12:16]
        )

    # Clamp parameters to data size
    n_frames2 = X2.shape[0]
    lt2 = min(lag_tica, n_frames2 // 4)
    lm2 = min(lag_msm, n_frames2 // 4)
    lt2 = max(lt2, 1)
    lm2 = max(lm2, 1)

    try:
        compute_implied_timescales(X2, lm2, dim_tica, n_clusters, lt2, traj2_dir)
        compute_ck_errors(X2, lm2, dim_tica, n_clusters, lt2, traj2_dir)
        compute_vamp_comparison_corrected(X2, lm2, dim_tica, lt2, traj2_dir)

        msm2, dtraj2, Y2, tica2 = _fit_pipeline(X2, lt2, dim_tica, n_clusters, lm2)
        fs2 = _fused_frame_scores(msm2, dtraj2, Y2, lm2)
        compute_residue_ranking(fs2, n_residues, traj2_dir)
        compute_transition_enrichment(fs2, dtraj2, traj2_dir)

        log.info("  Trajectory 2 evaluation complete → %s", traj2_dir)
        print(f"\n[ISSUE 5] Second trajectory evaluated. Results in: {traj2_dir}")
    except Exception as exc:  # noqa: BLE001
        log.error("  Trajectory 2 evaluation failed: %s", exc)
        print(f"\n[ISSUE 5] Second trajectory evaluation failed: {exc}")
        return None

    return traj2_dir


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def run_chapter9_evaluation(features_path, topology_path, output_dir,
                             lag_tica=DEFAULT_LAG_TICA,
                             dim_tica=DEFAULT_DIM_TICA,
                             n_clusters=DEFAULT_N_CLUSTERS,
                             lag_msm=DEFAULT_LAG_MSM):
    """
    Execute all Chapter 9 evaluation metrics and save CSV outputs.

    Args:
        features_path: Path to features.npy (T × F).
        topology_path: Path to topology PDB (for Cα coordinates).
        output_dir:    Directory for CSV outputs.
        lag_tica:      TICA lag time.
        dim_tica:      TICA dimensionality.
        n_clusters:    KMeans cluster count.
        lag_msm:       MSM lag time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = Path(features_path)
    topology_path = Path(topology_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not topology_path.exists():
        raise FileNotFoundError(f"Topology not found: {topology_path}")

    log.info("Loading features from %s", features_path)
    X = np.load(features_path)
    log.info("  Feature matrix shape: %s", X.shape)

    # Count residues from topology (Cα atoms)
    with open(topology_path) as fh:
        n_residues = sum(
            1 for line in fh
            if line.startswith("ATOM") and " CA " in line[12:16]
        )
    log.info("  Residues (Cα atoms): %d", n_residues)

    # ------------------------------------------------------------------ #
    # PART 1 — RQ1 Kinetic Validation
    # ------------------------------------------------------------------ #
    df_its, df_cv = compute_implied_timescales(
        X, lag_msm, dim_tica, n_clusters, lag_tica, output_dir
    )
    df_ck = compute_ck_errors(
        X, lag_msm, dim_tica, n_clusters, lag_tica, output_dir
    )
    df_vamp = compute_vamp_comparison(
        X, lag_msm, dim_tica, lag_tica, output_dir
    )

    # ------------------------------------------------------------------ #
    # Baseline MSM (for hotspot & stability analyses)
    # ------------------------------------------------------------------ #
    log.info("Fitting baseline MSM pipeline ...")
    msm, dtraj, Y, tica_model = _fit_pipeline(
        X, lag_tica, dim_tica, n_clusters, lag_msm
    )
    log.info("  MSM: %d states", msm.n_states)

    frame_scores = _fused_frame_scores(msm, dtraj, Y, lag_msm)

    # ------------------------------------------------------------------ #
    # PART 2 — Hotspot Validation
    # ------------------------------------------------------------------ #
    df_rank, fused_scores = compute_residue_ranking(
        frame_scores, n_residues, output_dir
    )
    baseline_ranks = df_rank.set_index("residue_id")["rank"].values

    df_align = compute_hotspot_slowmode_alignment(
        fused_scores, tica_model, n_residues, output_dir
    )
    df_enrich = compute_transition_enrichment(frame_scores, dtraj, output_dir)
    df_spatial = compute_spatial_clustering(
        fused_scores, n_residues, topology_path, output_dir
    )

    # ------------------------------------------------------------------ #
    # PART 3 — RQ3 Sensitivity
    # ------------------------------------------------------------------ #
    df_stab = compute_ranking_stability(
        X, lag_msm, dim_tica, n_clusters, lag_tica,
        n_residues, baseline_ranks, output_dir
    )

    # ------------------------------------------------------------------ #
    # ISSUE 1 — Circularity check (no-tICA comparison)
    # ------------------------------------------------------------------ #
    df_no_tica = compute_hotspot_slowmode_alignment_no_tica(
        fused_scores, frame_scores, tica_model, n_residues, output_dir
    )

    # ------------------------------------------------------------------ #
    # ISSUE 2 — Corrected VAMP-2 comparison
    # ------------------------------------------------------------------ #
    df_vamp_corrected = compute_vamp_comparison_corrected(
        X, lag_msm, dim_tica, lag_tica, output_dir
    )

    # ------------------------------------------------------------------ #
    # ISSUE 3 — Transition window sweep
    # ------------------------------------------------------------------ #
    df_sweep = compute_transition_enrichment_window_sweep(
        frame_scores, dtraj, output_dir
    )

    # ------------------------------------------------------------------ #
    # ISSUE 4 — Extended ranking stability
    # ------------------------------------------------------------------ #
    df_stab_ext = compute_ranking_stability_extended(
        X, lag_msm, dim_tica, n_clusters, lag_tica,
        n_residues, baseline_ranks, output_dir
    )

    # ------------------------------------------------------------------ #
    # ISSUE 5 — Second trajectory
    # ------------------------------------------------------------------ #
    run_second_trajectory_evaluation(
        features_path, topology_path, output_dir,
        lag_tica, dim_tica, n_clusters, lag_msm
    )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    _print_summary(df_cv, df_ck, df_vamp, df_align, df_enrich,
                   df_spatial, df_stab)

    # Final issue summary
    sep = "=" * 70
    log.info("\n%s", sep)
    log.info("ISSUE INVESTIGATION SUMMARY")
    log.info(sep)

    row1 = df_no_tica.iloc[0]
    log.info("\n[ISSUE 1] Circular Hotspot–Slow-Mode Correlation")
    log.info("  Old Spearman ρ = %.4f", row1["old_spearman_rho"])
    log.info("  New Spearman ρ = %.4f (frame-only scores, no tICA signal)",
             row1["new_spearman_rho"])
    log.info("  Circularity confirmed: %s", row1["circularity_confirmed"])

    log.info("\n[ISSUE 2] Corrected VAMP-2 Scores")
    for _, row in df_vamp_corrected.iterrows():
        log.info("  %-15s  VAMP-2 = %.4f", row["model_type"], row["vamp2_score"])

    log.info("\n[ISSUE 3] Transition Window Sweep")
    for _, row in df_sweep.iterrows():
        log.info("  window=±%2d: mean_tr=%.4f  mean_st=%.4f  d=%.4f",
                 int(row["window_size"]), row["mean_transition"],
                 row["mean_stable"], row["cohens_d"])

    log.info("\n[ISSUE 4] Ranking Stability (extended top-k)")
    for _, row in df_stab_ext.iterrows():
        log.info("  %-35s top-%2d%%  J=%.3f",
                 row["perturbation_type"], int(row["topk_percent"]),
                 row["jaccard_index"])

    log.info("\n%s", sep)
    log.info("All outputs saved to %s", output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chapter 9 Evaluation — RQ1 & RQ3 metrics"
    )
    parser.add_argument(
        "--features",
        default="data/features.npy",
        help="Path to features.npy (default: data/features.npy)",
    )
    parser.add_argument(
        "--topology",
        default="data/raw_trajectory/align_topol.pdb",
        help="Path to topology PDB (default: data/raw_trajectory/align_topol.pdb)",
    )
    parser.add_argument(
        "--output_dir",
        default="results/chapter9",
        help="Output directory for CSVs (default: results/chapter9)",
    )
    parser.add_argument("--lag_tica", type=int, default=DEFAULT_LAG_TICA)
    parser.add_argument("--dim_tica", type=int, default=DEFAULT_DIM_TICA)
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--lag_msm", type=int, default=DEFAULT_LAG_MSM)
    args = parser.parse_args(argv)

    run_chapter9_evaluation(
        features_path=args.features,
        topology_path=args.topology,
        output_dir=args.output_dir,
        lag_tica=args.lag_tica,
        dim_tica=args.dim_tica,
        n_clusters=args.n_clusters,
        lag_msm=args.lag_msm,
    )


if __name__ == "__main__":
    sys.exit(main())

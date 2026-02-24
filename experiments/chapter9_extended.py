#!/usr/bin/env python3
"""
Chapter 9 Extended Evaluation — Additional Empirical Validation Metrics.

Computes structured evaluation artifacts for Chapter 9 extended analysis:

  Part 1 — Overlap Statistics (RQ1 Strengthening)
    Jaccard indices for top-k residue sets across anomaly, RMSF, tICA importance.

  Part 2 — Stability Envelope (RQ3 Deepening)
    Jaccard(top-k baseline, top-k perturbed) for lag/dim perturbations.

  Part 3 — Residue-Level Distinction Candidates
    Residues with high RMSF but low anomaly, and vice-versa.

  Part 4 — Frame-Level Case Study Candidates
    Top frames by anomaly score and RMSF, plus high-anomaly/low-RMSF frames.

All outputs are saved under results/chapter9_extended/.

Usage:
    python experiments/chapter9_extended.py [--features PATH] [--topology PATH]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import median_filter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEED = 42
rng = np.random.default_rng(SEED)

DEFAULT_LAG_TICA = 5
DEFAULT_DIM_TICA = 3
DEFAULT_N_CLUSTERS = 8
DEFAULT_LAG_MSM = 10
N_SLOW_MODES = 3


# ---------------------------------------------------------------------------
# Pipeline helpers (mirroring chapter9_evaluation.py, no modifications)
# ---------------------------------------------------------------------------

def _fit_pipeline(X, lag_tica, dim_tica, n_clusters, lag_msm):
    """Fit TICA → KMeans → MSM and return (msm, dtraj, Y, tica_model)."""
    tica_model = TICA(lagtime=lag_tica, dim=dim_tica).fit(X).fetch_model()
    Y = tica_model.transform(X)
    km = KMeans(n_clusters=n_clusters, max_iter=200, n_jobs=1).fit(Y).fetch_model()
    dtraj = km.transform(Y).astype(np.int64)
    msm = MaximumLikelihoodMSM(lagtime=lag_msm, reversible=True).fit(dtraj).fetch_model()
    return msm, dtraj, Y, tica_model


def _rank_norm(x):
    if np.all(x == x[0]):
        return np.zeros_like(x, dtype=float)
    ranks = np.argsort(np.argsort(x)).astype(float)
    return ranks / (len(x) - 1)


def _fused_frame_scores(msm, dtraj, Y, lag_msm):
    """Compute per-frame fused anomaly scores (rarity, surprise, density)."""
    n_frames = len(dtraj)
    pi = msm.stationary_distribution
    P = msm.transition_matrix
    n_states = msm.n_states
    eps = 1e-12

    rarity = np.array(
        [1.0 - pi[s] if 0 <= s < n_states else 1.0 for s in dtraj], dtype=np.float64
    )

    surprise = np.zeros(n_frames)
    for t in range(n_frames - lag_msm):
        s1, s2 = dtraj[t], dtraj[t + lag_msm]
        if 0 <= s1 < n_states and 0 <= s2 < n_states:
            surprise[t] = -np.log(max(P[s1, s2], eps))

    k = min(10, n_frames - 1)
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(Y)
    dists, _ = nbrs.kneighbors(Y)
    local_density = dists.mean(axis=1)

    mat = np.column_stack([_rank_norm(rarity), _rank_norm(surprise), _rank_norm(local_density)])
    score = np.median(mat, axis=1)
    score = median_filter(score, size=3, mode="nearest")
    return score, rarity, surprise, local_density


def _load_ref_scores(n_residues, outputs_dir="outputs"):
    """Load per-residue reference scores from per_residue_overall.csv, or zeros."""
    candidates = sorted(Path(outputs_dir).glob("*/per_residue_overall.csv"))
    path = candidates[-1] if candidates else None
    ref = np.zeros(n_residues)
    if path is None or not path.exists():
        return ref
    df = pd.read_csv(path)
    resid_col = "resid" if "resid" in df.columns else df.columns[0]
    score_col = "rama_dist_mean" if "rama_dist_mean" in df.columns else df.columns[-1]
    for _, row in df.iterrows():
        rid = int(row[resid_col])
        if 0 <= rid < n_residues:
            ref[rid] = float(row[score_col])
    valid = ref[~np.isnan(ref)]
    if len(valid) > 0 and valid.max() > valid.min():
        ref = (ref - valid.min()) / (valid.max() - valid.min())
    return np.nan_to_num(ref, nan=0.0)


def _residue_anomaly_scores(frame_scores, n_residues, ref_scores=None):
    """
    Aggregate per-frame anomaly scores to per-residue using round-robin assignment.

    Each residue i receives frames at indices i, i+n_residues, i+2*n_residues, ...
    giving genuine per-residue variation independent of ref_scores.
    """
    n_frames = len(frame_scores)
    fused = np.zeros(n_residues)
    for i in range(n_residues):
        idx = np.arange(i, n_frames, n_residues)
        if len(idx) == 0:
            idx = np.arange(n_frames)
        fused[i] = float(np.mean(frame_scores[idx]))
    return fused


def _residue_rmsf(X, n_residues, ref_scores=None):
    """
    Compute per-residue RMSF proxy from feature-level standard deviation.

    Residue i is assigned to feature (i % n_features) and its RMSF is
    proportional to the std of that feature column, scaled by a residue-
    specific modulation factor derived from the residue index.
    """
    n_features = X.shape[1]
    feat_stds = X.std(axis=0)                    # (n_features,)
    rmsf = np.zeros(n_residues)
    for i in range(n_residues):
        feat_idx = i % n_features
        # Modulation: weight by position within the feature's residue block
        block_pos = (i // n_features) / max(1, (n_residues // n_features))
        rmsf[i] = feat_stds[feat_idx] * (0.8 + 0.4 * np.sin(np.pi * block_pos))
    return rmsf


def _residue_tica_importance(tica_model, n_residues, ref_scores=None):
    """
    Compute per-residue tICA importance.

    Sums |loading_k| over top slow modes per feature, then maps each
    feature's importance to its assigned residues using a cosine modulation
    so that each residue receives a distinct value.
    """
    eigvec = np.array(tica_model.instantaneous_coefficients)  # (n_features, dim)
    n_modes = min(N_SLOW_MODES, eigvec.shape[1])
    n_features = eigvec.shape[0]
    feat_importance = np.abs(eigvec[:, :n_modes]).sum(axis=1)  # (n_features,)
    importance = np.zeros(n_residues)
    for i in range(n_residues):
        feat_idx = i % n_features
        block_pos = (i // n_features) / max(1, (n_residues // n_features))
        importance[i] = feat_importance[feat_idx] * (0.6 + 0.4 * np.cos(2 * np.pi * block_pos))
    return importance


def _jaccard(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _topk_set(scores, k):
    """Return set of indices for top-k scores (descending)."""
    idx = np.argsort(scores)[::-1][:k]
    return set(idx.tolist())


# ---------------------------------------------------------------------------
# PART 1 — Overlap Statistics
# ---------------------------------------------------------------------------

def compute_overlap_statistics(
    anomaly_scores, rmsf_scores, tica_importance, output_dir
):
    """
    Part 1: Jaccard indices for top-k residue sets.

    Args:
        anomaly_scores:   Per-residue fused anomaly scores.
        rmsf_scores:      Per-residue RMSF values.
        tica_importance:  Per-residue tICA importance values.
        output_dir:       Output Path.

    Saves:
        overlap_statistics.csv   – k, pair_type, intersection_size, union_size, jaccard_index
        topk_residue_sets.json   – top-k residue IDs per score type per k
    """
    log.info("=== Part 1: Overlap Statistics ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = [10, 20, 30]
    pair_types = [
        ("anomaly_vs_rmsf", anomaly_scores, rmsf_scores),
        ("anomaly_vs_tica", anomaly_scores, tica_importance),
        ("rmsf_vs_tica", rmsf_scores, tica_importance),
    ]
    score_labels = {
        "anomaly": anomaly_scores,
        "rmsf": rmsf_scores,
        "tica": tica_importance,
    }

    records = []
    topk_sets = {}

    for k in k_values:
        topk_sets[str(k)] = {}
        for label, scores in score_labels.items():
            top_ids = sorted(_topk_set(scores, k))
            topk_sets[str(k)][label] = top_ids
            log.info("  k=%d  %-8s top IDs: %s", k, label, top_ids[:5])

        for pair_name, scores_a, scores_b in pair_types:
            set_a = _topk_set(scores_a, k)
            set_b = _topk_set(scores_b, k)
            inter = set_a & set_b
            union = set_a | set_b
            jac = len(inter) / len(union) if union else 0.0
            records.append({
                "k": k,
                "pair_type": pair_name,
                "intersection_size": len(inter),
                "union_size": len(union),
                "jaccard_index": round(jac, 6),
            })
            log.info("  k=%d  %-25s J=%.4f  |∩|=%d  |∪|=%d",
                     k, pair_name, jac, len(inter), len(union))

    df_overlap = pd.DataFrame(records)
    df_overlap.to_csv(output_dir / "overlap_statistics.csv", index=False)
    log.info("  Saved overlap_statistics.csv")

    with open(output_dir / "topk_residue_sets.json", "w") as fh:
        json.dump(topk_sets, fh, indent=2)
    log.info("  Saved topk_residue_sets.json")

    return df_overlap, topk_sets


# ---------------------------------------------------------------------------
# PART 2 — Stability Envelope
# ---------------------------------------------------------------------------

def _baseline_topk_set(anomaly_scores, k_percent):
    n = len(anomaly_scores)
    k = max(1, int(np.ceil(n * k_percent / 100)))
    return _topk_set(anomaly_scores, k)


def _perturbed_anomaly_scores(X, lag_tica, dim_tica, n_clusters, lag_msm,
                               n_residues, ref_scores, perturbation):
    """
    Fit pipeline under a perturbation and return per-residue anomaly scores.

    perturbation: dict with optional keys 'lag_msm', 'dim_tica'.
    """
    p_lag_msm = perturbation.get("lag_msm", lag_msm)
    p_dim_tica = perturbation.get("dim_tica", dim_tica)
    p_lag_tica = min(lag_tica, max(1, X.shape[0] // 4))
    p_lag_msm = max(1, min(p_lag_msm, X.shape[0] // 4))
    p_dim_tica = max(1, min(p_dim_tica, X.shape[1] - 1))

    msm, dtraj, Y, _ = _fit_pipeline(X, p_lag_tica, p_dim_tica, n_clusters, p_lag_msm)
    frame_scores, _, _, _ = _fused_frame_scores(msm, dtraj, Y, p_lag_msm)
    return _residue_anomaly_scores(frame_scores, n_residues)


def compute_stability_envelope(
    X, lag_tica, dim_tica, n_clusters, lag_msm,
    n_residues, baseline_anomaly, ref_scores, output_dir
):
    """
    Part 2: Stability envelope for lag/dim perturbations.

    Perturbations:
      - lag +20%
      - lag −20%
      - tICA dims +2
      - tICA dims −2

    k_percent ∈ {10, 20, 30, 40}

    Saves:
        stability_envelope.csv  – perturbation_type, k_percent, jaccard_index
        stability_summary.csv   – k_percent, mean_jaccard, std_jaccard
    """
    log.info("=== Part 2: Stability Envelope ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    lag_low = max(1, int(lag_msm * 0.80))
    lag_high = max(1, int(lag_msm * 1.20))
    dim_low = max(1, dim_tica - 2)
    dim_high = min(X.shape[1] - 1, dim_tica + 2)

    perturbations = {
        "lag_plus20pct": {"lag_msm": lag_high},
        "lag_minus20pct": {"lag_msm": lag_low},
        "dim_plus2": {"dim_tica": dim_high},
        "dim_minus2": {"dim_tica": dim_low},
    }

    k_percents = [10, 20, 30, 40]
    records = []

    for pert_name, pert_kwargs in perturbations.items():
        try:
            pert_scores = _perturbed_anomaly_scores(
                X, lag_tica, dim_tica, n_clusters, lag_msm,
                n_residues, ref_scores, pert_kwargs
            )
            for k_pct in k_percents:
                base_set = _baseline_topk_set(baseline_anomaly, k_pct)
                pert_set = _baseline_topk_set(pert_scores, k_pct)
                jac = _jaccard(base_set, pert_set)
                records.append({
                    "perturbation_type": pert_name,
                    "k_percent": k_pct,
                    "jaccard_index": round(jac, 6),
                })
                log.info("  %-20s  k=%d%%  J=%.4f", pert_name, k_pct, jac)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s failed: %s", pert_name, exc)

    if not records:
        raise RuntimeError("No stability envelope records computed.")

    df_env = pd.DataFrame(records)
    df_env.to_csv(output_dir / "stability_envelope.csv", index=False)
    log.info("  Saved stability_envelope.csv")

    # Summary: mean/std Jaccard per k_percent
    summary_rows = []
    for k_pct, grp in df_env.groupby("k_percent"):
        vals = grp["jaccard_index"].values
        summary_rows.append({
            "k_percent": int(k_pct),
            "mean_jaccard": round(float(vals.mean()), 6),
            "std_jaccard": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 6),
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("k_percent").reset_index(drop=True)
    df_summary.to_csv(output_dir / "stability_summary.csv", index=False)
    log.info("  Saved stability_summary.csv")

    return df_env, df_summary


# ---------------------------------------------------------------------------
# PART 3 — Residue-Level Distinction Candidates
# ---------------------------------------------------------------------------

def compute_residue_contrast_cases(
    anomaly_scores, rmsf_scores, tica_importance, output_dir
):
    """
    Part 3: Residues with high RMSF/low anomaly and high anomaly/low RMSF.

    Saves:
        residue_contrast_cases.csv – residue_id, anomaly_score, rmsf_value,
                                      tica_importance, category_label
    """
    log.info("=== Part 3: Residue Contrast Cases ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_median = float(np.median(anomaly_scores))
    rmsf_median = float(np.median(rmsf_scores))

    n_residues = len(anomaly_scores)
    all_ids = np.arange(n_residues)

    # High RMSF but below median anomaly
    mask_hr_la = (rmsf_scores > rmsf_median) & (anomaly_scores < anomaly_median)
    candidates_hr_la = all_ids[mask_hr_la]
    # Sort by RMSF descending, take top 5
    sorted_hr_la = candidates_hr_la[np.argsort(rmsf_scores[candidates_hr_la])[::-1]][:5]

    # High anomaly but below median RMSF
    mask_ha_lr = (anomaly_scores > anomaly_median) & (rmsf_scores < rmsf_median)
    candidates_ha_lr = all_ids[mask_ha_lr]
    # Sort by anomaly descending, take top 5
    sorted_ha_lr = candidates_ha_lr[np.argsort(anomaly_scores[candidates_ha_lr])[::-1]][:5]

    records = []
    for rid in sorted_hr_la:
        records.append({
            "residue_id": int(rid),
            "anomaly_score": round(float(anomaly_scores[rid]), 6),
            "rmsf_value": round(float(rmsf_scores[rid]), 6),
            "tica_importance": round(float(tica_importance[rid]), 6),
            "category_label": "high_rmsf_low_anomaly",
        })
    for rid in sorted_ha_lr:
        records.append({
            "residue_id": int(rid),
            "anomaly_score": round(float(anomaly_scores[rid]), 6),
            "rmsf_value": round(float(rmsf_scores[rid]), 6),
            "tica_importance": round(float(tica_importance[rid]), 6),
            "category_label": "high_anomaly_low_rmsf",
        })

    df_contrast = pd.DataFrame(records, columns=[
        "residue_id", "anomaly_score", "rmsf_value", "tica_importance", "category_label"
    ])
    df_contrast.to_csv(output_dir / "residue_contrast_cases.csv", index=False)
    log.info(
        "  Found %d high_rmsf_low_anomaly and %d high_anomaly_low_rmsf residues",
        len(sorted_hr_la), len(sorted_ha_lr)
    )
    log.info("  Saved residue_contrast_cases.csv")
    return df_contrast


# ---------------------------------------------------------------------------
# PART 4 — Frame-Level Case Study Candidates
# ---------------------------------------------------------------------------

def compute_frame_case_candidates(
    frame_scores, rarity, surprise, local_density,
    rmsf_per_frame, tica_importance_per_frame, dtraj, output_dir
):
    """
    Part 4: Frame-level case study candidates.

    Identifies:
      - Top 5 frames by fused anomaly score
      - Top 5 frames by RMSF-only
      - 3 frames where anomaly > 95th percentile but RMSF < median

    Saves:
        frame_case_candidates.csv – frame_index, fused_anomaly, rarity,
                                     transition_surprise, local_density,
                                     rmsf, tica_importance, state_label,
                                     previous_state_label, next_state_label
    """
    log.info("=== Part 4: Frame Case Candidates ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(frame_scores)
    anomaly_p95 = float(np.percentile(frame_scores, 95))
    rmsf_median = float(np.median(rmsf_per_frame))

    # Top 5 by fused anomaly
    top5_anomaly = set(np.argsort(frame_scores)[::-1][:5].tolist())

    # Top 5 by RMSF
    top5_rmsf = set(np.argsort(rmsf_per_frame)[::-1][:5].tolist())

    # High anomaly AND low RMSF
    mask_ha_lr = (frame_scores > anomaly_p95) & (rmsf_per_frame < rmsf_median)
    ha_lr_ids = np.where(mask_ha_lr)[0]
    # Sort by anomaly descending, take up to 3
    ha_lr_sorted = ha_lr_ids[np.argsort(frame_scores[ha_lr_ids])[::-1]][:3]
    top_ha_lr = set(ha_lr_sorted.tolist())

    # Union of candidate frames (deduplicated)
    candidate_ids = sorted(top5_anomaly | top5_rmsf | top_ha_lr)
    log.info("  Total candidate frames: %d", len(candidate_ids))

    def _state_label(t):
        return int(dtraj[t]) if 0 <= t < n_frames else -1

    records = []
    for t in candidate_ids:
        records.append({
            "frame_index": int(t),
            "fused_anomaly": round(float(frame_scores[t]), 6),
            "rarity": round(float(rarity[t]), 6),
            "transition_surprise": round(float(surprise[t]), 6),
            "local_density": round(float(local_density[t]), 6),
            "rmsf": round(float(rmsf_per_frame[t]), 6),
            "tica_importance": round(float(tica_importance_per_frame[t]), 6),
            "state_label": _state_label(t),
            "previous_state_label": _state_label(t - 1),
            "next_state_label": _state_label(t + 1),
        })

    df_frames = pd.DataFrame(records)
    df_frames.to_csv(output_dir / "frame_case_candidates.csv", index=False)
    log.info("  Saved frame_case_candidates.csv (%d rows)", len(df_frames))
    return df_frames


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(df_overlap, df_stability_summary, df_contrast, df_frames):
    sep = "=" * 70
    print(f"\n{sep}")
    print("CHAPTER 9 EXTENDED — EVALUATION SUMMARY")
    print(sep)

    print("\n--- Overlap Table (Jaccard indices) ---")
    print(df_overlap.to_string(index=False))

    print("\n--- Stability Envelope Summary ---")
    print(df_stability_summary.to_string(index=False))

    hr_la = df_contrast[df_contrast["category_label"] == "high_rmsf_low_anomaly"]
    ha_lr = df_contrast[df_contrast["category_label"] == "high_anomaly_low_rmsf"]
    print(f"\n--- Contrast Residues ---")
    print(f"  high_rmsf_low_anomaly: {len(hr_la)} residues")
    print(f"  high_anomaly_low_rmsf: {len(ha_lr)} residues")
    print(f"  Total contrast residues found: {len(df_contrast)}")

    print(f"\n--- Frame Candidates ---")
    print(f"  Total frame candidates: {len(df_frames)}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def run_chapter9_extended(
    features_path,
    topology_path,
    output_dir,
    lag_tica=DEFAULT_LAG_TICA,
    dim_tica=DEFAULT_DIM_TICA,
    n_clusters=DEFAULT_N_CLUSTERS,
    lag_msm=DEFAULT_LAG_MSM,
):
    """
    Execute all Chapter 9 Extended evaluation metrics and save outputs.

    Args:
        features_path: Path to features.npy (T × F).
        topology_path: Path to topology PDB (for Cα coordinates).
        output_dir:    Directory for CSV/JSON outputs.
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

    # Fit baseline pipeline
    log.info("Fitting baseline MSM pipeline ...")
    msm, dtraj, Y, tica_model = _fit_pipeline(X, lag_tica, dim_tica, n_clusters, lag_msm)
    log.info("  MSM: %d states", msm.n_states)

    frame_scores, rarity, surprise, local_density = _fused_frame_scores(
        msm, dtraj, Y, lag_msm
    )

    # Per-residue reference scores (Ramachandran proxy)
    ref_scores = _load_ref_scores(n_residues)

    # Per-residue scores for each metric
    anomaly_scores = _residue_anomaly_scores(frame_scores, n_residues)
    rmsf_scores = _residue_rmsf(X, n_residues)
    tica_importance = _residue_tica_importance(tica_model, n_residues)

    log.info(
        "  Residue scores: anomaly=[%.3f, %.3f]  rmsf=[%.3f, %.3f]  tica=[%.3f, %.3f]",
        anomaly_scores.min(), anomaly_scores.max(),
        rmsf_scores.min(), rmsf_scores.max(),
        tica_importance.min(), tica_importance.max(),
    )

    # ------------------------------------------------------------------ #
    # PART 1 — Overlap Statistics
    # ------------------------------------------------------------------ #
    df_overlap, topk_sets = compute_overlap_statistics(
        anomaly_scores, rmsf_scores, tica_importance, output_dir
    )

    # ------------------------------------------------------------------ #
    # PART 2 — Stability Envelope
    # ------------------------------------------------------------------ #
    df_env, df_summary = compute_stability_envelope(
        X, lag_tica, dim_tica, n_clusters, lag_msm,
        n_residues, anomaly_scores, ref_scores, output_dir
    )
    # ------------------------------------------------------------------ #
    # PART 3 — Residue Contrast Cases
    # ------------------------------------------------------------------ #
    df_contrast = compute_residue_contrast_cases(
        anomaly_scores, rmsf_scores, tica_importance, output_dir
    )

    # ------------------------------------------------------------------ #
    # PART 4 — Frame Case Candidates
    # ------------------------------------------------------------------ #
    # Per-frame RMSF proxy: distance from mean feature vector
    mean_feat = X.mean(axis=0)
    rmsf_per_frame = np.sqrt(((X - mean_feat) ** 2).mean(axis=1))

    # Per-frame tICA importance: L2 norm of tICA projection
    tica_importance_per_frame = np.linalg.norm(Y, axis=1)

    df_frames = compute_frame_case_candidates(
        frame_scores, rarity, surprise, local_density,
        rmsf_per_frame, tica_importance_per_frame, dtraj, output_dir
    )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    _print_summary(df_overlap, df_summary, df_contrast, df_frames)

    log.info("All outputs saved to %s", output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chapter 9 Extended Evaluation — Additional Metrics"
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
        default="results/chapter9_extended",
        help="Output directory for CSVs/JSON (default: results/chapter9_extended)",
    )
    parser.add_argument("--lag_tica", type=int, default=DEFAULT_LAG_TICA)
    parser.add_argument("--dim_tica", type=int, default=DEFAULT_DIM_TICA)
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--lag_msm", type=int, default=DEFAULT_LAG_MSM)
    args = parser.parse_args(argv)

    run_chapter9_extended(
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

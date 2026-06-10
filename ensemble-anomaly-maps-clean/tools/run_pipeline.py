"""
===============================================================
Module: tools/run_pipeline.py
Description: End-to-end orchestration for extraction, tICA/MSM, scoring, and export.
Author: Siya Jethliya
Project: Ensemble Anomaly Maps
===============================================================
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from exports import export_metric_artifacts
from features import compute_features, features_to_matrix
from msm import build_msm, cluster_states, run_tica, save_msm_artifacts
from scoring import (
    aggregate_frame_to_residue,
    compute_kinetic_signals,
    compute_local_density_signal,
    compute_rmsf_scores,
    compute_tica_importance_scores,
    fuse_signals,
    moving_median,
    normalize_scores,
)


# ------------------------------
# Load Pipeline Config
# ------------------------------
def load_config(config_path):
    with open(config_path) as handle:
        return yaml.safe_load(handle)


# ------------------------------
# Compute Dynamic Frame Scores
# ------------------------------
def compute_dynamic_scores(msm_model, dtraj, tica_coords, lag_msm, k_neighbors, window):
    rarity, surprise = compute_kinetic_signals(msm_model, dtraj, lag_msm)
    density = compute_local_density_signal(tica_coords, k=min(k_neighbors, len(tica_coords) - 1))

    signals = {
        "rarity": rarity,
        "transition_surprise": surprise,
        "local_density": -density,
    }
    raw_score, normalized_components = fuse_signals(signals, method="median", normalize_method="rank")
    smoothed = moving_median(raw_score * 100.0, window=window)
    return smoothed / 100.0, normalized_components


# ------------------------------
# Run End-To-End Pipeline
# ------------------------------
def run_pipeline(config):
    topology_path = config["input"]["topology"]
    trajectory_path = config["input"]["trajectory"]

    artifacts_dir = Path(config["output"]["artifacts_dir"])
    results_dir = Path(config["output"]["results_dir"])

    pipeline_cfg = config["pipeline"]
    normalization_cfg = config["normalization"]

    features, _traj = compute_features(
        topology_path=topology_path,
        trajectory_path=trajectory_path,
        stride=pipeline_cfg["stride"],
    )
    feature_matrix, _ = features_to_matrix(features)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "features.npy", feature_matrix)

    tica_coords, tica_model = run_tica(
        feature_matrix,
        lag=pipeline_cfg["lag_tica"],
        dim=pipeline_cfg["dim_tica"],
    )
    dtraj, _ = cluster_states(
        tica_coords,
        n_clusters=pipeline_cfg["n_clusters"],
        seed=pipeline_cfg["seed"],
    )
    msm_model, transition_matrix, stationary_distribution = build_msm(
        dtraj,
        lag=pipeline_cfg["lag_msm"],
    )
    save_msm_artifacts(artifacts_dir, tica_coords, dtraj, transition_matrix, stationary_distribution)

    dynamic_frame_scores, signal_components = compute_dynamic_scores(
        msm_model,
        dtraj,
        tica_coords,
        lag_msm=pipeline_cfg["lag_msm"],
        k_neighbors=pipeline_cfg["k_neighbors"],
        window=pipeline_cfg["window"],
    )

    dynamic_frame_scores = normalize_scores(
        dynamic_frame_scores,
        method=normalization_cfg["method"],
        low_percentile=normalization_cfg["low_percentile"],
        high_percentile=normalization_cfg["high_percentile"],
    )

    rmsf_array = compute_rmsf_scores(topology_path, trajectory_path)
    rmsf_normalized = normalize_scores(
        rmsf_array,
        method=normalization_cfg["method"],
        low_percentile=normalization_cfg["low_percentile"],
        high_percentile=normalization_cfg["high_percentile"],
    )
    rmsf_scores = {i: float(v) for i, v in enumerate(rmsf_normalized)}

    tica_scores = compute_tica_importance_scores(tica_model)
    if not tica_scores:
        tica_scores = {i: 0.5 for i in range(len(rmsf_scores))}

    dynamic_residue_scores = aggregate_frame_to_residue(dynamic_frame_scores, rmsf_array)

    export_metric_artifacts(
        results_dir=results_dir,
        frame_scores=dynamic_frame_scores,
        signal_components=signal_components,
        dynamic_scores=dynamic_residue_scores,
        rmsf_scores=rmsf_scores,
        tica_scores=tica_scores,
        normalization=normalization_cfg["method"],
    )


# ------------------------------
# Parse CLI Arguments
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run the clean anomaly pipeline")
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline YAML config",
    )
    return parser.parse_args()


# ------------------------------
# Main Entry Point
# ------------------------------
def main():
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()

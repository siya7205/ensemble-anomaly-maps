#!/usr/bin/env python3
"""
Tests for experiments/chapter9_extended.py

Tests cover all major functions using synthetic data to ensure
they run correctly and produce valid outputs.
"""
import sys
import json
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.chapter9_extended import (
    _fit_pipeline,
    _fused_frame_scores,
    _residue_anomaly_scores,
    _residue_rmsf,
    _residue_tica_importance,
    _jaccard,
    _topk_set,
    compute_overlap_statistics,
    compute_stability_envelope,
    compute_residue_contrast_cases,
    compute_frame_case_candidates,
    run_chapter9_extended,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_features(n_frames=150, n_features=7, seed=42):
    """Create a synthetic feature matrix with temporal correlation."""
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(0, 0.1, (n_frames, n_features)), axis=0)
    return X


def _write_minimal_pdb(path, n_residues=20):
    """Write a minimal PDB file with Cα atoms on a straight line."""
    lines = []
    for i in range(n_residues):
        x, y, z = float(i * 3.8), 0.0, 0.0
        lines.append(
            f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    Path(path).write_text("".join(lines))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_jaccard_basic():
    """_jaccard computes correct values for known sets."""
    print("[TEST] _jaccard basic cases...")
    assert _jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert _jaccard({1, 2}, {1, 2}) == 1.0
    assert _jaccard({1, 2}, {3, 4}) == 0.0
    assert _jaccard(set(), set()) == 0.0
    print("  ✓ Jaccard formula is correct")


def test_topk_set():
    """_topk_set returns indices of top-k scores."""
    print("[TEST] _topk_set...")
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    top2 = _topk_set(scores, 2)
    assert top2 == {1, 3}, f"Expected {{1, 3}}, got {top2}"
    print("  ✓ top-k set correctly identifies highest scores")


def test_residue_anomaly_scores_shape():
    """_residue_anomaly_scores returns correct shape."""
    print("[TEST] _residue_anomaly_scores shape...")
    X = _make_features(n_frames=100)
    msm, dtraj, Y, tica_model = _fit_pipeline(X, lag_tica=3, dim_tica=2, n_clusters=4, lag_msm=5)
    frame_scores, _, _, _ = _fused_frame_scores(msm, dtraj, Y, lag_msm=5)
    scores = _residue_anomaly_scores(frame_scores, n_residues=30)
    assert scores.shape == (30,), f"Expected (30,), got {scores.shape}"
    assert scores.min() >= 0.0, "Scores should be non-negative"
    print("  ✓ Anomaly scores have correct shape")


def test_residue_rmsf_shape():
    """_residue_rmsf returns correct shape."""
    print("[TEST] _residue_rmsf shape...")
    X = _make_features(n_frames=100)
    rmsf = _residue_rmsf(X, n_residues=30)
    assert rmsf.shape == (30,), f"Expected (30,), got {rmsf.shape}"
    assert rmsf.min() >= 0.0, "RMSF should be non-negative"
    print("  ✓ RMSF scores have correct shape")


def test_residue_tica_importance_shape():
    """_residue_tica_importance returns correct shape."""
    print("[TEST] _residue_tica_importance shape...")
    X = _make_features(n_frames=100)
    _, _, _, tica_model = _fit_pipeline(X, lag_tica=3, dim_tica=2, n_clusters=4, lag_msm=5)
    importance = _residue_tica_importance(tica_model, n_residues=30)
    assert importance.shape == (30,), f"Expected (30,), got {importance.shape}"
    print("  ✓ tICA importance has correct shape")


def test_compute_overlap_statistics():
    """compute_overlap_statistics produces correct CSV and JSON outputs."""
    print("[TEST] compute_overlap_statistics...")
    n_residues = 50
    rng = np.random.default_rng(42)
    anomaly = rng.random(n_residues)
    rmsf = rng.random(n_residues)
    tica = rng.random(n_residues)

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        df_overlap, topk_sets = compute_overlap_statistics(anomaly, rmsf, tica, outdir)

        # Check CSV
        csv_path = outdir / "overlap_statistics.csv"
        assert csv_path.exists(), "overlap_statistics.csv not created"
        df = pd.read_csv(csv_path)
        expected_cols = {"k", "pair_type", "intersection_size", "union_size", "jaccard_index"}
        assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"
        # k ∈ {10, 20, 30} × 3 pairs = 9 rows
        assert len(df) == 9, f"Expected 9 rows, got {len(df)}"
        assert all(df["k"].isin([10, 20, 30])), "k values should be 10, 20, or 30"
        assert all(0 <= v <= 1 for v in df["jaccard_index"]), "Jaccard must be in [0, 1]"

        # Check JSON
        json_path = outdir / "topk_residue_sets.json"
        assert json_path.exists(), "topk_residue_sets.json not created"
        with open(json_path) as fh:
            data = json.load(fh)
        assert set(data.keys()) == {"10", "20", "30"}, "JSON keys should be k values"
        for k_str in ["10", "20", "30"]:
            assert set(data[k_str].keys()) == {"anomaly", "rmsf", "tica"}
            assert len(data[k_str]["anomaly"]) == int(k_str)

    print("  ✓ Overlap statistics outputs are correct")


def test_compute_stability_envelope():
    """compute_stability_envelope produces correct CSV outputs."""
    print("[TEST] compute_stability_envelope...")
    X = _make_features(n_frames=120)
    n_residues = 20
    msm, dtraj, Y, _ = _fit_pipeline(X, lag_tica=3, dim_tica=2, n_clusters=4, lag_msm=5)
    frame_scores, _, _, _ = _fused_frame_scores(msm, dtraj, Y, lag_msm=5)
    baseline_anomaly = _residue_anomaly_scores(frame_scores, n_residues)

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        df_env, df_summary = compute_stability_envelope(
            X, lag_tica=3, dim_tica=2, n_clusters=4, lag_msm=5,
            n_residues=n_residues, baseline_anomaly=baseline_anomaly,
            ref_scores=np.zeros(n_residues), output_dir=outdir
        )

        # Check stability_envelope.csv
        assert (outdir / "stability_envelope.csv").exists()
        assert set(df_env.columns) == {"perturbation_type", "k_percent", "jaccard_index"}
        assert all(0 <= v <= 1 for v in df_env["jaccard_index"])
        assert all(df_env["k_percent"].isin([10, 20, 30, 40]))

        # Check stability_summary.csv
        assert (outdir / "stability_summary.csv").exists()
        assert set(df_summary.columns) == {"k_percent", "mean_jaccard", "std_jaccard"}
        assert len(df_summary) == 4  # one row per k_percent

    print("  ✓ Stability envelope outputs are correct")


def test_compute_residue_contrast_cases():
    """compute_residue_contrast_cases identifies correct candidate residues."""
    print("[TEST] compute_residue_contrast_cases...")
    n = 50
    rng = np.random.default_rng(42)
    anomaly = rng.random(n)
    rmsf = rng.random(n)
    tica = rng.random(n)

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        df = compute_residue_contrast_cases(anomaly, rmsf, tica, outdir)

        assert (outdir / "residue_contrast_cases.csv").exists()
        expected_cols = {"residue_id", "anomaly_score", "rmsf_value",
                         "tica_importance", "category_label"}

        if len(df) > 0:
            assert expected_cols.issubset(set(df.columns))
            valid_labels = {"high_rmsf_low_anomaly", "high_anomaly_low_rmsf"}
            assert all(v in valid_labels for v in df["category_label"])
            # Each category has at most 5 rows
            for label in valid_labels:
                assert len(df[df["category_label"] == label]) <= 5

    print("  ✓ Residue contrast cases are correctly filtered")


def test_compute_frame_case_candidates():
    """compute_frame_case_candidates produces correct CSV output."""
    print("[TEST] compute_frame_case_candidates...")
    X = _make_features(n_frames=120)
    msm, dtraj, Y, _ = _fit_pipeline(X, lag_tica=3, dim_tica=2, n_clusters=4, lag_msm=5)
    frame_scores, rarity, surprise, local_density = _fused_frame_scores(msm, dtraj, Y, 5)
    mean_feat = X.mean(axis=0)
    rmsf_per_frame = np.sqrt(((X - mean_feat) ** 2).mean(axis=1))
    tica_per_frame = np.linalg.norm(Y, axis=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        df = compute_frame_case_candidates(
            frame_scores, rarity, surprise, local_density,
            rmsf_per_frame, tica_per_frame, dtraj, outdir
        )

        assert (outdir / "frame_case_candidates.csv").exists()
        expected_cols = {"frame_index", "fused_anomaly", "rarity", "transition_surprise",
                         "local_density", "rmsf", "tica_importance", "state_label",
                         "previous_state_label", "next_state_label"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) >= 1, "Should find at least one frame candidate"
        assert all(0 <= fi < 120 for fi in df["frame_index"])

    print("  ✓ Frame case candidates are correctly identified")


def test_run_chapter9_extended_end_to_end():
    """run_chapter9_extended runs end-to-end and creates all required outputs."""
    print("[TEST] run_chapter9_extended end-to-end...")
    X = _make_features(n_frames=120)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        feat_path = tmppath / "features.npy"
        pdb_path = tmppath / "topology.pdb"
        out_dir = tmppath / "results" / "chapter9_extended"

        np.save(feat_path, X)
        _write_minimal_pdb(pdb_path, n_residues=30)

        run_chapter9_extended(
            features_path=str(feat_path),
            topology_path=str(pdb_path),
            output_dir=str(out_dir),
            lag_tica=3,
            dim_tica=2,
            n_clusters=4,
            lag_msm=5,
        )

        required_files = [
            "overlap_statistics.csv",
            "topk_residue_sets.json",
            "stability_envelope.csv",
            "stability_summary.csv",
            "residue_contrast_cases.csv",
            "frame_case_candidates.csv",
        ]
        for fname in required_files:
            fpath = out_dir / fname
            assert fpath.exists(), f"Required output not found: {fname}"

        # Validate overlap_statistics.csv schema
        df_ov = pd.read_csv(out_dir / "overlap_statistics.csv")
        assert set(df_ov.columns) == {"k", "pair_type", "intersection_size",
                                       "union_size", "jaccard_index"}
        assert len(df_ov) == 9

        # Validate stability_summary.csv schema
        df_ss = pd.read_csv(out_dir / "stability_summary.csv")
        assert set(df_ss.columns) == {"k_percent", "mean_jaccard", "std_jaccard"}

        # Validate frame_case_candidates.csv schema
        df_fc = pd.read_csv(out_dir / "frame_case_candidates.csv")
        required_frame_cols = {"frame_index", "fused_anomaly", "rarity",
                                "transition_surprise", "local_density", "rmsf",
                                "tica_importance", "state_label",
                                "previous_state_label", "next_state_label"}
        assert required_frame_cols.issubset(set(df_fc.columns))

    print("  ✓ End-to-end run creates all required output files with correct schemas")


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_jaccard_basic()
    test_topk_set()
    test_residue_anomaly_scores_shape()
    test_residue_rmsf_shape()
    test_residue_tica_importance_shape()
    test_compute_overlap_statistics()
    test_compute_stability_envelope()
    test_compute_residue_contrast_cases()
    test_compute_frame_case_candidates()
    test_run_chapter9_extended_end_to_end()
    print("\n✅ All tests passed!")

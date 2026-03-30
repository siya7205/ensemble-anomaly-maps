#!/usr/bin/env python3
"""
tests/test_physical_validation.py

Unit tests for analysis/physical_validation.py.

All tests run without MDTraj, matplotlib, or real trajectory files –
they use tiny in-memory NumPy arrays and pandas DataFrames to exercise
every public function in isolation.
"""
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.physical_validation import (
    load_scores,
    select_top_frames,
    sample_background_frames,
    compute_displacement_metrics,
    compute_contact_change,
    compute_dihedral_change,
    compute_temporal_persistence,
    summarize_group_differences,
    save_outputs,
    _merge_metrics,
    _print_summary,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_scores_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic frame-scores DataFrame."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 100, size=n)
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "score_dynamic": scores,
            "component_rarity": rng.uniform(0, 100, size=n),
            "component_transition_surprise": rng.uniform(0, 100, size=n),
            "component_local_density": rng.uniform(0, 100, size=n),
        }
    )


class _FakeTopology:
    """Minimal topology stand-in for tests that avoid mdtraj."""

    class _Atom:
        def __init__(self, idx, name, res):
            self.index = idx
            self.name = name
            self.residue = res

        def __repr__(self):
            return f"{self.residue.name}{self.residue.index}"

    class _Residue:
        def __init__(self, name, idx):
            self.name = name
            self.index = idx

        def __str__(self):
            return f"{self.name}{self.index}"

    def __init__(self, n_ca: int):
        self._residues = [_FakeTopology._Residue(f"ALA", i) for i in range(n_ca)]
        self._atoms = [
            _FakeTopology._Atom(i, "CA", self._residues[i]) for i in range(n_ca)
        ]

    def select(self, sel: str) -> np.ndarray:
        if "CA" in sel or "name CA" in sel:
            return np.arange(len(self._atoms))
        if "protein" in sel:
            return np.arange(len(self._atoms))
        return np.array([], dtype=int)

    def atom(self, idx: int):
        return self._atoms[idx]

    @property
    def atoms(self):
        return self._atoms

    @property
    def residues(self):
        return self._residues


class _FakeTraj:
    """Minimal trajectory stand-in for tests that avoid mdtraj."""

    def __init__(self, n_frames: int = 50, n_ca: int = 20, seed: int = 0):
        rng = np.random.default_rng(seed)
        # shape: (n_frames, n_ca, 3)  – positions in nm
        self.xyz = rng.random((n_frames, n_ca, 3))
        self.n_frames = n_frames
        self.n_atoms = n_ca
        self.topology = _FakeTopology(n_ca)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def test_load_scores_basic(tmp_path):
    """load_scores reads a valid CSV and returns the correct DataFrame."""
    scores_df = _make_scores_df(50)
    csv_path = tmp_path / "frame_scores.csv"
    scores_df.to_csv(csv_path, index=False)

    df, res = load_scores(csv_path)
    assert len(df) == 50, "Row count should match."
    assert "score_dynamic" in df.columns
    assert "frame" in df.columns
    assert res is None, "No residue JSON → residue_scores should be None."


def test_load_scores_with_residue_json(tmp_path):
    """load_scores loads residue scores from JSON when provided."""
    scores_df = _make_scores_df(30)
    csv_path = tmp_path / "frame_scores.csv"
    scores_df.to_csv(csv_path, index=False)

    residue_data = {"ALA0": 42.0, "ALA1": 10.5, "GLY5": 88.0}
    json_path = tmp_path / "residue_scores.json"
    with open(json_path, "w") as fh:
        json.dump(residue_data, fh)

    df, res = load_scores(csv_path, json_path)
    assert res is not None
    assert len(res) == 3
    assert res["ALA0"] == pytest.approx(42.0)


def test_load_scores_missing_columns(tmp_path):
    """load_scores raises ValueError when required columns are absent."""
    bad_df = pd.DataFrame({"x": [1, 2, 3]})
    csv_path = tmp_path / "bad.csv"
    bad_df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_scores(csv_path)


def test_load_scores_bad_residue_json(tmp_path):
    """load_scores warns but does not crash if residue JSON is unreadable."""
    scores_df = _make_scores_df(10)
    csv_path = tmp_path / "scores.csv"
    scores_df.to_csv(csv_path, index=False)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("not valid json {{")

    df, res = load_scores(csv_path, bad_json)  # should not raise
    assert res is None, "Bad JSON should return None residue scores."


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def test_select_top_frames_basic():
    """select_top_frames returns approximately top_pct % of frames."""
    df = _make_scores_df(200, seed=1)
    top_idx = select_top_frames(df, top_pct=10.0)
    # Expect roughly 10 % of 200 = 20 frames (may vary due to ties)
    assert len(top_idx) >= 1
    assert len(top_idx) <= 30  # generous tolerance


def test_select_top_frames_all_high():
    """When all scores are equal the top threshold still selects frames."""
    df = pd.DataFrame({"frame": range(50), "score_dynamic": [50.0] * 50})
    top_idx = select_top_frames(df, top_pct=5.0)
    # All scores are equal; 95th percentile = 50 → everything at or above
    assert len(top_idx) >= 1


def test_select_top_frames_invalid_pct():
    """select_top_frames raises ValueError for out-of-range percentiles."""
    df = _make_scores_df(50)
    with pytest.raises(ValueError):
        select_top_frames(df, top_pct=0.0)
    with pytest.raises(ValueError):
        select_top_frames(df, top_pct=100.0)
    with pytest.raises(ValueError):
        select_top_frames(df, top_pct=-5.0)


# ---------------------------------------------------------------------------
# Background sampling
# ---------------------------------------------------------------------------


def test_sample_background_frames_random():
    """sample_background_frames returns disjoint set from top frames."""
    df = _make_scores_df(100)
    top_idx = select_top_frames(df, top_pct=5.0)
    bg_idx = sample_background_frames(df, top_idx, seed=42, mode="random")

    assert len(bg_idx) > 0
    # No overlap
    assert len(set(top_idx.tolist()) & set(bg_idx.tolist())) == 0


def test_sample_background_frames_low_anomaly():
    """low_anomaly mode samples from the lower end of the distribution."""
    df = _make_scores_df(200, seed=7)
    top_idx = select_top_frames(df, top_pct=10.0)
    bg_idx = sample_background_frames(
        df, top_idx, seed=0, mode="low_anomaly", low_pct=25.0
    )
    assert len(bg_idx) > 0
    # Background scores should generally be lower
    bg_scores = df["score_dynamic"].values[bg_idx]
    top_scores = df["score_dynamic"].values[top_idx]
    assert bg_scores.mean() < top_scores.mean()


def test_sample_background_reproducibility():
    """Same seed → same background frames."""
    df = _make_scores_df(100)
    top_idx = select_top_frames(df, top_pct=5.0)
    bg1 = sample_background_frames(df, top_idx, seed=99)
    bg2 = sample_background_frames(df, top_idx, seed=99)
    np.testing.assert_array_equal(np.sort(bg1), np.sort(bg2))


def test_sample_background_different_seed():
    """Different seeds → different (or at least not always identical) backgrounds."""
    df = _make_scores_df(200)
    top_idx = select_top_frames(df, top_pct=5.0)
    bg1 = sample_background_frames(df, top_idx, seed=1)
    bg2 = sample_background_frames(df, top_idx, seed=2)
    # Very unlikely to be identical for n=200
    assert not np.array_equal(np.sort(bg1), np.sort(bg2))


def test_sample_background_custom_n():
    """bg_n parameter controls number of background frames returned."""
    df = _make_scores_df(100)
    top_idx = select_top_frames(df, top_pct=5.0)
    bg_idx = sample_background_frames(df, top_idx, n_samples=15, seed=0)
    assert len(bg_idx) == 15


# ---------------------------------------------------------------------------
# Structural metrics (using _FakeTraj)
# ---------------------------------------------------------------------------


def test_compute_displacement_metrics_basic():
    """compute_displacement_metrics returns expected columns and non-negative values."""
    traj = _FakeTraj(n_frames=50, n_ca=20)
    frames = np.array([0, 5, 10, 20, 30])
    result = compute_displacement_metrics(traj, frames, ref_frame=0, top_k=5)

    assert set(result.columns) >= {
        "frame",
        "mean_residue_displacement",
        "max_residue_displacement",
        "topk_residue_displacement",
    }
    assert len(result) == len(frames)
    assert (result["mean_residue_displacement"] >= 0).all()
    assert (result["max_residue_displacement"] >= result["mean_residue_displacement"]).all()
    assert (result["topk_residue_displacement"] >= result["mean_residue_displacement"]).all()


def test_compute_displacement_ref_frame_is_zero():
    """Reference frame displacement against itself should be ~0."""
    traj = _FakeTraj(n_frames=30, n_ca=15)
    result = compute_displacement_metrics(traj, frames=np.array([0]), ref_frame=0)
    assert result.iloc[0]["mean_residue_displacement"] == pytest.approx(0.0, abs=1e-10)


def test_compute_displacement_empty_frames():
    """Empty frames array returns an empty DataFrame with correct columns."""
    traj = _FakeTraj(n_frames=30, n_ca=15)
    result = compute_displacement_metrics(traj, frames=np.array([], dtype=int))
    assert len(result) == 0
    assert "frame" in result.columns


def test_compute_displacement_topk_clipped():
    """top_k larger than n_ca should not crash."""
    traj = _FakeTraj(n_frames=20, n_ca=5)
    result = compute_displacement_metrics(traj, frames=np.array([1, 2]), top_k=1000)
    assert len(result) == 2


def test_compute_contact_change_basic():
    """compute_contact_change returns fraction in [0, 1]."""
    traj = _FakeTraj(n_frames=30, n_ca=15)
    frames = np.array([1, 5, 10])
    result = compute_contact_change(traj, frames, ref_frame=0, cutoff_nm=0.5)
    assert set(result.columns) >= {"frame", "contact_change"}
    assert len(result) == 3
    assert (result["contact_change"] >= 0).all()
    assert (result["contact_change"] <= 1).all()


def test_compute_contact_change_ref_is_zero():
    """Contact change of the reference frame against itself is 0."""
    traj = _FakeTraj(n_frames=20, n_ca=10)
    result = compute_contact_change(traj, frames=np.array([0]), ref_frame=0)
    assert result.iloc[0]["contact_change"] == pytest.approx(0.0, abs=1e-10)


def test_compute_contact_change_empty():
    """Empty frames → empty DataFrame with correct columns."""
    traj = _FakeTraj(n_frames=20, n_ca=10)
    result = compute_contact_change(traj, frames=np.array([], dtype=int))
    assert len(result) == 0
    assert "contact_change" in result.columns


def test_compute_temporal_persistence_values():
    """Temporal persistence averages correctly within window."""
    # Craft a score series where the peak at frame 50 is clearly persistent
    n = 100
    scores = np.ones(n) * 10.0
    scores[48:53] = 80.0  # elevated region
    df = pd.DataFrame({"frame": np.arange(n), "score_dynamic": scores})
    top_idx = np.array([50])

    result = compute_temporal_persistence(df, top_idx, window=2)
    assert len(result) == n
    # frame 50 should have high persistence
    mid_persist = result[result["frame"] == 50]["local_persistence_score"].values[0]
    low_persist = result[result["frame"] == 0]["local_persistence_score"].values[0]
    assert mid_persist > low_persist


def test_compute_temporal_persistence_boundary():
    """Persistence at the very first / last frame does not raise."""
    df = pd.DataFrame(
        {"frame": np.arange(10), "score_dynamic": np.arange(10, dtype=float)}
    )
    result = compute_temporal_persistence(df, np.array([0, 9]), window=3)
    assert len(result) == 10
    assert result["local_persistence_score"].notna().all()


# ---------------------------------------------------------------------------
# Dihedral change (without mdtraj – test fallback path)
# ---------------------------------------------------------------------------


def test_compute_dihedral_change_no_mdtraj(monkeypatch):
    """compute_dihedral_change returns empty DataFrame when mdtraj is absent."""
    import analysis.physical_validation as pv

    monkeypatch.setattr(pv, "_try_import_mdtraj", lambda: None)
    traj = _FakeTraj(n_frames=20, n_ca=10)
    result = pv.compute_dihedral_change(traj, frames=np.array([0, 1, 2]))
    assert len(result) == 0
    assert "dihedral_change" in result.columns


def test_compute_dihedral_change_empty_frames():
    """Empty frames → empty DataFrame."""
    import analysis.physical_validation as pv

    traj = _FakeTraj(n_frames=20, n_ca=10)
    result = pv.compute_dihedral_change(traj, frames=np.array([], dtype=int))
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Summary / statistics
# ---------------------------------------------------------------------------


def _make_labelled_df(n_top=20, n_bg=20, seed=5):
    """Create a labelled per-frame DataFrame for testing summaries."""
    rng = np.random.default_rng(seed)
    n = 200
    scores = rng.uniform(0, 100, n)
    disp = rng.exponential(0.5, n)

    df = pd.DataFrame(
        {
            "frame": np.arange(n),
            "score_dynamic": scores,
            "mean_residue_displacement": disp,
            "contact_change": rng.uniform(0, 1, n),
            "is_top_anomaly": False,
            "is_background": False,
        }
    )
    top_idx = np.argsort(scores)[-n_top:]
    bg_idx = np.argsort(scores)[:n_bg]
    df.iloc[top_idx, df.columns.get_loc("is_top_anomaly")] = True
    df.iloc[bg_idx, df.columns.get_loc("is_background")] = True
    return df


def test_summarize_group_differences_structure():
    """summarize_group_differences returns expected keys."""
    df = _make_labelled_df()
    summary = summarize_group_differences(
        df, metric_cols=["score_dynamic", "mean_residue_displacement"]
    )
    assert "n_top_anomaly" in summary
    assert "n_background" in summary
    assert "metrics" in summary
    assert "score_dynamic" in summary["metrics"]
    metric = summary["metrics"]["score_dynamic"]
    assert "top_anomaly_mean" in metric
    assert "background_mean" in metric
    assert "cohens_d" in metric


def test_summarize_cohen_d_direction():
    """Cohen's d should be positive when top-anomaly scores are higher than BG."""
    # By construction, top group has highest scores
    df = _make_labelled_df(n_top=20, n_bg=20, seed=42)
    summary = summarize_group_differences(df, metric_cols=["score_dynamic"])
    assert summary["metrics"]["score_dynamic"]["cohens_d"] > 0


def test_summarize_missing_metric():
    """Metric column absent from df is silently skipped."""
    df = _make_labelled_df()
    summary = summarize_group_differences(
        df, metric_cols=["score_dynamic", "nonexistent_col"]
    )
    assert "nonexistent_col" not in summary["metrics"]


def test_summarize_mannwhitney_present():
    """Mann-Whitney p-value is included when scipy is available."""
    try:
        from scipy import stats  # noqa: F401
    except ImportError:
        pytest.skip("scipy not available")

    df = _make_labelled_df()
    summary = summarize_group_differences(df, metric_cols=["score_dynamic"])
    assert "mannwhitney_pvalue" in summary["metrics"]["score_dynamic"]
    pval = summary["metrics"]["score_dynamic"]["mannwhitney_pvalue"]
    assert 0 <= pval <= 1


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def test_merge_metrics_group_flags():
    """_merge_metrics assigns is_top_anomaly and is_background correctly."""
    scores_df = _make_scores_df(50)
    top_idx = np.array([10, 20, 30])
    bg_idx = np.array([0, 5, 40])

    empty_df = pd.DataFrame()
    persist_df = pd.DataFrame(
        {
            "frame": scores_df["frame"].values,
            "local_persistence_score": np.ones(50),
        }
    )

    result = _merge_metrics(
        scores_df, top_idx, bg_idx, empty_df, empty_df, empty_df, empty_df, persist_df
    )

    assert result["is_top_anomaly"].sum() == 3
    assert result["is_background"].sum() == 3
    # Verify no overlap
    overlap = result["is_top_anomaly"] & result["is_background"]
    assert overlap.sum() == 0


def test_merge_metrics_expected_columns():
    """_merge_metrics includes all expected metric columns (NaN-filled if absent)."""
    scores_df = _make_scores_df(20)
    top_idx = np.array([2, 4])
    bg_idx = np.array([0, 1])

    persist_df = pd.DataFrame(
        {
            "frame": scores_df["frame"].values,
            "local_persistence_score": np.zeros(20),
        }
    )

    result = _merge_metrics(
        scores_df, top_idx, bg_idx,
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), persist_df
    )

    for col in [
        "mean_residue_displacement",
        "max_residue_displacement",
        "hotspot_local_rmsd",
        "contact_change",
        "dihedral_change",
        "local_persistence_score",
    ]:
        assert col in result.columns, f"Expected column '{col}' not found."


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------


def test_save_outputs_creates_files(tmp_path):
    """save_outputs creates expected CSV and JSON files."""
    df = _make_labelled_df()
    summary = {
        "n_top_anomaly": 20,
        "n_background": 20,
        "metrics": {"score_dynamic": {"top_anomaly_mean": 80.0, "background_mean": 10.0}},
    }
    save_outputs(df, summary, tmp_path)

    assert (tmp_path / "frame_validation.csv").exists()
    assert (tmp_path / "validation_summary.json").exists()


def test_save_outputs_csv_integrity(tmp_path):
    """CSV written by save_outputs is readable and contains correct row count."""
    df = _make_labelled_df(n_top=5, n_bg=5)
    save_outputs(df, {"metrics": {}}, tmp_path)

    loaded = pd.read_csv(tmp_path / "frame_validation.csv")
    assert len(loaded) == len(df)


def test_save_outputs_json_config(tmp_path):
    """JSON written by save_outputs includes the config block."""
    df = _make_labelled_df()
    config = {"top_pct": 5.0, "seed": 42}
    save_outputs(df, {"metrics": {}}, tmp_path, config=config)

    with open(tmp_path / "validation_summary.json") as fh:
        data = json.load(fh)

    assert data["config"]["top_pct"] == 5.0
    assert data["config"]["seed"] == 42


def test_save_outputs_creates_directory(tmp_path):
    """save_outputs creates out_dir if it does not exist."""
    new_dir = tmp_path / "nested" / "output"
    df = _make_labelled_df()
    save_outputs(df, {"metrics": {}}, new_dir)
    assert new_dir.exists()


# ---------------------------------------------------------------------------
# Print summary (smoke test)
# ---------------------------------------------------------------------------


def test_print_summary_no_crash(capsys):
    """_print_summary should not raise and should produce some output."""
    summary = {
        "n_top_anomaly": 10,
        "n_background": 10,
        "metrics": {
            "score_dynamic": {
                "top_anomaly_mean": 90.0,
                "top_anomaly_std": 5.0,
                "background_mean": 20.0,
                "background_std": 8.0,
                "cohens_d": 8.5,
                "mannwhitney_pvalue": 0.0001,
            }
        },
    }
    _print_summary(summary)
    captured = capsys.readouterr()
    assert "PHYSICAL VALIDATION" in captured.out
    assert "score_dynamic" in captured.out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_help():
    """CLI --help exits cleanly."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "analysis.physical_validation", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "topology" in result.stdout


if __name__ == "__main__":
    # Allow running directly: python tests/test_physical_validation.py
    import pytest as _pytest

    _pytest.main([__file__, "-v"])

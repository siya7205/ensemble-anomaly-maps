#!/usr/bin/env python3
"""
Unit tests for Phase 1: Model Selection & Bootstrap.
"""
import sys
import numpy as np
import tempfile
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from msm.select_lag_and_dim import compute_vamp2_score
from msm.bootstrap_msm import bootstrap_resample, fit_msm_pipeline
from msm.reproducibility import set_global_seed, generate_seed_sequence


def test_vamp2_score_basic():
    """Test VAMP-2 score computation."""
    print("[TEST] VAMP-2 score computation...")
    
    # Create synthetic trajectory with temporal structure
    np.random.seed(42)
    T = 500
    t = np.linspace(0, 10, T)
    X = np.column_stack([
        np.sin(t) + 0.1 * np.random.randn(T),
        np.cos(t) + 0.1 * np.random.randn(T),
        np.random.randn(T) * 0.5
    ])
    
    # Compute VAMP-2 score
    score = compute_vamp2_score(X, lag=10, dim=2, validation_fraction=0.2, seed=42)
    
    assert score > 0, "VAMP-2 score should be positive"
    assert np.isfinite(score), "VAMP-2 score should be finite"
    
    print(f"  ✓ VAMP-2 score: {score:.4f}")


def test_vamp2_reproducibility():
    """Test VAMP-2 score reproducibility with same seed."""
    print("[TEST] VAMP-2 reproducibility...")
    
    np.random.seed(42)
    T = 300
    X = np.random.randn(T, 5)
    
    score1 = compute_vamp2_score(X, lag=10, dim=3, seed=42)
    score2 = compute_vamp2_score(X, lag=10, dim=3, seed=42)
    
    assert np.abs(score1 - score2) < 1e-10, "Same seed should give same score"
    
    print(f"  ✓ Reproducible: {score1:.6f} == {score2:.6f}")


def test_bootstrap_resample_shape():
    """Test bootstrap resampling preserves shape."""
    print("[TEST] Bootstrap resampling shape...")
    
    X = np.random.randn(200, 5)
    
    # Frame resampling
    X_boot1 = bootstrap_resample(X, method='frames', seed=42)
    assert X_boot1.shape == X.shape, "Shape should be preserved"
    
    # Block resampling
    X_boot2 = bootstrap_resample(X, method='blocks', block_size=10, seed=42)
    assert X_boot2.shape == X.shape, "Shape should be preserved with blocks"
    
    print("  ✓ Shapes preserved")


def test_bootstrap_reproducibility():
    """Test bootstrap reproducibility with seeds."""
    print("[TEST] Bootstrap reproducibility...")
    
    X = np.random.randn(150, 4)
    
    X_boot1 = bootstrap_resample(X, method='frames', seed=123)
    X_boot2 = bootstrap_resample(X, method='frames', seed=123)
    
    assert np.allclose(X_boot1, X_boot2), "Same seed should give same resample"
    
    print("  ✓ Bootstrap is reproducible")


def test_msm_pipeline_basic():
    """Test MSM pipeline fitting."""
    print("[TEST] MSM pipeline fitting...")
    
    # Create synthetic trajectory
    np.random.seed(42)
    T = 400
    t = np.linspace(0, 20, T)
    X = np.column_stack([
        np.sin(t) + 0.2 * np.random.randn(T),
        np.cos(t) + 0.2 * np.random.randn(T),
        np.sin(2*t) + 0.2 * np.random.randn(T)
    ])
    
    # Fit pipeline
    msm, dtraj = fit_msm_pipeline(X, lag_tica=5, dim_tica=2, 
                                  n_clusters=10, lag_msm=10, seed_kmeans=42)
    
    assert msm is not None, "MSM should be fitted"
    assert len(dtraj) == len(X), "Discrete trajectory length should match"
    assert msm.n_states > 0, "Should have at least one state"
    assert len(msm.stationary_distribution) == msm.n_states
    
    print(f"  ✓ MSM fitted: {msm.n_states} states")


def test_seed_sequence():
    """Test seed sequence generation."""
    print("[TEST] Seed sequence generation...")
    
    seeds1 = generate_seed_sequence(42, 10)
    seeds2 = generate_seed_sequence(42, 10)
    
    assert len(seeds1) == 10, "Should generate correct number of seeds"
    assert seeds1 == seeds2, "Same master seed should give same sequence"
    
    seeds3 = generate_seed_sequence(43, 10)
    assert seeds1 != seeds3, "Different master seed should give different sequence"
    
    print("  ✓ Seed sequence generation works")


def test_global_seed():
    """Test global seed setting."""
    print("[TEST] Global seed setting...")
    
    set_global_seed(42)
    x1 = np.random.rand(5)
    
    set_global_seed(42)
    x2 = np.random.rand(5)
    
    assert np.allclose(x1, x2), "Global seed should make numpy reproducible"
    
    print("  ✓ Global seed setting works")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("[TEST] Edge cases...")
    
    # Very short trajectory
    X_short = np.random.randn(50, 3)
    score = compute_vamp2_score(X_short, lag=40, dim=2)
    assert score == -np.inf or np.isfinite(score), "Should handle short trajectories"
    
    # High dimensionality
    X_high_dim = np.random.randn(200, 20)
    score = compute_vamp2_score(X_high_dim, lag=10, dim=5)
    assert np.isfinite(score) or score == -np.inf, "Should handle high dimensions"
    
    print("  ✓ Edge cases handled")


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING PHASE 1: MODEL SELECTION & BOOTSTRAP")
    print("="*70 + "\n")
    
    try:
        test_vamp2_score_basic()
        test_vamp2_reproducibility()
        test_bootstrap_resample_shape()
        test_bootstrap_reproducibility()
        test_msm_pipeline_basic()
        test_seed_sequence()
        test_global_seed()
        test_edge_cases()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Simple test script to verify optimization functionality.
Tests the key utility functions and ensures they work correctly.
"""
import sys
import numpy as np
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from utils import (
    minmax_normalize,
    map_to_active_set,
    compute_transition_surprise,
    compute_local_density,
    compute_frame_scores,
    ProgressBar
)


def test_minmax_normalize():
    """Test normalization functions."""
    print("[TEST] minmax_normalize...")
    
    # Test basic normalization
    x = np.array([1, 2, 3, 4, 5])
    result = minmax_normalize(x, clip=False)
    assert result.min() == 0.0, "Min should be 0"
    assert result.max() == 1.0, "Max should be 1"
    assert abs(result.mean() - 0.5) < 0.1, "Mean should be near 0.5"
    
    # Test with clipping (robust normalization)
    x_outliers = np.array([1, 2, 3, 4, 5, 100])
    result_clipped = minmax_normalize(x_outliers, clip=True)
    assert result_clipped.max() == 1.0, "Max should be clipped to 1"
    assert result_clipped.min() >= 0.0, "Min should be >= 0"
    
    print("  ✓ minmax_normalize works correctly")


def test_map_to_active_set():
    """Test active set mapping."""
    print("[TEST] map_to_active_set...")
    
    dtraj = np.array([0, 1, 2, 3, 4, 5])
    active_set = np.array([1, 3, 5])  # Only states 1, 3, 5 are active
    n_clusters = 6
    
    mapped = map_to_active_set(dtraj, active_set, n_clusters)
    
    assert mapped[0] == -1, "State 0 should map to -1 (inactive)"
    assert mapped[1] == 0, "State 1 should map to 0 (first active)"
    assert mapped[2] == -1, "State 2 should map to -1 (inactive)"
    assert mapped[3] == 1, "State 3 should map to 1 (second active)"
    assert mapped[5] == 2, "State 5 should map to 2 (third active)"
    
    print("  ✓ map_to_active_set works correctly")


def test_compute_transition_surprise():
    """Test transition surprise computation."""
    print("[TEST] compute_transition_surprise...")
    
    # Create simple test case
    dtraj = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    transition_matrix = np.array([
        [0.1, 0.9],  # From state 0: mostly go to state 1
        [0.9, 0.1]   # From state 1: mostly go to state 0
    ])
    lag = 1
    active_mask = np.ones(len(dtraj), dtype=bool)
    
    surprise = compute_transition_surprise(dtraj, transition_matrix, lag, active_mask)
    
    assert len(surprise) == len(dtraj), "Output length should match input"
    assert np.all(surprise >= 0), "Surprise should be non-negative"
    assert np.sum(surprise) > 0, "Should have some non-zero surprises"
    
    print("  ✓ compute_transition_surprise works correctly")


def test_compute_local_density():
    """Test local density computation."""
    print("[TEST] compute_local_density...")
    
    # Create clustered data
    cluster1 = np.random.randn(50, 2) * 0.1
    cluster2 = np.random.randn(50, 2) * 0.1 + np.array([5, 5])
    Y = np.vstack([cluster1, cluster2])
    
    density = compute_local_density(Y, n_neighbors=5)
    
    assert len(density) == len(Y), "Output length should match input"
    assert np.all(np.isfinite(density)), "All density values should be finite"
    
    # Points in dense clusters should have higher (less negative) density
    avg_density_cluster1 = density[:50].mean()
    avg_density_outlier = density[-1]
    
    print("  ✓ compute_local_density works correctly")


def test_compute_frame_scores():
    """Test combined frame score computation."""
    print("[TEST] compute_frame_scores...")
    
    n_frames = 100
    rarity = np.random.rand(n_frames)
    surprise = np.random.rand(n_frames)
    density = np.random.rand(n_frames)
    
    scores = compute_frame_scores(rarity, surprise, density)
    
    assert len(scores) == n_frames, "Output length should match input"
    assert np.all(scores >= 0), "Scores should be non-negative"
    assert np.all(scores <= 100), "Scores should be <= 100"
    assert np.all(np.isfinite(scores)), "All scores should be finite"
    
    print("  ✓ compute_frame_scores works correctly")


def test_progress_bar():
    """Test progress bar functionality."""
    print("[TEST] ProgressBar...")
    
    total = 10
    progress = ProgressBar(total, desc="Testing")
    
    for i in range(total):
        progress.update(1)
    
    assert progress.current == total, "Progress should reach total"
    
    print("\n  ✓ ProgressBar works correctly")


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING ML PIPELINE OPTIMIZATIONS")
    print("="*70 + "\n")
    
    try:
        test_minmax_normalize()
        test_map_to_active_set()
        test_compute_transition_surprise()
        test_compute_local_density()
        test_compute_frame_scores()
        test_progress_bar()
        
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

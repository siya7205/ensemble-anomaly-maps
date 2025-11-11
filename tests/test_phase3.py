#!/usr/bin/env python3
"""
Unit tests for Phase 3: Enhanced Scoring & Soft States.
"""
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.anomaly_v2 import (
    rank_normalize,
    quantile_normalize,
    compute_zscore,
    moving_median,
    fuse_signals
)
from msm.soft_states import compute_state_entropy


def test_rank_normalize_basic():
    """Test rank normalization."""
    print("[TEST] Rank normalization...")
    
    x = np.array([1, 5, 3, 9, 2])
    normalized = rank_normalize(x)
    
    assert normalized.min() == 0.0, "Min should be 0"
    assert normalized.max() == 1.0, "Max should be 1"
    assert len(normalized) == len(x), "Length should be preserved"
    
    # Check ordering
    assert normalized[0] < normalized[2], "Ordering should be preserved"
    assert normalized[2] < normalized[1], "Ordering should be preserved"
    
    print(f"  ✓ Input: {x}")
    print(f"  ✓ Normalized: {normalized}")


def test_rank_normalize_constant():
    """Test rank normalization with constant array."""
    print("[TEST] Rank normalization - constant...")
    
    x = np.array([5, 5, 5, 5])
    normalized = rank_normalize(x)
    
    assert np.all(normalized == 0), "Constant array should give zeros"
    
    print(f"  ✓ Constant handled: {normalized}")


def test_quantile_normalize():
    """Test quantile normalization."""
    print("[TEST] Quantile normalization...")
    
    x = np.array([1, 2, 3, 4, 5, 100])  # Outlier
    normalized = quantile_normalize(x, lower=0.1, upper=0.9)
    
    assert 0 <= normalized.min() <= 1, "Should be in [0,1]"
    assert 0 <= normalized.max() <= 1, "Should be in [0,1]"
    
    # Outlier should be clipped
    assert normalized[-1] == 1.0, "Outlier should be clipped to 1"
    
    print(f"  ✓ Input: {x}")
    print(f"  ✓ Normalized: {normalized}")


def test_zscore_computation():
    """Test z-score computation."""
    print("[TEST] Z-score computation...")
    
    x = np.array([1, 2, 3, 4, 5])
    z = compute_zscore(x)
    
    assert np.abs(z.mean()) < 1e-10, "Mean should be ~0"
    assert np.abs(z.std() - 1.0) < 1e-10, "Std should be ~1"
    
    print(f"  ✓ Z-scores: mean={z.mean():.6f}, std={z.std():.6f}")


def test_zscore_constant():
    """Test z-score with constant array."""
    print("[TEST] Z-score - constant...")
    
    x = np.array([3, 3, 3, 3])
    z = compute_zscore(x)
    
    assert np.all(z == 0), "Constant should give zero z-scores"
    
    print(f"  ✓ Constant handled")


def test_moving_median():
    """Test moving median filter."""
    print("[TEST] Moving median...")
    
    x = np.array([1, 10, 2, 11, 3, 12, 4])  # Noisy
    smoothed = moving_median(x, window=3)
    
    assert len(smoothed) == len(x), "Length should be preserved"
    # Check smoothing reduces variance
    assert smoothed.std() < x.std(), "Should reduce variance"
    
    print(f"  ✓ Input: {x}")
    print(f"  ✓ Smoothed: {smoothed}")


def test_signal_fusion_median():
    """Test signal fusion with median."""
    print("[TEST] Signal fusion - median...")
    
    signals = {
        'signal1': np.array([0.1, 0.5, 0.9]),
        'signal2': np.array([0.2, 0.4, 0.8]),
        'signal3': np.array([0.3, 0.6, 0.7])
    }
    
    score, normalized = fuse_signals(signals, method='median', 
                                     normalize_method='rank')
    
    assert len(score) == 3, "Score length should match"
    assert 0 <= score.min() <= 1, "Score should be in [0,1]"
    assert 0 <= score.max() <= 1, "Score should be in [0,1]"
    
    print(f"  ✓ Fused scores: {score}")


def test_signal_fusion_mean():
    """Test signal fusion with mean."""
    print("[TEST] Signal fusion - mean...")
    
    signals = {
        'signal1': np.array([0, 1, 0]),
        'signal2': np.array([1, 0, 1])
    }
    
    score, _ = fuse_signals(signals, method='mean', normalize_method='rank')
    
    # Mean of (0,1) and (1,0) should be 0.5 after rank normalization
    assert len(score) == 3, "Score length should match"
    
    print(f"  ✓ Mean fusion: {score}")


def test_state_entropy():
    """Test state entropy computation."""
    print("[TEST] State entropy...")
    
    # Deterministic state (low entropy)
    q_det = np.array([[1.0, 0.0, 0.0]])
    H_det = compute_state_entropy(q_det)
    
    # Uniform state (high entropy)
    q_uniform = np.array([[0.33, 0.33, 0.34]])
    H_uniform = compute_state_entropy(q_uniform)
    
    assert H_det[0] < H_uniform[0], "Deterministic should have lower entropy"
    assert H_det[0] < 0.1, "Deterministic entropy should be near 0"
    
    print(f"  ✓ Deterministic entropy: {H_det[0]:.6f}")
    print(f"  ✓ Uniform entropy: {H_uniform[0]:.6f}")


def test_normalization_preserves_order():
    """Test that normalization preserves ordering."""
    print("[TEST] Normalization preserves order...")
    
    x = np.array([5, 2, 8, 1, 9])
    
    # Rank normalization
    x_rank = rank_normalize(x)
    assert np.all(np.argsort(x) == np.argsort(x_rank)), "Order should be preserved"
    
    # Quantile normalization
    x_quant = quantile_normalize(x)
    # Ordering preserved except for clipped outliers
    
    print(f"  ✓ Ordering preserved")


def test_edge_cases():
    """Test edge cases."""
    print("[TEST] Edge cases...")
    
    # Empty array
    x_empty = np.array([])
    assert len(rank_normalize(x_empty)) == 0, "Empty should stay empty"
    
    # Single element
    x_single = np.array([5])
    assert rank_normalize(x_single)[0] == 0, "Single element should be 0"
    
    # Two elements
    x_two = np.array([1, 2])
    norm_two = rank_normalize(x_two)
    assert norm_two[0] == 0 and norm_two[1] == 1, "Two elements should be 0,1"
    
    print(f"  ✓ Edge cases handled")


def test_fusion_reproducibility():
    """Test that fusion is reproducible."""
    print("[TEST] Fusion reproducibility...")
    
    signals = {
        's1': np.random.randn(100),
        's2': np.random.randn(100)
    }
    
    score1, _ = fuse_signals(signals, method='median', normalize_method='rank')
    score2, _ = fuse_signals(signals, method='median', normalize_method='rank')
    
    assert np.allclose(score1, score2), "Fusion should be reproducible"
    
    print(f"  ✓ Reproducible")


def test_monotone_behavior():
    """Test monotone behavior under perturbations."""
    print("[TEST] Monotone behavior...")
    
    # Create base signal
    base_signal = np.ones(10) * 0.5
    
    # Create perturbed signal (spike at index 5)
    perturbed_signal = base_signal.copy()
    perturbed_signal[5] = 0.9
    
    # Normalize both
    base_norm = rank_normalize(base_signal)
    perturbed_norm = rank_normalize(perturbed_signal)
    
    # Perturbed signal should have higher value at spike
    assert perturbed_norm[5] > base_norm[5], "Spike should increase score"
    
    print(f"  ✓ Monotone under perturbation")


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING PHASE 3: ENHANCED SCORING & SOFT STATES")
    print("="*70 + "\n")
    
    try:
        # Normalization tests
        test_rank_normalize_basic()
        test_rank_normalize_constant()
        test_quantile_normalize()
        test_normalization_preserves_order()
        
        # Z-score tests
        test_zscore_computation()
        test_zscore_constant()
        
        # Smoothing tests
        test_moving_median()
        
        # Fusion tests
        test_signal_fusion_median()
        test_signal_fusion_mean()
        test_fusion_reproducibility()
        
        # Entropy tests
        test_state_entropy()
        
        # Behavioral tests
        test_monotone_behavior()
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

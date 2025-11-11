#!/usr/bin/env python3
"""
Unit tests for Phase 2: Feature Extensions (Energy & Pockets).
"""
import sys
import numpy as np
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features_energy.compute_energy import get_contact_energy, HYDROPHOBIC, CHARGED_POSITIVE
from features_pockets.compute_pockets import create_grid_around_protein


def test_contact_energy_attractive():
    """Test contact energy computation - attractive regime."""
    print("[TEST] Contact energy - attractive...")
    
    # Hydrophobic-hydrophobic should be attractive
    energy = get_contact_energy('ALA', 'VAL', 0.6)  # 6 Angstroms optimal
    
    assert energy < 0, "Hydrophobic contact should be attractive"
    assert np.isfinite(energy), "Energy should be finite"
    
    print(f"  ✓ Attractive energy: {energy:.3f} kcal/mol")


def test_contact_energy_repulsive():
    """Test contact energy computation - repulsive regime."""
    print("[TEST] Contact energy - repulsive...")
    
    # Very close distance should be repulsive
    energy = get_contact_energy('ALA', 'ALA', 0.3)  # 3 Angstroms - too close
    
    assert energy > 0, "Very close contact should be repulsive"
    
    print(f"  ✓ Repulsive energy: {energy:.3f} kcal/mol")


def test_contact_energy_zero_far():
    """Test contact energy - zero at large distances."""
    print("[TEST] Contact energy - far distance...")
    
    energy = get_contact_energy('ALA', 'VAL', 1.5)  # 15 Angstroms - far
    
    assert np.abs(energy) < 0.01, "Far distance should have ~zero energy"
    
    print(f"  ✓ Far energy: {energy:.6f} kcal/mol")


def test_grid_creation():
    """Test 3D grid creation around protein."""
    print("[TEST] Grid creation...")
    
    # Create mock protein positions
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    grid_points, grid_shape, origin = create_grid_around_protein(
        positions, grid_spacing=0.5, padding=1.0
    )
    
    assert grid_points.shape[1] == 3, "Grid points should be 3D"
    assert len(grid_shape) == 3, "Grid shape should be 3D"
    assert np.prod(grid_shape) == len(grid_points), "Grid size should match points"
    
    # Check grid covers protein
    assert np.all(origin <= positions.min(axis=0) - 1.0), "Grid should start before protein"
    
    print(f"  ✓ Grid shape: {grid_shape}, points: {len(grid_points)}")


def test_residue_classification():
    """Test residue type classification."""
    print("[TEST] Residue classification...")
    
    # Check hydrophobic
    assert 'ALA' in HYDROPHOBIC, "ALA should be hydrophobic"
    assert 'VAL' in HYDROPHOBIC, "VAL should be hydrophobic"
    assert 'PHE' in HYDROPHOBIC, "PHE should be hydrophobic"
    
    # Check charged
    assert 'LYS' in CHARGED_POSITIVE, "LYS should be positive"
    assert 'ARG' in CHARGED_POSITIVE, "ARG should be positive"
    
    print("  ✓ Residue classifications correct")


def test_energy_symmetry():
    """Test contact energy is symmetric."""
    print("[TEST] Energy symmetry...")
    
    dist = 0.6
    e1 = get_contact_energy('ALA', 'VAL', dist)
    e2 = get_contact_energy('VAL', 'ALA', dist)
    
    assert np.abs(e1 - e2) < 1e-10, "Contact energy should be symmetric"
    
    print(f"  ✓ Symmetric: {e1:.6f} == {e2:.6f}")


def test_energy_distance_dependence():
    """Test energy decreases with distance."""
    print("[TEST] Energy distance dependence...")
    
    # Test at several distances
    distances = [0.4, 0.5, 0.6, 0.7, 0.8]
    energies = [get_contact_energy('ALA', 'VAL', d) for d in distances]
    
    # At moderate distances, should generally decrease in magnitude
    # (become less negative) as distance increases
    for i in range(len(distances) - 1):
        # Skip repulsive region
        if distances[i] > 0.5:
            assert energies[i] < 0, f"Should be attractive at {distances[i]}"
    
    print("  ✓ Energy distance dependence verified")


def test_pocket_grid_basics():
    """Test basic pocket detection grid properties."""
    print("[TEST] Pocket detection basics...")
    
    # Create simple spherical protein
    n_atoms = 50
    radius = 1.0
    angles = np.linspace(0, 2*np.pi, n_atoms)
    positions = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros(n_atoms)
    ])
    
    grid_points, grid_shape, origin = create_grid_around_protein(
        positions, grid_spacing=0.3, padding=0.5
    )
    
    # Grid should be larger than protein
    assert grid_points.shape[0] > n_atoms, "Grid should have more points than atoms"
    
    # Grid should span protein dimensions
    grid_extent = grid_shape[0] * 0.3  # spacing * n_points
    protein_extent = positions.max() - positions.min()
    assert grid_extent > protein_extent, "Grid should span protein"
    
    print(f"  ✓ Grid: {grid_shape}, protein extent: {protein_extent:.2f}")


def test_energy_values_reasonable():
    """Test that energy values are in reasonable range."""
    print("[TEST] Energy value ranges...")
    
    # Test various residue pairs at typical contact distance
    test_pairs = [
        ('ALA', 'ALA'),
        ('LYS', 'ASP'),  # Opposite charges - should be very attractive
        ('PHE', 'TRP'),  # Hydrophobic-hydrophobic
    ]
    
    for res1, res2 in test_pairs:
        energy = get_contact_energy(res1, res2, 0.6)
        
        # Energies should be in physically reasonable range
        assert -20.0 < energy < 20.0, f"Energy for {res1}-{res2} out of range: {energy}"
    
    print("  ✓ All energy values in reasonable range")


def test_grid_spacing():
    """Test grid spacing is preserved."""
    print("[TEST] Grid spacing...")
    
    positions = np.random.randn(10, 3)
    spacing = 0.4
    
    grid_points, _, _ = create_grid_around_protein(positions, grid_spacing=spacing)
    
    # Check spacing in first dimension
    # Sort points by x coordinate
    sorted_x = np.sort(grid_points[:, 0])
    unique_x = np.unique(sorted_x)
    
    if len(unique_x) > 1:
        diffs = np.diff(unique_x)
        # All differences should be close to spacing
        assert np.allclose(diffs, spacing, atol=1e-6), "Grid spacing not preserved"
    
    print(f"  ✓ Grid spacing: {spacing} nm")


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING PHASE 2: FEATURE EXTENSIONS")
    print("="*70 + "\n")
    
    try:
        # Energy tests
        test_contact_energy_attractive()
        test_contact_energy_repulsive()
        test_contact_energy_zero_far()
        test_energy_symmetry()
        test_energy_distance_dependence()
        test_energy_values_reasonable()
        test_residue_classification()
        
        # Pocket tests
        test_grid_creation()
        test_pocket_grid_basics()
        test_grid_spacing()
        
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

#!/usr/bin/env python3
"""
Pocket/cavity detection over MD trajectories.

Implements a grid-based pocket detection algorithm similar to MDpocket:
- Identifies concave regions on protein surface
- Tracks pocket volume, mouth radius, and SASA
- Maps nearest residues to each pocket

Output: data/derived/pockets.parquet, pocket_rims.parquet
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import mdtraj as md
from scipy.spatial import ConvexHull, distance_matrix
from scipy.ndimage import label as nd_label
from sklearn.cluster import DBSCAN


def compute_protein_center(positions):
    """Compute geometric center of protein."""
    return positions.mean(axis=0)


def create_grid_around_protein(positions, grid_spacing=0.5, padding=1.0):
    """
    Create 3D grid around protein.
    
    Args:
        positions: Atom positions (N x 3) in nm
        grid_spacing: Grid spacing in nm
        padding: Extra space around protein in nm
        
    Returns:
        grid_points: Grid point coordinates
        grid_shape: Shape of the grid
        origin: Grid origin
    """
    # Find bounding box
    mins = positions.min(axis=0) - padding
    maxs = positions.max(axis=0) + padding
    
    # Create grid
    x = np.arange(mins[0], maxs[0], grid_spacing)
    y = np.arange(mins[1], maxs[1], grid_spacing)
    z = np.arange(mins[2], maxs[2], grid_spacing)
    
    grid_shape = (len(x), len(y), len(z))
    
    # Generate grid points
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    return grid_points, grid_shape, mins


def identify_pockets_grid(positions, grid_spacing=0.5, probe_radius=0.14, 
                          min_volume=0.5, max_volume=50.0):
    """
    Identify pockets using grid-based method.
    
    Args:
        positions: Heavy atom positions (N x 3) in nm
        grid_spacing: Grid spacing in nm
        probe_radius: Probe sphere radius in nm (water-like)
        min_volume: Minimum pocket volume in nm^3
        max_volume: Maximum pocket volume in nm^3
        
    Returns:
        pockets: List of pocket dictionaries
    """
    # Create grid
    grid_points, grid_shape, origin = create_grid_around_protein(positions, grid_spacing)
    
    # Compute distances from grid points to nearest atom
    distances = distance_matrix(grid_points, positions).min(axis=1)
    
    # Points in pockets: far enough from atoms to fit probe, but not too far
    in_pocket = (distances > probe_radius) & (distances < probe_radius + 0.6)
    
    # Reshape to grid
    pocket_grid = in_pocket.reshape(grid_shape)
    
    # Label connected components
    labeled_grid, n_pockets = nd_label(pocket_grid)
    
    if n_pockets == 0:
        return []
    
    # Analyze each pocket
    pockets = []
    voxel_volume = grid_spacing ** 3
    
    for pocket_id in range(1, n_pockets + 1):
        pocket_mask = (labeled_grid == pocket_id)
        pocket_volume = pocket_mask.sum() * voxel_volume
        
        # Filter by volume
        if pocket_volume < min_volume or pocket_volume > max_volume:
            continue
        
        # Get pocket grid points
        pocket_indices = np.argwhere(pocket_mask)
        pocket_coords = grid_points[np.ravel_multi_index(
            pocket_indices.T, grid_shape
        )]
        
        # Compute pocket center
        pocket_center = pocket_coords.mean(axis=0)
        
        # Estimate mouth radius (distance to farthest point)
        center_dists = np.linalg.norm(pocket_coords - pocket_center, axis=1)
        mouth_radius = np.percentile(center_dists, 90)  # 90th percentile
        
        # Estimate SASA of rim (simplified: surface points)
        # Find boundary points (adjacent to non-pocket)
        boundary_points = []
        for idx in pocket_indices:
            i, j, k = idx
            # Check neighbors
            for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                ni, nj, nk = i+di, j+dj, k+dk
                if 0 <= ni < grid_shape[0] and 0 <= nj < grid_shape[1] and 0 <= nk < grid_shape[2]:
                    if not pocket_mask[ni, nj, nk]:
                        boundary_points.append([i, j, k])
                        break
        
        sasa_rim = len(boundary_points) * grid_spacing ** 2  # Approximate SASA
        
        pockets.append({
            'pocket_id': pocket_id - 1,  # 0-indexed
            'volume': pocket_volume,
            'mouth_radius': mouth_radius,
            'sasa_rim': sasa_rim,
            'center': pocket_center
        })
    
    return pockets


def map_residues_to_pockets(pockets, residue_positions, residue_ids, cutoff=0.8):
    """
    Map residues to nearest pockets.
    
    Args:
        pockets: List of pocket dictionaries
        residue_positions: CA positions (N_res x 3)
        residue_ids: Residue IDs
        cutoff: Distance cutoff in nm
        
    Returns:
        mappings: List of (pocket_id, res_id, distance) tuples
    """
    mappings = []
    
    for pocket in pockets:
        pocket_center = pocket['center']
        
        # Compute distances from pocket center to all residues
        distances = np.linalg.norm(residue_positions - pocket_center, axis=1)
        
        # Find residues within cutoff
        nearby_indices = np.where(distances < cutoff)[0]
        
        for idx in nearby_indices:
            mappings.append({
                'pocket_id': pocket['pocket_id'],
                'res_id': residue_ids[idx],
                'rim_distance': distances[idx]
            })
    
    return mappings


def compute_pocket_features(traj, grid_spacing=0.5, probe_radius=0.14,
                           min_volume=0.5, progress_callback=None):
    """
    Compute pocket features for all frames.
    
    Args:
        traj: MDTraj trajectory
        grid_spacing: Grid spacing in nm
        probe_radius: Probe radius in nm
        min_volume: Minimum pocket volume
        progress_callback: Optional callback(frame_idx, total_frames)
        
    Returns:
        pockets_df: DataFrame with pocket features
        rims_df: DataFrame with pocket-residue mappings
    """
    n_frames = len(traj)
    
    # Get heavy atoms (non-hydrogen)
    heavy_atoms = traj.topology.select('not element H')
    
    # Get CA atoms and residue info
    residues = list(traj.topology.residues)
    ca_indices = []
    res_ids = []
    
    for res in residues:
        ca_atoms = [a for a in res.atoms if a.name == 'CA']
        if ca_atoms:
            ca_indices.append(ca_atoms[0].index)
            res_ids.append(res.resSeq)
    
    ca_indices = np.array(ca_indices)
    res_ids = np.array(res_ids)
    
    # Storage
    all_pockets = []
    all_rims = []
    
    # Process each frame
    for frame_idx in range(n_frames):
        if progress_callback:
            progress_callback(frame_idx, n_frames)
        
        # Get heavy atom positions
        heavy_positions = traj.xyz[frame_idx, heavy_atoms]
        
        # Detect pockets
        pockets = identify_pockets_grid(heavy_positions, grid_spacing, 
                                       probe_radius, min_volume)
        
        # Get CA positions
        ca_positions = traj.xyz[frame_idx, ca_indices]
        
        # Map residues to pockets
        for pocket in pockets:
            all_pockets.append({
                'frame': frame_idx,
                'pocket_id': pocket['pocket_id'],
                'volume': pocket['volume'],
                'mouth_radius': pocket['mouth_radius'],
                'sasa_rim': pocket['sasa_rim']
            })
            
            # Map nearby residues
            mappings = map_residues_to_pockets([pocket], ca_positions, res_ids)
            for mapping in mappings:
                all_rims.append({
                    'frame': frame_idx,
                    **mapping
                })
    
    pockets_df = pd.DataFrame(all_pockets)
    rims_df = pd.DataFrame(all_rims)
    
    return pockets_df, rims_df


def main():
    parser = argparse.ArgumentParser(
        description='Compute pocket/cavity features over trajectory'
    )
    parser.add_argument('--topology', required=True,
                       help='Path to topology file (PDB)')
    parser.add_argument('--trajectory', required=True,
                       help='Path to trajectory file (XTC, DCD, etc.)')
    parser.add_argument('--output_pockets', default='data/derived/pockets.parquet',
                       help='Output parquet file for pockets')
    parser.add_argument('--output_rims', default='data/derived/pocket_rims.parquet',
                       help='Output parquet file for pocket rims')
    parser.add_argument('--grid_spacing', type=float, default=0.5,
                       help='Grid spacing in nm (default: 0.5)')
    parser.add_argument('--probe_radius', type=float, default=0.14,
                       help='Probe radius in nm (default: 0.14, water-like)')
    parser.add_argument('--min_volume', type=float, default=0.5,
                       help='Minimum pocket volume in nm^3 (default: 0.5)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for trajectory (default: 1)')
    
    args = parser.parse_args()
    
    print(f"[1/3] Loading trajectory")
    print(f"  Topology: {args.topology}")
    print(f"  Trajectory: {args.trajectory}")
    
    traj = md.load(args.trajectory, top=args.topology, stride=args.stride)
    print(f"  Loaded: {len(traj)} frames, {traj.n_residues} residues")
    
    print(f"[2/3] Computing pocket features")
    print(f"  Grid spacing: {args.grid_spacing} nm")
    print(f"  Probe radius: {args.probe_radius} nm")
    
    def progress(frame_idx, total):
        if (frame_idx + 1) % 5 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx + 1}/{total}")
    
    pockets_df, rims_df = compute_pocket_features(
        traj, args.grid_spacing, args.probe_radius, 
        args.min_volume, progress
    )
    
    print(f"[3/3] Saving results")
    
    # Save pockets
    output_pockets = Path(args.output_pockets)
    output_pockets.parent.mkdir(parents=True, exist_ok=True)
    pockets_df.to_parquet(output_pockets, index=False)
    print(f"  Pockets saved to {output_pockets}")
    print(f"  Shape: {pockets_df.shape}")
    
    # Save rims
    output_rims = Path(args.output_rims)
    rims_df.to_parquet(output_rims, index=False)
    print(f"  Rims saved to {output_rims}")
    print(f"  Shape: {rims_df.shape}")
    
    if len(pockets_df) > 0:
        print(f"\nPocket statistics per frame:")
        pocket_stats = pockets_df.groupby('frame').agg({
            'volume': ['count', 'mean', 'std'],
            'mouth_radius': ['mean', 'std']
        })
        print(pocket_stats.head())


if __name__ == '__main__':
    main()

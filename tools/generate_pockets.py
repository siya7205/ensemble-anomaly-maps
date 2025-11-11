#!/usr/bin/env python3
"""
Generate pocket/cavity dynamics features.

Wrapper for features_pockets/compute_pockets.py with caching support.
"""
import argparse
import sys
from pathlib import Path
import hashlib
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features_pockets.compute_pockets import compute_pocket_features
import mdtraj as md


def compute_file_hash(filepath):
    """Compute MD5 hash of file for caching."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_cache(topology_path, trajectory_path, output_pockets, output_rims, cache_info_path):
    """Check if cached output is valid."""
    if not Path(output_pockets).exists() or not Path(output_rims).exists():
        return False
    
    if not Path(cache_info_path).exists():
        return False
    
    # Load cache info
    with open(cache_info_path) as f:
        cache_info = json.load(f)
    
    # Check if input files have changed
    top_hash = compute_file_hash(topology_path)
    traj_hash = compute_file_hash(trajectory_path)
    
    if cache_info.get('topology_hash') != top_hash:
        return False
    if cache_info.get('trajectory_hash') != traj_hash:
        return False
    
    return True


def save_cache_info(topology_path, trajectory_path, cache_info_path):
    """Save cache metadata."""
    cache_info = {
        'topology_hash': compute_file_hash(topology_path),
        'trajectory_hash': compute_file_hash(trajectory_path)
    }
    
    with open(cache_info_path, 'w') as f:
        json.dump(cache_info, f)


def main():
    parser = argparse.ArgumentParser(
        description='Generate pocket/cavity dynamics features',
        epilog="""
Examples:
  # Basic usage
  python tools/generate_pockets.py --topology data/top.pdb --trajectory data/traj.xtc
  
  # With custom parameters
  python tools/generate_pockets.py --topology data/top.pdb --trajectory data/traj.xtc \\
      --grid_spacing 0.3 --probe_radius 0.14 --min_volume 1.0
  
  # Disable caching
  python tools/generate_pockets.py --topology data/top.pdb --trajectory data/traj.xtc \\
      --no-cache
        """
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
                       help='Probe radius in nm (default: 0.14)')
    parser.add_argument('--min_volume', type=float, default=0.5,
                       help='Minimum pocket volume in nm^3 (default: 0.5)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for trajectory (default: 1)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    
    args = parser.parse_args()
    
    output_pockets = Path(args.output_pockets)
    output_rims = Path(args.output_rims)
    cache_info_path = output_pockets.parent / '.pockets_cache.json'
    
    # Check cache
    if not args.no_cache:
        if check_cache(args.topology, args.trajectory, output_pockets, 
                      output_rims, cache_info_path):
            print(f"[CACHED] Using cached pockets from:")
            print(f"  {output_pockets}")
            print(f"  {output_rims}")
            print("  Use --no-cache to recompute")
            return
    
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
    output_pockets.parent.mkdir(parents=True, exist_ok=True)
    pockets_df.to_parquet(output_pockets, index=False)
    print(f"  Pockets saved to {output_pockets}")
    print(f"  Shape: {pockets_df.shape}")
    
    # Save rims
    rims_df.to_parquet(output_rims, index=False)
    print(f"  Rims saved to {output_rims}")
    print(f"  Shape: {rims_df.shape}")
    
    # Save cache info
    if not args.no_cache:
        save_cache_info(args.topology, args.trajectory, cache_info_path)
        print(f"  Cache info saved")
    
    if len(pockets_df) > 0:
        print(f"\nPocket statistics per frame:")
        pocket_stats = pockets_df.groupby('frame').agg({
            'volume': ['count', 'mean', 'std'],
            'mouth_radius': ['mean', 'std']
        })
        print(pocket_stats.head())


if __name__ == '__main__':
    main()

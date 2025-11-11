#!/usr/bin/env python3
"""
Generate per-residue energetic features.

Wrapper for features_energy/compute_energy.py with caching support.
"""
import argparse
import sys
from pathlib import Path
import hashlib
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features_energy.compute_energy import compute_residue_energies
import mdtraj as md


def compute_file_hash(filepath):
    """Compute MD5 hash of file for caching."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_cache(topology_path, trajectory_path, output_path, cache_info_path):
    """Check if cached output is valid."""
    if not Path(output_path).exists():
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
        description='Generate per-residue energetic features',
        epilog="""
Examples:
  # Basic usage
  python tools/generate_energy.py --topology data/top.pdb --trajectory data/traj.xtc
  
  # With custom output
  python tools/generate_energy.py --topology data/top.pdb --trajectory data/traj.xtc \\
      --output data/derived/my_energy.parquet
  
  # Disable caching
  python tools/generate_energy.py --topology data/top.pdb --trajectory data/traj.xtc \\
      --no-cache
        """
    )
    
    parser.add_argument('--topology', required=True,
                       help='Path to topology file (PDB)')
    parser.add_argument('--trajectory', required=True,
                       help='Path to trajectory file (XTC, DCD, etc.)')
    parser.add_argument('--output', default='data/derived/residue_energy.parquet',
                       help='Output parquet file')
    parser.add_argument('--contact_cutoff', type=float, default=0.8,
                       help='Contact cutoff in nm (default: 0.8)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for trajectory (default: 1)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    cache_info_path = output_path.parent / '.energy_cache.json'
    
    # Check cache
    if not args.no_cache:
        if check_cache(args.topology, args.trajectory, output_path, cache_info_path):
            print(f"[CACHED] Using cached energies from {output_path}")
            print("  Use --no-cache to recompute")
            return
    
    print(f"[1/3] Loading trajectory")
    print(f"  Topology: {args.topology}")
    print(f"  Trajectory: {args.trajectory}")
    
    traj = md.load(args.trajectory, top=args.topology, stride=args.stride)
    print(f"  Loaded: {len(traj)} frames, {traj.n_residues} residues")
    
    print(f"[2/3] Computing per-residue energies")
    
    def progress(frame_idx, total):
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx + 1}/{total}")
    
    df = compute_residue_energies(traj, args.contact_cutoff, progress)
    
    print(f"[3/3] Saving results")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Shape: {df.shape}")
    
    # Save cache info
    if not args.no_cache:
        save_cache_info(args.topology, args.trajectory, cache_info_path)
        print(f"  Cache info saved")
    
    print(f"\nSummary statistics:")
    print(df.groupby('frame')['energy'].agg(['mean', 'std', 'min', 'max']))


if __name__ == '__main__':
    main()

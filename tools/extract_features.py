#!/usr/bin/env python3
"""
Extract features from MD trajectory for anomaly detection pipeline.

Usage:
    python tools/extract_features.py \
        --topology data/raw_trajectory/align_topol.pdb \
        --trajectory data/raw_trajectory/trajectory_0.xtc \
        --output data/features.npy
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from features.compute_md_features import compute_features, features_to_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Extract MD features from trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python tools/extract_features.py \\
        --topology data/raw_trajectory/align_topol.pdb \\
        --trajectory data/raw_trajectory/trajectory_0.xtc \\
        --output data/features.npy
        
    # With stride (faster, every 5th frame)
    python tools/extract_features.py \\
        --topology data/raw_trajectory/align_topol.pdb \\
        --trajectory data/raw_trajectory/trajectory_0.xtc \\
        --output data/features.npy \\
        --stride 5
        """
    )
    
    parser.add_argument('--topology', required=True,
                       help='Path to topology file (PDB)')
    parser.add_argument('--trajectory', required=True,
                       help='Path to trajectory file (XTC, DCD, etc.)')
    parser.add_argument('--output', required=True,
                       help='Output path for features file (.npy)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Load every stride-th frame (default: 1)')
    parser.add_argument('--reference-frame', type=int, default=0,
                       help='Reference frame for RMSD (default: 0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    topology_path = Path(args.topology)
    trajectory_path = Path(args.trajectory)
    output_path = Path(args.output)
    
    if not topology_path.exists():
        print(f"Error: Topology file not found: {topology_path}")
        sys.exit(1)
        
    if not trajectory_path.exists():
        print(f"Error: Trajectory file not found: {trajectory_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Topology:   {topology_path}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Output:     {output_path}")
    print(f"Stride:     {args.stride}")
    print()
    
    # Extract features
    print("[1/2] Computing features...")
    try:
        features, traj = compute_features(
            topology_path=str(topology_path),
            trajectory_path=str(trajectory_path),
            stride=args.stride,
            reference_frame=args.reference_frame
        )
    except Exception as e:
        print(f"Error computing features: {e}")
        sys.exit(1)
    
    # Convert to matrix
    print("[2/2] Converting to feature matrix...")
    X, keys = features_to_matrix(features)
    
    # Save
    np.save(output_path, X)
    
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Frames:     {X.shape[0]}")
    print(f"Features:   {X.shape[1]}")
    print(f"Keys:       {', '.join(keys)}")
    print(f"Saved to:   {output_path}")
    

if __name__ == '__main__':
    main()

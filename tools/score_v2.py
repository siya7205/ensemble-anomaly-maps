#!/usr/bin/env python3
"""
Wrapper CLI for enhanced anomaly scoring v2.

Orchestrates multi-signal fusion for anomaly detection.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.anomaly_v2 import (
    load_signal_data,
    compute_anomaly_scores_v2
)
import numpy as np
from deeptime.markov.msm import MarkovStateModel


def main():
    parser = argparse.ArgumentParser(
        description='Compute enhanced anomaly scores v2 with multi-signal fusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with kinetic signals only
  python tools/score_v2.py --features data/features.npy --msm_dir outputs/msm
  
  # With energy and pocket features
  python tools/score_v2.py --features data/features.npy --msm_dir outputs/msm \\
      --energy data/derived/residue_energy.parquet \\
      --pockets data/derived/pockets.parquet
  
  # With soft states
  python tools/score_v2.py --features data/features.npy --msm_dir outputs/msm \\
      --soft_dtraj data/derived/soft_dtraj.npy \\
      --state_entropy data/derived/state_entropy.npy
  
  # Custom parameters
  python tools/score_v2.py --features data/features.npy --msm_dir outputs/msm \\
      --window 7 --normalize quantile --fusion mean
        """
    )
    
    parser.add_argument('--features', required=True,
                       help='Path to features.npy')
    parser.add_argument('--msm_dir', required=True,
                       help='Directory with MSM outputs')
    parser.add_argument('--vamp2_best', default='reports/vamp2_best.json',
                       help='Path to VAMP-2 best parameters')
    parser.add_argument('--energy', default=None,
                       help='Path to residue_energy.parquet (optional)')
    parser.add_argument('--pockets', default=None,
                       help='Path to pockets.parquet (optional)')
    parser.add_argument('--soft_dtraj', default=None,
                       help='Path to soft_dtraj.npy (optional)')
    parser.add_argument('--state_entropy', default=None,
                       help='Path to state_entropy.npy (optional)')
    parser.add_argument('--output_scores', default='data/derived/frame_scores_v2.csv',
                       help='Output CSV for scores')
    parser.add_argument('--output_summary', default='reports/scoring_v2_summary.json',
                       help='Output JSON for summary')
    parser.add_argument('--window', type=int, default=5,
                       help='Window size for smoothing (default: 5)')
    parser.add_argument('--normalize', choices=['rank', 'quantile'], default='rank',
                       help='Normalization method (default: rank)')
    parser.add_argument('--fusion', choices=['median', 'mean'], default='median',
                       help='Fusion method (default: median)')
    
    args = parser.parse_args()
    
    print("[Phase 3] Enhanced Anomaly Scoring v2")
    print("="*70)
    
    # Load data
    print("\n[Loading data...]")
    data = load_signal_data(
        args.features,
        args.vamp2_best,
        args.energy,
        args.pockets,
        args.soft_dtraj,
        args.state_entropy
    )
    
    print(f"  Loaded {data['n_frames']} frames")
    print(f"  TICA params: lag={data['lag_tica']}, dim={data['dim_tica']}")
    
    # Load MSM outputs
    msm_dir = Path(args.msm_dir)
    dtraj = np.load(msm_dir / 'dtraj.npy')
    tica_coords = np.load(msm_dir / 'tica_coords.npy')
    
    # Reconstruct MSM
    P = np.load(msm_dir / 'P.npy')
    pi = np.load(msm_dir / 'pi.npy')
    msm = MarkovStateModel(P, stationary_distribution=pi)
    
    print(f"  MSM: {msm.n_states} states")
    print(f"  TICA: {tica_coords.shape}")
    
    # Configure scoring
    config = {
        'lag_msm': 30,
        'k_neighbors': 20,
        'window_size': args.window,
        'normalize_method': args.normalize,
        'fusion_method': args.fusion
    }
    
    # Compute scores
    print("\n[Computing anomaly scores v2...]")
    scores_df, summary = compute_anomaly_scores_v2(data, msm, dtraj, 
                                                   tica_coords, config)
    
    # Save outputs
    print("\n[Saving results...]")
    output_scores = Path(args.output_scores)
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_scores, index=False)
    print(f"  Scores saved to {output_scores}")
    
    import json
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {output_summary}")
    
    print("\n[Summary]")
    print(f"  Signals used: {', '.join(summary['signals'])}")
    print(f"  Score range: [{summary['score_stats']['raw_min']:.1f}, "
          f"{summary['score_stats']['raw_max']:.1f}]")
    print(f"  Mean score: {summary['score_stats']['raw_mean']:.1f} ± "
          f"{summary['score_stats']['raw_std']:.1f}")
    
    print("\n" + "="*70)
    print("[Phase 3 Complete!]")


if __name__ == '__main__':
    main()

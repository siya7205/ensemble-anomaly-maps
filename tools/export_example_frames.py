#!/usr/bin/env python3
"""
Export example frames for presentation visualization.

This script creates bar charts and heatmaps for representative frames
(Normal, Anomalous, Ambiguous) to illustrate the anomaly detection pipeline.

Usage:
    python tools/export_example_frames.py
    python tools/export_example_frames.py --out-dir outputs/summary
    python tools/export_example_frames.py --scores outputs/frame_scores.csv
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RANDOM_SEED = 42
EPSILON = 1e-10  # Small constant for numerical stability

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    logger.error("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def find_representative_frames(
    scores: np.ndarray,
    n_frames: int = 3
) -> Dict[str, Tuple[int, float]]:
    """
    Find representative frames for visualization.
    
    Returns dict with:
        'normal': (frame_id, score) - low anomaly score
        'anomalous': (frame_id, score) - high anomaly score
        'ambiguous': (frame_id, score) - median anomaly score
    """
    if len(scores) == 0:
        return {}
    
    sorted_idx = np.argsort(scores)
    n = len(scores)
    
    # Normal: 5th percentile
    normal_idx = sorted_idx[int(n * 0.05)]
    # Anomalous: 95th percentile
    anomalous_idx = sorted_idx[int(n * 0.95)]
    # Ambiguous: 50th percentile
    ambiguous_idx = sorted_idx[int(n * 0.50)]
    
    return {
        'normal': (int(normal_idx), float(scores[normal_idx])),
        'anomalous': (int(anomalous_idx), float(scores[anomalous_idx])),
        'ambiguous': (int(ambiguous_idx), float(scores[ambiguous_idx]))
    }


def create_score_bar_chart(
    frame_id: int,
    score: float,
    label: str,
    output_path: Path,
    residue_scores: Optional[np.ndarray] = None,
    top_n: int = 10
) -> None:
    """Create a bar chart showing frame score and optionally top residue contributions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Color based on score
    if score < 0.3:
        color = '#3b82f6'  # Blue (normal)
    elif score > 0.7:
        color = '#ef4444'  # Red (anomalous)
    else:
        color = '#f59e0b'  # Orange (ambiguous)
    
    if residue_scores is not None and len(residue_scores) > 0:
        # Sort residues by score and take top N
        sorted_idx = np.argsort(residue_scores)[::-1][:top_n]
        labels = [f"Res {i}" for i in sorted_idx]
        values = residue_scores[sorted_idx]
        
        colors = ['#ef4444' if v > 0.5 else '#3b82f6' for v in values]
        
        ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor='black')
        ax.set_xlabel('Anomaly Contribution', fontsize=12)
        ax.set_ylabel('Residue', fontsize=12)
    else:
        # Simple single bar showing the frame score
        ax.barh(['Frame Score'], [score], color=color, edgecolor='black', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Fused Anomaly Score', fontsize=12)
    
    ax.set_title(f'{label} Frame (ID: {frame_id}, Score: {score:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved bar chart: {output_path}")


def create_example_panel(
    frames: Dict[str, Tuple[int, float]],
    output_path: Path
) -> None:
    """Create a 3-panel figure showing Normal, Anomalous, and Ambiguous frames."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    configs = [
        ('normal', 'Normal', '#3b82f6'),
        ('ambiguous', 'Ambiguous', '#f59e0b'),
        ('anomalous', 'Anomalous', '#ef4444'),
    ]
    
    for ax, (key, label, color) in zip(axes, configs):
        if key in frames:
            frame_id, score = frames[key]
            ax.bar([label], [score], color=color, edgecolor='black', width=0.6)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Anomaly Score', fontsize=10)
            ax.set_title(f'{label}\n(Frame {frame_id})', fontsize=12, fontweight='bold')
            ax.text(0, score + 0.05, f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=12)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved example panel: {output_path}")


def generate_synthetic_example(out_dir: Path, seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, Tuple[int, float]]:
    """Generate synthetic example frames when no real data is available."""
    # Create synthetic frame scores (200 frames with ~20% anomaly rate)
    np.random.seed(seed)
    n_frames = 200
    
    # Normal distribution for majority, some high scores
    scores = np.random.beta(2, 8, n_frames)
    # Add some anomalies
    anomaly_mask = np.random.random(n_frames) < 0.22
    scores[anomaly_mask] = np.random.beta(8, 2, anomaly_mask.sum())
    
    frames = find_representative_frames(scores)
    
    # Save synthetic data for reference
    synthetic_df = pd.DataFrame({
        'frame': range(n_frames),
        'y_score': scores,
        'y_true': anomaly_mask.astype(int)
    })
    synthetic_df.to_csv(out_dir / 'synthetic_frame_scores.csv', index=False)
    logger.info(f"Generated synthetic frame scores: {out_dir / 'synthetic_frame_scores.csv'}")
    
    return frames


def load_frame_scores(path: Optional[Path], base_dir: Path) -> Optional[pd.DataFrame]:
    """Load frame scores from various sources."""
    if path and path.exists():
        if path.suffix == '.parquet':
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)
    
    # Try auto-detect
    candidates = [
        base_dir / 'outputs' / 'summary' / 'latest_angles_scored.parquet',
        base_dir / 'outputs' / 'frame_scores.csv',
        base_dir / 'outputs' / 'summary' / 'frame_scores.csv',
        base_dir / 'outputs' / 'metrics' / 'frame_scores_dynamic.csv',
    ]
    
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Found frame scores: {candidate}")
            if candidate.suffix == '.parquet':
                return pd.read_parquet(candidate)
            else:
                return pd.read_csv(candidate)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Export example frames for presentation visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--scores', '-s',
        type=Path,
        default=None,
        help='Path to frame scores CSV/Parquet'
    )
    parser.add_argument(
        '--out-dir', '-o',
        type=Path,
        default=Path('outputs/summary'),
        help='Output directory (default: outputs/summary)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic example data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f'Random seed for synthetic data generation (default: {DEFAULT_RANDOM_SEED})'
    )
    
    args = parser.parse_args()
    
    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("EXPORT EXAMPLE FRAMES")
    logger.info("="*60)
    
    # Load or generate scores
    if args.synthetic:
        frames = generate_synthetic_example(args.out_dir, seed=args.seed)
        scores_df = pd.read_csv(args.out_dir / 'synthetic_frame_scores.csv')
        score_col = 'y_score'
    else:
        scores_df = load_frame_scores(args.scores, repo_root)
        
        if scores_df is None:
            logger.warning("No frame scores found. Generating synthetic example data.")
            frames = generate_synthetic_example(args.out_dir, seed=args.seed)
            scores_df = pd.read_csv(args.out_dir / 'synthetic_frame_scores.csv')
            score_col = 'y_score'
        else:
            # Find score column
            score_cols = ['y_score', 'score', 'anomaly_score', 'fused_score', 'if_score']
            score_col = None
            for col in score_cols:
                if col in scores_df.columns:
                    score_col = col
                    break
            
            if score_col is None:
                logger.error(f"No score column found. Available: {list(scores_df.columns)}")
                sys.exit(1)
            
            scores = scores_df[score_col].values
            
            # If scores are negative (like if_score), invert them
            if scores.min() < 0:
                scores = -scores  # Higher is more anomalous
                scores = (scores - scores.min()) / (scores.max() - scores.min() + EPSILON)
            
            frames = find_representative_frames(scores)
    
    logger.info(f"Representative frames: {frames}")
    
    # Create example panel
    create_example_panel(frames, args.out_dir / 'example_frames_panel.png')
    
    # Create individual bar charts
    for key, (frame_id, score) in frames.items():
        label = key.capitalize()
        create_score_bar_chart(
            frame_id, score, label,
            args.out_dir / f'example_frame_{frame_id}_bar.png'
        )
    
    # Save frame selection metadata
    metadata = {
        'frames': {k: {'frame_id': v[0], 'score': v[1]} for k, v in frames.items()},
        'score_column': score_col if 'score_col' in dir() else 'y_score',
        'n_frames_total': len(scores_df) if scores_df is not None else 0
    }
    
    with open(args.out_dir / 'example_frames_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {args.out_dir / 'example_frames_metadata.json'}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ EXAMPLE FRAMES EXPORT COMPLETE")
    logger.info("="*60)
    logger.info(f"\nOutput files in: {args.out_dir}")
    logger.info("  - example_frames_panel.png")
    for key in frames:
        logger.info(f"  - example_frame_{frames[key][0]}_bar.png")
    logger.info("  - example_frames_metadata.json")


if __name__ == '__main__':
    main()

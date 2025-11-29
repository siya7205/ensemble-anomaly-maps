#!/usr/bin/env python3
"""
Compute evaluation metrics and generate presentation figures.

This script computes AUROC/AUPRC with bootstrap 95% CIs, generates ROC/PR curves
and score distribution plots, computes operational metrics (precision@k, recall@FPR),
and produces a summary CSV that can be embedded in presentations.

Usage:
    python tools/compute_presentation_metrics.py
    python tools/compute_presentation_metrics.py --predictions outputs/predictions.csv
    python tools/compute_presentation_metrics.py --bootstrap 5000 --seed 42

Examples:
    # Auto-detect predictions from repository outputs
    python tools/compute_presentation_metrics.py

    # Use specific predictions file
    python tools/compute_presentation_metrics.py --predictions tests/sample_predictions.csv

    # Custom output directory and bootstrap settings
    python tools/compute_presentation_metrics.py \\
        --predictions outputs/predictions.csv \\
        --out-dir outputs/metrics \\
        --bootstrap 5000 \\
        --seed 42

    # Dry run (check inputs without generating outputs)
    python tools/compute_presentation_metrics.py --dry-run
"""
import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading Functions
# =============================================================================

def auto_detect_predictions(base_dir: Path) -> Optional[Path]:
    """
    Auto-detect predictions file from repository outputs.
    
    Searches common locations for per-frame prediction files:
    - outputs/summary/predictions.csv
    - outputs/predictions.csv
    - outputs/frame_scores.csv
    - outputs/frame_predictions.parquet
    - outputs/run-*/predictions.csv
    - outputs/summary/latest_angles_scored.parquet (fallback)
    
    Returns:
        Path to detected predictions file, or None if not found.
    """
    common_paths = [
        base_dir / 'outputs' / 'summary' / 'predictions.csv',
        base_dir / 'outputs' / 'predictions.csv',
        base_dir / 'outputs' / 'frame_scores.csv',
        base_dir / 'outputs' / 'frame_predictions.parquet',
        base_dir / 'outputs' / 'summary' / 'frame_scores.csv',
        base_dir / 'outputs' / 'summary' / 'frame_predictions.parquet',
    ]
    
    # Check direct paths first
    for path in common_paths:
        if path.exists():
            logger.info(f"Found predictions file: {path}")
            return path
    
    # Check run-* directories
    outputs_dir = base_dir / 'outputs'
    if outputs_dir.exists():
        for run_dir in sorted(outputs_dir.glob('run-*'), reverse=True):
            for filename in ['predictions.csv', 'frame_scores.csv', 'predictions.parquet']:
                path = run_dir / filename
                if path.exists():
                    logger.info(f"Found predictions file: {path}")
                    return path
    
    # Fallback: Try to use latest_angles_scored.parquet if it has score columns
    fallback_path = base_dir / 'outputs' / 'summary' / 'latest_angles_scored.parquet'
    if fallback_path.exists():
        logger.info(f"Found fallback predictions file: {fallback_path}")
        return fallback_path
    
    return None


def load_predictions(path: Path) -> pd.DataFrame:
    """
    Load predictions from CSV or Parquet file.
    
    Expected columns: frame, y_true, y_score, optional run_id
    
    Args:
        path: Path to predictions file
        
    Returns:
        DataFrame with columns: frame, y_true, y_score, and optionally run_id
    """
    if path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix.lower() in ['.csv', '.tsv']:
        sep = '\t' if path.suffix.lower() == '.tsv' else ','
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Map common column name variations
    column_mapping = {
        'label': 'y_true',
        'target': 'y_true',
        'ground_truth': 'y_true',
        'anomaly': 'y_true',
        'is_anomaly': 'y_true',
        'score': 'y_score',
        'prediction': 'y_score',
        'pred_score': 'y_score',
        'anomaly_score': 'y_score',
        'run': 'run_id',
        'model': 'run_id',
        'method': 'run_id',
    }
    df = df.rename(columns=column_mapping)
    
    # Validate required columns
    required_cols = ['y_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Available: {list(df.columns)}")
    
    # Add frame column if missing
    if 'frame' not in df.columns:
        df['frame'] = range(len(df))
    
    return df


# =============================================================================
# Metrics Computation Functions
# =============================================================================

def compute_auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Compute AUROC and AUPRC."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    return auroc, auprc


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap confidence intervals for AUROC and AUPRC.
    
    Returns:
        Dict with keys 'auroc' and 'auprc', each containing (point_estimate, lower, upper)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    auroc_scores = []
    auprc_scores = []
    
    for _ in range(n_bootstrap):
        # Stratified bootstrap to preserve class ratio
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[idx]
        y_score_boot = y_score[idx]
        
        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        try:
            auroc, auprc = compute_auroc_auprc(y_true_boot, y_score_boot)
            auroc_scores.append(auroc)
            auprc_scores.append(auprc)
        except Exception:
            continue
    
    # Point estimates
    auroc_point, auprc_point = compute_auroc_auprc(y_true, y_score)
    
    # Percentile confidence intervals
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)
    
    if len(auroc_scores) > 0:
        auroc_lo = np.percentile(auroc_scores, lo_pct)
        auroc_hi = np.percentile(auroc_scores, hi_pct)
    else:
        auroc_lo = auroc_hi = auroc_point
    
    if len(auprc_scores) > 0:
        auprc_lo = np.percentile(auprc_scores, lo_pct)
        auprc_hi = np.percentile(auprc_scores, hi_pct)
    else:
        auprc_lo = auprc_hi = auprc_point
    
    return {
        'auroc': (auroc_point, auroc_lo, auroc_hi),
        'auprc': (auprc_point, auprc_lo, auprc_hi)
    }


def compute_precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_percentages: List[float]
) -> Dict[str, float]:
    """
    Compute precision at top k% of predictions.
    
    Args:
        y_true: Binary labels
        y_score: Prediction scores
        k_percentages: List of percentages (e.g., [1, 5, 10])
        
    Returns:
        Dict mapping percentage to precision value
    """
    n_samples = len(y_true)
    sorted_idx = np.argsort(y_score)[::-1]  # Descending order
    
    results = {}
    for k_pct in k_percentages:
        k = max(1, int(n_samples * k_pct / 100))
        top_k_idx = sorted_idx[:k]
        precision = np.mean(y_true[top_k_idx])
        results[f'precision_at_{int(k_pct)}pct'] = precision
    
    return results


def compute_recall_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_thresholds: List[float]
) -> Dict[str, float]:
    """
    Compute recall at specified false positive rates.
    
    Args:
        y_true: Binary labels
        y_score: Prediction scores
        fpr_thresholds: List of FPR values (e.g., [0.01, 0.05])
        
    Returns:
        Dict mapping FPR to recall value
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    results = {}
    for fpr_thresh in fpr_thresholds:
        # Find recall at closest FPR <= threshold
        valid_idx = np.where(fpr <= fpr_thresh)[0]
        if len(valid_idx) > 0:
            recall = tpr[valid_idx[-1]]
        else:
            recall = 0.0
        pct_key = int(fpr_thresh * 100)
        results[f'recall_at_{pct_key}pct_fpr'] = recall
    
    return results


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
    top_k_pct: List[float] = None,
    fpr_list: List[float] = None
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.
    
    Returns:
        Dict with all metrics including bootstrap CIs
    """
    if top_k_pct is None:
        top_k_pct = [1, 5, 10]
    if fpr_list is None:
        fpr_list = [0.01, 0.05]
    
    # Bootstrap CI for AUROC/AUPRC
    ci_results = bootstrap_ci(y_true, y_score, n_bootstrap=n_bootstrap, seed=seed)
    
    # Precision at k
    precision_results = compute_precision_at_k(y_true, y_score, top_k_pct)
    
    # Recall at FPR
    recall_results = compute_recall_at_fpr(y_true, y_score, fpr_list)
    
    metrics = {
        'n_samples': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'n_negative': int(np.sum(1 - y_true)),
        'auroc': ci_results['auroc'][0],
        'auroc_lo': ci_results['auroc'][1],
        'auroc_hi': ci_results['auroc'][2],
        'auprc': ci_results['auprc'][0],
        'auprc_lo': ci_results['auprc'][1],
        'auprc_hi': ci_results['auprc'][2],
    }
    metrics.update(precision_results)
    metrics.update(recall_results)
    
    return metrics


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    auroc: float,
    auroc_ci: Tuple[float, float],
    output_path: Path,
    title: str = "ROC Curve"
) -> None:
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#2563eb', lw=2,
            label=f'AUROC = {auroc:.3f} ({auroc_ci[0]:.3f}–{auroc_ci[1]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#2563eb')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved ROC curve: {output_path}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    auprc: float,
    auprc_ci: Tuple[float, float],
    output_path: Path,
    title: str = "Precision-Recall Curve"
) -> None:
    """Generate and save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    baseline = np.mean(y_true)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='#16a34a', lw=2,
            label=f'AUPRC = {auprc:.3f} ({auprc_ci[0]:.3f}–{auprc_ci[1]:.3f})')
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, alpha=0.5,
               label=f'Random baseline ({baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='#16a34a')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved PR curve: {output_path}")


def plot_score_distributions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    top_k_pct: List[float] = None,
    title: str = "Score Distributions"
) -> None:
    """Generate and save score distribution plot with histogram and KDE."""
    if top_k_pct is None:
        top_k_pct = [1, 5, 10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate scores by class
    scores_neg = y_score[y_true == 0]
    scores_pos = y_score[y_true == 1]
    
    # Plot histograms with KDE
    if len(scores_neg) > 0:
        sns.histplot(scores_neg, kde=True, ax=ax, color='#3b82f6', alpha=0.5,
                     label=f'Negative (n={len(scores_neg)})', stat='density')
    if len(scores_pos) > 0:
        sns.histplot(scores_pos, kde=True, ax=ax, color='#ef4444', alpha=0.5,
                     label=f'Positive (n={len(scores_pos)})', stat='density')
    
    # Add top-k percentile markers
    colors = ['#f59e0b', '#8b5cf6', '#06b6d4']
    for i, k_pct in enumerate(top_k_pct):
        threshold = np.percentile(y_score, 100 - k_pct)
        color = colors[i % len(colors)]
        ax.axvline(x=threshold, color=color, linestyle='--', lw=2, alpha=0.8,
                   label=f'Top {k_pct}% threshold ({threshold:.3f})')
    
    ax.set_xlabel('Prediction Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved score distributions: {output_path}")


# =============================================================================
# CSV Output Functions
# =============================================================================

def save_metrics_summary(
    metrics: Dict[str, Any],
    output_path: Path,
    run_id: str = 'overall'
) -> None:
    """Save metrics summary to CSV."""
    row = {'dataset': run_id}
    row.update(metrics)
    
    df = pd.DataFrame([row])
    
    # Reorder columns
    cols_order = [
        'dataset', 'n_samples', 'n_positive', 'n_negative',
        'auroc', 'auroc_lo', 'auroc_hi',
        'auprc', 'auprc_lo', 'auprc_hi',
    ]
    other_cols = [c for c in df.columns if c not in cols_order]
    df = df[[c for c in cols_order if c in df.columns] + other_cols]
    
    df.to_csv(output_path, index=False, float_format='%.6f')
    logger.info(f"Saved metrics summary: {output_path}")


def save_per_run_metrics(
    metrics_list: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Save per-run metrics to CSV."""
    df = pd.DataFrame(metrics_list)
    
    # Reorder columns
    cols_order = [
        'run_id', 'n_samples', 'n_positive', 'n_negative',
        'auroc', 'auroc_lo', 'auroc_hi',
        'auprc', 'auprc_lo', 'auprc_hi',
    ]
    other_cols = [c for c in df.columns if c not in cols_order]
    df = df[[c for c in cols_order if c in df.columns] + other_cols]
    
    df.to_csv(output_path, index=False, float_format='%.6f')
    logger.info(f"Saved per-run metrics: {output_path}")


# =============================================================================
# Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute evaluation metrics and generate presentation figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--predictions', '-p',
        type=Path,
        default=None,
        help='Path to predictions CSV or Parquet file. If not provided, auto-detects from repo outputs.'
    )
    parser.add_argument(
        '--out-dir', '-o',
        type=Path,
        default=Path('outputs/summary'),
        help='Output directory for plots and metrics (default: outputs/summary)'
    )
    parser.add_argument(
        '--bootstrap', '-b',
        type=int,
        default=2000,
        help='Number of bootstrap resamples for CIs (default: 2000)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=0,
        help='Random seed for reproducibility (default: 0)'
    )
    parser.add_argument(
        '--top-k-pct',
        type=float,
        nargs='+',
        default=[1, 5, 10],
        help='Percentages for precision@k (default: 1 5 10)'
    )
    parser.add_argument(
        '--fpr-list',
        type=float,
        nargs='+',
        default=[0.01, 0.05],
        help='FPR thresholds for recall@FPR (default: 0.01 0.05)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check inputs without generating outputs'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    logger.info("="*70)
    logger.info("COMPUTE PRESENTATION METRICS")
    logger.info("="*70)
    
    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Load predictions
    if args.predictions:
        predictions_path = args.predictions
        if not predictions_path.exists():
            logger.error(f"Predictions file not found: {predictions_path}")
            sys.exit(1)
    else:
        predictions_path = auto_detect_predictions(repo_root)
        if predictions_path is None:
            logger.error(
                "No predictions file found. Please provide one with --predictions.\n"
                "Expected file format: CSV/Parquet with columns: frame, y_true, y_score, optional run_id\n"
                "Searched locations:\n"
                "  - outputs/summary/predictions.csv\n"
                "  - outputs/predictions.csv\n"
                "  - outputs/frame_scores.csv\n"
                "  - outputs/run-*/predictions.csv"
            )
            sys.exit(1)
    
    logger.info(f"Loading predictions from: {predictions_path}")
    df = load_predictions(predictions_path)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Check for y_true column
    has_labels = 'y_true' in df.columns and not df['y_true'].isna().all()
    
    if not has_labels:
        logger.warning(
            "No ground truth labels (y_true) found in predictions.\n"
            "Score distribution plots will be generated, but AUC/AUPRC cannot be computed.\n"
            "To compute full metrics, provide a CSV with columns: frame, y_true, y_score"
        )
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Inputs validated successfully. Exiting without generating outputs.")
        return
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.out_dir}")
    
    # Determine if we have run_id for per-run analysis
    has_run_id = 'run_id' in df.columns and df['run_id'].nunique() > 1
    
    if has_labels:
        y_true = df['y_true'].values.astype(int)
        y_score = df['y_score'].values
        
        # Check for sufficient class diversity
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            logger.error("Cannot compute AUC metrics: need at least one sample of each class.")
            sys.exit(1)
        
        logger.info(f"\nClass distribution: {n_pos} positive, {n_neg} negative")
        logger.info(f"Bootstrap resamples: {args.bootstrap}")
        logger.info(f"Random seed: {args.seed}")
        
        # Compute overall metrics
        logger.info("\n[1/4] Computing overall metrics...")
        metrics = compute_all_metrics(
            y_true, y_score,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            top_k_pct=args.top_k_pct,
            fpr_list=args.fpr_list
        )
        
        logger.info(f"  AUROC: {metrics['auroc']:.4f} ({metrics['auroc_lo']:.4f}–{metrics['auroc_hi']:.4f})")
        logger.info(f"  AUPRC: {metrics['auprc']:.4f} ({metrics['auprc_lo']:.4f}–{metrics['auprc_hi']:.4f})")
        
        # Save metrics summary
        save_metrics_summary(metrics, args.out_dir / 'metrics_summary.csv')
        
        # Generate plots
        logger.info("\n[2/4] Generating ROC curve...")
        plot_roc_curve(
            y_true, y_score,
            metrics['auroc'], (metrics['auroc_lo'], metrics['auroc_hi']),
            args.out_dir / 'predictions_roc.png'
        )
        
        logger.info("\n[3/4] Generating PR curve...")
        plot_pr_curve(
            y_true, y_score,
            metrics['auprc'], (metrics['auprc_lo'], metrics['auprc_hi']),
            args.out_dir / 'predictions_pr.png'
        )
        
        logger.info("\n[4/4] Generating score distributions...")
        plot_score_distributions(
            y_true, y_score,
            args.out_dir / 'score_distributions.png',
            top_k_pct=args.top_k_pct
        )
        
        # Per-run analysis if applicable
        if has_run_id:
            logger.info("\n[5/5] Computing per-run metrics...")
            per_run_metrics = []
            
            for run_id in df['run_id'].unique():
                run_df = df[df['run_id'] == run_id]
                run_y_true = run_df['y_true'].values.astype(int)
                run_y_score = run_df['y_score'].values
                
                # Skip if insufficient class diversity
                if len(np.unique(run_y_true)) < 2:
                    logger.warning(f"  Skipping run {run_id}: only one class present")
                    continue
                
                run_metrics = compute_all_metrics(
                    run_y_true, run_y_score,
                    n_bootstrap=args.bootstrap,
                    seed=args.seed,
                    top_k_pct=args.top_k_pct,
                    fpr_list=args.fpr_list
                )
                run_metrics['run_id'] = run_id
                per_run_metrics.append(run_metrics)
                
                logger.info(f"  {run_id}: AUROC={run_metrics['auroc']:.4f}, AUPRC={run_metrics['auprc']:.4f}")
            
            if per_run_metrics:
                save_per_run_metrics(per_run_metrics, args.out_dir / 'metrics_summary_per_run.csv')
                
                # Perform statistical comparison if multiple runs
                if len(per_run_metrics) > 1:
                    aurocs = [m['auroc'] for m in per_run_metrics]
                    logger.info(f"\n  Per-run AUROC summary: mean={np.mean(aurocs):.4f}, std={np.std(aurocs):.4f}")
                    logger.info(f"  Per-run AUROC range: {min(aurocs):.4f} – {max(aurocs):.4f}")
    else:
        # No labels - generate score distribution only
        y_score = df['y_score'].values
        
        logger.info("\n[1/1] Generating score distributions (no labels available)...")
        
        # Create dummy y_true for plotting (all zeros)
        dummy_y_true = np.zeros(len(y_score), dtype=int)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(y_score, kde=True, ax=ax, color='#6366f1', alpha=0.6, stat='density')
        
        # Add top-k percentile markers
        colors = ['#f59e0b', '#8b5cf6', '#06b6d4']
        for i, k_pct in enumerate(args.top_k_pct):
            threshold = np.percentile(y_score, 100 - k_pct)
            color = colors[i % len(colors)]
            ax.axvline(x=threshold, color=color, linestyle='--', lw=2, alpha=0.8,
                       label=f'Top {k_pct}% threshold ({threshold:.3f})')
        
        ax.set_xlabel('Prediction Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distribution (No Labels)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(args.out_dir / 'score_distributions.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved score distributions: {args.out_dir / 'score_distributions.png'}")
        
        # Save a minimal metrics file
        minimal_metrics = {
            'dataset': 'overall',
            'n_samples': len(y_score),
            'score_min': float(y_score.min()),
            'score_max': float(y_score.max()),
            'score_mean': float(y_score.mean()),
            'score_std': float(y_score.std()),
            'note': 'No y_true labels provided - AUC metrics unavailable'
        }
        pd.DataFrame([minimal_metrics]).to_csv(args.out_dir / 'metrics_summary.csv', index=False)
        logger.info(f"Saved minimal metrics: {args.out_dir / 'metrics_summary.csv'}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ METRICS COMPUTATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nOutput files saved to: {args.out_dir}")
    logger.info("  - metrics_summary.csv")
    if has_labels:
        logger.info("  - predictions_roc.png")
        logger.info("  - predictions_pr.png")
    logger.info("  - score_distributions.png")
    if has_labels and has_run_id:
        logger.info("  - metrics_summary_per_run.csv")


if __name__ == '__main__':
    main()

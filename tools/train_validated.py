#!/usr/bin/env python3
"""
Scientifically validated training pipeline.

This script wraps the standard training pipeline with comprehensive
input validation and scientific best practices.

Usage:
    python tools/train_validated.py --features data/features.npy --output outputs/validated
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from msm.input_validation import validate_input_data, print_validation_summary
from msm.reproducibility import set_global_seed

try:
    from tools.run_phase1 import run_phase1_pipeline
except ImportError as e:
    print(f"Error: Failed to import run_phase1 module: {e}")
    print("Please ensure the tools/ directory is in your Python path")
    sys.exit(EXIT_FAILURE)


def train_with_validation(features_path: str,
                         output_dir: str,
                         config_path: Optional[str] = None,
                         traj_path: Optional[str] = None,
                         top_path: Optional[str] = None,
                         lag_tica: Optional[int] = None,
                         lag_msm: Optional[int] = None,
                         n_clusters: Optional[int] = None,
                         skip_validation: bool = False,
                         force: bool = False):
    """
    Train ML model with comprehensive validation.
    
    Args:
        features_path: Path to features.npy
        output_dir: Output directory
        config_path: Path to config YAML
        traj_path: Path to trajectory (for validation)
        top_path: Path to topology (for validation)
        lag_tica: TICA lag time (for validation)
        lag_msm: MSM lag time (for validation)
        n_clusters: Number of clusters (for validation)
        skip_validation: Skip input validation
        force: Proceed even if validation fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SCIENTIFICALLY VALIDATED TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Input Validation
    if not skip_validation:
        print("\n[STEP 1/3] Validating Input Data")
        print("-"*70)
        
        is_valid, validation_report = validate_input_data(
            features_path=features_path,
            traj_path=traj_path,
            top_path=top_path,
            lag_tica=lag_tica,
            lag_msm=lag_msm,
            n_clusters=n_clusters
        )
        
        # Save validation report
        validation_file = output_dir / 'input_validation_report.json'
        with open(validation_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        print(f"\n✓ Validation report saved to {validation_file}")
        
        # Print summary
        print_validation_summary(validation_report)
        
        # Check if we should proceed
        if not is_valid and not force:
            print("\n✗ Validation FAILED. Fix errors before training.")
            print("  Use --force to proceed anyway (not recommended).")
            return EXIT_FAILURE
        elif not is_valid and force:
            print("\n⚠ WARNING: Proceeding despite validation failures (--force used)")
    else:
        print("\n[STEP 1/3] Skipping input validation (--skip-validation used)")
    
    # Step 2: Model Selection and Training
    print("\n[STEP 2/3] Running Model Selection and Training")
    print("-"*70)
    
    try:
        run_phase1_pipeline(
            features_path,
            output_dir / 'phase1',
            config_path=config_path,
            skip_vamp2=False,
            skip_bootstrap=False
        )
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return EXIT_FAILURE
    
    # Step 3: Post-Training Validation
    print("\n[STEP 3/3] Post-Training Validation")
    print("-"*70)
    print("To validate the trained model, run:")
    print(f"  python tools/validate_model.py \\")
    print(f"    --msm_dir {output_dir}/phase1/models/msm_bootstrap \\")
    print(f"    --output_dir {output_dir}/validation")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  • phase1/ - Model selection and bootstrap results")
    print(f"  • input_validation_report.json - Input validation report")
    print("\nNext steps:")
    print("  1. Review input validation report")
    print("  2. Run model validation (see command above)")
    print("  3. Review validation plots and metrics")
    print("  4. If validation passes, proceed with anomaly detection")
    
    return EXIT_SUCCESS


def main():
    parser = argparse.ArgumentParser(
        description='Scientifically validated ML training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validated training
  python tools/train_validated.py \\
      --features data/features.npy \\
      --output outputs/validated \\
      --topology data/raw_trajectory/topology.pdb \\
      --trajectory data/raw_trajectory/trajectory.xtc \\
      --lag_tica 10 --lag_msm 30 --n_clusters 30
  
  # Skip input validation (not recommended)
  python tools/train_validated.py \\
      --features data/features.npy \\
      --output outputs/validated \\
      --skip-validation
  
  # Force training despite validation errors (dangerous!)
  python tools/train_validated.py \\
      --features data/features.npy \\
      --output outputs/validated \\
      --force

This pipeline performs:
  1. Input data validation (trajectory quality, feature quality, parameter compatibility)
  2. VAMP-2 based model selection
  3. Bootstrap MSM for uncertainty quantification
  4. Reproducible configuration saving

For scientific best practices, see: SCIENTIFIC_TRAINING_GUIDE.md
        """
    )
    
    # Required arguments
    parser.add_argument('--features', required=True,
                       help='Path to features.npy file')
    parser.add_argument('--output', required=True,
                       help='Output directory')
    
    # Optional validation inputs
    parser.add_argument('--topology',
                       help='Path to topology file (for validation)')
    parser.add_argument('--trajectory',
                       help='Path to trajectory file (for validation)')
    parser.add_argument('--lag_tica', type=int,
                       help='TICA lag time (for validation)')
    parser.add_argument('--lag_msm', type=int,
                       help='MSM lag time (for validation)')
    parser.add_argument('--n_clusters', type=int,
                       help='Number of clusters (for validation)')
    
    # Configuration
    parser.add_argument('--config',
                       help='Path to pipeline.yaml config')
    
    # Control flags
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip input validation (not recommended)')
    parser.add_argument('--force', action='store_true',
                       help='Proceed even if validation fails (dangerous!)')
    
    args = parser.parse_args()
    
    return train_with_validation(
        features_path=args.features,
        output_dir=args.output,
        config_path=args.config,
        traj_path=args.trajectory,
        top_path=args.topology,
        lag_tica=args.lag_tica,
        lag_msm=args.lag_msm,
        n_clusters=args.n_clusters,
        skip_validation=args.skip_validation,
        force=args.force
    )


if __name__ == '__main__':
    sys.exit(main())

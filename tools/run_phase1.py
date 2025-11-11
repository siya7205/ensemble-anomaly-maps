#!/usr/bin/env python3
"""
Unified CLI tool for Phase 1: Model Selection and Bootstrap.

This tool orchestrates:
1. VAMP-2 model selection for optimal TICA parameters
2. Bootstrap MSM for uncertainty quantification
3. Saving run configuration for reproducibility
"""
import argparse
import sys
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from msm.select_lag_and_dim import select_lag_and_dim
from msm.bootstrap_msm import bootstrap_msm
from msm.reproducibility import save_run_config, set_global_seed


def load_config(config_path='configs/pipeline.yaml'):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_phase1_pipeline(features_path, output_base, config_path=None, 
                       skip_vamp2=False, skip_bootstrap=False):
    """
    Run Phase 1 pipeline: model selection + bootstrap.
    
    Args:
        features_path: Path to features.npy
        output_base: Base output directory
        config_path: Path to config YAML
        skip_vamp2: Skip VAMP-2 selection (use config defaults)
        skip_bootstrap: Skip bootstrap analysis
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(config_path) if config_path else load_config()
    
    if config:
        # Set global seed
        global_seed = config['seeds']['global']
        set_global_seed(global_seed)
        print(f"Set global seed: {global_seed}")
    
    # Stage 1: VAMP-2 model selection
    lag_tica = None
    dim_tica = None
    
    if not skip_vamp2:
        print("\n" + "="*70)
        print("STAGE 1: VAMP-2 Model Selection")
        print("="*70)
        
        reports_dir = output_base / 'reports'
        lag_tica, dim_tica = select_lag_and_dim(
            features_path, reports_dir, config_path
        )[0]
        
        print(f"\n✓ Selected: lag={lag_tica}, dim={dim_tica}")
    else:
        print("\nSkipping VAMP-2 selection (using config defaults)")
        if config:
            lag_tica = config['tica']['default_lag']
            dim_tica = config['tica']['default_dim']
            print(f"  Using: lag={lag_tica}, dim={dim_tica}")
    
    # Stage 2: Bootstrap MSM
    if not skip_bootstrap:
        print("\n" + "="*70)
        print("STAGE 2: Bootstrap MSM")
        print("="*70)
        
        models_dir = output_base / 'models' / 'msm_bootstrap'
        bootstrap_msm(
            features_path, models_dir, config_path,
            lag_tica, dim_tica
        )
        
        print("\n✓ Bootstrap complete")
    else:
        print("\nSkipping bootstrap analysis")
    
    # Save run configuration
    print("\n" + "="*70)
    print("Saving run configuration")
    print("="*70)
    
    run_config = {
        'features_path': str(features_path),
        'output_base': str(output_base),
        'lag_tica': lag_tica,
        'dim_tica': dim_tica,
        'skip_vamp2': skip_vamp2,
        'skip_bootstrap': skip_bootstrap
    }
    
    if config:
        run_config['seeds'] = config['seeds']
        run_config['msm_params'] = config['msm']
        run_config['bootstrap_params'] = config['bootstrap']
    
    run_path = output_base / 'run.json'
    save_run_config(run_path, run_config)
    print(f"✓ Saved run config to {run_path}")
    
    print("\n" + "="*70)
    print("Phase 1 Complete!")
    print("="*70)
    print(f"\nOutputs:")
    if not skip_vamp2:
        print(f"  • VAMP-2 results: {output_base / 'reports'}")
    if not skip_bootstrap:
        print(f"  • Bootstrap MSMs: {output_base / 'models' / 'msm_bootstrap'}")
    print(f"  • Run config: {run_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Model Selection & Bootstrap MSM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with VAMP-2 selection and bootstrap
  python tools/run_phase1.py --features data/features.npy --output outputs/phase1
  
  # Skip VAMP-2, use config defaults
  python tools/run_phase1.py --features data/features.npy --output outputs/phase1 --skip-vamp2
  
  # Only run VAMP-2 selection
  python tools/run_phase1.py --features data/features.npy --output outputs/phase1 --skip-bootstrap
  
  # Use custom config
  python tools/run_phase1.py --features data/features.npy --output outputs/phase1 --config my_config.yaml
        """
    )
    
    parser.add_argument('--features', required=True,
                       help='Path to features.npy file')
    parser.add_argument('--output', required=True,
                       help='Base output directory')
    parser.add_argument('--config', default=None,
                       help='Path to pipeline.yaml config (default: configs/pipeline.yaml)')
    parser.add_argument('--skip-vamp2', action='store_true',
                       help='Skip VAMP-2 model selection')
    parser.add_argument('--skip-bootstrap', action='store_true',
                       help='Skip bootstrap MSM analysis')
    
    args = parser.parse_args()
    
    run_phase1_pipeline(
        args.features,
        args.output,
        args.config,
        args.skip_vamp2,
        args.skip_bootstrap
    )


if __name__ == '__main__':
    main()

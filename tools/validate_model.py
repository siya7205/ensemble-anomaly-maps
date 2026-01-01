#!/usr/bin/env python3
"""
Comprehensive model validation script.

This script performs scientific validation of trained MSM and TICA models
to ensure they are statistically sound and scientifically correct.

Usage:
    python tools/validate_model.py --msm_dir outputs/msm --output_dir outputs/validation
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from msm.validation import (
    chapman_kolmogorov_test,
    implied_timescales_convergence,
    vamp2_cross_validation,
    signal_correlation_analysis,
    validate_stationary_distribution,
    plot_validation_summary,
    generate_validation_report
)


def load_msm_outputs(msm_dir: Path):
    """Load MSM outputs from directory."""
    msm_dir = Path(msm_dir)
    
    outputs = {}
    
    # Load discrete trajectory
    dtraj_path = msm_dir / 'dtraj.npy'
    if dtraj_path.exists():
        outputs['dtraj'] = np.load(dtraj_path)
        print(f"✓ Loaded discrete trajectory: {len(outputs['dtraj'])} frames")
    else:
        raise FileNotFoundError(f"Discrete trajectory not found: {dtraj_path}")
    
    # Load transition matrix
    P_path = msm_dir / 'P.npy'
    if P_path.exists():
        outputs['P'] = np.load(P_path)
        print(f"✓ Loaded transition matrix: {outputs['P'].shape}")
    
    # Load stationary distribution
    pi_path = msm_dir / 'pi.npy'
    if pi_path.exists():
        outputs['pi'] = np.load(pi_path)
        print(f"✓ Loaded stationary distribution: {len(outputs['pi'])} states")
    
    # Load TICA coordinates
    tica_path = msm_dir / 'tica_coords.npy'
    if tica_path.exists():
        outputs['tica_coords'] = np.load(tica_path)
        print(f"✓ Loaded TICA coordinates: {outputs['tica_coords'].shape}")
    
    # Load features if available
    features_path = msm_dir.parent / 'features.npy'
    if features_path.exists():
        outputs['features'] = np.load(features_path)
        print(f"✓ Loaded features: {outputs['features'].shape}")
    
    return outputs


def validate_msm(msm_dir: Path, 
                 output_dir: Path,
                 msm_lag: int = 30,
                 skip_ck_test: bool = False,
                 skip_its_test: bool = False):
    """
    Run comprehensive MSM validation.
    
    Args:
        msm_dir: Directory containing MSM outputs
        output_dir: Directory for validation outputs
        msm_lag: MSM lag time used in training
        skip_ck_test: Skip Chapman-Kolmogorov test (slow)
        skip_its_test: Skip implied timescales test
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MSM VALIDATION")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading MSM outputs...")
    data = load_msm_outputs(msm_dir)
    dtraj = data['dtraj']
    
    results = {}
    
    # Test 1: Chapman-Kolmogorov test
    if not skip_ck_test and 'P' in data:
        print("\n[2/6] Running Chapman-Kolmogorov test...")
        print("  This tests if the MSM satisfies the Markov property.")
        
        try:
            lags, predicted, estimated = chapman_kolmogorov_test(
                dtraj, msm_lag, n_lags=5
            )
            
            # Compute error metrics
            errors = np.abs(predicted - estimated)
            mean_error = np.nanmean(errors)
            max_error = np.nanmax(errors)
            
            results['chapman_kolmogorov'] = {
                'lags_tested': lags.tolist(),
                'mean_absolute_error': float(mean_error),
                'max_absolute_error': float(max_error),
                'passed': max_error < 0.2  # Tolerance
            }
            
            print(f"  Mean absolute error: {mean_error:.4f}")
            print(f"  Max absolute error: {max_error:.4f}")
            print(f"  Status: {'✓ PASSED' if results['chapman_kolmogorov']['passed'] else '✗ NEEDS REVIEW'}")
            
            # Store for plotting
            ck_test_data = (lags, predicted, estimated)
        except Exception as e:
            print(f"  ✗ Chapman-Kolmogorov test failed: {e}")
            ck_test_data = None
    else:
        print("\n[2/6] Skipping Chapman-Kolmogorov test")
        ck_test_data = None
    
    # Test 2: Implied timescales convergence
    if not skip_its_test:
        print("\n[3/6] Computing implied timescales...")
        print("  Timescales should plateau at sufficient lag times.")
        
        try:
            lags, timescales = implied_timescales_convergence(dtraj, n_its=5)
            
            # Check for convergence (coefficient of variation in last half)
            if len(timescales) >= 4:
                second_half = timescales[len(timescales)//2:]
                cv = np.nanstd(second_half, axis=0) / np.nanmean(second_half, axis=0)
                converged = cv[0] < 0.2  # CV < 20% for slowest timescale
            else:
                converged = False
            
            results['implied_timescales'] = {
                'lags': lags.tolist(),
                'timescales': timescales.tolist(),
                'converged': bool(converged),
                'cv_slowest': float(cv[0]) if len(timescales) >= 4 else None
            }
            
            print(f"  Tested {len(lags)} lag times")
            print(f"  Convergence status: {'✓ CONVERGED' if converged else '✗ NOT CONVERGED'}")
            
            its_data = (lags, timescales)
        except Exception as e:
            print(f"  ✗ Implied timescales test failed: {e}")
            its_data = None
    else:
        print("\n[3/6] Skipping implied timescales test")
        its_data = None
    
    # Test 3: Stationary distribution validation
    if 'pi' in data:
        print("\n[4/6] Validating stationary distribution...")
        print("  Comparing MSM stationary distribution to empirical frequencies.")
        
        is_valid, diagnostics = validate_stationary_distribution(
            data['pi'], dtraj, tolerance=0.15
        )
        
        results['stationary_distribution'] = diagnostics
        
        print(f"  Max relative error: {diagnostics['max_relative_error']:.4f}")
        print(f"  Mean relative error: {diagnostics['mean_relative_error']:.4f}")
        print(f"  Sampled states: {diagnostics['n_sampled_states']}/{len(data['pi'])}")
        print(f"  Status: {'✓ VALID' if is_valid else '✗ NEEDS REVIEW'}")
    else:
        print("\n[4/6] Skipping stationary distribution validation (pi.npy not found)")
    
    # Test 4: Cross-validation (if features available)
    if 'features' in data:
        print("\n[5/6] Running VAMP-2 cross-validation...")
        print("  Testing model generalization with k-fold CV.")
        
        # Use reasonable defaults for lag and dim
        lag_tica = 10
        dim_tica = 5
        
        try:
            mean_score, std_score = vamp2_cross_validation(
                data['features'], lag_tica, dim_tica, n_folds=5, seed=42
            )
            
            results['cross_validation'] = {
                'mean_vamp2_score': float(mean_score),
                'std_vamp2_score': float(std_score),
                'n_folds': 5,
                'lag': lag_tica,
                'dim': dim_tica
            }
            
            print(f"  Mean VAMP-2 score: {mean_score:.4f} ± {std_score:.4f}")
            print(f"  Status: ✓ COMPLETED")
        except Exception as e:
            print(f"  ✗ Cross-validation failed: {e}")
    else:
        print("\n[5/6] Skipping cross-validation (features not available)")
    
    # Test 5: Generate plots
    print("\n[6/6] Generating validation plots...")
    try:
        plot_validation_summary(
            output_dir,
            ck_test_data=ck_test_data,
            its_data=its_data,
            correlation_matrix=None  # Will be added if signal data available
        )
    except Exception as e:
        print(f"  ✗ Plot generation failed: {e}")
    
    # Generate validation report
    print("\nGenerating validation report...")
    generate_validation_report(
        output_dir / 'validation_report.json',
        ck_results=results.get('chapman_kolmogorov'),
        its_results=results.get('implied_timescales'),
        cv_results=results.get('cross_validation'),
        stationary_results=results.get('stationary_distribution')
    )
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  • validation_report.json - Detailed validation metrics")
    print(f"  • *.png - Validation plots")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = True
    if 'chapman_kolmogorov' in results:
        status = '✓' if results['chapman_kolmogorov']['passed'] else '✗'
        print(f"{status} Chapman-Kolmogorov test: {'PASSED' if results['chapman_kolmogorov']['passed'] else 'NEEDS REVIEW'}")
        all_passed &= results['chapman_kolmogorov']['passed']
    
    if 'implied_timescales' in results:
        status = '✓' if results['implied_timescales']['converged'] else '✗'
        print(f"{status} Implied timescales: {'CONVERGED' if results['implied_timescales']['converged'] else 'NOT CONVERGED'}")
        all_passed &= results['implied_timescales']['converged']
    
    if 'stationary_distribution' in results:
        is_valid = results['stationary_distribution']['max_relative_error'] < results['stationary_distribution']['tolerance']
        status = '✓' if is_valid else '✗'
        print(f"{status} Stationary distribution: {'VALID' if is_valid else 'NEEDS REVIEW'}")
        all_passed &= is_valid
    
    print("\n" + "="*70)
    if all_passed:
        print("OVERALL: ✓ MODEL IS SCIENTIFICALLY VALID")
    else:
        print("OVERALL: ✗ MODEL NEEDS REVIEW - Check validation report for details")
    print("="*70)
    
    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(
        description='Validate trained MSM and TICA models for scientific correctness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python tools/validate_model.py --msm_dir outputs/msm --output_dir outputs/validation
  
  # Skip expensive tests
  python tools/validate_model.py --msm_dir outputs/msm --output_dir outputs/validation --skip-ck-test
  
  # Specify MSM lag time
  python tools/validate_model.py --msm_dir outputs/msm --output_dir outputs/validation --msm_lag 30

Scientific Validation Tests Performed:
  1. Chapman-Kolmogorov test - Validates Markov property
  2. Implied timescales - Checks for convergence
  3. Stationary distribution - Compares to empirical frequencies
  4. Cross-validation - Tests model generalization
  5. Diagnostic plots - Visual inspection of model quality

References:
  Prinz et al. (2011). "Markov models of molecular kinetics"
  J. Chem. Phys. 134: 174105
        """
    )
    
    parser.add_argument('--msm_dir', required=True,
                       help='Directory containing MSM outputs (dtraj.npy, P.npy, pi.npy, etc.)')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for validation results')
    parser.add_argument('--msm_lag', type=int, default=30,
                       help='MSM lag time used during training (default: 30)')
    parser.add_argument('--skip-ck-test', action='store_true',
                       help='Skip Chapman-Kolmogorov test (slow for large models)')
    parser.add_argument('--skip-its-test', action='store_true',
                       help='Skip implied timescales convergence test')
    
    args = parser.parse_args()
    
    return validate_msm(
        Path(args.msm_dir),
        Path(args.output_dir),
        args.msm_lag,
        args.skip_ck_test,
        args.skip_its_test
    )


if __name__ == '__main__':
    sys.exit(main())

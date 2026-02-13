#!/usr/bin/env python3
"""
Comprehensive validation test runner.

Executes all validation tests and generates summary report.
"""
import sys
import subprocess
from pathlib import Path
import time


def run_test_file(test_file):
    """Run a single test file and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file.name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Determine if passed
        passed = result.returncode == 0
        
        return {
            'file': test_file.name,
            'passed': passed,
            'returncode': result.returncode,
            'elapsed': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT after {elapsed:.1f}s")
        return {
            'file': test_file.name,
            'passed': False,
            'returncode': -1,
            'elapsed': elapsed,
            'stdout': '',
            'stderr': 'Test timed out'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ ERROR: {e}")
        return {
            'file': test_file.name,
            'passed': False,
            'returncode': -1,
            'elapsed': elapsed,
            'stdout': '',
            'stderr': str(e)
        }


def main():
    """Run all validation tests."""
    print("="*70)
    print("COMPREHENSIVE VALIDATION TEST SUITE")
    print("="*70)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    
    # Find all validation test files
    tests_dir = Path(__file__).parent
    
    test_files = [
        tests_dir / 'test_dataset_validation.py',
        tests_dir / 'test_statistical_validation.py',
        tests_dir / 'test_reproducibility.py',
        tests_dir / 'test_scientific_validation.py',
    ]
    
    # Filter to existing files
    test_files = [f for f in test_files if f.exists()]
    
    if not test_files:
        print("✗ No test files found!")
        return 1
    
    print(f"\nFound {len(test_files)} test suites to run\n")
    
    # Run all tests
    results = []
    for test_file in test_files:
        result = run_test_file(test_file)
        results.append(result)
    
    # Generate summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['passed'])
    failed_tests = total_tests - passed_tests
    total_time = sum(r['elapsed'] for r in results)
    
    print(f"\nTest Files Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Total Time: {total_time:.2f}s")
    
    print("\nDetailed Results:")
    print("-" * 70)
    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"{status:8} | {result['file']:40} | {result['elapsed']:6.2f}s")
    
    # Overall status
    print("\n" + "="*70)
    if failed_tests == 0:
        print("✓ ALL VALIDATION TESTS PASSED")
        print("="*70)
        print("\nThe ML pipeline has been comprehensively validated:")
        print("  • Dataset quality verified")
        print("  • Statistical rigor confirmed")
        print("  • Reproducibility established")
        print("  • Scientific methods validated")
        print("\nThis pipeline is ready for publication-quality research.")
        return 0
    else:
        print(f"✗ {failed_tests} TEST SUITE(S) FAILED")
        print("="*70)
        print("\nFailed tests:")
        for result in results:
            if not result['passed']:
                print(f"  • {result['file']}")
                if result['stderr']:
                    print(f"    Error: {result['stderr'][:100]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

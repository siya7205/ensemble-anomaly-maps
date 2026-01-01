# Scientific Validation for ML Model Training

## Overview

This directory contains comprehensive tools for ensuring ML model training is scientifically correct and follows best practices from the computational biology literature.

## What Was Implemented

### 1. Core Validation Library (`msm/validation.py`)

**Scientific validation methods:**
- **Chapman-Kolmogorov Test** - Validates that the MSM satisfies the Markov property
- **Implied Timescales Convergence** - Checks that timescales plateau with lag time
- **VAMP-2 Cross-Validation** - K-fold CV for robust parameter selection
- **Signal Correlation Analysis** - Ensures anomaly signals are independent
- **Stationary Distribution Validation** - Compares MSM populations to empirical frequencies

**References:**
- Prinz et al. (2011). "Markov models of molecular kinetics" *J. Chem. Phys.* 134: 174105
- Wu & Noé (2020). "Variational Approach for Learning Markov Processes" *J. Nonlinear Sci.* 30: 23-66
- Trendelkamp-Schroer et al. (2015). "Estimation and uncertainty of reversible Markov models" *J. Chem. Phys.* 143: 174101

### 2. Input Data Validation (`msm/input_validation.py`)

**Pre-training quality checks:**
- Trajectory quality (clashes, unfolding, sufficient sampling)
- Feature quality (NaN, Inf, zero variance, extreme values)
- Parameter compatibility (lag times vs trajectory length, frames per cluster)

**Prevents common errors:**
- Insufficient data
- Corrupted features
- Invalid parameter combinations
- Numerical instabilities

### 3. CLI Tools

#### Model Validation Tool (`tools/validate_model.py`)

Comprehensive post-training validation:

```bash
python tools/validate_model.py \
    --msm_dir outputs/msm \
    --output_dir outputs/validation \
    --msm_lag 30
```

**Outputs:**
- `validation_report.json` - Detailed metrics
- `chapman_kolmogorov.png` - Markov property test plot
- `implied_timescales.png` - Convergence test plot
- Overall pass/fail status

#### Validated Training Pipeline (`tools/train_validated.py`)

End-to-end training with validation:

```bash
python tools/train_validated.py \
    --features data/features.npy \
    --output outputs/validated \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc \
    --lag_tica 10 --lag_msm 30 --n_clusters 30
```

**Features:**
- Pre-training input validation
- VAMP-2 model selection
- Bootstrap uncertainty quantification
- Post-training validation guidance
- Fail-safe mechanisms

### 4. Documentation

#### Comprehensive Guides
- **`SCIENTIFIC_TRAINING_GUIDE.md`** (12KB)
  - Pre-training checklist
  - Parameter selection guidelines
  - Validation requirements
  - Common pitfalls and solutions
  - Scientific references

- **`SCIENTIFIC_QUICKSTART.md`** (6.5KB)
  - Step-by-step quick start
  - Expected outputs
  - Troubleshooting common issues
  - Next steps after validation

#### Updated Main README
- Added scientifically validated training workflow
- Examples of proper usage
- Links to validation documentation

### 5. Comprehensive Testing

**Test Suite (`tests/test_scientific_validation.py`):**
- 10 unit tests covering all validation functions
- Tests for basic functionality
- Tests for edge cases (short trajectories, disconnected states)
- Reproducibility tests
- All tests passing ✅

## Scientific Correctness Guarantees

When you use these tools, your model training will:

1. ✅ **Use proper validation methods** from peer-reviewed literature
2. ✅ **Check for data quality issues** before training
3. ✅ **Validate Markov property** is satisfied
4. ✅ **Ensure convergence** of timescales
5. ✅ **Quantify uncertainty** via bootstrap
6. ✅ **Prevent overfitting** via cross-validation
7. ✅ **Be reproducible** with documented seeds and parameters

## Quick Start

### Option 1: Validated Training (Recommended)

```bash
# One command with validation
python tools/train_validated.py \
    --features data/features.npy \
    --output outputs/validated \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc
```

### Option 2: Validate Existing Model

```bash
# Validate a model you already trained
python tools/validate_model.py \
    --msm_dir outputs/msm \
    --output_dir outputs/validation
```

## Validation Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                   VALIDATED TRAINING WORKFLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Input Validation                                             │
│     ├── Trajectory quality check                                 │
│     ├── Feature quality check                                    │
│     └── Parameter compatibility check                            │
│                                                                  │
│  2. Model Training                                               │
│     ├── VAMP-2 grid search                                       │
│     ├── Optimal parameter selection                             │
│     └── Bootstrap MSM (100 iterations)                          │
│                                                                  │
│  3. Model Validation                                             │
│     ├── Chapman-Kolmogorov test                                  │
│     ├── Implied timescales convergence                          │
│     ├── Stationary distribution check                           │
│     ├── Cross-validation                                        │
│     └── Diagnostic plots                                        │
│                                                                  │
│  4. Final Report                                                 │
│     ├── Validation report (JSON)                                │
│     ├── Confidence intervals                                    │
│     ├── Pass/fail status                                        │
│     └── Recommendations                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Validation Criteria

### Input Data (Pre-Training)
- ✅ Trajectory: ≥ 1000 frames
- ✅ No NaN or Inf in features
- ✅ No zero-variance features
- ✅ Sufficient frames per cluster (≥ 50)

### Model Quality (Post-Training)
- ✅ Chapman-Kolmogorov error < 0.2
- ✅ Implied timescales converged (CV < 0.2)
- ✅ Stationary distribution error < 0.15
- ✅ Cross-validation std < mean/2

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Too few frames | Run longer simulations or reduce clusters |
| Zero variance features | Check alignment, remove constant features |
| CK test fails | Increase MSM lag time |
| Timescales not converged | Increase lag or add tICA dimensions |
| High bootstrap uncertainty | Collect more data or reduce complexity |

## Files Delivered

```
ensemble-anomaly-maps/
├── msm/
│   ├── validation.py              # Core validation algorithms (429 lines)
│   └── input_validation.py        # Input quality checks (366 lines)
├── tools/
│   ├── validate_model.py          # CLI validation tool (333 lines)
│   └── train_validated.py         # Validated training pipeline (222 lines)
├── tests/
│   └── test_scientific_validation.py  # Test suite (358 lines)
├── SCIENTIFIC_TRAINING_GUIDE.md   # Comprehensive guide (420 lines)
├── SCIENTIFIC_QUICKSTART.md       # Quick start guide (230 lines)
└── README.md                      # Updated with validation workflow
```

**Total:** ~2,400 lines of code and documentation

## Testing

Run the test suite:

```bash
python tests/test_scientific_validation.py
```

Expected output:
```
======================================================================
TESTING SCIENTIFIC VALIDATION TOOLS
======================================================================

[TEST] Chapman-Kolmogorov test - basic functionality
  ✓ Test completed

[TEST] Implied timescales convergence
  ✓ Test completed

[TEST] VAMP-2 cross-validation
  ✓ Test completed

... (7 more tests) ...

======================================================================
RESULTS: 10 passed, 0 failed
======================================================================
```

## Support

- **Documentation:** See `SCIENTIFIC_TRAINING_GUIDE.md` for comprehensive documentation
- **Quick Start:** See `SCIENTIFIC_QUICKSTART.md` for step-by-step guide
- **Issues:** Open a GitHub issue with your validation report attached
- **Scientific Questions:** Check references in the guide or documentation

## References

Key scientific papers for the implemented methods:

1. **MSM Validation:** Prinz et al. (2011) J. Chem. Phys.
2. **VAMP Score:** Wu & Noé (2020) J. Nonlinear Sci.
3. **tICA:** Pérez-Hernández et al. (2013) J. Chem. Phys.
4. **Bootstrap:** Trendelkamp-Schroer et al. (2015) J. Chem. Phys.
5. **Best Practices:** Noé et al. (2009) PNAS

## License

Same as main project.

---

**Questions?** Read `SCIENTIFIC_TRAINING_GUIDE.md` or open an issue.

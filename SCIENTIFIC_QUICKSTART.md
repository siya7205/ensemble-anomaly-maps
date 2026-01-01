# Quick Start: Scientific ML Training

This guide shows you how to train your ML model with scientific best practices in just a few commands.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements_phase1.txt
```

## Step-by-Step Training

### 1. Prepare Your Data

```bash
# Ensure you have:
# - topology.pdb (protein structure)
# - trajectory.xtc (MD simulation trajectory)
# - features.npy (extracted features) OR extract them now

# Extract features if needed:
python features/compute_md_features.py \
    --topology data/raw_trajectory/topology.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --output data/features.npy
```

### 2. Train with Validation

```bash
# Run scientifically validated training pipeline
python tools/train_validated.py \
    --features data/features.npy \
    --output outputs/validated \
    --topology data/raw_trajectory/topology.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30

# This will:
# ✓ Validate input data quality
# ✓ Check parameter compatibility
# ✓ Select optimal parameters via VAMP-2
# ✓ Compute bootstrap confidence intervals
```

**Expected output:**
```
==================================================================
SCIENTIFICALLY VALIDATED TRAINING PIPELINE
==================================================================

[STEP 1/3] Validating Input Data
------------------------------------------------------------------
Checking trajectory quality...
  ✓ Trajectory check PASSED
Checking feature quality...
  ✓ Feature check PASSED
Checking parameter compatibility...
  ✓ Parameter check PASSED

[STEP 2/3] Running Model Selection and Training
------------------------------------------------------------------
STAGE 1: VAMP-2 Model Selection
  Testing 42 parameter combinations...
  ✓ Selected: lag=10, dim=5

STAGE 2: Bootstrap MSM
  Running 100 bootstrap iterations...
  ✓ Bootstrap complete

[STEP 3/3] Post-Training Validation
------------------------------------------------------------------
✓ Training complete
```

### 3. Validate the Model

```bash
# Validate the trained model for scientific correctness
python tools/validate_model.py \
    --msm_dir outputs/validated/phase1/models/msm_bootstrap \
    --output_dir outputs/validated/validation \
    --msm_lag 30

# This performs:
# ✓ Chapman-Kolmogorov test (Markov property)
# ✓ Implied timescales convergence
# ✓ Stationary distribution validation
# ✓ Cross-validation for generalization
```

**Expected output:**
```
==================================================================
MSM VALIDATION
==================================================================

[1/6] Loading MSM outputs...
  ✓ Loaded discrete trajectory: 5000 frames
  ✓ Loaded transition matrix: (30, 30)

[2/6] Running Chapman-Kolmogorov test...
  Mean absolute error: 0.12
  ✓ PASSED

[3/6] Computing implied timescales...
  ✓ CONVERGED

[4/6] Validating stationary distribution...
  Max relative error: 0.09
  ✓ VALID

[5/6] Running VAMP-2 cross-validation...
  Mean VAMP-2 score: 2.45 ± 0.12
  ✓ COMPLETED

==================================================================
OVERALL: ✓ MODEL IS SCIENTIFICALLY VALID
==================================================================
```

### 4. Review Validation Results

```bash
# Check the validation report
cat outputs/validated/validation/validation_report.json

# View validation plots
ls outputs/validated/validation/*.png
# - chapman_kolmogorov.png
# - implied_timescales.png
```

## What If Validation Fails?

### Common Issues and Solutions

#### Issue: "Too few frames"
```bash
# Solution: Run longer simulations or combine multiple trajectories
# OR use fewer clusters:
python tools/train_validated.py --n_clusters 15 ...
```

#### Issue: "TICA lag too large"
```bash
# Solution: Reduce lag time
python tools/train_validated.py --lag_tica 5 --lag_msm 15 ...
```

#### Issue: "Chapman-Kolmogorov test failed"
```bash
# Solution: Increase MSM lag time
python tools/train_validated.py --lag_msm 50 ...
```

#### Issue: "Zero variance features"
```bash
# Solution: Check trajectory alignment and remove constant features
python -c "
import numpy as np
X = np.load('data/features.npy')
var = np.var(X, axis=0)
good_features = var > 1e-6
X_filtered = X[:, good_features]
np.save('data/features_filtered.npy', X_filtered)
"
```

## Next Steps

Once validation passes:

```bash
# 1. Build final MSM with validated parameters
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30

# 2. Compute anomaly scores
python tools/compute_all_metrics.py \
    --topology data/raw_trajectory/topology.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics

# 3. Visualize results
python viewer/app.py \
    --topology data/raw_trajectory/topology.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --hotspots outputs/metrics/hotspots_unified.json
```

## Scientific Best Practices Checklist

Before publishing or making scientific claims based on your results:

- [ ] All validation tests passed (Chapman-Kolmogorov, implied timescales, etc.)
- [ ] Bootstrap confidence intervals are narrow (< 50% relative uncertainty)
- [ ] Results are reproducible (random seeds set, configuration saved)
- [ ] Parameters are documented and justified
- [ ] Anomaly signals are independent (correlation < 0.7)
- [ ] Results validated against known biology (if available)
- [ ] Sensitivity analysis performed (test different parameters)

## Getting Help

- **Documentation:** See `SCIENTIFIC_TRAINING_GUIDE.md` for detailed best practices
- **Scientific background:** See `SCIENTIFIC_DOCUMENTATION.md` for theory
- **Troubleshooting:** Check validation report and error messages
- **Issues:** Open a GitHub issue with your validation report attached

## References

Key papers for the methods used:

1. **VAMP-2:** Wu & Noé (2020). "Variational Approach for Learning Markov Processes"
2. **MSM Validation:** Prinz et al. (2011). "Markov models of molecular kinetics"
3. **tICA:** Pérez-Hernández et al. (2013). "Identification of slow molecular order parameters"
4. **Bootstrap:** Trendelkamp-Schroer et al. (2015). "Estimation and uncertainty of reversible Markov models"

---

**Questions?** Read `SCIENTIFIC_TRAINING_GUIDE.md` for comprehensive documentation.

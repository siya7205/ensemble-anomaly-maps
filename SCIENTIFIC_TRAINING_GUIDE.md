# Scientific Training Best Practices

This document provides guidelines for ensuring your ML model training is scientifically correct and reproducible.

## Table of Contents
1. [Overview](#overview)
2. [Pre-Training Checklist](#pre-training-checklist)
3. [Model Selection](#model-selection)
4. [Training Procedure](#training-procedure)
5. [Validation Requirements](#validation-requirements)
6. [Post-Training Analysis](#post-training-analysis)
7. [Common Pitfalls](#common-pitfalls)
8. [Troubleshooting](#troubleshooting)

## Overview

Training machine learning models for molecular dynamics analysis requires careful attention to:
- **Parameter selection** - Using appropriate lag times, dimensions, and cluster numbers
- **Statistical validation** - Ensuring models are well-converged and statistically sound
- **Reproducibility** - Using random seeds and documenting all parameters
- **Scientific rigor** - Following established best practices from the literature

## Pre-Training Checklist

Before training your model, verify:

### Data Quality
- [ ] Trajectory is properly aligned (remove global translation/rotation)
- [ ] No artifacts or unphysical conformations (check for clashes, unfolding)
- [ ] Sufficient sampling (recommended: >1000 frames minimum)
- [ ] Features are properly normalized (check for NaN or Inf values)

### System Requirements
```bash
# Check trajectory quality
python -c "
import mdtraj as md
traj = md.load('trajectory.xtc', top='topology.pdb')
print(f'Frames: {traj.n_frames}')
print(f'Atoms: {traj.n_atoms}')
print(f'Time span: {traj.time[0]:.2f} - {traj.time[-1]:.2f} ps')
"

# Check for NaN in features
python -c "
import numpy as np
X = np.load('data/features.npy')
print(f'Features shape: {X.shape}')
print(f'NaN count: {np.isnan(X).sum()}')
print(f'Inf count: {np.isinf(X).sum()}')
print(f'Feature ranges: min={X.min():.2f}, max={X.max():.2f}')
"
```

### Configuration
- [ ] Set all random seeds in `configs/pipeline.yaml`
- [ ] Review parameter ranges for your system size
- [ ] Ensure output directories exist and are writable

## Model Selection

### VAMP-2 Based Selection (Recommended)

The pipeline automatically selects optimal parameters using VAMP-2 scoring:

```bash
# Run Phase 1 for automatic model selection
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1 \
    --config configs/pipeline.yaml
```

**What this does:**
1. Grid search over lag times: [5, 10, 15, 20, 30, 50] frames
2. Grid search over dimensions: [2, 3, 4, 5, 6, 8, 10]
3. Selects parameters maximizing VAMP-2 score on validation set
4. Performs bootstrap MSM for uncertainty quantification

**Scientific basis:**
- VAMP-2 score measures how well the model captures slow dynamics
- Higher scores indicate better separation of slow and fast motions
- Validation on held-out data prevents overfitting

### Manual Parameter Selection

If you need to manually select parameters:

**Lag Time Selection:**
- **Too small** (<5 frames): Captures noise, not dynamics
- **Too large** (>trajectory/10): Poor statistics
- **Rule of thumb**: Start with 10-30 frames (1-3 ns for 100 ps timesteps)

**Dimensionality Selection:**
- **Too few** (<2): Miss important motions
- **Too many** (>10): Include noise
- **Rule of thumb**: Use 3-6 dimensions for most proteins

**Number of Clusters:**
- **Too few** (<10): Oversimplify conformational space
- **Too many** (>50): Poor sampling per state
- **Rule of thumb**: Aim for 100-200 frames per cluster on average

```python
# Calculate recommended number of clusters
n_frames = 5000
target_frames_per_cluster = 150
recommended_clusters = n_frames // target_frames_per_cluster
print(f"Recommended clusters: {recommended_clusters}")
```

## Training Procedure

### Step 1: Feature Extraction

```bash
# Extract features from trajectory
python features/compute_md_features.py \
    --topology data/raw_trajectory/topology.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --output data/features.npy
```

**Scientific considerations:**
- Use backbone atoms (N, CA, C, O) for alignment
- Sin/cos encoding for dihedral angles (avoids periodicity issues)
- Include both local (dihedrals) and global (RMSD, Rg) features

### Step 2: Model Selection and Training

```bash
# Recommended: Use Phase 1 pipeline
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1

# Read selected parameters
cat outputs/phase1/reports/vamp2_best.json
```

**Key outputs:**
- `vamp2_best.json` - Selected lag and dimensionality
- `vamp2_grid.csv` - Full grid search results
- `msm_bootstrap/` - Bootstrap confidence intervals

### Step 3: Build Final MSM

```bash
# Use selected parameters
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30
```

## Validation Requirements

**CRITICAL:** Always validate your trained model before using it for analysis.

### Required Validation Tests

Run comprehensive validation:

```bash
python tools/validate_model.py \
    --msm_dir outputs/msm \
    --output_dir outputs/validation \
    --msm_lag 30
```

This performs:

#### 1. Chapman-Kolmogorov Test
**What it tests:** Markov property is satisfied  
**Passing criteria:** Predicted and estimated transition probabilities agree within 20%  
**What it means:** The MSM correctly captures the kinetics at longer timescales

#### 2. Implied Timescales Convergence
**What it tests:** Model convergence with respect to lag time  
**Passing criteria:** Timescales plateau at chosen lag  
**What it means:** The lag time is long enough for Markov property to hold

#### 3. Stationary Distribution Validation
**What it tests:** MSM populations match empirical frequencies  
**Passing criteria:** Relative error < 15% for well-sampled states  
**What it means:** The MSM correctly captures equilibrium populations

#### 4. Cross-Validation
**What it tests:** Model generalization to unseen data  
**Passing criteria:** Consistent VAMP-2 scores across folds  
**What it means:** The model doesn't overfit to training data

### Interpreting Validation Results

The validation script produces:
- `validation_report.json` - Detailed metrics
- `chapman_kolmogorov.png` - Markov property test
- `implied_timescales.png` - Convergence test

**Example of good validation:**
```json
{
  "chapman_kolmogorov": {
    "max_absolute_error": 0.12,
    "passed": true
  },
  "implied_timescales": {
    "converged": true,
    "cv_slowest": 0.08
  },
  "stationary_distribution": {
    "max_relative_error": 0.09
  }
}
```

**Red flags:**
- ✗ Chapman-Kolmogorov errors > 0.3
- ✗ Implied timescales still increasing at longest lag
- ✗ Stationary distribution errors > 0.3
- ✗ Large cross-validation standard deviation

## Post-Training Analysis

### Confidence Intervals

Bootstrap confidence intervals are computed during Phase 1:

```python
import pandas as pd
import numpy as np

# Load bootstrap results
pi_ci = pd.read_parquet('outputs/phase1/models/msm_bootstrap/pi_ci.parquet')

# Check which states have reliable populations
reliable = (pi_ci['pi_upper'] - pi_ci['pi_lower']) / pi_ci['pi_mean'] < 0.5
print(f"States with <50% relative uncertainty: {reliable.sum()}/{len(pi_ci)}")
```

**Using confidence intervals:**
- Only trust anomaly scores for states with narrow CIs
- Report uncertainty ranges when making claims
- Consider increasing sampling for high-uncertainty states

### Signal Correlation Analysis

Check that anomaly signals are independent:

```python
from msm.validation import signal_correlation_analysis

signals = {
    'rarity': rarity_scores,
    'transition_surprise': surprise_scores,
    'local_density': density_scores
}

corr = signal_correlation_analysis(signals)
print(corr)

# Correlation matrix should have off-diagonal < 0.7
max_correlation = corr.abs().values[np.triu_indices_from(corr, k=1)].max()
print(f"Max pairwise correlation: {max_correlation:.2f}")
```

## Common Pitfalls

### 1. Insufficient Sampling
**Problem:** Trajectory too short or poorly sampled  
**Symptoms:** 
- Few frames per MSM state (< 20)
- High bootstrap uncertainty
- Disconnected MSM states

**Solution:**
- Run longer simulations
- Combine multiple trajectories
- Use fewer clusters
- Use adaptive sampling to target rare states

### 2. Wrong Lag Time
**Problem:** Lag time doesn't satisfy Markov property  
**Symptoms:**
- Chapman-Kolmogorov test fails
- Implied timescales haven't plateaued
- Poor anomaly score quality

**Solution:**
- Increase lag time until implied timescales converge
- Check if you need more tICA dimensions
- Verify trajectory is properly aligned

### 3. Overfitting
**Problem:** Model memorizes training data  
**Symptoms:**
- High VAMP-2 on training, low on validation
- Large cross-validation variance
- Unrealistic anomaly patterns

**Solution:**
- Reduce tICA dimensions
- Increase regularization
- Use cross-validation for parameter selection
- Increase training data

### 4. Ignoring Uncertainty
**Problem:** Treating point estimates as ground truth  
**Symptoms:**
- Making strong claims without CIs
- Ignoring poorly-sampled states
- No sensitivity analysis

**Solution:**
- Always compute and report bootstrap CIs
- Flag high-uncertainty regions
- Test robustness to parameter choices

## Troubleshooting

### Model Training Fails

**Error: "Singular covariance matrix"**
```bash
# Check for zero-variance features
python -c "
import numpy as np
X = np.load('data/features.npy')
var = np.var(X, axis=0)
print(f'Zero variance features: {(var == 0).sum()}')
print(f'Very low variance (<1e-6): {(var < 1e-6).sum()}')
"

# Solution: Remove constant features or add regularization
```

**Error: "Not enough data for lag time"**
```bash
# Reduce lag time or stride
python tools/run_msm_tica.py --lag_tica 5 --lag_msm 10 ...
```

### Validation Fails

**Chapman-Kolmogorov test fails:**
1. Increase MSM lag time
2. Increase tICA dimensions
3. Check for artifacts in trajectory
4. Verify Markov assumption is appropriate

**Implied timescales don't converge:**
1. Test longer lag times
2. Check sampling quality
3. May need more tICA dimensions
4. Consider if system has very slow modes

**High bootstrap uncertainty:**
1. Collect more data
2. Reduce number of clusters
3. Focus on well-sampled states
4. Use regularized estimators

## References

### Scientific Validation Methods

1. **Chapman-Kolmogorov Test:**
   - Prinz et al. (2011). "Markov models of molecular kinetics: Generation and validation." *J. Chem. Phys.* 134: 174105.

2. **Implied Timescales:**
   - Swope et al. (2004). "Describing protein folding kinetics by molecular dynamics simulations." *J. Phys. Chem. B* 108: 6571-6581.

3. **VAMP Score:**
   - Wu & Noé (2020). "Variational Approach for Learning Markov Processes from Time Series Data." *J. Nonlinear Sci.* 30: 23-66.

4. **Bootstrap Methods:**
   - Trendelkamp-Schroer et al. (2015). "Estimation and uncertainty of reversible Markov models." *J. Chem. Phys.* 143: 174101.

### Best Practices

5. **MSM Construction:**
   - Noé et al. (2009). "Constructing the equilibrium ensemble of folding pathways." *PNAS* 106: 19011-19016.

6. **tICA for MD:**
   - Pérez-Hernández et al. (2013). "Identification of slow molecular order parameters for Markov model construction." *J. Chem. Phys.* 139: 015102.

7. **Model Selection:**
   - McGibbon & Pande (2015). "Variational cross-validation of slow dynamical modes in molecular kinetics." *J. Chem. Phys.* 142: 124105.

---

## Quick Reference Card

### Minimum Requirements
- **Trajectory length:** ≥1000 frames
- **Frames per state:** ≥50
- **Bootstrap samples:** ≥100
- **Cross-validation folds:** ≥5

### Parameter Rules of Thumb
- **tICA lag:** 10-50 frames (1-5 ns typical)
- **tICA dim:** 3-6 for most proteins
- **MSM lag:** 2-5× tICA lag
- **Clusters:** n_frames / 150

### Validation Thresholds
- **Chapman-Kolmogorov error:** <0.2
- **Timescale CV:** <0.2 for convergence
- **Stationary error:** <0.15
- **Signal correlation:** <0.7

### When to Seek Help
- ✗ All validation tests fail
- ✗ Results contradict known biology
- ✗ Extremely high uncertainty (>50% CI)
- ✗ Anomaly scores show no variation

**Support:** Check `SCIENTIFIC_DOCUMENTATION.md` or open an issue on GitHub.

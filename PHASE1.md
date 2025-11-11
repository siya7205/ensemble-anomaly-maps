# Phase 1: Model Selection & Bootstrap MSM

This phase implements scientifically rigorous model selection and uncertainty quantification for the MD anomaly detection pipeline.

## Overview

Phase 1 adds two critical capabilities:
1. **VAMP-2 Model Selection**: Automated selection of optimal TICA lag time and dimensionality
2. **Bootstrap MSM**: Uncertainty quantification via bootstrap confidence intervals

## Features

### 1. VAMP-2 Model Selection (`msm/select_lag_and_dim.py`)

Performs grid search over TICA parameters to maximize the VAMP-2 score:

```bash
python msm/select_lag_and_dim.py \
    --features data/features.npy \
    --output_dir reports \
    --config configs/pipeline.yaml
```

**Outputs:**
- `reports/vamp2_grid.csv` - Full grid search results
- `reports/vamp2_best.json` - Best parameters (L*, d*)

**Algorithm:**
- Grid search over lag times L ∈ {5, 10, 15, 20, 30, 50}
- Grid search over dimensions d ∈ {2, 3, 4, 5, 6, 8, 10}
- Splits data into train/validation (80/20)
- Fits VAMP on training, evaluates on validation
- VAMP-2 score = Σ(σᵢ²) where σᵢ are singular values of C₀₁

### 2. Bootstrap MSM (`msm/bootstrap_msm.py`)

Computes confidence intervals for MSM parameters:

```bash
python msm/bootstrap_msm.py \
    --features data/features.npy \
    --output_dir models/msm_bootstrap \
    --config configs/pipeline.yaml
```

**Outputs:**
- `models/msm_bootstrap/pi_ci.parquet` - Stationary distribution CIs
- `models/msm_bootstrap/P_ci.npz` - Transition matrix CIs
- `models/msm_bootstrap/mfpt_ci.parquet` - Mean first passage time CIs
- `models/msm_bootstrap/bootstrap_metadata.json` - Run metadata

**Algorithm:**
- Bootstrap resampling (B=100 iterations)
- Two methods: 'frames' (random frame resampling) or 'blocks' (block bootstrap)
- Fits complete pipeline for each bootstrap: TICA → KMeans → MSM
- Computes percentile-based CIs (default: 95%)

### 3. Unified CLI (`tools/run_phase1.py`)

Orchestrates the complete Phase 1 pipeline:

```bash
# Full pipeline
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1

# Skip VAMP-2 (use config defaults)
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1 \
    --skip-vamp2

# Only VAMP-2 selection
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1 \
    --skip-bootstrap

# Custom config
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1 \
    --config my_config.yaml
```

**Outputs:**
- All VAMP-2 and bootstrap outputs
- `outputs/phase1/run.json` - Complete run configuration for reproducibility

## Configuration

Edit `configs/pipeline.yaml` to customize:

```yaml
# Random seeds for reproducibility
seeds:
  global: 42
  kmeans: 42
  bootstrap: 123
  vamp: 456

# TICA parameters
tica:
  lag_candidates: [5, 10, 15, 20, 30, 50]
  dim_candidates: [2, 3, 4, 5, 6, 8, 10]
  default_lag: 10
  default_dim: 5

# MSM parameters
msm:
  lag: 30
  n_clusters: 30

# Bootstrap parameters
bootstrap:
  n_iterations: 100
  method: 'frames'  # or 'blocks'
  confidence_level: 0.95
```

## Reproducibility

All components use deterministic seeding:
- `seeds.global` - Master random seed
- `seeds.kmeans` - KMeans clustering
- `seeds.bootstrap` - Bootstrap resampling
- `seeds.vamp` - VAMPnet (Phase 3)

Every run saves `run.json` with complete configuration for exact reproduction.

## Testing

Run unit tests:

```bash
python tests/test_phase1.py
```

Tests cover:
- VAMP-2 score computation
- Bootstrap resampling
- MSM pipeline fitting
- Reproducibility with seeds
- Edge cases

## Performance

**VAMP-2 Selection:**
- ~30 grid points × 2-5 seconds = 1-3 minutes
- Parallelization possible (future enhancement)

**Bootstrap MSM:**
- 100 iterations × 3-5 seconds = 5-10 minutes
- Memory: ~500MB per iteration
- Can reduce to 50 iterations for faster turnaround

## Scientific Justification

### Why VAMP-2?

VAMP (Variational Approach for Markov Processes) scores quantify how well a dimensionality reduction captures slow dynamics:
- Higher VAMP-2 = better separation of slow/fast processes
- Validated approach in molecular dynamics literature
- Robust to overfitting through train/validation split

**References:**
- Wu & Noé (2020). "Variational approach for learning Markov processes from time series data"
- Nüske et al. (2014). "Variational approach to molecular kinetics"

### Why Bootstrap MSMs?

Bootstrap provides non-parametric uncertainty quantification:
- No assumptions about distribution of errors
- Captures sampling uncertainty in finite trajectories
- Critical for assessing statistical significance of findings
- Standard practice in computational biophysics

**References:**
- Efron & Tibshirani (1993). "An Introduction to the Bootstrap"
- Trendelkamp-Schroer et al. (2015). "Estimation and uncertainty of reversible Markov models"

## Integration with Existing Pipeline

Phase 1 is **fully backward compatible**:
- Existing scripts (run_msm_tica.py, etc.) continue to work
- New features are opt-in via explicit calls
- No changes to existing output formats
- Can use VAMP-2 selected parameters in existing scripts:

```python
import json
with open('reports/vamp2_best.json') as f:
    params = json.load(f)

lag_tica = params['lag']
dim_tica = params['dim']

# Use in existing pipeline...
```

## Next Steps

Phase 2 will add:
- Per-residue energetic features
- Pocket/tunnel dynamics
- Integration with anomaly scoring

Phase 3 will add:
- Soft state assignments (HMM/VAMPnet)
- Enhanced multi-signal anomaly scoring
- Windowed scoring for jitter reduction

Phase 4 will add:
- Visualization extensions
- API endpoints for new features
- Frontend overlays

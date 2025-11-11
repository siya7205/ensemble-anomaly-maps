# Phase 3: Enhanced Scoring & Soft States

This phase implements multi-signal anomaly scoring that fuses kinetic, structural, and energetic features, along with optional soft state assignments via HMM.

## Overview

Phase 3 adds two critical capabilities:
1. **Enhanced Anomaly Scoring v2**: Multi-signal fusion of kinetic, structural, and energetic features
2. **Soft State Assignments**: HMM-based probabilistic state assignments with entropy quantification

## Features

### 1. Enhanced Anomaly Scoring v2 (`scoring/anomaly_v2.py`)

Fuses multiple signals into a unified anomaly score:

**Kinetic Signals:**
- **Rarity**: `1 - π[state]` - how rare is this state?
- **Transition surprise**: `-log(P[s_t → s_{t+1}])` - how unexpected is transition?

**Structural Signals:**
- **Local density**: k-NN distance in TICA space - is this an outlier?
- **Soft entropy** (optional): State assignment ambiguity

**Energetic Signals:**
- **Energy stress**: Total or top-k unfavorable residue energies
- **Pocket volatility**: Frame-to-frame changes in pocket volume/mouth radius

```bash
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm \
    --energy data/derived/residue_energy.parquet \
    --pockets data/derived/pockets.parquet
```

**Outputs:**
- `data/derived/frame_scores_v2.csv` - Scores with components
- `reports/scoring_v2_summary.json` - Metadata

**Output Schema** (`frame_scores_v2.csv`):
```
frame: int                    - Frame index
score_raw: float             - Raw fused score [0,100]
score_windowed: float        - Smoothed score [0,100]
component_rarity: float      - Normalized rarity [0,100]
component_transition_surprise: float
component_local_density: float
component_energy_stress: float       (if energy available)
component_pocket_volatility: float   (if pockets available)
component_soft_entropy: float        (if soft states available)
```

**Algorithm:**

1. **Collect signals** from various sources
2. **Normalize** each signal to [0,1]:
   - `rank`: Rank-based scaling (robust to outliers)
   - `quantile`: Quantile-based scaling (configurable percentiles)
3. **Fuse** signals:
   - `median`: Robust to outliers (default)
   - `mean`: Arithmetic average
4. **Scale** to [0,100]
5. **Window** with moving median (reduces frame jitter)

**Formulas:**
```
rarity_t = 1 − π[s_t]
surprise_t = −log(max(P[s_t, s_{t+1}], ε))
density_t = rank_scale(kNNdist(Y_t; k))
entropy_t = −∑_i q_{t,i} log(q_{t,i})
energy_stress_t = zscore(∑_{r} energy_{t,r})
pocket_volatility_t = zscore(|volume_t − volume_{t−1}|)

score_raw_t = median(normalize([rarity_t, surprise_t, density_t, ...]))
score_windowed_t = moving_median(score_raw_t, w)
```

### 2. Soft State Assignments (`msm/soft_states.py`)

Computes probabilistic state assignments using HMM:

```bash
python tools/train_soft_states.py --dtraj outputs/msm/dtraj.npy
```

**Outputs:**
- `data/derived/soft_dtraj.npy` - Soft assignments (T × n_states)
- `data/derived/state_entropy.npy` - Per-frame entropy
- `reports/soft_states_meta.json` - Metadata

**Algorithm:**
1. Initialize Gaussian HMM with n_states from discrete trajectory
2. Fit using EM algorithm (default 100 iterations)
3. Compute posterior probabilities: `q_{t,i} = P(state=i | observations)`
4. Compute entropy: `H_t = −∑_i q_{t,i} log(q_{t,i})`

**Interpretation:**
- **Low entropy** (H ≈ 0): Deterministic state, confident assignment
- **High entropy** (H ≈ log(n_states)): Ambiguous, uncertain assignment
- High entropy frames are transitional or anomalous

## Configuration

### Scoring Parameters

Controlled via command-line arguments or config:

```bash
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm \
    --window 7 \                    # Smoothing window
    --normalize quantile \          # rank or quantile
    --fusion mean                   # median or mean
```

### Soft States Parameters

```bash
python tools/train_soft_states.py \
    --dtraj outputs/msm/dtraj.npy \
    --n_states 30 \                 # Number of states
    --n_iter 200 \                  # EM iterations
    --seed 42                       # Random seed
```

## Usage Examples

### Basic Workflow

```bash
# 1. Generate all features (if not done)
python tools/generate_energy.py --topology data/top.pdb --trajectory data/traj.xtc
python tools/generate_pockets.py --topology data/top.pdb --trajectory data/traj.xtc

# 2. Run Phase 1 for MSM
python tools/run_phase1.py --features data/features.npy --output outputs/phase1

# 3. Compute soft states (optional)
python tools/train_soft_states.py --dtraj outputs/msm/dtraj.npy

# 4. Compute enhanced anomaly scores v2
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm \
    --energy data/derived/residue_energy.parquet \
    --pockets data/derived/pockets.parquet \
    --soft_dtraj data/derived/soft_dtraj.npy \
    --state_entropy data/derived/state_entropy.npy
```

### Custom Scoring

```bash
# Use quantile normalization and mean fusion
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm \
    --normalize quantile \
    --fusion mean \
    --window 7

# Kinetic signals only (no energy/pockets)
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm
```

## Analysis Examples

### Load and Plot Scores

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load scores
df = pd.read_csv('data/derived/frame_scores_v2.csv')

# Plot raw vs windowed
plt.figure(figsize=(12, 4))
plt.plot(df['frame'], df['score_raw'], alpha=0.3, label='Raw')
plt.plot(df['frame'], df['score_windowed'], label='Windowed')
plt.xlabel('Frame')
plt.ylabel('Anomaly Score')
plt.legend()
plt.show()

# Identify top anomalies
top_anomalies = df.nlargest(10, 'score_windowed')
print("Top 10 anomalous frames:")
print(top_anomalies[['frame', 'score_windowed']])
```

### Component Analysis

```python
import pandas as pd
import seaborn as sns

df = pd.read_csv('data/derived/frame_scores_v2.csv')

# Get component columns
component_cols = [c for c in df.columns if c.startswith('component_')]

# Correlation heatmap
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(df[component_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Signal Correlation')
plt.show()

# Which signal dominates for top anomalies?
top_frames = df.nlargest(50, 'score_windowed')
print("\nMean component values for top 50 anomalies:")
print(top_frames[component_cols].mean())
```

### Entropy Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Load entropy
entropy = np.load('data/derived/state_entropy.npy')

# High entropy frames are ambiguous/transitional
threshold = np.percentile(entropy, 90)
high_entropy_frames = np.where(entropy > threshold)[0]

print(f"High entropy frames (top 10%): {len(high_entropy_frames)}")

# Plot entropy over time
plt.figure(figsize=(12, 3))
plt.plot(entropy)
plt.axhline(threshold, color='r', linestyle='--', label='90th percentile')
plt.xlabel('Frame')
plt.ylabel('State Entropy')
plt.legend()
plt.show()
```

## Integration with Existing Pipeline

Phase 3 scores integrate seamlessly:

```python
import pandas as pd
import numpy as np

# Load v1 scores (from original pipeline)
df_v1 = pd.read_csv('outputs/msm/frame_scores.csv')

# Load v2 scores (from Phase 3)
df_v2 = pd.read_csv('data/derived/frame_scores_v2.csv')

# Compare
comparison = pd.DataFrame({
    'frame': df_v1['frame'],
    'score_v1': df_v1['score'],
    'score_v2': df_v2['score_windowed']
})

# Correlation
print(f"Correlation: {comparison['score_v1'].corr(comparison['score_v2']):.3f}")

# Identify frames that changed most
comparison['delta'] = comparison['score_v2'] - comparison['score_v1']
changed_most = comparison.nlargest(10, 'delta')
print("\nFrames with largest score increase in v2:")
print(changed_most)
```

## Testing

Run unit tests:

```bash
python tests/test_phase3.py
```

Tests cover:
- Rank and quantile normalization
- Z-score computation
- Moving median smoothing
- Signal fusion (median/mean)
- State entropy computation
- Monotone behavior under perturbations
- Edge cases

## Performance

**Scoring v2:**
- Time: ~5-10 seconds for 1000 frames (all signals)
- Time: ~2-3 seconds for 1000 frames (kinetic only)
- Memory: ~50-100 MB

**Soft States:**
- Time: ~30-60 seconds for 1000 frames, 30 states, 100 EM iters
- Memory: ~20-50 MB

**Optimization:**
- Use fewer EM iterations (50-100 usually sufficient)
- Reduce number of states if needed
- Downsample trajectory with stride

## Scientific Justification

### Why Multi-Signal Fusion?

Single-signal anomaly detection can miss important events:
- **Kinetic only**: Misses energetically stressed but kinetically accessible states
- **Structural only**: Misses kinetically forbidden transitions
- **Energetic only**: Misses structurally unusual but low-energy states

Multi-signal fusion captures:
- **Comprehensive anomalies**: Events unusual in multiple dimensions
- **Robust detection**: Less sensitive to noise in any single signal
- **Interpretable**: Component scores reveal why frame is anomalous

**References:**
- Chandola et al. (2009). "Anomaly detection: A survey"
- Aggarwal (2017). "Outlier Analysis"

### Why Soft States?

Hard state assignments ignore uncertainty:
- **Transition regions**: Frames between states
- **Metastable states**: Shallow free energy minima
- **Noise**: Measurement/sampling uncertainty

Soft assignments provide:
- **Probabilistic interpretation**: Confidence in assignments
- **Transition detection**: High entropy = transitional
- **Improved MSM**: Better estimation of transition rates

**References:**
- Rabiner (1989). "A tutorial on hidden Markov models"
- Nüske et al. (2017). "Markov state models from short non-equilibrium simulations"

### Why Rank Normalization?

Rank normalization is robust:
- **Outlier-resistant**: Not affected by extreme values
- **Distribution-free**: Works with any distribution
- **Monotone**: Preserves ordering exactly
- **Standard practice**: Used in bioinformatics (GSEA, etc.)

**References:**
- Subramanian et al. (2005). "Gene set enrichment analysis"

## Comparison: v1 vs v2

| Feature | v1 (Original) | v2 (Enhanced) |
|---------|---------------|---------------|
| Signals | 3 (kinetic + density) | 6+ (kinetic + structural + energetic) |
| Normalization | Min-max | Rank/quantile (robust) |
| Fusion | Median | Median/mean (configurable) |
| Smoothing | None | Moving median window |
| Soft states | No | Yes (optional) |
| Energy | No | Yes (if available) |
| Pockets | No | Yes (if available) |
| Reproducibility | Partial | Full (seeded, logged) |

## Limitations & Future Work

### Current Limitations
1. **HMM assumptions**: Gaussian emissions, Markovian
2. **Signal independence**: Assumes signals are independent
3. **Fixed weights**: All signals weighted equally
4. **Linear fusion**: Median/mean only

### Future Enhancements
1. **VAMPNet**: Deep learning soft states
2. **Weighted fusion**: Learn signal weights
3. **Nonlinear fusion**: Neural network combination
4. **Online scoring**: Streaming anomaly detection

## Next Steps

Phase 4 will add visualization extensions:
- Flask API endpoints for energy/pockets/entropy
- Frontend overlays (energy halos, pocket meshes)
- Timeline tracks for multi-signal scores

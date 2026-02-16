# ML Pipeline Outputs: Quick Reference for Thesis

**Purpose**: Concise summary of ML pipeline output contracts for thesis systems/architecture chapter.

**Full Specification**: See [OUTPUT_INTERFACE_SPECIFICATION.md](OUTPUT_INTERFACE_SPECIFICATION.md) for complete details.

---

## 1. Output Artifacts

### Primary Visualization Outputs

| File | Format | Granularity | Purpose |
|------|--------|-------------|---------|
| `hotspots_unified.json` | JSON | Per-residue | **Primary interface**: All metric channels in one file |
| `frame_scores_dynamic.csv` | CSV | Per-frame | Temporal timeseries with component breakdown |
| `residue_scores_dynamic.json` | JSON | Per-residue | Legacy: dynamic anomaly only |
| `residue_scores_rmsf.json` | JSON | Per-residue | Legacy: RMSF/stability only |
| `residue_scores_tica_importance.json` | JSON | Per-residue | Legacy: tICA importance only |

**Key Point**: New consumers should use `hotspots_unified.json` for all per-residue metrics.

---

## 2. Signal Semantics

### Three Independent Metric Channels

#### Dynamic Anomaly (Kinetic + Structural)
- **Physical Meaning**: Involvement in rare/unexpected conformational dynamics
- **Components**: State rarity + Transition surprise + Local density (fused)
- **Normalization**: [0, 1], default percentile-based
- **Interpretation**: 
  - `[0.0-0.3]` = Normal dynamics
  - `[0.7-1.0]` = High anomaly (rare/unexpected)

#### RMSF (Structural Flexibility)
- **Physical Meaning**: Root Mean Square Fluctuation (positional variability)
- **Units**: Angstroms (raw), normalized to [0, 1] in output
- **Interpretation**:
  - `[0.0-0.3]` = Rigid (core, secondary structure)
  - `[0.7-1.0]` = Flexible (loops, termini)
- **Relation**: RMSF ≈ B-factor × sqrt(3/(8π²))

#### tICA Importance (Slow-Mode Contribution)
- **Physical Meaning**: Contribution to slow collective motions
- **Calculation**: L2 norm of loadings across top 5 slowest components
- **Interpretation**:
  - `[0.0-0.3]` = Passive/follower residues
  - `[0.7-1.0]` = Drivers of slow modes (hinges, allosteric nodes)
- **Complementarity**: Can have low RMSF (rigid) but high importance (critical hinge)

### Normalization Methods

| Method | Formula | Pros | Cons | Use Case |
|--------|---------|------|------|----------|
| **Percentile** (default) | Clip to [p_low, p_high] → [0,1] | Robust to outliers | Loses extreme values | Standard visualization |
| **Rank** | argsort(argsort(x)) / (n-1) | Maximally robust | Loses magnitude | Highly skewed distributions |
| **Z-score** | sigmoid((x - μ) / σ) | Interpretable | Assumes Gaussian | Normally distributed scores |

**Default**: Percentile with [5th, 95th] percentile clipping

---

## 3. Alignment Guarantees

### Frame Indexing
- **Contract**: Output frame index `i` → input trajectory frame `i` (0-based, after alignment)
- **No gaps**: All input frames processed (no dropping/skipping)
- **1:1 correspondence**: `len(frame_scores_dynamic.csv)` == `n_trajectory_frames`

### Residue Indexing
- **Contract**: Residue IDs match topology file numbering (0-based in output)
- **PDB mapping**: JSON key `"0"` → PDB residue 1 (if PDB uses 1-based)
- **Chain info lost**: Multi-chain proteins have sequential IDs across chains
- **Sparse**: Not all residues present (only those analyzed)

### Temporal Resolution
- **Frame rate**: Preserved 1:1 (no resampling)
- **Smoothing**: Median filter (window=5 frames) on fused scores only
- **Physical timescale**: Depends on trajectory timestep (typically 2-10 ps/frame)

---

## 4. Intended Consumer

### Visualization-Agnostic Design

**Pipeline Provides**:
- ✅ Normalized scores in [0, 1]
- ✅ Separate channels for different properties
- ✅ Metadata (normalization method, percentile ranges)

**Pipeline Does NOT Provide**:
- ❌ Color maps (RGB values)
- ❌ Visualization thresholds
- ❌ 3D rendering parameters
- ❌ UI interactions

**Consumer Responsibility**:
- Map scores → colors (e.g., blue-white-red gradient)
- Choose visualization modality (surface, cartoon, spheres)
- Implement interactive controls (thresholds, animation)

### Human Interpretability

**Design Philosophy**: Scores interpretable by domain scientists without ML expertise

**Features**:
1. **Normalized scales**: All scores in [0, 1]
2. **Physical grounding**: Each metric tied to concrete property
3. **Component breakdown**: Individual signals in CSV for debugging
4. **Metadata**: Normalization method documented in JSON

**Example Interpretation**:
```
Residue 42 scores:
- dynamic_anomaly: 0.92 → Involved in rare dynamics
- rmsf: 0.15          → Rigid (not flexible)
- tica_importance: 0.87 → Critical for slow modes

→ Likely a HINGE RESIDUE: small but essential motion
```

---

## 5. Known Limitations

### Methodological

| Limitation | Impact | Detection | Mitigation |
|------------|--------|-----------|------------|
| **Markovian assumption** | Inaccurate kinetic signals if memory effects present | Chapman-Kolmogorov test | Increase MSM lag time; use longer trajectories |
| **Insufficient sampling** | High variance, unreliable statistics | Bootstrap CI width > 0.2 | Use ≥10K frames; multiple independent runs |
| **Feature choice** | Misses side-chain, solvent effects | N/A | Add Phase 2 energy features; validate experimentally |

### Normalization

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Percentile clipping** | Extreme outliers underweighted | Use rank normalization; adjust percentile range |
| **Global normalization** | Scores trajectory-dependent | Interpret as relative rankings within trajectory |

### Temporal

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **MSM lag constraint** | Misses fast processes (< 60-300 ps) | Adjust `--lag_msm`; validate implied timescales |
| **Smoothing artifacts** | Brief spikes (<5 frames) suppressed | Inspect raw scores; reduce `--window` |

### Interpretation

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **No absolute significance** | Cannot determine biological importance | Validate against experimental data; compare to known sites |
| **Correlation ≠ causation** | High score ≠ functional importance | Check trajectory quality; use domain knowledge |

### Computational

| Aspect | Limit | Behavior |
|--------|-------|----------|
| **Scalability** | Tested to 500K frames | Auto-optimization for >100K frames (k-NN subsampling) |
| **Memory** | ~8 GB for 100K frames | Scales linearly with n_frames × n_features |
| **Determinism** | Guaranteed with fixed seeds | Stochastic steps (k-means, bootstrap) use `random_state=42` |

---

## 6. Quick Reference Examples

### Example 1: Load Unified Scores
```python
import json

with open('outputs/metrics/hotspots_unified.json') as f:
    data = json.load(f)

# Extract specific metric
dynamic_scores = data['per_residue']['dynamic_anomaly']
rmsf_scores = data['per_residue']['rmsf']

# Check metadata
n_frames = data['meta']['n_frames']
normalization = data['meta']['normalization']  # e.g., "percentile"
```

### Example 2: Frame Timeseries
```python
import pandas as pd

df = pd.read_csv('outputs/metrics/frame_scores_dynamic.csv')

# Plot fused score over time
plt.plot(df['frame'], df['score'])

# Check individual components
plt.plot(df['frame'], df['component_rarity'], label='Rarity')
plt.plot(df['frame'], df['component_transition_surprise'], label='Surprise')
```

### Example 3: Identify Hinge Residues
```python
# Hinges: rigid (low RMSF) but important (high tICA, high anomaly)
hinges = []
for res_id in dynamic_scores.keys():
    if (dynamic_scores[res_id] > 0.7 and 
        rmsf_scores[res_id] < 0.3 and
        tica_importance[res_id] > 0.7):
        hinges.append(int(res_id))

print(f"Potential hinge residues: {hinges}")
```

---

## Validation Checklist

Before consuming outputs, verify:

- [ ] `meta.n_frames` matches expected trajectory length
- [ ] Frame indices are contiguous: 0, 1, 2, ..., n-1
- [ ] All scores in [0.0, 1.0] range
- [ ] No NaN/Inf values
- [ ] Residue IDs match topology (accounting for 0-based indexing)
- [ ] Normalization method is appropriate (`meta.normalization`)

---

## File Locations

```
outputs/metrics/
├── hotspots_unified.json           ← PRIMARY: Use this
├── frame_scores_dynamic.csv        ← Timeseries
├── residue_scores_dynamic.json     (legacy)
├── residue_scores_rmsf.json        (legacy)
└── residue_scores_tica_importance.json (legacy)
```

---

## Pipeline Command Reference

### Generate All Metrics
```bash
python tools/compute_all_metrics.py \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --normalization percentile \
    --low-percentile 0.05 \
    --high-percentile 0.95
```

### Key Parameters
- `--normalization {percentile,rank,zscore}`: Normalization method
- `--low-percentile FLOAT`: Lower clip (default: 0.05)
- `--high-percentile FLOAT`: Upper clip (default: 0.95)
- `--per-frame-norm`: Per-frame instead of global normalization
- `--window INT`: Smoothing window size (default: 5)
- `--fusion {median,mean}`: Signal fusion method (default: median)
- `--robust`: Enable conservative parameters for challenging trajectories

---

## Key Takeaways for Thesis

1. **Output Format**: JSON + CSV, visualization-ready, no post-processing required
2. **Three Independent Channels**: Dynamic anomaly, RMSF, tICA importance (don't mix them)
3. **Alignment**: 1:1 frame correspondence, 0-based residue indexing
4. **Normalization**: [0, 1] scale, percentile-based by default, trajectory-relative
5. **Limitations**: Relative scores (not absolute), requires sufficient sampling (≥10K frames), assumes Markovian dynamics

**Primary Output**: `hotspots_unified.json` contains all per-residue metrics in standardized format.

**For Timeseries**: `frame_scores_dynamic.csv` provides per-frame scores with component breakdown.

**Visualization-Agnostic**: Outputs are semantic scores, not rendering instructions. Consumers map scores → visual encoding (colors, sizes, etc.).

---

**Full Documentation**: [OUTPUT_INTERFACE_SPECIFICATION.md](OUTPUT_INTERFACE_SPECIFICATION.md)

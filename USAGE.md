# Usage Guide: Dynamic Hotspot Detection Pipeline

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements_phase1.txt
pip install -r requirements_phase2.txt  
pip install -r requirements_phase3.txt

# Requires: numpy, scipy, pandas, scikit-learn, deeptime, mdtraj, hmmlearn
```

### Basic Pipeline (Raw MD → Scores)

```bash
# 1. Prepare your data
# You need:
#   - topology.pdb (protein structure)
#   - trajectory.xtc (MD trajectory)

# 2. Generate features from trajectory
# (This step extracts backbone angles, distances, etc.)
# Note: Specific implementation depends on your feature extraction script

# 3. Run Phase 1: tICA + MSM
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1 \
    --config configs/pipeline.yaml

# This creates:
#   outputs/phase1/reports/vamp2_best.json  # Optimal tICA parameters
#   outputs/phase1/models/msm_bootstrap/    # Bootstrap MSMs

# 4. Build MSM with optimal parameters
python tools/run_msm_tica.py \
    data/features.npy \
    outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30

# This creates:
#   outputs/msm/dtraj.npy         # Discrete state trajectory
#   outputs/msm/tica_coords.npy   # tICA-projected coordinates
#   outputs/msm/P.npy             # Transition matrix
#   outputs/msm/pi.npy            # Stationary distribution

# 5. Compute all metrics (NEW!)
python tools/compute_all_metrics.py \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --normalization percentile \
    --low-percentile 0.05 \
    --high-percentile 0.95

# This creates:
#   outputs/metrics/hotspots_unified.json           # All metrics in one file
#   outputs/metrics/residue_scores_dynamic.json     # Dynamic anomaly only
#   outputs/metrics/residue_scores_rmsf.json        # RMSF/stability only
#   outputs/metrics/residue_scores_tica_importance.json  # tICA importance only
#   outputs/metrics/frame_scores_dynamic.csv        # Per-frame scores
#   outputs/metrics/hotspots_residue.json           # Legacy format
```

---

## Detailed Usage

### compute_all_metrics.py - Unified Metrics Computation

This new tool computes and exports three separate metric channels:

#### Required Arguments:
- `--topology`: Path to topology file (PDB format)
- `--trajectory`: Path to trajectory file (XTC, DCD, etc.)
- `--msm_dir`: Directory containing MSM outputs (dtraj.npy, P.npy, etc.)

#### Optional Arguments:

**Output:**
- `--output_dir`: Where to save metric files (default: `outputs/metrics`)
- `--vamp2_best`: Path to VAMP-2 best parameters JSON (optional)

**Normalization:**
- `--normalization {rank,percentile,zscore}`: Normalization method (default: `percentile`)
- `--low-percentile FLOAT`: Lower percentile for clipping (default: 0.05)
- `--high-percentile FLOAT`: Upper percentile for clipping (default: 0.95)
- `--per-frame-norm`: Use per-frame normalization instead of global

**Processing:**
- `--lag_msm INT`: MSM lag time in frames (default: 30)
- `--k_neighbors INT`: Number of neighbors for density estimation (default: 20)
- `--window INT`: Window size for temporal smoothing (default: 5)
- `--fusion {median,mean}`: Signal fusion method (default: median)

#### Examples:

**Basic usage:**
```bash
python tools/compute_all_metrics.py \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics
```

**With percentile-based normalization (recommended for avoiding color compression):**
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

**Per-frame normalization (highlights frame-specific hotspots):**
```bash
python tools/compute_all_metrics.py \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --per-frame-norm
```

**Custom processing parameters:**
```bash
python tools/compute_all_metrics.py \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --lag_msm 50 \
    --k_neighbors 30 \
    --window 7 \
    --fusion mean
```

---

### Understanding Normalization Options

#### Why Normalization Matters

Raw anomaly scores can be heavily skewed, with most frames/residues having very low scores and only a few extreme outliers. This leads to poor visualization where everything looks blue except a few red tips.

Proper normalization spreads the scores across the full [0, 1] range, making the dynamic range visible.

#### Normalization Methods

**1. Rank Normalization** (`--normalization rank`)
```
score_norm = rank(score) / (n-1)
```
- **Pros**: Robust to outliers, distribution-free, preserves exact ordering
- **Cons**: Uniform output distribution (may not match physical intuition)
- **Use when**: You want equal representation across the full score range

**2. Percentile Normalization** (`--normalization percentile`) **[RECOMMENDED]**
```
score_norm = clip((score - p5) / (p95 - p5), 0, 1)
```
where p5 and p95 are 5th and 95th percentiles
- **Pros**: Focuses on bulk of distribution, clips extreme outliers naturally
- **Cons**: Loses information about extreme outliers
- **Use when**: You want to see gradations in the main distribution (most cases)
- **Parameters**: `--low-percentile 0.05 --high-percentile 0.95`

**3. Z-score Normalization** (`--normalization zscore`)
```
score_norm = sigmoid(z-score)
```
- **Pros**: Natural for Gaussian-distributed scores
- **Cons**: Sensitive to outliers, assumes Gaussian distribution
- **Use when**: Your scores are approximately Gaussian

#### Global vs. Per-Frame Normalization

**Global Normalization** (default):
- Compares all residues/frames across the entire trajectory
- **Interpretation**: "Which residues are most important overall?"
- **Visualization**: Constitutive hotspots stand out
- **Use for**: Finding residues that are consistently important

**Per-Frame Normalization** (`--per-frame-norm`):
- Normalizes within each frame independently
- **Interpretation**: "Which residues are most important in this specific conformation?"
- **Visualization**: Highlights change over time, different hotspots per frame
- **Use for**: Time-resolved analysis, finding frame-specific dynamics

---

## Output Files

### hotspots_unified.json

**Format:**
```json
{
  "meta": {
    "n_frames": 200,
    "n_residues": 150,
    "metrics": ["dynamic_anomaly", "rmsf", "tica_importance"],
    "normalization": "percentile",
    "percentile_range": [0.05, 0.95],
    "description": {
      "dynamic_anomaly": "Involvement in rare/unexpected dynamics",
      "rmsf": "Root Mean Square Fluctuation - flexibility metric",
      "tica_importance": "Contribution to slow collective motions"
    }
  },
  "per_residue": {
    "dynamic_anomaly": {"0": 0.45, "1": 0.23, ...},
    "rmsf": {"0": 0.12, "1": 0.67, ...},
    "tica_importance": {"0": 0.89, "1": 0.15, ...}
  }
}
```

**Usage in viewer:**
- Load this file to get all three metric channels
- Switch between channels in UI to see different aspects
- Residue IDs are strings (JSON requirement)

### residue_scores_dynamic.json

Per-residue dynamic anomaly scores only.
```json
{
  "0": 0.45,
  "1": 0.23,
  ...
}
```

### residue_scores_rmsf.json

Per-residue RMSF/stability scores.
```json
{
  "0": 0.12,
  "1": 0.67,
  ...
}
```

**Interpretation:**
- **High RMSF (>0.7)**: Flexible/floppy regions (loops, termini)
- **Medium RMSF (0.3-0.7)**: Moderately flexible
- **Low RMSF (<0.3)**: Rigid/stable regions (core, helices)

### residue_scores_tica_importance.json

Per-residue importance for slow collective motions.
```json
{
  "0": 0.89,
  "1": 0.15,
  ...
}
```

**Interpretation:**
- **High importance (>0.7)**: Drives slow motions (hinges, allosteric nodes)
- **Medium importance (0.3-0.7)**: Moderately involved
- **Low importance (<0.3)**: Passive or fast-motion only

### frame_scores_dynamic.csv

Per-frame dynamic anomaly scores with component breakdown.

```csv
frame,score,component_rarity,component_transition_surprise,component_local_density
0,0.23,0.15,0.31,0.22
1,0.45,0.52,0.41,0.43
...
```

**Columns:**
- `frame`: Frame index
- `score`: Final fused and smoothed anomaly score
- `component_*`: Individual signal components (before fusion)

**Usage:**
- Plot score vs. frame to see anomaly timeseries
- Identify which frames are most anomalous
- Decompose score to understand why a frame is anomalous

### hotspots_residue.json (legacy format)

Backward-compatible format for existing viewers.
```json
{
  "scores": [
    {"label": "Res 0", "score": 0.45},
    {"label": "Res 1", "score": 0.23},
    ...
  ]
}
```

---

## Interpreting Combined Metrics

### Score Combination Matrix

| Dynamic Anomaly | RMSF | tICA Importance | Interpretation |
|----------------|------|-----------------|----------------|
| High | High | High | **Prime hotspot**: Flexible, drives slow modes, involved in rare events |
| High | High | Low | **Flexible anomaly**: Floppy region with unusual local dynamics |
| High | Low | High | **Rigid hotspot**: Stable region critical for slow modes and anomalies |
| High | Low | Low | **Local rearrangement**: Unusual motion but not functionally central |
| Low | High | High | **Constitutive driver**: Flexible region that drives slow modes predictably |
| Low | High | Low | **Surface loop**: Flexible but not functionally important |
| Low | Low | High | **Stable driver**: Rigid region that enables slow modes (e.g., hinge) |
| Low | Low | Low | **Core residue**: Stable, buried, not dynamically active |

### Example Biological Interpretations

**Kinase ATP-binding site:**
- Dynamic anomaly: High (opens/closes during catalysis)
- RMSF: Medium-High (activation loop is flexible)
- tICA importance: High (controls catalytic state)

**Allosteric hinge:**
- Dynamic anomaly: High (rare rotations enable communication)
- RMSF: Low (small but critical rotations)
- tICA importance: High (drives domain movements)

**Surface loop (non-functional):**
- Dynamic anomaly: Low (predictable fluctuations)
- RMSF: High (very flexible)
- tICA importance: Low (fast local motions)

**Hydrophobic core:**
- Dynamic anomaly: Low (stable)
- RMSF: Low (rigid)
- tICA importance: Low (passive support)

---

## Validation Workflow

### 1. Visual Inspection

Load outputs in the molecular viewer:
```bash
# In the asvs viewer repo
python app.py \
    --topology path/to/topology.pdb \
    --trajectory path/to/trajectory.xtc \
    --hotspots path/to/hotspots_unified.json
```

**What to check:**
- Do hotspots localize to known functional regions?
- Are scores distributed across the full color range (not all blue/red)?
- Do different metrics highlight different regions as expected?

### 2. Compare to Known Sites

```python
# Example validation script
import json

# Load your hotspots
with open('outputs/metrics/hotspots_unified.json') as f:
    hotspots = json.load(f)

# Define known functional sites (from literature/UniProt)
active_site = [45, 46, 47, 100, 101]  # Example residue IDs
allosteric_site = [20, 21, 22, 75, 76]

# Check enrichment
dynamic_scores = hotspots['per_residue']['dynamic_anomaly']
active_scores = [float(dynamic_scores.get(str(i), 0)) for i in active_site]
allosteric_scores = [float(dynamic_scores.get(str(i), 0)) for i in allosteric_site]

print(f"Active site mean: {np.mean(active_scores):.3f}")
print(f"Allosteric site mean: {np.mean(allosteric_scores):.3f}")
print(f"Overall mean: {np.mean(list(dynamic_scores.values())):.3f}")
```

### 3. Sequence Conservation Analysis

```bash
# Generate MSA using Clustal Omega or MUSCLE
clustalo -i sequences.fasta -o alignment.aln

# Map conservation to residues
# Compare high-conservation residues to high-scoring hotspots
```

**Expected**:
- Functional hotspots should be enriched in conserved residues
- But not all hotspots need to be conserved (organism-specific regulation)

### 4. Experimental Validation

If available, compare to:
- **NMR/HDX-MS**: Compare RMSF scores to experimental dynamics
- **Crystallographic B-factors**: Should correlate with RMSF
- **Mutagenesis data**: High-scoring hotspots should be sensitive to mutation
- **Known binding sites**: Should have high dynamic anomaly during binding

---

## Troubleshooting

### Issue: All scores are blue with a few red tips

**Cause**: Poor normalization strategy or extreme outliers compressing the range.

**Solution**:
```bash
# Use percentile normalization with appropriate range
python tools/compute_all_metrics.py \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc \
    --msm_dir outputs/msm \
    --normalization percentile \
    --low-percentile 0.10 \
    --high-percentile 0.90  # More aggressive clipping
```

### Issue: RMSF computation fails

**Cause**: Missing MDTraj or incompatible trajectory format.

**Solution**:
```bash
# Install MDTraj
pip install mdtraj

# Convert trajectory to compatible format if needed
# (XTC and DCD are well-supported)
```

### Issue: Different metrics give contradictory results

**Cause**: This is expected! Metrics capture different aspects.

**Interpretation**:
- High dynamic anomaly + low RMSF = rigid hotspot (e.g., allosteric hinge)
- Low dynamic anomaly + high RMSF = flexible but predictable (e.g., surface loop)
- Check the score combination matrix above for interpretation

### Issue: Scores change drastically with normalization method

**Cause**: Different normalization methods emphasize different aspects.

**Solution**: Try multiple methods and pick the one that best matches your goals:
- **Rank**: Best for uniform distribution visualization
- **Percentile**: Best for focusing on bulk distribution (recommended)
- **Zscore**: Best if your scores are Gaussian

### Issue: Pipeline fails for short trajectories

**Cause**: Not enough frames for reliable tICA/MSM estimation.

**Solution**: Use robust mode:
```bash
python tools/compute_all_metrics.py \
    --topology data/topology.pdb \
    --trajectory data/trajectory.xtc \
    --msm_dir outputs/msm \
    --robust
```

### Issue: NaN or Inf in outputs

**Cause**: Numerical instability from ill-conditioned matrices or log(0).

**Solution**:
- Use `--robust` mode which adds numerical safeguards
- Check for constant features or disconnected MSM states
- Try reducing number of clusters or increasing lag time

### Issue: Empty or degenerate clusters

**Cause**: Too many clusters for the amount of data.

**Solution**:
```bash
# Reduce number of clusters when building MSM
python tools/run_msm_tica.py data/features.npy outputs/msm \
    --n_clusters 15  # Reduced from default 30
```

---

## Robust Mode

The `--robust` flag enables conservative settings for challenging trajectories:

### When to Use Robust Mode

- Trajectory has fewer than 500 frames
- Multiple short trajectories instead of one long trajectory  
- High noise or artifacts in trajectory
- MSM estimation fails with default parameters
- Anomaly scores have no contrast (all similar values)

### What Robust Mode Does

1. **Reduces k_neighbors** to 10 (from default 20)
   - Better for shorter trajectories where k-NN can fail

2. **Reduces lag_msm** to 20 (from default 30)
   - Shorter lag time works better with limited data

3. **Increases smoothing window** to 7 (from default 5)
   - More aggressive smoothing reduces noise

4. **Uses percentile normalization** with [0.10, 0.90]
   - Clips more extreme values for better visualization

### Example

```bash
python tools/compute_all_metrics.py \
    --topology data/topology.pdb \
    --trajectory data/short_trajectory.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --robust
```

Output:
```
[ROBUST MODE ENABLED]
  Using conservative parameters for challenging trajectories:
  - k_neighbors reduced to 10
  - lag_msm reduced to 20
  - window size increased to 7
  - normalization: percentile [0.1, 0.9]
```

---

## Limitations and Edge Cases

### Minimum Requirements

| Parameter | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Trajectory length | 100 frames | 1000+ frames | Fewer frames → poor MSM statistics |
| Frames per state | 10 | 100-200 | Below this, state populations unreliable |
| tICA lag | 5 frames | 10-50 frames | Must be < trajectory length / 10 |
| MSM lag | 10 frames | 20-50 frames | Must satisfy Markov property |

### Known Limitations

1. **Very short trajectories (< 100 frames)**
   - MSM statistics become unreliable
   - Consider using density-based scoring only
   - Use `--robust` mode

2. **Disconnected MSM states**
   - Some states may never transition to others
   - Pipeline handles this gracefully with pseudo-counts
   - Check for warnings about disconnected states

3. **Highly heterogeneous trajectories**
   - Multiple distinct conformational basins
   - May need trajectory-specific normalization
   - Consider per-frame normalization (`--per-frame-norm`)

4. **Trajectory artifacts**
   - Unphysical conformations will appear as strong anomalies
   - Inspect top anomalous frames visually
   - Filter trajectory before analysis if needed

### Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Trajectory too short" | < 2 frames | Check trajectory loading |
| "No atoms found with selection" | Bad atom selection | Check topology file |
| "Singular covariance matrix" | Zero variance features | Remove frozen atoms |
| "MSM has 0 states" | All frames in one cluster | Reduce n_clusters |

---

## Advanced Usage

### Custom Signal Weights

Currently, all signals (rarity, transition surprise, density) are weighted equally in fusion. To implement custom weights:

```python
# In your custom script
from scoring.signals import compute_dynamic_anomaly_scores

signals = compute_dynamic_anomaly_scores(msm, dtraj, tica_coords)

# Custom weighting
weighted_score = (
    0.5 * signals['rarity'] +
    0.3 * signals['transition_surprise'] +
    0.2 * signals['local_density']
)
```

### Time-Windowed Analysis

To analyze dynamics in specific trajectory segments:

```python
# Analyze first half vs. second half
n_frames = len(dtraj)
half = n_frames // 2

# First half
signals_first = compute_dynamic_anomaly_scores(
    msm, dtraj[:half], tica_coords[:half]
)

# Second half
signals_second = compute_dynamic_anomaly_scores(
    msm, dtraj[half:], tica_coords[half:]
)

# Compare
diff = signals_second['rarity'] - signals_first['rarity']
# Positive diff = more rare in second half
```

### Integration with Energy/Pocket Features

If you've computed Phase 2 features (energy, pockets):

```bash
# Use enhanced scoring with all signals
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm \
    --energy data/derived/residue_energy.parquet \
    --pockets data/derived/pockets.parquet \
    --output_scores outputs/frame_scores_v2.csv
```

---

## See Also

- **PIPELINE_SUMMARY_FOR_BIOCHEMISTS.md**: Detailed scientific explanation
- **SCIENTIFIC_DOCUMENTATION.md**: Original comprehensive documentation
- **PHASE1.md**, **PHASE2.md**, **PHASE3.md**: Phase-specific documentation
- **README.md**: Repository overview

---

## Citation

If you use this pipeline in your research, please cite:

[Add relevant citations for tICA, MSM, deeptime, etc.]

---

## Support

For questions or issues:
1. Check this USAGE.md and other documentation
2. Review existing issues on GitHub
3. Open a new issue with:
   - Command you ran
   - Error message
   - Output of `python --version` and `pip list`

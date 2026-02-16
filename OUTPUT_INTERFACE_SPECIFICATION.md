# ML Pipeline Output Interface Specification
## For External Visualization System Integration

**Document Purpose**: This specification describes the data contracts, formats, and semantics of outputs produced by the ensemble-anomaly-maps ML pipeline for consumption by external visualization systems. Written for inclusion in a systems/architecture chapter of a CS+ML capstone thesis.

**Last Updated**: 2026-02-16  
**Pipeline Version**: Phase 3 (Multi-Signal Fusion)

---

## Table of Contents

1. [Output Artifacts](#1-output-artifacts)
2. [Signal Semantics](#2-signal-semantics)
3. [Alignment Guarantees](#3-alignment-guarantees)
4. [Intended Consumer](#4-intended-consumer)
5. [Known Limitations](#5-known-limitations)
6. [Version Compatibility](#6-version-compatibility)

---

## 1. Output Artifacts

### 1.1 Primary Output Files

The pipeline exports the following files for downstream visualization consumption:

#### **hotspots_unified.json** (Primary Interface)
**Format**: JSON  
**Granularity**: Per-residue aggregated scores  
**Purpose**: Unified interface for all metric channels

**Schema**:
```json
{
  "meta": {
    "n_frames": int,
    "n_residues": int,
    "metrics": ["dynamic_anomaly", "rmsf", "tica_importance"],
    "normalization": "percentile|rank|zscore",
    "percentile_range": [float, float],
    "description": {
      "dynamic_anomaly": "...",
      "rmsf": "...",
      "tica_importance": "..."
    }
  },
  "per_residue": {
    "dynamic_anomaly": {"0": 0.75, "1": 0.23, ...},
    "rmsf": {"0": 0.62, "1": 0.89, ...},
    "tica_importance": {"0": 0.41, "1": 0.15, ...}
  }
}
```

**Data Contract**:
- All scores normalized to `[0.0, 1.0]`
- Residue IDs are **string keys** (not integers)
- Missing residues have no entry (sparse representation)
- Metrics are **independent channels** (not mixed)
- Normalization method documented in `meta.normalization`

**Consumption Pattern**:
```python
# Load and extract specific metric
with open('hotspots_unified.json') as f:
    data = json.load(f)
    
# Get dynamic anomaly scores for visualization
scores = data['per_residue']['dynamic_anomaly']
residues = [int(k) for k in scores.keys()]  # Convert to int
values = [scores[str(r)] for r in residues]

# Apply custom color mapping: [0,1] -> RGB
```

---

#### **frame_scores_dynamic.csv** (Temporal Series)
**Format**: CSV  
**Granularity**: Per-frame timeseries  
**Purpose**: Frame-level anomaly scores with component breakdown

**Schema**:
```csv
frame,score,component_rarity,component_transition_surprise,component_local_density
0,0.234,0.12,0.45,0.89
1,0.567,0.23,0.34,0.76
...
```

**Data Contract**:
- `frame`: Integer index, 0-based, **contiguous** (no gaps)
- `score`: Fused anomaly score in `[0.0, 1.0]`
- `component_*`: Individual signal contributions in `[0.0, 1.0]`
- Row count **exactly matches** trajectory frame count
- Component columns present only if signal was computed

**Alignment**:
- Frame index `i` corresponds to trajectory frame `i` (after any preprocessing)
- No temporal resampling or interpolation applied
- 1:1 correspondence with input trajectory frames

**Consumption Pattern**:
```python
import pandas as pd

df = pd.read_csv('frame_scores_dynamic.csv')

# Temporal animation: map frame -> color
frame_colors = {
    row['frame']: score_to_rgb(row['score'])
    for _, row in df.iterrows()
}

# Plot timeseries
plt.plot(df['frame'], df['score'])
```

---

#### **Legacy Formats** (Backward Compatibility)

##### **residue_scores_dynamic.json**
**Format**: JSON (simple dict)  
**Schema**: `{"0": 0.75, "1": 0.23, ...}`  
**Purpose**: Single-channel dynamic anomaly scores

##### **residue_scores_rmsf.json**
**Format**: JSON (simple dict)  
**Schema**: `{"0": 0.62, "1": 0.89, ...}`  
**Purpose**: RMSF/stability scores only

##### **residue_scores_tica_importance.json**
**Format**: JSON (simple dict)  
**Schema**: `{"0": 0.41, "1": 0.15, ...}`  
**Purpose**: tICA component importance scores

##### **hotspots_residue.json** (Deprecated)
**Format**: JSON  
**Schema**:
```json
{
  "scores": [
    {"label": "Res 0", "score": 0.75},
    {"label": "Res 1", "score": 0.23},
    ...
  ]
}
```
**Status**: Maintained for backward compatibility. New consumers should use `hotspots_unified.json`.

---

### 1.2 Intermediate Artifacts

These files are produced during pipeline execution but not intended for direct visualization consumption:

#### **MSM Outputs** (`outputs/msm/`)
- `dtraj.npy`: Discrete state trajectory (NumPy array, shape `[n_frames]`, dtype int32)
- `tica_coords.npy`: tICA projection (NumPy array, shape `[n_frames, n_dims]`, dtype float64)
- `P.npy`: Transition probability matrix (shape `[n_states, n_states]`, dtype float64)
- `pi.npy`: Stationary distribution (shape `[n_states]`, dtype float64)
- `tica_model.npz`: Serialized tICA model (eigenvectors, eigenvalues)

**Purpose**: Input to scoring functions; can be used for advanced analytics but not visualization-ready.

#### **Feature Files** (`data/`)
- `features.npy`: Raw feature matrix (shape `[n_frames, n_features]`)
- `angles.parquet`: Backbone dihedral angles per frame/residue
- `residue_energy.parquet`: Per-residue energies per frame
- `pockets.parquet`: Pocket/cavity metrics per frame

**Format Notes**:
- Parquet files use PyArrow schema with columns: `frame (int32)`, `res_id (int32)`, `<metric> (float64)`
- All frame indices 0-based and match trajectory ordering

---

### 1.3 Output Directory Structure

```
outputs/
├── metrics/                          # Primary visualization outputs
│   ├── hotspots_unified.json         ← Use this
│   ├── frame_scores_dynamic.csv      ← Use this
│   ├── residue_scores_dynamic.json   (legacy)
│   ├── residue_scores_rmsf.json      (legacy)
│   ├── residue_scores_tica_importance.json (legacy)
│   └── hotspots_residue.json         (deprecated)
│
├── msm/                              # MSM artifacts (intermediate)
│   ├── dtraj.npy
│   ├── tica_coords.npy
│   ├── P.npy
│   ├── pi.npy
│   └── tica_model.npz
│
└── phase1/                           # VAMP-2 selection results
    └── reports/
        └── vamp2_best.json           # Optimal hyperparameters
```

---

## 2. Signal Semantics

### 2.1 Signal Types

The pipeline produces three independent metric channels, each capturing distinct aspects of protein dynamics:

#### **Dynamic Anomaly** (Kinetic + Structural)
**Physical Meaning**: Involvement in rare or unexpected conformational dynamics

**Component Signals** (fused):
1. **State Rarity**: `1 - π[state]` where π is MSM stationary distribution
   - Rare states in thermodynamic equilibrium
   - High values indicate transient, low-population conformations
   
2. **Transition Surprise**: `-log(P[s_t → s_{t+τ}])` where P is MSM transition matrix
   - Unexpected state-to-state transitions
   - High values indicate rare barrier-crossing events
   
3. **Local Density**: k-NN distance in tICA space (inverted)
   - Structural isolation in reduced coordinate space
   - High values indicate geometric outliers

**Fusion Method**: Rank-normalized median fusion
- Each component independently rank-normalized to [0,1]
- Median aggregation across components
- Temporal smoothing with median filter (window=5 frames)

**Interpretation**:
- `[0.0, 0.3]`: Normal/common dynamics
- `[0.3, 0.7]`: Moderate anomaly
- `[0.7, 1.0]`: High anomaly (rare/unexpected)

---

#### **RMSF** (Structural Flexibility)
**Physical Meaning**: Root Mean Square Fluctuation - average positional deviation from mean structure

**Calculation**: 
```
RMSF_i = sqrt( mean_t( ||r_i(t) - <r_i>||^2 ) )
```
where `r_i(t)` is position of residue i at time t after alignment.

**Units**: Angstroms (Å) in raw form, normalized to [0,1] for output

**Interpretation**:
- `[0.0, 0.3]`: Rigid (buried core, secondary structure)
- `[0.3, 0.7]`: Moderate flexibility
- `[0.7, 1.0]`: Highly flexible (loops, termini)

**Typical Raw Values**:
- Stable core: 0.5-1.5 Å
- Mobile loops: 2-5 Å
- Disordered termini: >5 Å

**Relation to B-factors**: RMSF ≈ B-factor × sqrt(3/(8π²)) for X-ray structures

---

#### **tICA Importance** (Slow-Mode Contribution)
**Physical Meaning**: Contribution to slow collective motions identified by tICA

**Calculation**:
```
importance_i = sqrt( sum_k( loading_ik^2 ) )
```
where `loading_ik` is eigenvector weight for residue i in component k, summed over top 5 slowest components.

**Interpretation**:
- `[0.0, 0.3]`: Passive/follower residues
- `[0.3, 0.7]`: Moderate contribution to slow modes
- `[0.7, 1.0]`: Driver of slow collective motions (hinges, allosteric nodes)

**Biological Significance**:
- High-importance residues often correspond to:
  - Hinge regions in domain motions
  - Allosteric communication pathways
  - Functional sites where mutation disrupts dynamics

**Complementarity to RMSF**:
- RMSF measures **magnitude** of motion (how much a residue moves)
- tICA importance measures **functional relevance** (how much a residue contributes to slow modes)
- A residue can be rigid (low RMSF) but critical (high importance) if it acts as a hinge with small but essential rotation

---

### 2.2 Normalization

#### **Global vs. Per-Frame Normalization**

**Default**: Global normalization across entire trajectory

**Global Normalization**:
- Compute statistics over all frames
- Preserves absolute magnitude relationships
- Recommended for identifying globally exceptional residues/frames

**Per-Frame Normalization** (optional via `--per-frame-norm`):
- Normalize within each frame independently
- Highlights relative importance within each snapshot
- Recommended for temporal animation where frame-to-frame variation is more important than absolute scale

#### **Normalization Methods**

##### **Percentile** (Default: `--normalization percentile`)
```python
# Clip to [low_percentile, high_percentile] then scale to [0,1]
q_low = quantile(scores, 0.05)   # default: 5th percentile
q_high = quantile(scores, 0.95)  # default: 95th percentile
normalized = clip((scores - q_low) / (q_high - q_low), 0, 1)
```

**Advantages**:
- Robust to extreme outliers
- Prevents "all blue with one red tip" visualization
- Focuses dynamic range on bulk of data

**Use Case**: Default for most visualizations

##### **Rank** (`--normalization rank`)
```python
# Replace values with their rank position
ranks = argsort(argsort(scores))
normalized = ranks / (len(scores) - 1)
```

**Advantages**:
- Maximally robust (only ordering matters)
- Uniform output distribution by construction
- Immune to outliers and heavy tails

**Disadvantages**:
- Loses magnitude information
- Treats all gaps equally

**Use Case**: When distribution is highly skewed or outlier-dominated

##### **Z-score** (`--normalization zscore`)
```python
z = (scores - mean(scores)) / std(scores)
normalized = sigmoid(z)  # Map to [0,1] via 1/(1+exp(-z))
```

**Advantages**:
- Interpretable (units of standard deviation)
- Preserves relative magnitudes

**Disadvantages**:
- Assumes approximately Gaussian distribution
- Sensitive to outliers (affects mean/std)

**Use Case**: When scores are normally distributed and you want to preserve magnitude relationships

---

### 2.3 Temporal Resolution

**Frame Rate Preservation**: 
- Output frame indices match input trajectory 1:1
- No temporal resampling, interpolation, or decimation
- If input trajectory has 10,000 frames, output has 10,000 entries

**Temporal Smoothing**:
- Applied only to fused dynamic anomaly scores
- Method: Median filter with window size W (default W=5)
- Preserves edges (unlike mean filter)
- Window size configurable via `--window` parameter

**Physical Timescale**:
- Depends on input trajectory timestep (typically 2-10 ps/frame)
- Example: 5-frame window at 5 ps/frame = 25 ps smoothing
- Sufficient to suppress thermal noise while preserving conformational transitions

**Lag Times** (Internal Parameters):
- **tICA lag** (τ_tICA): Typical 10 frames (20-100 ps)
  - Defines timescale separation for slow modes
- **MSM lag** (τ_MSM): Typical 30 frames (60-300 ps)
  - Defines Markov property timescale
  - Transitions must equilibrate within this time

**Important**: Lag times are internal to model construction. Output scores are **per-frame** at the trajectory's native resolution.

---

## 3. Alignment Guarantees

### 3.1 Frame Indexing

**Contract**: Output frame indices are 0-based and exactly correspond to input trajectory frame indices after preprocessing.

**Preprocessing Steps** (applied before feature extraction):
1. **Trajectory loading**: Full trajectory loaded via MDAnalysis/MDTraj
2. **Alignment**: Trajectory superposed to reference frame (frame 0) using Cα atoms
   - Removes global rotation/translation
   - Preserves internal coordinates
3. **Frame indexing**: 0-based, contiguous, no gaps

**Example Alignment**:
```
Input trajectory:  frame_0, frame_1, ..., frame_9999
Output CSV:        0,       1,       ..., 9999
```

**No Frame Dropping**: All input frames are processed. Frames are never skipped or dropped during pipeline execution.

---

### 3.2 Residue Indexing

**Contract**: Residue IDs match the **topology file** (PDB) residue numbering.

**Indexing Convention**:
- Residue IDs are **0-based** in output files (internal representation)
- If PDB uses 1-based numbering (e.g., "residue 1" in PDB), it maps to `"0"` in JSON output
- **Chain information is lost** in simplified output formats
- For multi-chain proteins, residue IDs are sequential across chains (chain A residue 1, chain A residue 2, ..., chain B residue 1, ...)

**Mapping to PDB**:
```python
# If using MDTraj/MDAnalysis
residue_id_json = 0
residue_id_pdb = topology.residue(residue_id_json).resSeq
chain_id = topology.residue(residue_id_json).chain.chain_id
```

**Sparse Representation**: 
- Not all residues may appear in output
- Missing residues typically indicate:
  - Not selected for analysis (e.g., solvent, ions filtered out)
  - Zero contribution to metrics (e.g., no slow-mode loading)

---

### 3.3 Feature-to-Frame Alignment

**Guarantee**: Features extracted at frame `i` correspond to trajectory frame `i`.

**MSM Discrete Trajectory Alignment**:
- `dtraj[i]` = MSM state assignment for frame `i`
- `tica_coords[i, :]` = tICA projection for frame `i`
- No lag-induced offsets in output (lags used only for transition statistics internally)

**Energy/Pocket Features**:
- If present, `residue_energy.parquet` has `frame` column matching trajectory indices
- Rows may be per-residue per-frame (long format), requiring grouping by frame for aggregation

---

### 3.4 Assumptions on Input Trajectory

The pipeline assumes:

1. **Single protein system** (or primary protein with solvent/ligands that are filtered)
2. **Pre-equilibrated trajectory** (no drift, no progressive unfolding)
3. **Consistent topology** (atom count and ordering do not change across frames)
4. **Sufficient sampling** (recommended: ≥1000 frames for MSM statistics)
5. **Fixed timestep** (frames evenly spaced in time)

**Violated Assumptions → Undefined Behavior**:
- If topology changes mid-trajectory (e.g., atoms added/removed), indexing breaks
- If trajectory is not equilibrated (e.g., RMSD increases monotonically), MSM stationarity assumption fails
- If frames are irregularly sampled (e.g., some frames 1 ps apart, others 10 ps), lag time interpretation is incorrect

---

## 4. Intended Consumer

### 4.1 Visualization-Agnostic Design

**Principle**: The pipeline exports **semantically meaningful scores**, not rendering instructions.

**What the Pipeline Provides**:
- Normalized scores in [0, 1] range
- Separate channels for different physical properties
- Metadata for interpretation (normalization method, percentile ranges)

**What the Pipeline Does NOT Provide**:
- Color maps (RGB values)
- Visualization thresholds
- 3D rendering parameters
- UI interactions

**Consumer Responsibility**:
- Map scores to colors (e.g., blue-white-red gradient)
- Choose visualization modality (surface, cartoon, spheres)
- Implement interactive controls (thresholds, time scrubbing)
- Combine multiple metric channels (e.g., overlay dynamic + RMSF)

**Supported Visualization Patterns**:
1. **Per-Residue Heatmap**: Color protein by residue score (static or animated)
2. **Temporal Timeseries**: Plot frame scores over time
3. **Trajectory Animation**: Color frames dynamically as trajectory plays
4. **Multi-Channel Overlay**: Render each metric channel with different visual encoding

---

### 4.2 Human Interpretability

**Design Philosophy**: Scores should be interpretable by domain scientists without ML expertise.

**Interpretability Features**:

1. **Normalized Scales**: All scores in [0, 1] for consistency
2. **Physical Grounding**: Each metric tied to concrete physical property:
   - Dynamic anomaly → rare dynamics
   - RMSF → flexibility
   - tICA importance → slow-mode contribution
3. **Component Breakdown**: `frame_scores_dynamic.csv` includes individual signals for debugging/interpretation
4. **Metadata**: Normalization method and percentile ranges documented in output

**Example Interpretation Workflow**:
```
User Question: "Why is residue 42 highlighted?"

1. Check hotspots_unified.json:
   - dynamic_anomaly: 0.92 (high)
   - rmsf: 0.15 (low)
   - tica_importance: 0.87 (high)

2. Interpretation:
   - Residue 42 is RIGID (low RMSF) but highly IMPORTANT (high tICA)
   - Involved in RARE dynamics (high anomaly)
   - Likely a hinge residue: small but critical motion

3. Validation:
   - Check frame_scores_dynamic.csv for temporal pattern
   - Inspect component_transition_surprise: are there rare transitions?
   - Correlate with known functional sites or experimental data
```

---

### 4.3 Post-Processing Requirements

**Minimal Post-Processing**: Outputs are visualization-ready, but consumers may apply:

1. **Custom Normalization**: Re-normalize to different scale (e.g., [0, 100])
2. **Filtering**: Threshold to show only high-scoring residues (e.g., top 10%)
3. **Smoothing**: Additional temporal smoothing for noisy trajectories
4. **Aggregation**: Combine residue scores by secondary structure or domain
5. **Coloring**: Map [0,1] scores to custom color schemes

**Example**: Blue-White-Red Color Map
```python
def score_to_rgb(score):
    """Map [0,1] score to blue (low) - white (mid) - red (high)"""
    if score < 0.5:
        # Blue to white
        t = score * 2  # [0, 1]
        return (t, t, 1.0)
    else:
        # White to red
        t = (score - 0.5) * 2  # [0, 1]
        return (1.0, 1-t, 1-t)
```

**No Mandatory Processing**: Scores can be used directly without transformation.

---

### 4.4 Reference Visualization Implementation

The repository includes a reference implementation: `experiments/trame/anomaly_viewer_3d.py`

**Features**:
- Loads scores from JSON (`/api/points` endpoint)
- VTK-based 3D rendering with color mapping
- Interactive threshold slider
- Frame-by-frame animation support

**Not Required**: Consumers can implement visualizations in any framework (PyMOL, ChimeraX, VMD, NGL Viewer, custom WebGL, etc.)

---

## 5. Known Limitations

### 5.1 Methodological Limitations

#### **Markovian Assumption**
**Limitation**: MSM assumes memoryless dynamics (transitions depend only on current state, not history).

**Failure Modes**:
- Slow processes that haven't equilibrated within MSM lag time (e.g., proline isomerization, buried water rearrangement)
- Multi-timescale systems where slow and fast processes are coupled

**Detection**: Chapman-Kolmogorov test in `tools/validate_model.py`

**Impact on Outputs**:
- State rarity and transition surprise signals may be inaccurate if Markov property violated
- Spurious "rare" states may appear if sampling is insufficient

**Mitigation**:
- Increase MSM lag time if CK test fails
- Use longer trajectories (>10 μs recommended for complex systems)
- Validate against experimental observables (e.g., NMR order parameters)

---

#### **Insufficient Sampling**
**Limitation**: Short trajectories (<1000 frames) yield unreliable statistics.

**Failure Modes**:
- Rare states visited once → artificially high rarity scores
- Poor transition statistics → noisy surprise signals
- Wide confidence intervals on MSM parameters

**Detection**: Bootstrap analysis in Phase 1 (`tools/run_phase1.py`)

**Impact on Outputs**:
- High variance in scores across trajectory subsets
- Scores not reproducible on independent simulations

**Mitigation**:
- Use trajectories with ≥10,000 frames (recommended)
- Check bootstrap confidence intervals (should be <0.2 for stationary distribution)
- Compare multiple independent runs

---

#### **Feature Choice Limitations**
**Limitation**: Pipeline uses backbone dihedrals and Cα distances, which may miss:
- Side-chain rotamer dynamics
- Metal coordination geometry
- Explicit solvent effects
- Electrostatic reorganization

**Impact on Outputs**:
- Hotspots biased toward backbone-driven processes
- May miss functional sites driven by side-chain chemistry

**Mitigation**:
- Use Phase 2 energy features to capture some side-chain effects
- Validate hotspots against experimental data (mutagenesis, binding assays)
- Consider supplementing with all-atom contact analysis

---

### 5.2 Normalization Limitations

#### **Percentile Clipping**
**Limitation**: Clipping to [5th, 95th] percentile discards extreme values.

**Impact**:
- True outliers (e.g., single very rare state) may be underweighted
- Bimodal distributions may be poorly represented

**Mitigation**:
- Use rank normalization if outliers are important
- Adjust percentile range via `--low-percentile` / `--high-percentile`

---

#### **Global Normalization Bias**
**Limitation**: Global normalization makes scores trajectory-dependent.

**Impact**:
- Score of 0.8 in trajectory A ≠ score of 0.8 in trajectory B
- Scores not comparable across different proteins or simulation conditions

**Interpretation**: Scores are **relative rankings** within a single trajectory, not absolute measures.

---

### 5.3 Temporal Resolution Limitations

#### **Lag Time Constraints**
**Limitation**: MSM lag time (τ_MSM) defines minimum resolvable timescale.

**Impact**:
- Fast processes (< τ_MSM) are not captured by kinetic signals
- Catalytic motions faster than 60-300 ps may be missed

**Mitigation**:
- Adjust lag time based on system (see `--lag_msm` parameter)
- Use shorter lags for fast-timescale systems (enzymatic reactions)
- Validate implied timescales to ensure separation

---

#### **Smoothing Artifacts**
**Limitation**: Median filter with window=5 smooths over 10-50 ps.

**Impact**:
- Brief spikes (<5 frames) are suppressed
- Rare single-frame events may be invisible

**Mitigation**:
- Inspect raw scores in `frame_scores_dynamic.csv` (`score` vs `score_windowed`)
- Reduce window size for high-temporal-resolution analysis (`--window 1`)

---

### 5.4 Interpretation Limitations

#### **No Absolute Significance**
**Limitation**: Pipeline provides **relative scores** (which residues are most anomalous), not **absolute significance** (is this anomaly biologically important?).

**Impact**:
- Cannot distinguish biologically meaningful hotspots from statistical fluctuations without external validation

**Mitigation**:
- Validate against known functional sites (active sites, allosteric sites)
- Compare to experimental data (mutation studies, chemical shift perturbations)
- Use multiple independent simulations to assess reproducibility

---

#### **Correlation ≠ Causation**
**Limitation**: High anomaly score indicates involvement in rare dynamics, not necessarily functional importance.

**Impact**:
- Artifact-driven motions (force field errors, unfolding) score high
- Functionally passive but rare states may be highlighted

**Mitigation**:
- Cross-reference with structural knowledge (is this residue near active site?)
- Check trajectory quality (RMSD stability, secondary structure preservation)
- Use domain expertise to filter false positives

---

### 5.5 Computational Limitations

#### **Scalability**
**Performance**: 
- Trajectory loading: O(n_frames × n_atoms)
- tICA projection: O(n_frames × n_features²)
- MSM construction: O(n_frames × n_states)
- k-NN density: O(n_frames²) without optimization

**Large Trajectory Handling** (n_frames > 100K):
- Auto-optimization enabled by default (`compute_all_metrics.py`)
- Subsampling for k-NN (fits on 50K random frames, queries all)
- Complexity reduced from O(n²) to O(n × 50K)

**Limits**:
- Tested up to 500K frames
- Memory requirement: ~8 GB for 100K frames (depends on n_features)

---

#### **Determinism**
**Guarantee**: Pipeline is deterministic with fixed random seeds.

**Stochastic Steps**:
- k-means clustering for MSM (uses `random_state=42`)
- Bootstrap sampling (uses fixed seed)
- k-NN subsampling (uses `np.random.seed(42)`)

**Impact**: Same input → same output (bit-level reproducibility).

---

### 5.6 Known Edge Cases

#### **Single-State Trajectories**
**Scenario**: Trajectory remains in one conformational state (e.g., well-folded, no transitions).

**Behavior**:
- MSM has 1 state → rarity ≈ 0 everywhere
- Transition surprise undefined
- Local density still computes (may find subtle variations)

**Output**: Scores will be uniformly low or zero. Check `meta.n_states` in logs.

---

#### **Disconnected State Spaces**
**Scenario**: MSM has disconnected components (some states never transition to others).

**Behavior**:
- Stationary distribution computed only for largest connected component
- Disconnected states may have undefined or zero probability
- Warning issued during MSM construction

**Impact**: Rare states in disconnected components may not score correctly.

**Detection**: `tools/validate_model.py` checks connectivity.

---

#### **Missing Residues in Topology**
**Scenario**: PDB has missing residues (e.g., disordered loops not resolved in crystal structure).

**Behavior**:
- Those residues have no entries in output
- Residue numbering may have gaps

**Impact**: Visualization must handle sparse residue sets.

---

## 6. Version Compatibility

### 6.1 Output Format Versioning

**Current Version**: Phase 3 (Multi-Signal Fusion)

**Format Changes**:
- **Phase 1 → Phase 2**: Added energy and pocket signals (optional)
- **Phase 2 → Phase 3**: Introduced unified JSON format (`hotspots_unified.json`)

**Backward Compatibility**:
- Legacy formats (`residue_scores_dynamic.json`, etc.) still produced
- Old visualization code will continue to work

**Forward Compatibility**:
- New consumers should use `hotspots_unified.json`
- Check `meta.metrics` array for available channels

---

### 6.2 Deprecation Policy

**Deprecated Formats**:
- `hotspots_residue.json` (array-based schema)
  - Status: Maintained for backward compatibility
  - Removal timeline: Not before version 2.0

**Recommended Migration**:
```python
# Old (deprecated)
with open('hotspots_residue.json') as f:
    data = json.load(f)
    scores = {int(item['label'].split()[1]): item['score'] 
              for item in data['scores']}

# New (recommended)
with open('hotspots_unified.json') as f:
    data = json.load(f)
    scores = data['per_residue']['dynamic_anomaly']
```

---

### 6.3 Dependency Versions

**Critical Dependencies**:
- NumPy ≥ 1.20 (array operations, random seeding)
- Pandas ≥ 1.3 (Parquet I/O)
- scikit-learn ≥ 1.0 (k-NN)
- deeptime ≥ 0.4 (MSM, tICA)
- MDTraj ≥ 1.9 (trajectory loading, RMSF)

**Breaking Changes**:
- deeptime 0.4 → 0.5: MSM API changed (transition_matrix attribute)
- NumPy 1.x → 2.0: Random number generation API changed

**Mitigation**: Version pinning in `requirements_phase3.txt`

---

## Appendix A: Quick Reference

### Common Consumption Patterns

#### Pattern 1: Load Residue Scores for Static Visualization
```python
import json

with open('outputs/metrics/hotspots_unified.json') as f:
    data = json.load(f)

# Extract dynamic anomaly
scores = data['per_residue']['dynamic_anomaly']

# Map to visualization
for res_id, score in scores.items():
    residue = protein.residue(int(res_id))
    color = score_to_color(score)
    residue.set_color(color)
```

#### Pattern 2: Animate Trajectory by Frame Scores
```python
import pandas as pd

df = pd.read_csv('outputs/metrics/frame_scores_dynamic.csv')

for frame_idx, row in df.iterrows():
    trajectory.seek(frame_idx)
    score = row['score']
    render_with_global_color(trajectory, score)
```

#### Pattern 3: Multi-Channel Comparison
```python
import json

with open('outputs/metrics/hotspots_unified.json') as f:
    data = json.load(f)

dynamic = data['per_residue']['dynamic_anomaly']
rmsf = data['per_residue']['rmsf']

# Identify rigid but dynamic residues (hinges)
for res_id in dynamic.keys():
    if dynamic[res_id] > 0.7 and rmsf[res_id] < 0.3:
        print(f"Potential hinge: Residue {res_id}")
```

---

## Appendix B: Validation Checklist

Before consuming pipeline outputs, verify:

- [ ] `meta.n_frames` matches expected trajectory length
- [ ] `meta.normalization` is appropriate for visualization (default: percentile)
- [ ] Frame indices in `frame_scores_dynamic.csv` are contiguous (0, 1, 2, ..., n-1)
- [ ] Residue IDs in JSON match PDB topology (accounting for 0-based indexing)
- [ ] All scores are in [0.0, 1.0] range
- [ ] No NaN or Inf values in score arrays
- [ ] `meta.metrics` array contains expected channels

If validation fails, check pipeline logs for warnings about:
- Disconnected MSM states
- Insufficient sampling
- Failed Chapman-Kolmogorov test
- Missing feature files

---

## Appendix C: Glossary

- **Dynamic Anomaly**: Measure of involvement in rare or unexpected conformational dynamics
- **Frame**: Single snapshot of MD trajectory at time t
- **MSM**: Markov State Model - discrete-state kinetic model of protein dynamics
- **Residue**: Amino acid in protein sequence (indexed 0-based in outputs)
- **RMSF**: Root Mean Square Fluctuation - measure of positional variability
- **tICA**: Time-lagged Independent Component Analysis - dimensionality reduction for slow modes
- **Normalization**: Transformation to standardized scale (e.g., [0, 1])
- **Lag Time** (τ): Time interval for computing transitions or covariances
- **Stationary Distribution** (π): Equilibrium population of MSM states
- **Transition Matrix** (P): Conditional probabilities of state-to-state transitions

---

## Document Maintenance

**Update Triggers**:
- Change to output file schema
- Change to normalization methods
- Addition of new metric channels
- Discovery of new limitations or edge cases

**Review Cycle**: After each pipeline phase completion

**Contact**: See repository README.md for maintainer information

---

**End of Specification**

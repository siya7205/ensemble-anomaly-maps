# Pipeline Summary for Biochemists
## Dynamic Hotspot Detection in MD Trajectories

### 1-Page Quick Summary

**What are we doing?**  
We analyze molecular dynamics (MD) trajectories to identify "dynamic hotspots" - protein residues that exhibit unusual, rare, or functionally important motions during simulation.

**Why does it matter?**  
Dynamic hotspots often correspond to:
- Functional sites (active sites, allosteric sites)
- Druggable pockets (especially cryptic pockets)
- Regulatory regions that control protein function
- Sites where mutations would significantly impact dynamics

**How does it work?**  
We use a multi-stage machine learning pipeline:

1. **Feature Extraction**: Convert raw atomic coordinates into meaningful features (backbone angles, inter-residue contacts, energies)
2. **tICA (Dimensionality Reduction)**: Identify slow collective motions that are functionally relevant
3. **MSM (Markov State Model)**: Build a kinetic model to understand state populations and transition rates
4. **Anomaly Detection**: Compute multiple signals that detect rare, unusual, or strained conformations
5. **Hotspot Scoring**: Aggregate frame-level anomalies into per-residue importance scores

**What are the key outputs?**
- **Dynamic anomaly scores**: Which residues are involved in rare/unexpected motions
- **RMSF/stability scores**: Which residues are flexible vs. rigid
- **tICA importance scores**: Which residues contribute most to slow collective motions
- **Interactive visualization**: 3D viewer showing hotspots colored by frame

---

## Detailed Pipeline Overview

### Stage 1: Feature Extraction

**Input**: 
- `topology.pdb` - Protein structure
- `trajectory.xtc` - MD trajectory (T frames)

**What we extract**:
1. **Backbone dihedral angles** (φ, ψ) - Local backbone conformation
2. **Cα-Cα distances** - Inter-residue contacts and spatial relationships
3. **Per-residue energies** (optional) - Knowledge-based contact potentials
4. **Pocket/cavity metrics** (optional) - Binding site dynamics

**Output**: Feature matrix X of shape [n_frames, n_features]

**Why these features?**
- Dihedral angles capture local backbone flexibility
- Distance-based features capture tertiary structure changes
- Energy features identify strained/unstable conformations
- Pocket features track druggable cavity dynamics

---

### Stage 2: tICA (Time-lagged Independent Component Analysis)

**Purpose**: Reduce high-dimensional feature space to a few "slow modes" that capture functionally relevant motions.

**Key Concept**: Unlike PCA (which finds directions of maximum variance), tICA finds directions of **slowest motion**. Protein function typically involves slow conformational changes (domain movements, loop rearrangements), not fast vibrations.

**Mathematical Approach**:
- Compute covariance matrices at time t and time t+lag
- Find linear combinations of features that decorrelate slowly
- Project trajectory onto top slow modes (typically 3-8 dimensions)

**Why tICA?**
- **Functionally relevant**: Slow motions often control binding, catalysis, allostery
- **Markovian**: Provides coordinate system where Markov models are valid
- **Interpretable**: Component loadings show which residues contribute to each slow mode

**Output**: 
- Low-dimensional coordinates Y [n_frames, n_dims]
- Per-residue loadings for each tICA component

---

### Stage 3: MSM (Markov State Model)

**Purpose**: Discretize continuous conformational space and model transitions as a Markov chain.

**Construction**:
1. **Clustering**: Group similar conformations in tICA space (e.g., k-means with k=20-50 states)
2. **Count transitions**: Track state-to-state transitions at lag time τ
3. **Estimate matrix**: Build transition probability matrix P
4. **Compute equilibrium**: Find stationary distribution π (long-term populations)

**Why build MSMs?**
- **Kinetic information**: Know not just what states exist, but how fast they interconvert
- **Equilibrium populations**: Identify rare vs. common states
- **Timescale extension**: Extrapolate dynamics beyond simulation length

**Key Outputs**:
- **Transition matrix P**: Probability of transitioning between states
- **Stationary distribution π**: Equilibrium population of each state
- **Implied timescales**: Relaxation timescales for slow processes

**Physical Interpretation**:
- Low π[state] → rare, potentially important state
- Low P[i→j] → rare, barrier-crossing transition
- Long implied timescales → slow relaxation processes

---

### Stage 4: Anomaly Detection (Multi-Signal Fusion)

**Purpose**: Identify frames and residues that deviate from typical behavior along multiple dimensions.

#### Signals We Compute:

**1. State Rarity** (Kinetic Signal)
```
rarity(t) = 1 - π[state(t)]
```
- **Physical meaning**: How rare is this conformation in equilibrium?
- **High values indicate**: Rarely-visited states that may be:
  - Transition intermediates
  - High-energy barriers
  - Functionally important but transient states

**2. Transition Surprise** (Kinetic Signal)
```
surprise(t) = -log(P[state(t) → state(t+τ)])
```
- **Physical meaning**: How unexpected is this transition?
- **High values indicate**: Rare transitions that may involve:
  - Large conformational changes
  - Barrier crossings
  - Allosteric communication

**3. Local Density** (Structural Signal)
```
density(t) = k-NN distance in tICA space
```
- **Physical meaning**: How isolated is this conformation?
- **High values (low density) indicate**: Structural outliers that:
  - Sample rare regions of conformational space
  - May be artifacts or interesting edge cases

**4. Soft Entropy** (Optional - Kinetic/Structural Signal)
```
entropy(t) = -Σ p(state|t) log p(state|t)
```
- **Physical meaning**: How ambiguous is the state assignment?
- **High values indicate**: Transitional conformations that:
  - Fluctuate between states
  - Are at state boundaries
  - Show high local disorder

**5. Energy Stress** (Optional - Energetic Signal)
```
stress(t) = Σ E_contact(residue, t)
```
- **Physical meaning**: Total unfavorable interaction energy
- **High values indicate**: Energetically strained conformations with:
  - Unfavorable contacts
  - Frustrated packing
  - Potential for relaxation or functional importance (e.g., transition states)

**6. Pocket Volatility** (Optional - Structural Signal)
```
volatility(t) = |Volume(t) - Volume(t-1)|
```
- **Physical meaning**: Rate of change in binding pocket geometry
- **High values indicate**: Dynamic pocket breathing that affects:
  - Ligand binding kinetics
  - Cryptic pocket opening/closing
  - Allosteric responses

#### Signal Fusion Strategy:

1. **Normalize**: Convert each signal to [0,1] scale using rank normalization (robust to outliers)
2. **Fuse**: Combine signals using median (robust) or mean (sensitive)
3. **Smooth**: Apply moving median filter to reduce frame-to-frame noise
4. **Scale**: Convert to [0,100] range for visualization

**Why multiple signals?**
- **Comprehensive**: Captures anomalies along kinetic, structural, and energetic dimensions
- **Robust**: Multiple signals must agree for high scores (reduces false positives)
- **Interpretable**: Can decompose score to understand *why* a frame is anomalous

---

### Stage 5: Hotspot Scoring (Frame → Residue Mapping)

**Purpose**: Convert per-frame anomaly scores to per-residue importance scores.

#### Methods:

**Method 1: tICA Component Weights**
- Each tICA component has feature loadings
- Residues with high loadings contribute strongly to slow modes
- Saved as: `ic*_residue_weights.json`

**Method 2: Anomaly Contribution**
- Select top 10% anomalous frames
- For each residue, compute mean energy/deviation in those frames
- Rank residues by contribution to anomalies

**Method 3: Time-Aggregated Metrics**
- Compute per-residue statistics across trajectory:
  - **RMSF (Root Mean Square Fluctuation)**: Measures flexibility/rigidity
  - **Mean energy stress**: Average unfavorable contacts
  - **Pocket rim frequency**: How often residue is at a pocket boundary

#### Three Separate Metric Channels:

**1. Dynamic Anomaly Score**
- **What**: Involvement in rare/unexpected dynamics
- **Interpretation**: High scores → involved in unusual motions
- **Use case**: Identify allosteric pathways, transition intermediates

**2. RMSF/Stability Score**  
- **What**: Overall flexibility vs. rigidity
- **Interpretation**: High scores → flexible/floppy regions; Low scores → rigid/stable
- **Use case**: Compare to experimental B-factors, identify flexible loops

**3. tICA Importance Score**
- **What**: Contribution to slow collective motions
- **Interpretation**: High scores → drive functionally relevant slow modes
- **Use case**: Identify hinge residues, allosteric nodes

---

## Normalization Strategies

### Why Normalization Matters:

Raw anomaly scores can be heavily skewed, leading to poor visualization (e.g., "all blue with red tips"). Proper normalization makes the full dynamic range visible.

### Normalization Options:

**1. Rank-Based Normalization** (Default)
```
score_norm(x) = rank(x) / (n-1)
```
- **Advantages**: Robust to outliers, preserves exact ordering, distribution-free
- **Use when**: You want equal representation across the score range

**2. Percentile-Based Normalization**
```
score_norm(x) = clip((x - p5) / (p95 - p5), 0, 1)
```
where p5 and p95 are 5th and 95th percentiles
- **Advantages**: Focuses on bulk of distribution, clips extreme outliers
- **Use when**: You have artifacts or extreme outliers skewing the range

**3. Global vs. Per-Frame Normalization**

**Global Normalization** (Default):
- Compare all residues across entire trajectory
- **Interpretation**: "Which residues are most important overall?"
- **Best for**: Finding constitutively important hotspots

**Per-Frame Normalization**:
- Normalize within each frame independently
- **Interpretation**: "Which residues are most important in this specific conformation?"
- **Best for**: Time-resolved visualization, finding frame-specific hotspots

### CLI Parameters:

```bash
--normalization global|per_frame   # Normalization scope
--low-percentile 0.05              # Lower clip percentile
--high-percentile 0.95             # Upper clip percentile
```

---

## Output Format for Visualization

### JSON Schema:

```json
{
  "meta": {
    "n_frames": 200,
    "n_residues": 150,
    "metrics": ["dynamic_anomaly", "rmsf", "tica_importance"],
    "normalization": "global",
    "percentile_range": [0.05, 0.95]
  },
  "per_frame": {
    "0": {
      "dynamic_anomaly": {"1": 0.31, "2": 0.02, ...},
      "rmsf": {"1": 0.12, "2": 0.34, ...}
    },
    ...
  },
  "per_residue": {
    "dynamic_anomaly": {"1": 0.45, "2": 0.15, ...},
    "rmsf": {"1": 0.12, "2": 0.34, ...},
    "tica_importance": {"1": 0.88, "2": 0.05, ...}
  }
}
```

### File Outputs:

1. **`frame_scores_dynamic.csv`** - Per-frame dynamic anomaly scores + components
2. **`residue_scores_dynamic.json`** - Per-residue dynamic anomaly aggregation
3. **`residue_scores_rmsf.json`** - Per-residue RMSF/stability scores
4. **`residue_scores_tica_importance.json`** - Per-residue slow-mode importance
5. **`hotspots_unified.json`** - Combined format for viewer consumption

---

## Validation & Interpretation

### How to Validate Hotspots:

**1. Compare to Known Functional Sites**
- Active sites (from UniProt, literature)
- Allosteric sites (from experiments)
- Binding sites (from crystal structures)

**2. Conservation Analysis**
- Run multiple sequence alignment (MSA)
- Check if predicted hotspots are conserved
- Caveat: Not all hotspots are conserved (organism-specific regulation)

**3. Experimental Validation**
- **Mutagenesis**: Test if hotspot mutations affect function
- **NMR/HDX-MS**: Compare to experimental dynamics measurements
- **Crystallography**: Compare to B-factors (RMSF analog)

### Interpreting Score Combinations:

| Dynamic Anomaly | RMSF | Interpretation |
|----------------|------|----------------|
| High | High | Flexible region involved in rare motions (e.g., active site loop) |
| High | Low | Rigid region with unusual dynamics (e.g., allosteric hinge) |
| Low | High | Constitutively flexible, not anomalous (e.g., surface loop) |
| Low | Low | Stable, rigid core (e.g., hydrophobic core) |

| tICA Importance | Dynamic Anomaly | Interpretation |
|----------------|-----------------|----------------|
| High | High | Critical for both slow modes AND rare events (prime hotspot candidate) |
| High | Low | Drives slow motions but not anomalous (functional but well-sampled) |
| Low | High | Anomalous but not slow-mode driver (local rearrangement) |

---

## Best Practices

### Parameter Selection:

**tICA lag time**: 10-50 frames (1-5 ns for 100 ps timesteps)
- Too small: Captures fast noise
- Too large: Loses temporal resolution
- **Optimal**: Use VAMP-2 score maximization

**MSM lag time**: 20-50 frames
- Must be long enough for Markov property to hold
- Validate with implied timescales plot (should plateau)

**Number of MSM states**: 20-50
- Too few: Lose detail
- Too many: Poor sampling per state
- **Rule of thumb**: ~100-200 frames per state on average

**Smoothing window**: 5-7 frames (0.5-0.7 ns)
- Reduces frame-to-frame jitter
- Should be shorter than timescale of interest

### Computational Tips:

- **Caching**: Use hash-based caching for energy/pocket features (expensive)
- **Stride**: Process every Nth frame for initial exploration
- **Alignment**: Always align trajectory to remove global rotation/translation
- **Selection**: Use backbone heavy atoms (N, CA, C, O) for features

---

## References & Further Reading

### Key Papers:

1. **tICA**: Pérez-Hernández et al. (2013) J. Chem. Phys. - Original tICA for MD
2. **MSM**: Prinz et al. (2011) J. Chem. Phys. - Comprehensive MSM methodology
3. **VAMP**: Wu & Noé (2020) J. Nonlinear Sci. - Variational approach theory
4. **Cryptic Pockets**: Beglov et al. (2018) PNAS - Structural origins of druggable cryptic sites
5. **Allostery**: Nussinov & Tsai (2013) Cell - Allostery in disease and drug discovery

### Software Documentation:

- **deeptime**: https://deeptime-ml.github.io/ - Modern Python library for tICA/MSM
- **MDTraj**: https://mdtraj.org/ - MD trajectory analysis
- **PyEMMA**: http://emma-project.org/ - Alternative MSM/tICA implementation

---

## Summary

This pipeline provides a **rigorous, multi-scale approach** to understanding protein dynamics:

✓ **Kinetic**: MSMs capture state populations and transition rates  
✓ **Structural**: tICA and density metrics capture geometric anomalies  
✓ **Energetic**: Contact energies identify strained conformations  
✓ **Functional**: Pocket dynamics reveal binding site behavior  

By combining these signals and providing **separate metric channels** (dynamic anomaly, RMSF, tICA importance), we enable:
- **Better visualization** with distinct color channels
- **Mechanistic interpretation** of which type of dynamics matter
- **Targeted validation** of specific hotspot predictions
- **Drug discovery** insights into allosteric and cryptic sites

**Key Innovation**: We don't just find important residues - we explain *why* they're important (flexibility? rare dynamics? slow-mode contribution?) and *when* they're important (frame-specific dynamics).

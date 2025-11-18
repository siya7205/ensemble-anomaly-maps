# Scientific Documentation: Dynamic Hotspot Detection in Molecular Dynamics

## Table of Contents
1. [Introduction](#introduction)
2. [Overview of the ML Pipeline](#overview-of-the-ml-pipeline)
3. [Feature Extraction: What and Why](#feature-extraction-what-and-why)
4. [Dimensionality Reduction with tICA](#dimensionality-reduction-with-tica)
5. [Markov State Models (MSM)](#markov-state-models-msm)
6. [Anomaly Detection](#anomaly-detection)
7. [Hotspot Scoring: Significance and Interpretation](#hotspot-scoring-significance-and-interpretation)
8. [Dynamic Hotspots: Definition and Biological Importance](#dynamic-hotspots-definition-and-biological-importance)
9. [Multi-Signal Fusion Strategy](#multi-signal-fusion-strategy)
10. [Scientific Validation and Best Practices](#scientific-validation-and-best-practices)
11. [References](#references)

---

## Introduction

This document provides a comprehensive scientific explanation of the machine learning pipeline used to detect and analyze **dynamic hotspots** in molecular dynamics (MD) simulations of proteins. Our approach combines established computational methods from statistical mechanics, machine learning, and structural biology to identify regions of proteins that exhibit anomalous or functionally significant dynamic behavior.

### Key Questions Addressed:
- **What are we doing?** Building a multi-stage pipeline to detect unusual conformational states and structural anomalies in proteins
- **Why does it matter?** Dynamic hotspots often correspond to functional sites, allosteric regulatory regions, and druggable pockets
- **How does it work?** Through dimensionality reduction, kinetic modeling, and multi-signal anomaly detection

---

## Overview of the ML Pipeline

### Pipeline Architecture

Our pipeline consists of six main stages:

```
MD Trajectory → Feature Extraction → tICA Projection → MSM Construction → 
Anomaly Scoring → Dynamic Hotspot Identification
```

### Why This Architecture?

1. **Dimensionality Problem**: Raw MD trajectories contain millions of atomic coordinates across thousands of frames. This high-dimensional data is redundant and computationally intractable.

2. **Timescale Separation**: Protein motions span 12+ orders of magnitude in time (femtoseconds to seconds). We need methods that focus on functionally relevant slow motions.

3. **Kinetic Information**: Structural similarity alone is insufficient. We must understand which conformations are kinetically accessible and how transitions occur.

4. **Multi-scale Detection**: Anomalies manifest across multiple dimensions: kinetic (rare states), structural (geometric outliers), and energetic (strained conformations).

---

## Feature Extraction: What and Why

### What We Extract

#### 1. Geometric Features
- **Backbone dihedral angles (φ, ψ)**: Capture local backbone conformation
- **C-alpha distances**: Inter-residue spatial relationships
- **Backbone RMSD**: Global structural similarity

**Why?** Dihedral angles are the natural degrees of freedom for protein backbone motion. They are rotation-invariant and provide a compact representation of local structure.

#### 2. Energetic Features (Phase 2)
- **Per-residue contact energies**: Knowledge-based potentials (simplified Miyazawa-Jernigan)
- **Hydrogen bond networks**: Stabilizing interactions
- **Electrostatic contributions**: Charge-charge interactions

**Why?** Energetic strain often precedes structural changes. High-energy residues indicate frustrated conformations that may be functionally important (e.g., enzyme active sites, allosteric switches).

**Scientific Basis**: Knowledge-based potentials are derived from statistical analysis of known protein structures. Residue pairs that occur frequently in stable structures receive favorable (negative) energies.

**Formula**:
```
E_contact(i,j) = E_base(res_i, res_j) × f(d_ij)
```
where `f(d)` is a distance-dependent weighting function (Lennard-Jones-like).

#### 3. Pocket/Cavity Features (Phase 2)
- **Pocket volume**: Size of binding cavities
- **Mouth radius**: Accessibility of pockets
- **SASA (rim)**: Solvent-accessible surface area at pocket entrance

**Why?** Most drugs and small molecules bind in protein pockets. Transient (cryptic) pockets that appear only during dynamics are increasingly recognized as druggable targets.

**Scientific Basis**: Pockets are identified using grid-based methods similar to MDpocket. A probe sphere (radius ≈ 1.4 Å, water-sized) is used to identify concave regions on the protein surface.

**Algorithm**:
1. Create 3D grid around protein (spacing ~ 0.5 nm)
2. Identify grid points where probe fits but bulk solvent doesn't reach
3. Label connected components as pockets
4. Track volume, mouth radius, and rim residues over time

### Why These Specific Features?

- **Biological Relevance**: Each feature type captures distinct aspects of protein function
  - **Geometric**: Conformational state
  - **Energetic**: Stability and frustration
  - **Pocket**: Binding site dynamics

- **Computational Tractability**: These features reduce the dimensionality from ~10⁴-10⁵ atom coordinates to ~10²-10³ features while preserving essential information

- **Timescale Coverage**: Different features report on different timescales:
  - Dihedral angles: ps-ns (fast)
  - Pocket dynamics: ns-μs (intermediate)
  - Energy landscapes: μs-ms (slow)

---

## Dimensionality Reduction with tICA

### What is tICA?

**Time-lagged Independent Component Analysis (tICA)** is a dimensionality reduction method specifically designed for dynamical systems. Unlike PCA (which finds directions of maximum variance), tICA finds directions of **slowest motion**.

### Why Use tICA?

#### Problem with Standard Methods:
- **PCA** maximizes variance but may capture fast, unimportant fluctuations
- **Standard ICA** ignores temporal structure

#### tICA Solution:
tICA identifies **collective variables** that change slowly over time, which are typically the functionally relevant motions in proteins (domain movements, loop rearrangements, allosteric transitions).

### Mathematical Formulation

Given feature vectors **x**_t at time t:

1. **Instantaneous covariance matrix**: C₀ = ⟨x_t ⊗ x_t⟩
2. **Time-lagged covariance matrix**: C_τ = ⟨x_t ⊗ x_{t+τ}⟩

tICA finds transformation **W** that diagonalizes:
```
C₀⁻¹ C_τ
```

The eigenvectors (independent components) are ordered by their **timescales**:
```
τᵢ = -lag / ln(λᵢ)
```
where λᵢ are eigenvalues.

### Scientific Significance

**Markovian Approximation**: tICA components approximate the leading eigenfunctions of the transfer operator, meaning they capture the slowest relaxation processes of the system.

**Variational Principle**: tICA maximizes the **VAMP score** (Variational Approach for Markov Processes), providing a rigorous theoretical foundation.

### Parameter Selection (Phase 1)

#### Lag Time (τ)
- **Too small**: Captures fast noise, not slow dynamics
- **Too large**: Loses temporal resolution
- **Optimal**: Chosen via VAMP-2 score maximization

We test lags: 5, 10, 15, 20, 30, 50 frames

#### Dimensionality (d)
- **Too few**: Miss important slow modes
- **Too many**: Include fast modes and noise
- **Optimal**: Balance via VAMP-2 score

We test dimensions: 2, 3, 4, 5, 6, 8, 10

**VAMP-2 Score**:
```
VAMP-2 = Σᵢ σᵢ²
```
where σᵢ are singular values of the correlation matrix C₀₁.

### Why This Matters for Hotspot Detection

By projecting onto slow modes, we:
1. **Remove noise**: Fast local fluctuations are filtered out
2. **Focus on function**: Slow motions are often functional (binding, catalysis, allostery)
3. **Enable kinetic modeling**: tICA provides a reduced space where Markov models are valid

---

## Markov State Models (MSM)

### What is an MSM?

A **Markov State Model** discretizes continuous conformational space into a set of **metastable states** and models transitions between them as a Markov chain.

### Why Build MSMs?

#### Problems MSMs Solve:
1. **Timescale limitation**: MD simulations are limited to microseconds, but biological processes occur on milliseconds or longer
2. **Trajectory integration**: How to combine information from multiple short trajectories?
3. **Rare event sampling**: Important transitions may never be observed in any single trajectory

#### MSM Solutions:
- **Timescale extension**: By modeling transition probabilities, MSMs extrapolate dynamics beyond simulation timescales
- **Statistical power**: Multiple trajectories are combined into a single model
- **Quantitative kinetics**: Calculate transition rates, equilibrium populations, and mean first passage times (MFPT)

### MSM Construction

#### Step 1: Clustering
After tICA projection, we cluster conformations into **discrete states** using k-means:
```
Y (tICA coords) → K-means (k=30) → Discrete trajectory d(t)
```

**Why k-means?** Simple, fast, and adequate when tICA has already separated slow modes.

#### Step 2: Transition Counting
Count transitions between states at lag time τ:
```
C_ij = # transitions i → j in time τ
```

#### Step 3: Transition Matrix Estimation
Estimate transition probability matrix:
```
P_ij = P(state j at t+τ | state i at t) = C_ij / Σⱼ C_ij
```

#### Step 4: Validation
- **Implied timescales**: Converge as lag increases (Chapman-Kolmogorov test)
- **Bootstrap uncertainty**: Confidence intervals from resampling (Phase 1)

### Key MSM Quantities

#### 1. Stationary Distribution (π)
```
π = eigenvector of P^T with eigenvalue 1
```
**Interpretation**: Long-term equilibrium population of each state. Rare states have low π.

#### 2. Transition Matrix (P)
**Interpretation**: P_ij gives probability of transitioning i→j in time τ.

#### 3. Implied Timescales (τᵢ)
```
τᵢ = -τ_lag / ln(λᵢ)
```
where λᵢ are eigenvalues of P.

**Interpretation**: Relaxation timescales of the system. Longest timescales correspond to rare, slow transitions.

### Scientific Foundation

**Markov Property**: Assumes future depends only on present, not past:
```
P(x_{t+τ} | x_t, x_{t-τ}, ...) = P(x_{t+τ} | x_t)
```

**Validity**: Markov property holds in the tICA-projected space when:
1. Lag time τ is sufficiently long
2. tICA captures all slow degrees of freedom

**References**:
- Prinz et al. (2011). "Markov models of molecular kinetics"
- Noé et al. (2009). "Constructing the equilibrium ensemble of folding pathways"

---

## Anomaly Detection

### What is an Anomaly?

In our context, an **anomaly** is a conformation or trajectory segment that:
1. **Deviates from typical behavior** in one or more dimensions
2. **Occurs rarely** in equilibrium sampling
3. **May indicate functional importance** (binding events, transitions, etc.)

### Why Detect Anomalies?

#### Scientific Motivation:
- **Functional sites**: Often involve unusual conformations (catalytic intermediates, binding-competent states)
- **Allosteric pathways**: Rare transitions that propagate signals across proteins
- **Druggable states**: Transient conformations may be more targetable than ground states

#### Practical Motivation:
- **Data summarization**: Identify most interesting frames from millions
- **Hypothesis generation**: Focus experimental validation on promising regions
- **Quality control**: Detect artifacts or unphysical states

### Multi-Signal Approach

We compute anomaly scores from **six complementary signals**:

#### Kinetic Signals (from MSM)

##### 1. State Rarity
```
rarity(t) = 1 - π[s(t)]
```
**What**: How rare is the current state?
**Why**: Rare states may be functionally important transition intermediates or high-energy barriers.

**Biological Example**: A rare open-pocket state might be the only conformation where a drug can enter.

##### 2. Transition Surprise
```
surprise(t) = -log(P[s(t) → s(t+τ)])
```
**What**: How unexpected is the observed transition?
**Why**: Forbidden or rare transitions may indicate large conformational changes, barrier crossings, or artifacts.

**Biological Example**: A transition from inactive to active state that rarely occurs spontaneously.

#### Structural Signals

##### 3. Local Density
```
density(t) = k-NN distance in tICA space
```
**What**: How isolated is this conformation from others?
**Why**: Structural outliers may represent rarely-visited regions of conformational space.

**Method**: k-nearest neighbors (k=20) in tICA-projected space. Large distances indicate isolation.

##### 4. Soft Entropy (Phase 3, optional)
```
entropy(t) = -Σᵢ q(i|t) log q(i|t)
```
where q(i|t) are probabilistic state assignments from HMM.

**What**: How ambiguous is the state assignment?
**Why**: High entropy indicates:
- Conformations at state boundaries (transitional)
- Rapid fluctuations between states
- Poorly defined structure

**Biological Example**: Active site loops that fluctuate between open/closed states.

#### Energetic Signals (Phase 2)

##### 5. Energy Stress
```
stress(t) = Σ_residues E_contact(t, residue)
```
**What**: Total unfavorable contact energy at frame t
**Why**: High-energy conformations are:
- Unstable (may relax quickly)
- Functionally strained (e.g., enzyme transition states)
- Potential artifacts (if extremely high)

**Biological Example**: Transition state of an enzymatic reaction has high energy but is catalytically essential.

##### 6. Pocket Volatility
```
volatility(t) = |Volume(t) - Volume(t-1)|
```
**What**: Rate of change in pocket volume/geometry
**Why**: Rapid pocket changes indicate:
- Breathing motions (important for binding kinetics)
- Opening/closing of cryptic pockets
- Allosteric responses to distant events

**Biological Example**: GPCRs exhibit rapid pocket breathing that modulates ligand binding rates.

### Why Multiple Signals?

**Single-signal limitations**:
- Kinetic-only: Misses energetically strained but accessible states
- Structural-only: Misses kinetically forbidden transitions
- Energetic-only: Misses structurally unusual but low-energy states

**Multi-signal advantages**:
- **Comprehensive**: Captures anomalies along multiple dimensions
- **Robust**: Less sensitive to noise in any single signal
- **Interpretable**: Decomposition reveals *why* a frame is anomalous

### Fusion Strategy (Phase 3)

#### Step 1: Normalization
Each signal is normalized to [0, 1] using **rank normalization**:
```
normalized(x) = rank(x) / (N-1)
```

**Why rank normalization?**
- Robust to outliers
- Distribution-free (works for any data distribution)
- Preserves ordering exactly
- Standard in bioinformatics (e.g., GSEA)

**Alternative**: Quantile normalization (uses percentiles)

#### Step 2: Fusion
Combine normalized signals using **median** (default) or **mean**:
```
score_raw(t) = median([signal₁(t), signal₂(t), ..., signal₆(t)])
```

**Why median?**
- Robust to outliers in any single signal
- Requires multiple signals to agree for high scores
- Standard in ensemble methods

#### Step 3: Smoothing
Apply **moving median filter** (window = 5-7 frames):
```
score_windowed(t) = median(score_raw[t-w:t+w])
```

**Why smoothing?**
- Reduces frame-to-frame jitter
- Identifies sustained anomalies (not transient noise)
- More robust for visual interpretation

#### Step 4: Scaling
Scale to interpretable range [0, 100]:
```
score_final(t) = score_windowed(t) × 100
```

### Interpretation of Scores

| Score Range | Interpretation |
|------------|----------------|
| 0-25 | Typical, well-sampled conformation |
| 25-50 | Moderate anomaly in 1-2 dimensions |
| 50-75 | Significant anomaly in multiple dimensions |
| 75-100 | Extreme anomaly, requires careful inspection |

---

## Hotspot Scoring: Significance and Interpretation

### What is a Hotspot?

A **hotspot** is a **residue** that:
1. Contributes significantly to anomaly signals
2. Undergoes unusual motions or strain
3. May be functionally or structurally important

### Why Identify Hotspots?

#### Scientific Value:
- **Functional annotation**: Predict important residues without prior knowledge
- **Mutagenesis targets**: Guide experimental validation
- **Drug discovery**: Identify binding site residues
- **Allosteric networks**: Map communication pathways

#### Practical Value:
- **Visual focus**: Highlight regions in 3D visualization
- **Validation**: Compare predicted hotspots to known functional sites
- **Hypothesis generation**: Suggest mechanisms for observed phenomena

### Hotspot Scoring Methods

#### Method 1: tICA Component Weights
Each tICA component has feature loadings that indicate residue contributions:
```
weight(residue, IC) = |loading(feature_residue, IC)|
```

**Saved as**: `ic*_residue_weights.json`

**Interpretation**: Residues with high weights drive slow motions captured by that IC.

#### Method 2: Anomaly Contribution
For frames with high anomaly scores, identify which residues have:
- High energy stress
- Large geometric deviations
- Proximity to volatile pockets

**Method**: 
1. Select top 10% anomalous frames
2. For each residue, compute mean energy/deviation in those frames
3. Rank residues by contribution

#### Method 3: SASA and Pocket Proximity (Phase 2)
Residues at pocket rims are tracked in `pocket_rims.parquet`:
```
rim_distance(residue, pocket) < 0.8 nm
```

**Interpretation**: These residues control pocket accessibility and may be allosteric switches.

### Visualization of Hotspots

In the interactive viewer (`viewer/app.py`):
- **Color scale**: Blue (low score) → White (medium) → Red (high score)
- **Temporal animation**: Watch hotspots evolve over trajectory
- **Threshold control**: Adjust sensitivity to focus on top hotspots

### Validation Strategies

#### 1. Known Functional Sites
Compare predicted hotspots to:
- Active site residues (from UniProt/literature)
- Allosteric sites (from experiments)
- Mutation studies (ΔΔG, activity changes)

#### 2. Conservation Analysis
- Functionally important residues are often conserved
- Use multiple sequence alignment (MSA)
- Check if predicted hotspots are conserved

#### 3. Experimental Validation
- **Mutagenesis**: Test if hotspot mutations affect function
- **NMR/HDX**: Compare predicted dynamics to experimental order parameters
- **Crystallography**: Check if hotspots overlap with crystallographic B-factors

---

## Dynamic Hotspots: Definition and Biological Importance

### What is a Dynamic Hotspot?

A **dynamic hotspot** is a residue that:
1. **Changes its behavior over time** (not static)
2. **Exhibits anomalous motions** in specific trajectory segments
3. **May transition between different functional roles**

**Contrast with static hotspots**:
- **Static**: Always important (e.g., catalytic triad in enzymes)
- **Dynamic**: Importance varies with conformation (e.g., gatekeeper residues)

### Why Are Dynamic Hotspots Important?

#### 1. Allosteric Regulation
Many proteins are regulated by distant binding events (allostery). Dynamic hotspots form the **communication pathways** that transmit signals.

**Example**: In kinases, an allosteric inhibitor binding >20 Å away can deactivate the active site. Dynamic hotspots trace the pathway.

**Mechanism**:
```
Ligand binds → Dynamic hotspot 1 responds → Signal propagates → 
Dynamic hotspot 2 responds → Active site changes
```

#### 2. Conformational Selection
Proteins exist as **ensembles** of conformations. Ligands select specific substates by stabilizing them.

**Example**: A cryptic pocket appears transiently. When a ligand binds, the pocket-forming residues become static hotspots.

**Dynamic hotspots** are the residues whose behavior changes between unbound and bound ensembles.

#### 3. Enzyme Catalysis
Enzymes achieve rate enhancement through **transition state stabilization**. Dynamic hotspots are residues that:
- Stabilize the transition state
- Destabilize the ground state
- Change protonation states during reaction

**Example**: In serine proteases, the oxyanion hole (dynamic hotspot) stabilizes the tetrahedral intermediate but not the substrate.

#### 4. Drug Discovery
**Targeting dynamic hotspots** offers advantages:
- **Specificity**: Dynamic hotspots may be unique to specific conformational states
- **Allosteric modulation**: Avoid competitive inhibition at active site
- **Resistance**: Mutations at dynamic hotspots are more likely to disrupt function

**Example**: Cryptic pockets that open during dynamics are increasingly exploited as drug targets (e.g., in BCL-XL, K-Ras).

### Types of Dynamic Hotspots

#### 1. Gatekeeper Residues
**Function**: Control access to binding pockets
**Dynamics**: Flip between open/closed states
**Example**: Tyr in kinase ATP pocket

#### 2. Hinge Residues
**Function**: Enable domain movements
**Dynamics**: Undergo large conformational changes
**Example**: Linker residues in multi-domain proteins

#### 3. Allosteric Nodes
**Function**: Relay signals across protein
**Dynamics**: Respond to distant perturbations
**Example**: PDZ domain residues that couple binding to structure

#### 4. Catalytic Assistants
**Function**: Assist chemistry without direct catalytic role
**Dynamics**: Reorganize to position catalytic residues
**Example**: Second-shell residues in enzyme active sites

### How Do We Detect Dynamic Hotspots?

#### Method 1: Time-Varying Anomaly Scores
Residues whose anomaly contribution varies over time:
```
variability(residue) = std(anomaly_contribution(residue, t))
```

High variability → dynamic hotspot

#### Method 2: State-Dependent Analysis
Compare hotspot scores across MSM states:
```
For each state s:
    hotspots(s) = top residues by anomaly in state s
```

Residues that appear in some states but not others are dynamic.

#### Method 3: Pocket Rim Tracking (Phase 2)
Residues that alternate between being at pocket rims and being buried:
```
rim_frequency(residue) = # frames at rim / total frames
```

Residues with intermediate frequency (20-80%) are dynamic pocket formers.

#### Method 4: Energy Volatility
Residues whose energy fluctuates significantly:
```
energy_volatility(residue) = std(energy(residue, t))
```

High volatility suggests dynamic stabilization/destabilization.

### Biological Examples

#### Example 1: GPCR Activation
**Static hotspots**: Conserved DRY motif (always essential)
**Dynamic hotspots**: 
- Ionic lock (breaks during activation)
- Extracellular loop 2 (rearranges to allow ligand entry)
- Helix 6 (moves outward in active state)

**Detection**: Dynamic hotspots show high anomaly scores only in transition frames.

#### Example 2: Kinase Inhibitors
**Static hotspots**: DFG motif, hinge region (always define ATP pocket)
**Dynamic hotspots**:
- Gatekeeper residue (controls type I/II inhibitor access)
- Activation loop (phosphorylation-dependent remodeling)
- Allosteric spine (stabilizes active conformation)

**Detection**: Pocket volatility signal captures activation loop motions.

#### Example 3: Allosteric Enzyme Regulation
**Static hotspots**: Catalytic triad (always needed for chemistry)
**Dynamic hotspots**:
- Allosteric site residues (only important when effector bound)
- Hinge residues (couple allosteric site to active site)
- Interface residues (change upon oligomerization)

**Detection**: State rarity and transition surprise identify allosteric transitions.

---

## Multi-Signal Fusion Strategy

### Why Fusion?

No single signal captures all aspects of protein dynamics. Fusion provides:

1. **Redundancy**: Multiple signals detecting the same event increases confidence
2. **Complementarity**: Different signals detect different types of anomalies
3. **Robustness**: Noise in one signal doesn't dominate final score

### Signal Characteristics

| Signal | Reports On | Timescale | Noise Level |
|--------|-----------|-----------|-------------|
| Rarity | Population statistics | Equilibrium | Low |
| Surprise | Transition kinetics | ~τ_lag (30 frames) | Medium |
| Density | Structural outliers | Instantaneous | Medium |
| Entropy | State uncertainty | ~τ_lag | Medium |
| Energy | Strain/frustration | Instantaneous | High |
| Pocket Vol. | Cavity breathing | ~1 frame | High |

### Fusion Principles

#### 1. Independence Assumption
We assume signals are **approximately independent** conditioned on the true anomaly state.

**Justification**: 
- Kinetic vs. structural vs. energetic are distinct physical properties
- Different computational methods (MSM vs. geometry vs. potentials)
- Different timescales reduce correlations

**Validation**: Check signal correlation matrix (should be < 0.7).

#### 2. Rank-Based Normalization
Each signal is normalized using ranks, not raw values.

**Advantages**:
- Handles different scales (probabilities, distances, energies)
- Robust to outliers (extreme values don't dominate)
- Distribution-free (works for any distribution)

**Formula**:
```
normalized_i(x) = rank(x_i) / (N - 1) ∈ [0, 1]
```

#### 3. Median Fusion
We use **median** (not mean) for robustness.

**Why median?**
- Outlier-resistant (one bad signal doesn't skew result)
- Requires multiple signals to agree (increases specificity)
- Reduces false positives

**Trade-off**: May reduce sensitivity if only one signal detects an anomaly.

**Alternative**: Mean fusion (higher sensitivity, lower specificity).

### Practical Implementation

#### Signal Weights
Currently, all signals have **equal weight**:
```
score = median([s₁, s₂, s₃, s₄, s₅, s₆])
```

**Future enhancement**: Learn optimal weights from data:
```
score = w₁·s₁ + w₂·s₂ + ... + w₆·s₆
```
where weights are optimized to maximize AUC for known functional sites.

#### Conditional Fusion
Some signals may be unavailable:
- Energy features (Phase 2) are optional
- Pocket features (Phase 2) are optional  
- Soft entropy (Phase 3) is optional

**Handling**: Fusion uses only available signals:
```
score = median([available_signals])
```

**Minimum**: At least 3 signals (rarity, surprise, density) are always available.

#### Smoothing Window
Raw scores are smoothed using **moving median** with window size w=5-7:
```
score_smooth(t) = median(score_raw[t-w:t+w])
```

**Effect**:
- Reduces frame-to-frame jitter
- Identifies sustained (not transient) anomalies
- Makes visual interpretation easier

**Trade-off**: Reduces temporal resolution by ~w frames.

---

## Scientific Validation and Best Practices

### Model Validation

#### 1. MSM Validation (Phase 1)

##### Chapman-Kolmogorov Test
Predict transition probabilities at long lag times using short-lag MSM:
```
P(τ_long) ≈ P(τ_short)^k    where k = τ_long/τ_short
```

**Pass criterion**: Predicted and observed match within confidence intervals.

##### Implied Timescales
Plot timescales vs. lag time. Should **plateau** beyond certain lag:
```
τᵢ(lag) ≈ constant for lag > lag_min
```

**Interpretation**: Markov property is satisfied, MSM is valid.

##### Bootstrap Confidence Intervals (Phase 1)
Resample trajectory, rebuild MSM, compute statistics:
```
CI(π, P, MFPT) from B=100 bootstrap iterations
```

**Use**: Assess statistical uncertainty in all MSM-derived quantities.

#### 2. tICA Validation (Phase 1)

##### VAMP-2 Score
Higher scores indicate better separation of slow modes:
```
VAMP-2 = Σᵢ σᵢ²
```

**Optimization**: Grid search over (lag, dim) to maximize VAMP-2.

##### Train/Validation Split
- Fit tICA on 80% of data
- Evaluate VAMP-2 on held-out 20%
- Prevents overfitting

#### 3. Anomaly Detection Validation

##### Known Functional Sites
If available, compare predicted hotspots to:
- Active sites
- Allosteric sites  
- Binding sites
- Mutation data (ΔΔG)

**Metric**: Precision/Recall, AUC-ROC

##### Conservation Analysis
Predicted hotspots should be:
- Enriched in conserved residues (precision)
- Cover most conserved residues (recall)

**Caveat**: Not all hotspots are conserved (some are organism-specific).

##### Visual Inspection
Always inspect top-scoring frames:
- Do they look physically reasonable?
- Are artifacts (clashes, unfolding) excluded?
- Do hotspots localize to known functional regions?

### Best Practices

#### Reproducibility (Phase 1)
- **Seed all random operations**: KMeans, bootstrap, HMM
- **Save configuration**: `run.json` with all parameters
- **Version control**: Git tag for each analysis

#### Computational Efficiency
- **Feature caching**: Hash-based caching for energy/pockets
- **Stride parameter**: Process every Nth frame for exploration
- **Grid spacing**: Coarser grids (0.7-1.0 nm) for pockets

#### Trajectory Preprocessing
- **Alignment**: Remove global rotation/translation (align to reference)
- **Selection**: Use backbone heavy atoms (N, CA, C, O)
- **Filtering**: Remove frames with artifacts (clashes, unfolded)

#### Parameter Selection
- **Lag time**: 10-50 frames (1-5 ns for 100 ps timesteps)
- **Number of states**: 20-50 (balance detail vs. sampling)
- **Window size**: 5-7 frames (0.5-0.7 ns)

#### Interpretation Guidelines
1. **High scores are hypotheses**, not facts
2. **Validate with orthogonal methods** (NMR, mutagenesis, etc.)
3. **Consider context**: Known biology, sequence conservation, structures
4. **Inspect components**: Why is this frame anomalous? (kinetic? energetic?)

---

## References

### Dimensionality Reduction & tICA
1. **Pérez-Hernández, G. et al. (2013)**. "Identification of slow molecular order parameters for Markov model construction." *J. Chem. Phys.* 139(1): 015102.
   - Original tICA paper for molecular dynamics

2. **Schwantes, C.R. & Pande, V.S. (2013)**. "Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9." *J. Chem. Theory Comput.* 9(4): 2000-2009.
   - tICA vs PCA comparison

3. **Wu, H. & Noé, F. (2020)**. "Variational Approach for Learning Markov Processes from Time Series Data." *J. Nonlinear Sci.* 30: 23-66.
   - VAMP theory and score

### Markov State Models
4. **Prinz, J.-H. et al. (2011)**. "Markov models of molecular kinetics: Generation and validation." *J. Chem. Phys.* 134(17): 174105.
   - Comprehensive MSM methodology

5. **Noé, F. et al. (2009)**. "Constructing the equilibrium ensemble of folding pathways from short off-equilibrium simulations." *Proc. Natl. Acad. Sci.* 106(45): 19011-19016.
   - MSM for rare events

6. **Trendelkamp-Schroer, B. et al. (2015)**. "Estimation and uncertainty of reversible Markov models." *J. Chem. Phys.* 143(17): 174101.
   - Bootstrap MSMs for uncertainty quantification

### Energetic Features
7. **Miyazawa, S. & Jernigan, R.L. (1996)**. "Residue-residue potentials with a favorable contact pair term and an unfavorable high packing density term, for simulation and threading." *J. Mol. Biol.* 256(3): 623-644.
   - Knowledge-based potentials

8. **Moal, I.H. & Fernández-Recio, J. (2013)**. "SKEMPI: a Structural Kinetic and Energetic database of Mutant Protein Interactions and its use in empirical models." *Bioinformatics* 29(20): 2637-2644.
   - Energetics and mutations

### Pocket Detection
9. **Schmidtke, P. et al. (2011)**. "MDpocket: open-source cavity detection and characterization on molecular dynamics trajectories." *Bioinformatics* 27(23): 3276-3285.
   - Grid-based pocket detection

10. **Kokh, D.B. et al. (2018)**. "Estimation of drug-target residence times by τ-random acceleration molecular dynamics simulations." *J. Chem. Theory Comput.* 14(7): 3859-3869.
    - Pocket dynamics and drug binding

### Anomaly Detection
11. **Chandola, V. et al. (2009)**. "Anomaly detection: A survey." *ACM Comput. Surv.* 41(3): 1-58.
    - General anomaly detection theory

12. **Aggarwal, C.C. (2017)**. *Outlier Analysis*. 2nd edition. Springer.
    - Comprehensive outlier detection methods

### Bootstrap Methods
13. **Efron, B. & Tibshirani, R.J. (1993)**. *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
    - Bootstrap statistical methods

### Dynamic Allostery
14. **Nussinov, R. & Tsai, C.-J. (2013)**. "Allostery in disease and in drug discovery." *Cell* 153(2): 293-305.
    - Allostery and drug discovery

15. **Tsai, C.-J. et al. (2008)**. "Protein allostery, signal transmission and dynamics: a classification scheme of allosteric mechanisms." *Mol. BioSyst.* 5(3): 207-216.
    - Allosteric mechanisms

### Cryptic Pockets
16. **Beglov, D. et al. (2018)**. "Exploring the structural origins of cryptic sites on proteins." *Proc. Natl. Acad. Sci.* 115(15): E3416-E3425.
    - Cryptic pockets in drug discovery

### Software & Tools
17. **McGibbon, R.T. et al. (2015)**. "MDTraj: A Modern Open Library for the Analysis of Molecular Dynamics Trajectories." *Biophys. J.* 109(8): 1528-1532.
    - MDTraj library

18. **Hoffmann, M. et al. (2021)**. "Deeptime: a Python library for machine learning dynamical models from time series data." *Mach. Learn.: Sci. Technol.* 3(1): 015009.
    - Deeptime library (successor to PyEMMA)

---

## Summary

This pipeline represents a **comprehensive, multi-scale approach** to understanding protein dynamics:

1. **Feature extraction** captures geometry, energetics, and topology
2. **tICA** reduces dimensionality while preserving slow dynamics
3. **MSMs** model kinetics and identify metastable states
4. **Multi-signal anomaly detection** identifies unusual conformations
5. **Hotspot scoring** localizes important residues
6. **Dynamic hotspots** reveal time-varying functional sites

The integration of these methods provides:
- **Scientific rigor**: Grounded in statistical mechanics and machine learning theory
- **Biological insight**: Reveals functional mechanisms and druggable sites
- **Practical utility**: Guides experiments and drug discovery efforts

**Key Innovation**: Unlike static structure analysis, we capture **conformational dynamics** and identify regions whose importance varies over time—the **dynamic hotspots** that often control protein function.

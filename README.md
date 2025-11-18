


# Ensemble-Anomaly-Maps  
Dynamic Hotspot Detection in Molecular Dynamics Trajectories Using Machine Learning and Interactive Visualization

---

## Overview
Ensemble-Anomaly-Maps is a computational biology pipeline designed to detect and visualize dynamic structural anomalies in proteins.  
It combines machine-learning-based motion analysis with interactive molecular visualization to identify regions (residues) that exhibit abnormal movements across molecular dynamics (MD) simulations.

**📖 For comprehensive scientific documentation**, see [SCIENTIFIC_DOCUMENTATION.md](SCIENTIFIC_DOCUMENTATION.md) which explains:
- What we're doing in the ML pipeline and why
- The significance of hotspot scoring
- What dynamic hotspots are and why they matter
- Scientific rationale and validation methods

---

## Pipeline Architecture

### 1. Trajectory Parsing
- **Inputs**
  - `topology.pdb` – static atom and residue definitions  
  - `trajectory.xtc` – MD simulation trajectory (frames over time)
- Parsed using **MDAnalysis** to extract backbone coordinates and residue-wise motion.

### 2. Feature Extraction
- Calculates geometric and dihedral-angle features for every frame.  
- Outputs:
  - `features.npy`, `angles.parquet`, etc.

### 3. Temporal Dimensionality Reduction (tICA)
- Uses **Time-lagged Independent Component Analysis (tICA)** via **PyEMMA** or **deeptime** to capture slow collective motions.  
- Results saved in `data/tica/` as low-dimensional embeddings.

### 4. Anomaly Detection
- Applies unsupervised ML (e.g., One-Class SVM or reconstruction-error models) to detect conformations that deviate from normal motion.  
- Outputs:
  - `frames.json` – per-frame, per-residue anomaly scores  
  - `ic*_residue_weights.json` – residue contributions to each independent component

### 5. Visualization
- A **Trame + VTK**-based interactive viewer (`viewer/app.py`) renders the trajectory in 3D.  
- Residues are dynamically colored based on anomaly intensity (blue → white → red).  
- Includes playback and threshold controls for frame-wise animation.

---

## Repository Structure

ensemble-anomaly-maps/
│
├── viewer/                   # Visualization frontend
│   ├── app.py                # Trame/VTK interactive viewer
│   ├── topology.pdb          # Example topology
│   ├── trajectory.xtc        # Example MD trajectory
│   └── frames.json           # Frame-wise anomaly data
│
├── tools/                    # ML + feature generation scripts
│   ├── generate_features.py
│   ├── run_tica.py
│   └── generate_hotspots.py
│
├── data/                     # Generated data artifacts
│   ├── angles.parquet
│   ├── tica/
│   └── bioemu/
│
└── README.md

---
---

## Workflow Summary

### Basic Pipeline

```bash
# 1. Generate geometric features
python tools/generate_features.py

# 2. Perform tICA projection
python tools/run_tica.py

# 3. Compute anomaly (hotspot) scores
python tools/generate_hotspots.py

# 4. Visualize trajectory interactively
python viewer/app.py
```

### Enhanced Pipeline (Phase 1: Model Selection & Bootstrap)

```bash
# 1. Extract features (if not already done)
# Features should be in data/features.npy

# 2. Run Phase 1: VAMP-2 model selection + Bootstrap MSM
python tools/run_phase1.py --features data/features.npy --output outputs/phase1

# This will:
#   - Select optimal TICA lag and dimensionality via VAMP-2
#   - Compute bootstrap confidence intervals for MSM parameters
#   - Save reproducible run configuration

# 3. Use selected parameters in subsequent analysis
# Best parameters saved in: outputs/phase1/reports/vamp2_best.json
# Bootstrap CIs saved in: outputs/phase1/models/msm_bootstrap/

# See PHASE1.md for detailed documentation
```

### Enhanced Pipeline (Phase 2: Feature Extensions)

```bash
# 1. Generate per-residue energetic features
python tools/generate_energy.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc

# Output: data/derived/residue_energy.parquet
# Columns: frame, res_id, chain, energy, hbonds

# 2. Generate pocket/cavity dynamics features  
python tools/generate_pockets.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc

# Outputs:
#   data/derived/pockets.parquet (volume, mouth_radius, sasa_rim)
#   data/derived/pocket_rims.parquet (residue-pocket mappings)

# Both tools support caching and custom parameters
# See PHASE2.md for detailed documentation
```

### Enhanced Pipeline (Phase 3: Multi-Signal Anomaly Scoring)

```bash
# 1. (Optional) Compute soft state assignments
python tools/train_soft_states.py --dtraj outputs/msm/dtraj.npy

# Outputs:
#   data/derived/soft_dtraj.npy (probabilistic state assignments)
#   data/derived/state_entropy.npy (per-frame entropy)

# 2. Compute enhanced anomaly scores v2
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm \
    --energy data/derived/residue_energy.parquet \
    --pockets data/derived/pockets.parquet \
    --soft_dtraj data/derived/soft_dtraj.npy \
    --state_entropy data/derived/state_entropy.npy

# Outputs:
#   data/derived/frame_scores_v2.csv (multi-signal fused scores)
#   reports/scoring_v2_summary.json (metadata)

# Fuses 6+ signals:
#   - Kinetic: rarity, transition surprise
#   - Structural: local density, soft entropy
#   - Energetic: energy stress, pocket volatility
#
# Features rank/quantile normalization, median/mean fusion,
# and windowed smoothing for robust anomaly detection.
#
# See PHASE3.md for detailed documentation
```

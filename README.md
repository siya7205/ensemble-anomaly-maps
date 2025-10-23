


# Ensemble-Anomaly-Maps  
Dynamic Hotspot Detection in Molecular Dynamics Trajectories Using Machine Learning and Interactive Visualization

---

## Overview
Ensemble-Anomaly-Maps is a computational biology pipeline designed to detect and visualize dynamic structural anomalies in proteins.  
It combines machine-learning-based motion analysis with interactive molecular visualization to identify regions (residues) that exhibit abnormal movements across molecular dynamics (MD) simulations.

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

```bash
# 1. Generate geometric features
python tools/generate_features.py

# 2. Perform tICA projection
python tools/run_tica.py

# 3. Compute anomaly (hotspot) scores
python tools/generate_hotspots.py

# 4. Visualize trajectory interactively
python viewer/app.py

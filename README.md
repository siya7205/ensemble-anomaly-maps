


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
\section*{Workflow Summary}

\begin{verbatim}
# 1. Generate geometric features
python tools/generate_features.py

# 2. Perform tICA projection
python tools/run_tica.py

# 3. Compute anomaly (hotspot) scores
python tools/generate_hotspots.py

# 4. Visualize trajectory interactively
python viewer/app.py
\end{verbatim}

\noindent\rule{\textwidth}{0.4pt}

\section*{Dependencies}

\begin{center}
\begin{tabular}{|p{4cm}|p{9cm}|}
\hline
\textbf{Category} & \textbf{Libraries} \\
\hline
Trajectory processing & MDAnalysis, numpy, pandas \\
\hline
Machine learning & scikit-learn, PyEMMA, deeptime \\
\hline
Visualization & pyvista, vtk, trame, trame-vtk, trame-vuetify \\
\hline
Utilities & json, threading, time, os, argparse \\
\hline
\end{tabular}
\end{center}

\noindent\rule{\textwidth}{0.4pt}

\section*{Integration with Muskan’s Capstone}

Muskan’s capstone provides a stable static PDB visualization frontend built on Flask, Trame, and VTK.  
This project extends it by streaming MD trajectories instead of single structures and applying ML-derived per-frame anomaly coloring.

\subsection*{Integration Goals}
\begin{itemize}
    \item Use this repository as the machine learning and data engine.
    \item Integrate Muskan’s renderer as the visualization frontend.
    \item Enable real-time playback and residue-level dynamic coloring for enhanced scientific visualization.
\end{itemize}

\noindent\rule{\textwidth}{0.4pt}

\section*{Roadmap}

\begin{itemize}
    \item Implement trajectory feature extraction.
    \item Integrate tICA for temporal decomposition.
    \item Add anomaly-detection module.
    \item Prototype Trame/VTK viewer.
    \item Merge Muskan’s visualization frontend for stability.
    \item Add color legend, residue selection, and metastable state overlays.
    \item Export movie snapshots and analysis reports.
\end{itemize}


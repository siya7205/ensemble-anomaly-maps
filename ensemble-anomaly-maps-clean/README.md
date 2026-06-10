# Ensemble Anomaly Maps (Clean Integration Subset)

## Overview
This folder contains the production-focused subset of Ensemble Anomaly Maps needed to run trajectory-driven anomaly detection and generate SciViz artifacts.

## Pipeline architecture
Trajectory
↓
Feature Extraction
↓
tICA
↓
MSM
↓
Anomaly Scoring
↓
Metric Export
↓
Visualizer

## Repository structure
- `configs/` runtime configuration
- `features/` trajectory parsing and feature extraction
- `msm/` tICA and Markov state modeling
- `scoring/` dynamic and residue-level scoring
- `exports/` artifact serialization
- `tools/` end-to-end runner
- `docs/` dependency traceability
- `examples/` runnable command examples

## Installation
1. Create Python 3.9+ environment
2. Install required libraries used by this subset:
   - `numpy`, `pandas`, `scipy`, `scikit-learn`, `deeptime`, `mdtraj`, `pyyaml`

## Running the pipeline
```bash
python tools/run_pipeline.py --config configs/pipeline.yaml
```

## Outputs produced
- `artifacts/features.npy`
- `artifacts/tica_coords.npy`
- `artifacts/dtraj.npy`
- `artifacts/P.npy`
- `artifacts/pi.npy`
- `results/frame_scores_dynamic.csv`
- `results/residue_scores_dynamic.json`
- `results/residue_scores_rmsf.json`
- `results/residue_scores_tica_importance.json`
- `results/hotspots_unified.json`

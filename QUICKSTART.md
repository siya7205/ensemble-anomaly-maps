# QUICKSTART: Running the Ensemble-Anomaly-Maps Pipeline

This guide provides the exact step-by-step terminal commands to run the dynamic hotspot detection pipeline.

---

## Prerequisites

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements_phase1.txt
pip install -r requirements_phase2.txt
pip install -r requirements_phase3.txt
```

### 2. Prepare Your Data

You need:
- **Topology file**: `topology.pdb` (protein structure)
- **Trajectory file**: `trajectory.xtc` (MD simulation)

Example data location in this repo:
- Topology: `data/raw_trajectory/align_topol.pdb`
- Trajectories: `data/raw_trajectory/trajectory_0.xtc` (and others)

---

## Pipeline Execution Sequence

### STEP 1: Generate Features from Trajectory

Extract geometric features from your MD trajectory:

```bash
python tools/extract_features.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --output data/features.npy
```

---

### STEP 2: Run tICA + MSM Pipeline

Perform dimensionality reduction (tICA) and build a Markov State Model (MSM):

```bash
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30
```

**Outputs:**
- `outputs/msm/tica_coords.npy` - tICA-projected coordinates
- `outputs/msm/dtraj.npy` - Discrete state trajectory
- `outputs/msm/P.npy` - Transition probability matrix
- `outputs/msm/pi.npy` - Stationary distribution
- `outputs/msm/frame_scores.csv` - Per-frame anomaly scores

---

### STEP 3: Compute All Metrics (Recommended)

Compute dynamic anomaly, RMSF, and tICA importance scores:

```bash
python tools/compute_all_metrics.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics \
    --normalization percentile \
    --low-percentile 0.05 \
    --high-percentile 0.95
```

**Outputs:**
- `outputs/metrics/hotspots_unified.json` - All metrics for viewer
- `outputs/metrics/residue_scores_dynamic.json` - Dynamic anomaly scores
- `outputs/metrics/residue_scores_rmsf.json` - RMSF/flexibility scores
- `outputs/metrics/residue_scores_tica_importance.json` - Slow-mode importance
- `outputs/metrics/frame_scores_dynamic.csv` - Per-frame scores

---

### STEP 4: Launch the Visualization Server

Start the Flask web server to view results:

```bash
python app/app.py
```

Then open in browser: **http://localhost:5051**

---

## Complete Quick Run (All Steps)

Copy and paste this entire block to run the basic pipeline:

```bash
# Step 1: Create directories
mkdir -p outputs/msm outputs/metrics

# Step 2: Extract features
python tools/extract_features.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --output data/features.npy

# Step 3: Build MSM + TICA model
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 10 \
    --lag_msm 30 \
    --n_clusters 30

# Step 4: Compute unified metrics
python tools/compute_all_metrics.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc \
    --msm_dir outputs/msm \
    --output_dir outputs/metrics

# Step 5: View results
python app/app.py
```

---

## Enhanced Pipeline (Optional)

### Phase 1: Model Selection with VAMP-2

For optimal parameter selection:

```bash
python tools/run_phase1.py \
    --features data/features.npy \
    --output outputs/phase1 \
    --config configs/pipeline.yaml
```

### Phase 2: Energy & Pocket Features

Generate additional energetic and structural features:

```bash
# Energy features
python tools/generate_energy.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc

# Pocket/cavity features  
python tools/generate_pockets.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory_0.xtc
```

### Phase 3: Enhanced Scoring

Use multi-signal anomaly scoring:

```bash
python tools/score_v2.py \
    --features data/features.npy \
    --msm_dir outputs/msm \
    --energy data/derived/residue_energy.parquet \
    --pockets data/derived/pockets.parquet
```

---

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the repository root
cd /path/to/ensemble-anomaly-maps

# Install dependencies
pip install -r requirements_phase1.txt
pip install mdtraj deeptime pyemma scikit-learn pandas numpy
```

### "File not found" errors
```bash
# Check if your data files exist
ls -la data/raw_trajectory/

# Expected files:
#   align_topol.pdb
#   trajectory_0.xtc
```

### MSM convergence issues
```bash
# Try with fewer clusters or different lag times
python tools/run_msm_tica.py \
    --features data/features.npy \
    --out_dir outputs/msm \
    --lag_tica 5 \
    --lag_msm 20 \
    --n_clusters 15
```

---

## Summary of Key Files

| File                                | Purpose                           |
|-------------------------------------|-----------------------------------|
| `tools/extract_features.py`         | Extract features from trajectory  |
| `tools/run_msm_tica.py`             | Build tICA + MSM models           |
| `tools/compute_all_metrics.py`      | Compute all hotspot metrics       |
| `tools/run_phase1.py`               | VAMP-2 model selection            |
| `tools/generate_energy.py`          | Energy features (Phase 2)         |
| `tools/generate_pockets.py`         | Pocket features (Phase 2)         |
| `tools/score_v2.py`                 | Enhanced scoring (Phase 3)        |
| `app/app.py`                        | Visualization web server          |

---

## Next Steps

- See **[USAGE.md](USAGE.md)** for detailed parameter explanations
- See **[PIPELINE_SUMMARY_FOR_BIOCHEMISTS.md](PIPELINE_SUMMARY_FOR_BIOCHEMISTS.md)** for scientific background
- See **[PHASE1.md](PHASE1.md)**, **[PHASE2.md](PHASE2.md)**, **[PHASE3.md](PHASE3.md)** for advanced features

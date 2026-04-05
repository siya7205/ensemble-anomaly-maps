# Case Study — `case_study_10ns_1001frames`

This document explains how to run the existing ensemble-anomaly-maps pipeline
on the 10 ns / 1001-frame case study trajectory and consume the outputs in the
ASVS visualizer.

---

## How this case study is configured

| Item | Value |
|------|-------|
| Case study ID | `case_study_10ns_1001frames` |
| Topology | `data/case_study_10ns_1001frames/input/md.gro` |
| Trajectory | `data/case_study_10ns_1001frames/input/md.xtc` |
| Parameters | `configs/case_study_10ns_1001frames.yaml` |
| Runner | `run_case_study.py` |

`run_case_study.py` is a thin wrapper around the existing `run_pipeline()`
function in `run_all_proteins.py`.  It adds no new logic — it simply wires the
case study inputs to the standard pipeline and writes outputs to the standard
directory layout.

---

## How to run

From the repository root, run:

```bash
python run_case_study.py
```

To override any default parameter:

```bash
python run_case_study.py \
    --lag_tica 10 \
    --dim_tica 5 \
    --n_clusters 20 \
    --lag_msm 10 \
    --k_neighbors 10 \
    --window 5 \
    --seed 42
```

Run `python run_case_study.py --help` to see all options.

### Dependencies

Install the required Python packages before running:

```bash
pip install -r requirements_phase2.txt   # MDTraj, deeptime, scikit-learn, …
```

---

## Where outputs go

The pipeline writes to the repository's standard top-level directories,
under a `case_study_10ns_1001frames/` subdirectory in each:

```
artifacts/case_study_10ns_1001frames/
├── tica_coords.npy      – tICA embedding  (T × dim)
├── dtraj.npy            – discrete state trajectory  (T,)
├── P.npy                – MSM transition matrix  (S × S)
└── pi.npy               – MSM stationary distribution  (S,)

results/case_study_10ns_1001frames/
├── frame_scores_dynamic.csv      – per-frame anomaly scores + signal components
└── residue_scores_dynamic.json   – per-residue anomaly scores

exports/case_study_10ns_1001frames/
├── hotspots_residue.json    ← visualizer input
├── anomaly_residue.json     ← visualizer input
├── rmsf_residue.json        ← visualizer input
├── tica_importance.json     ← visualizer input
├── topology.gro             ← visualizer input
└── trajectory.xtc           ← visualizer input

outputs/case_study_10ns_1001frames/
├── run_metadata.json        – full run record (inputs, parameters, output paths)
└── pipeline_summary.txt     – human-readable summary of all produced files
```

---

## What each output file contains

| File | Description |
|------|-------------|
| `tica_coords.npy` | tICA-projected coordinates for every frame |
| `dtraj.npy` | KMeans cluster assignment per frame (discrete trajectory) |
| `P.npy` | MSM row-stochastic transition matrix |
| `pi.npy` | MSM stationary distribution (equilibrium occupancy per state) |
| `frame_scores_dynamic.csv` | Per-frame anomaly score (0–100) + individual signal components (rarity, transition\_surprise, local\_density) |
| `residue_scores_dynamic.json` | Per-residue combined anomaly score (RMSF × mean frame score) |
| `hotspots_residue.json` | Per-frame × per-residue hotspot scores in ASVS format |
| `anomaly_residue.json` | Per-frame × per-residue anomaly scores in ASVS format |
| `rmsf_residue.json` | Per-residue RMSF (normalised flexibility) in ASVS format |
| `tica_importance.json` | Per-residue contribution to slow collective motions in ASVS format |
| `run_metadata.json` | JSON record of all inputs, parameters, and output paths |
| `pipeline_summary.txt` | Plain-text listing of every produced file |

---

## What to load into the visualizer

Copy or point the ASVS viewer at the `exports/case_study_10ns_1001frames/`
directory.  The visualizer-facing files are:

```
exports/case_study_10ns_1001frames/
├── hotspots_residue.json
├── anomaly_residue.json
├── rmsf_residue.json
├── tica_importance.json
├── topology.gro
└── trajectory.xtc
```

Start the viewer:

```bash
cd app        # or wherever app.py lives
python app.py
```

Then open `http://localhost:5000/viewer` and load the files above.

---

## Pipeline steps (summary)

| Step | What happens | Key module |
|------|-------------|------------|
| 1 | Feature extraction (RMSD, Rg, contacts, φ/ψ dihedrals) | `features/compute_md_features.py` |
| 2 | tICA dimensionality reduction | `run_all_proteins.py::run_tica()` |
| 3 | KMeans clustering → discrete trajectory | `run_all_proteins.py::cluster_states()` |
| 4 | Maximum-likelihood reversible MSM | `run_all_proteins.py::build_msm()` |
| 5 | Anomaly scoring (rarity + transition surprise + local density) | `scoring/anomaly_v2.py` |
| 6 | Per-residue aggregation (RMSF blend) | `run_all_proteins.py::compute_residue_scores()` |
| 7 | ASVS JSON export | `tools/export_for_asvs.py` |

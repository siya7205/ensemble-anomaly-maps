# Changelog

All notable changes to the **ensemble-anomaly-maps** project are documented here.

---

## [Unreleased] — batch runner & code-quality improvements

### Added

#### `batch_runner.py` — End-to-End Batch Runner (new file)

A single, beginner-friendly entry point that automates the entire pipeline for
one or more PDB targets without any manual steps.

**What it does (4 automated stages):**

| Stage | What happens |
|-------|-------------|
| **1 — Download** | Fetches `topology.pdb` from RCSB for each PDB ID (default: `9UNN`, `9O6O`, `1CRN`). Skips proteins already on disk. |
| **2 — Toy MD** | Runs a short OpenMM simulation (AMBER14 + implicit GBn2 solvent, no explicit water box) to produce `traj.xtc`. Fast enough to finish on CPU in under a minute per protein. |
| **3 — ML Pipeline** | Delegates to `run_all_proteins.py` — runs tICA → KMeans clustering → Markov State Model → anomaly scoring — writing per-protein outputs to isolated directories so proteins never overwrite each other. |
| **4 — ASVS Export** | Calls `tools/export_for_asvs.py` and copies topology + trajectory alongside the JSON so the ASVS viewer is self-contained. |

**Output directory layout:**

```
data/<PDB_ID>/
  topology.pdb          ← downloaded from RCSB
  traj.xtc              ← toy MD trajectory from OpenMM

artifacts/<PDB_ID>/
  tica_coords.npy       ← tICA-projected coordinates
  dtraj.npy             ← discrete state trajectory
  P.npy                 ← MSM transition matrix
  pi.npy                ← stationary distribution

results/<PDB_ID>/
  frame_scores_dynamic.csv      ← per-frame anomaly scores
  residue_scores_dynamic.json   ← per-residue anomaly scores

exports/<PDB_ID>/
  hotspots_residue.json   ┐
  anomaly_residue.json    │ ASVS-compatible JSON files
  rmsf_residue.json       │
  tica_importance.json    ┘
  topology.pdb            ← copy for the ASVS viewer
  trajectory.xtc          ← copy for the ASVS viewer
```

**Usage examples:**

```bash
# Run all three default proteins end-to-end
python batch_runner.py

# Custom PDB IDs and output directory
python batch_runner.py --pdb_ids 1CRN 4LZT --work_dir /tmp/batch

# Skip steps already completed
python batch_runner.py --skip_download --skip_md

# Generate more trajectory frames for a richer signal
python batch_runner.py --md_frames 300 --md_steps_per_frame 500
```

**Key flags:**

| Flag | Purpose |
|------|---------|
| `--pdb_ids` | One or more PDB IDs to process |
| `--work_dir` | Root directory for downloaded PDB files and trajectories |
| `--md_frames` | Number of frames in the toy trajectory (default: 100) |
| `--md_steps_per_frame` | Integration steps between saved frames (default: 250) |
| `--skip_download` | Skip RCSB download (reuse existing `topology.pdb`) |
| `--skip_md` | Skip OpenMM MD (reuse existing `traj.xtc`) |
| `--skip_pipeline` | Skip ML pipeline (reuse existing results) |
| `--skip_export` | Skip ASVS JSON export |

---

### Changed

#### `batch_runner.py` — Code-quality fixes (code review pass)

After the initial implementation, the following improvements were applied:

1. **Imports moved to module level** — `import tempfile`, `import os`, and
   `import pandas as pd` were moved from inside functions to the top of the
   file, following Python's standard import convention.

2. **Safer temporary-file handling** — Replaced `NamedTemporaryFile(delete=False)`
   + manual `close()` with `tempfile.mkstemp()` wrapped in a `try/finally`
   block. This ensures the intermediate DCD file is always deleted even when an
   exception occurs during MD integration or XTC conversion.

3. **Removed duplicate import** — A redundant `from openmm import app as omm_app`
   inside `_best_openmm_platform()` was removed; the function now only imports
   `openmm` itself, which is all it needs.

---

## Summary of files changed in this PR

| File | Status | Description |
|------|--------|-------------|
| `batch_runner.py` | **Added** | New end-to-end batch runner (stages 1–4 above) |
| `CHANGELOG.md` | **Added** | This file |

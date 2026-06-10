# Pipeline Traceability

## Dependency chain

`topology.pdb + trajectory.xtc`
→ `features/compute_md_features.py::compute_features`
→ `artifacts/features.npy`
→ `msm/pipeline.py::run_tica`
→ `artifacts/tica_coords.npy`
→ `msm/pipeline.py::cluster_states`
→ `artifacts/dtraj.npy`
→ `msm/pipeline.py::build_msm`
→ `artifacts/P.npy + artifacts/pi.npy`
→ `scoring/anomaly_v2.py::compute_kinetic_signals`
→ `scoring/anomaly_v2.py::compute_local_density_signal`
→ `scoring/anomaly_v2.py::fuse_signals`
→ `scoring/anomaly_v2.py::moving_median`
→ `scoring/signals.py::compute_rmsf_scores`
→ `scoring/signals.py::compute_tica_importance_scores`
→ `scoring/signals.py::aggregate_frame_to_residue`
→ `exports/artifact_export.py::export_metric_artifacts`
→ `results/frame_scores_dynamic.csv`
→ `results/residue_scores_dynamic.json`
→ `results/residue_scores_rmsf.json`
→ `results/residue_scores_tica_importance.json`
→ `results/hotspots_unified.json`

## Retained file dependency map

- `tools/run_pipeline.py`
  - Inputs: `configs/pipeline.yaml`, `topology.pdb`, `trajectory.xtc`
  - Calls: `features`, `msm`, `scoring`, `exports`
  - Generates: full artifact set in configured `artifacts_dir` and `results_dir`

- `features/compute_md_features.py`
  - Inputs: topology + trajectory
  - Called by: `tools/run_pipeline.py`
  - Generates: in-memory feature dictionary, then `features.npy`

- `msm/pipeline.py`
  - Inputs: feature matrix
  - Called by: `tools/run_pipeline.py`
  - Generates: `tica_coords.npy`, `dtraj.npy`, `P.npy`, `pi.npy`

- `scoring/anomaly_v2.py`
  - Inputs: MSM model + `dtraj` + `tica_coords`
  - Called by: `tools/run_pipeline.py`
  - Generates: dynamic frame scores + per-signal components

- `scoring/signals.py`
  - Inputs: topology + trajectory + tICA model + frame scores
  - Called by: `tools/run_pipeline.py`
  - Generates: RMSF, tICA importance, per-residue dynamic aggregates

- `exports/artifact_export.py`
  - Inputs: frame and residue score structures
  - Called by: `tools/run_pipeline.py`
  - Generates: final integration artifacts including `hotspots_unified.json`

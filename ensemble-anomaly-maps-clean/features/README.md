# Features Stage

## Purpose
Parse trajectories and compute ML-ready molecular descriptors.

## Inputs
- `topology.pdb` (or compatible topology)
- `trajectory.xtc` (or compatible trajectory)

## Outputs
- `features.npy` (frame × feature matrix)

## Core algorithms
- RMSD to reference frame
- Radius of gyration
- CA-contact counts
- Mean sin/cos backbone dihedrals

## Important files
- `features/compute_md_features.py`

## Example execution
`python tools/run_pipeline.py --config configs/pipeline.yaml`

## Connection to next stage
Produces feature matrix consumed by `msm/pipeline.py` for tICA and clustering.

# Scoring Stage

## Purpose
Compute dynamic anomaly signals and aggregate per-residue metrics.

## Inputs
- `dtraj.npy`, `P.npy`, `pi.npy`, `tica_coords.npy`
- `topology.pdb`, `trajectory.xtc`

## Outputs
- Dynamic frame-score series
- Per-residue dynamic, RMSF, and tICA-importance scores

## Core algorithms
- State rarity and transition surprise
- kNN-based local density signal
- Rank/percentile/zscore normalization
- Median signal fusion and temporal smoothing

## Important files
- `scoring/anomaly_v2.py`
- `scoring/signals.py`

## Example execution
`python tools/run_pipeline.py --config configs/pipeline.yaml`

## Connection to next stage
Normalized scoring dictionaries are exported as integration artifacts in `exports/`.

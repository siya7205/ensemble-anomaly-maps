# MSM Stage

## Purpose
Generate tICA coordinates, cluster states, and construct an MSM kinetic model.

## Inputs
- `features.npy`

## Outputs
- `tica_coords.npy`
- `dtraj.npy`
- `P.npy`
- `pi.npy`

## Core algorithms
- tICA projection (`deeptime.decomposition.TICA`)
- KMeans state discretization (`deeptime.clustering.KMeans`)
- Reversible MLE MSM (`deeptime.markov.msm.MaximumLikelihoodMSM`)

## Important files
- `msm/pipeline.py`

## Example execution
`python tools/run_pipeline.py --config configs/pipeline.yaml`

## Connection to next stage
MSM and state trajectory feed anomaly scoring in `scoring/`.

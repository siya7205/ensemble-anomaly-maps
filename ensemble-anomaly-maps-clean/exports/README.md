# Exports Stage

## Purpose
Serialize scoring outputs into integration-ready artifact files.

## Inputs
- Frame scores and signal components
- Dynamic residue scores
- RMSF residue scores
- tICA-importance residue scores

## Outputs
- `frame_scores_dynamic.csv`
- `residue_scores_dynamic.json`
- `residue_scores_rmsf.json`
- `residue_scores_tica_importance.json`
- `hotspots_unified.json`

## Core algorithms
- Structured CSV/JSON serialization
- Unified hotspot payload assembly

## Important files
- `exports/artifact_export.py`

## Example execution
`python tools/run_pipeline.py --config configs/pipeline.yaml`

## Connection to next stage
Artifacts are consumed directly by SciViz visualizer timelines and residue color channels.

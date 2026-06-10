# README_INTEGRATION

This document is for SciViz developers consuming ML outputs.

## Required pipeline inputs
- `topology.pdb`: structure/topology for residue indexing
- `trajectory.xtc`: frame-ordered MD trajectory

## Produced artifacts and visualizer mapping

### `frame_scores_dynamic.csv`
- Use for timeline view
- Use for anomaly peak detection
- Use for frame navigation synchronization

### `residue_scores_dynamic.json`
- Use for dynamic anomaly residue channel
- Use for anomaly-focused coloring overlays

### `residue_scores_rmsf.json`
- Use for flexibility/stability residue channel
- Use for context layer against dynamic anomalies

### `residue_scores_tica_importance.json`
- Use for slow-mode importance residue channel
- Use for allosteric-motion inspection

### `hotspots_unified.json`
- Use as primary residue coloring payload
- Use for channel switching (`dynamic_anomaly`, `rmsf`, `tica_importance`)
- Use for hotspot inspection panels and residue drill-down

## Consumption notes
- Residue keys are strings and should be cast to integer indices as needed.
- `frame` indices in CSV are contiguous and aligned to trajectory frames.
- Use `hotspots_unified.json` as source-of-truth for per-residue multi-channel rendering.

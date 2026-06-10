"""
===============================================================
Module: exports/artifact_export.py
Description: Export aggregated scoring artifacts for visualization integration.
Author: Siya Jethliya
Project: Ensemble Anomaly Maps
===============================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------
# Build Unified Hotspot Payload
# ------------------------------
def _build_unified_payload(frame_scores, dynamic_scores, rmsf_scores, tica_scores, normalization):
    return {
        "meta": {
            "n_frames": int(len(frame_scores)),
            "n_residues": int(max(len(dynamic_scores), len(rmsf_scores), len(tica_scores))),
            "metrics": ["dynamic_anomaly", "rmsf", "tica_importance"],
            "normalization": normalization,
        },
        "per_residue": {
            "dynamic_anomaly": {str(k): float(v) for k, v in dynamic_scores.items()},
            "rmsf": {str(k): float(v) for k, v in rmsf_scores.items()},
            "tica_importance": {str(k): float(v) for k, v in tica_scores.items()},
        },
    }


# ------------------------------
# Write Export Artifacts
# ------------------------------
def export_metric_artifacts(results_dir, frame_scores, signal_components, dynamic_scores, rmsf_scores, tica_scores, normalization="percentile"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    frame_df = pd.DataFrame({"frame": np.arange(len(frame_scores)), "score": frame_scores, "score_dynamic": frame_scores})
    for signal_name, signal_values in signal_components.items():
        frame_df[f"component_{signal_name}"] = signal_values
    frame_df.to_csv(results_dir / "frame_scores_dynamic.csv", index=False)

    with open(results_dir / "residue_scores_dynamic.json", "w") as handle:
        json.dump({str(k): float(v) for k, v in dynamic_scores.items()}, handle, indent=2)

    with open(results_dir / "residue_scores_rmsf.json", "w") as handle:
        json.dump({str(k): float(v) for k, v in rmsf_scores.items()}, handle, indent=2)

    with open(results_dir / "residue_scores_tica_importance.json", "w") as handle:
        json.dump({str(k): float(v) for k, v in tica_scores.items()}, handle, indent=2)

    unified_payload = _build_unified_payload(
        frame_scores,
        dynamic_scores,
        rmsf_scores,
        tica_scores,
        normalization,
    )
    with open(results_dir / "hotspots_unified.json", "w") as handle:
        json.dump(unified_payload, handle, indent=2)

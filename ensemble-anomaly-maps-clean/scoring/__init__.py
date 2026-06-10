"""
===============================================================
Module: scoring/__init__.py
Description: Dynamic anomaly scoring and residue metric aggregation package.
Author: Siya Jethliya
Project: Ensemble Anomaly Maps
===============================================================
"""

from .anomaly_v2 import (
    compute_kinetic_signals,
    compute_local_density_signal,
    fuse_signals,
    moving_median,
)
from .signals import (
    compute_rmsf_scores,
    compute_tica_importance_scores,
    normalize_scores,
    aggregate_frame_to_residue,
)

__all__ = [
    "compute_kinetic_signals",
    "compute_local_density_signal",
    "fuse_signals",
    "moving_median",
    "compute_rmsf_scores",
    "compute_tica_importance_scores",
    "normalize_scores",
    "aggregate_frame_to_residue",
]

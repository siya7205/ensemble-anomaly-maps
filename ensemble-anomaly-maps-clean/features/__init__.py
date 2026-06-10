"""
===============================================================
Module: features/__init__.py
Description: Feature extraction package for trajectory-derived ML inputs.
Author: Siya Jethliya
Project: Ensemble Anomaly Maps
===============================================================
"""

from .compute_md_features import compute_features, features_to_matrix

__all__ = ["compute_features", "features_to_matrix"]

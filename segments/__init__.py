"""
Trajectory segmentation module.
"""
from .segmenter import segment_by_turning_angle, fixed_windows

__all__ = ['segment_by_turning_angle', 'fixed_windows']

#!/usr/bin/env python3
"""
Trajectory segmentation methods.

Provides methods to segment MD trajectories into windows:
- Fixed-size sliding windows
- Turning angle-based dynamic segmentation
"""
import numpy as np

# Numerical tolerance for zero vector detection
NORM_TOLERANCE = 1e-12


def fixed_windows(n_frames, win=250, overlap=50):
    """
    Create fixed-size overlapping windows.
    
    Args:
        n_frames: Total number of frames
        win: Window size
        overlap: Overlap between consecutive windows
        
    Returns:
        segments: List of (start, end) tuples
    """
    segments = []
    step = win - overlap
    
    start = 0
    while start < n_frames:
        end = min(start + win - 1, n_frames - 1)
        segments.append((start, end))
        start += step
        
        # Avoid tiny final segments
        if n_frames - start < win // 2 and start < n_frames:
            # Extend last segment to include remaining frames
            if segments:
                last_start, _ = segments[-1]
                segments[-1] = (last_start, n_frames - 1)
            break
    
    return segments


def compute_turning_angles(X):
    """
    Compute turning angles along a trajectory in feature space.
    
    The turning angle at frame t is the angle between:
    - Vector from frame t-1 to t
    - Vector from frame t to t+1
    
    Args:
        X: Feature matrix (n_frames x n_features)
        
    Returns:
        angles: Turning angles in degrees (n_frames - 2,)
    """
    n_frames = X.shape[0]
    
    if n_frames < 3:
        return np.array([])
    
    angles = np.zeros(n_frames - 2)
    
    for t in range(1, n_frames - 1):
        v1 = X[t] - X[t - 1]
        v2 = X[t + 1] - X[t]
        
        # Handle zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < NORM_TOLERANCE or norm2 < NORM_TOLERANCE:
            # Zero vector: no movement, treat as not a boundary (high angle = straight)
            angles[t - 1] = 180.0
            continue
        
        # Compute cosine of angle
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angles[t - 1] = np.degrees(np.arccos(cos_angle))
    
    return angles


def segment_by_turning_angle(X, angle_thresh_deg=135.0, min_len=50, max_len=2000):
    """
    Segment trajectory based on turning angles in feature space.
    
    A segment boundary is placed where the turning angle drops below
    the threshold (indicating a sharp change in direction).
    
    Args:
        X: Feature matrix (n_frames x n_features)
        angle_thresh_deg: Threshold angle in degrees (lower = sharper turn = boundary)
        min_len: Minimum segment length
        max_len: Maximum segment length (forced split if exceeded)
        
    Returns:
        segments: List of (start, end) tuples
    """
    n_frames = X.shape[0]
    
    if n_frames < 3:
        return [(0, n_frames - 1)]
    
    # Compute turning angles
    angles = compute_turning_angles(X)
    
    # Find boundary candidates (low angle = sharp turn)
    # angles[i] corresponds to frame i+1
    boundary_mask = angles < angle_thresh_deg
    
    # Build segments
    segments = []
    seg_start = 0
    
    for i, is_boundary in enumerate(boundary_mask):
        frame = i + 1  # angles[i] corresponds to frame i+1
        seg_len = frame - seg_start
        
        # Check for forced boundary (max length exceeded)
        if seg_len >= max_len:
            segments.append((seg_start, frame - 1))
            seg_start = frame
            continue
        
        # Check for detected boundary
        if is_boundary and seg_len >= min_len:
            segments.append((seg_start, frame - 1))
            seg_start = frame
    
    # Add final segment
    if seg_start < n_frames:
        segments.append((seg_start, n_frames - 1))
    
    # Merge tiny trailing segments
    if len(segments) > 1:
        last_start, last_end = segments[-1]
        if last_end - last_start + 1 < min_len:
            prev_start, _ = segments[-2]
            segments = segments[:-2]
            segments.append((prev_start, last_end))
    
    return segments

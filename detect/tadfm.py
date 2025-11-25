#!/usr/bin/env python3
"""
TAD-FM style anomaly detection.

Implements segment-based anomaly detection using:
- Shallow statistics per segment
- DBSCAN clustering with cosine distance
- Anomaly scoring based on cluster membership
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


def segment_shallow_stats(X, segments):
    """
    Compute shallow statistics for each segment.
    
    Statistics computed per segment:
    - Mean of each feature
    - Std of each feature
    - Min and max of each feature
    - Segment length
    
    Args:
        X: Feature matrix (n_frames x n_features)
        segments: List of (start, end) tuples
        
    Returns:
        stats: Statistics matrix (n_segments x n_stats)
    """
    n_features = X.shape[1]
    stats_list = []
    
    for start, end in segments:
        seg_data = X[start:end+1]
        
        # Basic statistics
        seg_mean = seg_data.mean(axis=0)
        seg_std = seg_data.std(axis=0)
        seg_min = seg_data.min(axis=0)
        seg_max = seg_data.max(axis=0)
        seg_len = np.array([end - start + 1])
        
        # Combine all stats
        seg_stats = np.concatenate([seg_mean, seg_std, seg_min, seg_max, seg_len])
        stats_list.append(seg_stats)
    
    return np.array(stats_list)


def dbscan_cosine(X, eps=0.3, min_samples=10):
    """
    Perform DBSCAN clustering with cosine distance.
    
    Points in small clusters or noise are considered anomalous.
    
    Args:
        X: Data matrix (n_samples x n_features)
        eps: DBSCAN epsilon (distance threshold)
        min_samples: Minimum samples for core point
        
    Returns:
        labels: Cluster labels (-1 for noise)
        scores: Anomaly scores (0-100, higher = more anomalous)
    """
    n_samples = X.shape[0]
    
    if n_samples < min_samples:
        # Not enough samples for clustering
        return np.full(n_samples, -1), np.full(n_samples, 100.0)
    
    # Normalize for cosine distance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_norm = X_scaled / norms
    
    # Compute cosine distance matrix
    # cosine_distance = 1 - cosine_similarity
    cos_sim = X_norm @ X_norm.T
    cos_dist = 1 - cos_sim
    cos_dist = np.clip(cos_dist, 0, 2)  # Numerical stability
    
    # DBSCAN with precomputed distance
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(cos_dist)
    
    # Compute anomaly scores based on cluster properties
    scores = compute_cluster_anomaly_scores(X_norm, labels, cos_dist)
    
    return labels, scores


def compute_cluster_anomaly_scores(X, labels, distance_matrix):
    """
    Compute anomaly scores based on cluster membership and distances.
    
    Scoring criteria:
    - Noise points (label=-1) get high scores
    - Points in small clusters get higher scores
    - Points far from cluster center get higher scores
    
    Args:
        X: Normalized data matrix
        labels: Cluster labels
        distance_matrix: Precomputed distance matrix
        
    Returns:
        scores: Anomaly scores (0-100)
    """
    n_samples = len(labels)
    scores = np.zeros(n_samples)
    
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    
    if n_clusters == 0:
        # All noise
        return np.full(n_samples, 100.0)
    
    # Compute cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        if label >= 0:
            cluster_sizes[label] = np.sum(labels == label)
    
    total_clustered = sum(cluster_sizes.values())
    
    for i in range(n_samples):
        label = labels[i]
        
        if label == -1:
            # Noise point - high anomaly score
            # Base score of 80, plus distance-based adjustment
            nearest_clustered = np.min([
                distance_matrix[i, j] 
                for j in range(n_samples) if labels[j] >= 0
            ]) if total_clustered > 0 else 1.0
            scores[i] = min(100.0, 80.0 + nearest_clustered * 20.0)
        else:
            # Clustered point
            cluster_size = cluster_sizes[label]
            
            # Small cluster penalty
            size_score = 1.0 - (cluster_size / total_clustered)
            
            # Distance from cluster center
            cluster_members = labels == label
            cluster_center = X[cluster_members].mean(axis=0)
            dist_to_center = np.linalg.norm(X[i] - cluster_center)
            
            # Combine scores
            scores[i] = (size_score * 50.0 + dist_to_center * 30.0)
    
    # Scale to 0-100
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min()) * 100.0
    
    return scores

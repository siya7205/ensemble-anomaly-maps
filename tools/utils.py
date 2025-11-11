"""
Optimized utility functions for the ML pipeline.
Consolidates common operations and provides performance optimizations.
"""
import numpy as np
from functools import lru_cache
from pathlib import Path
import hashlib
import pickle


def minmax_normalize(x, clip=True):
    """
    Fast min-max normalization with optional clipping.
    
    Args:
        x: Input array
        clip: Whether to use robust normalization with quantiles
    
    Returns:
        Normalized array in [0, 1] range
    """
    x = np.asarray(x, dtype=np.float64)
    
    if clip:
        # Use quantiles for robust normalization
        q1, q99 = np.quantile(x, [0.01, 0.99])
        return np.clip((x - q1) / (q99 - q1 + 1e-12), 0, 1)
    else:
        # Standard min-max normalization
        mn, mx = np.nanmin(x), np.nanmax(x)
        return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn)


def map_to_active_set(dtraj, active_set, n_clusters):
    """
    Vectorized mapping of cluster labels to active set indices.
    
    Args:
        dtraj: Original discrete trajectory (cluster labels)
        active_set: Active state indices from MSM
        n_clusters: Total number of clusters
    
    Returns:
        Mapped trajectory with -1 for inactive states
    """
    mapping = -np.ones(n_clusters, dtype=np.int64)
    active_set = np.asarray(active_set, dtype=np.int64)
    mapping[active_set] = np.arange(len(active_set))
    return mapping[np.asarray(dtraj, dtype=np.int64)]


def compute_transition_surprise(dtraj, transition_matrix, lag, active_mask):
    """
    Vectorized computation of transition surprise scores.
    
    Args:
        dtraj: Discrete trajectory (mapped to active set)
        transition_matrix: MSM transition matrix
        lag: Lag time for transitions
        active_mask: Boolean mask indicating frames in active set
    
    Returns:
        Array of transition surprise scores (smeared over lag time)
    """
    T = len(dtraj)
    surprise = np.zeros(T, dtype=np.float64)
    
    if T <= lag:
        return surprise
    
    # Identify valid transition pairs
    valid_pairs = active_mask[:-lag] & active_mask[lag:]
    
    if not np.any(valid_pairs):
        return surprise
    
    # Vectorized transition probability lookup
    idx_from = np.nonzero(valid_pairs)[0]
    states_from = dtraj[idx_from]
    states_to = dtraj[idx_from + lag]
    
    # Compute transition probabilities and surprises
    trans_probs = transition_matrix[states_from, states_to]
    jump_costs = -np.log(np.maximum(trans_probs, 1e-12))
    
    # Smear surprise scores over lag interval (vectorized)
    for t0, cost in zip(idx_from, jump_costs):
        surprise[t0:t0 + lag] += cost / lag
    
    return surprise


def load_features_cached(feat_path, cache_dir=None):
    """
    Load features with optional disk caching.
    
    Args:
        feat_path: Path to feature file (.npy or .csv)
        cache_dir: Optional directory for caching loaded features
    
    Returns:
        Feature array (T x F)
    """
    feat_path = Path(feat_path)
    
    # Compute cache key based on file path and modification time
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        stat = feat_path.stat()
        cache_key = hashlib.md5(
            f"{feat_path.absolute()}_{stat.st_mtime}_{stat.st_size}".encode()
        ).hexdigest()
        cache_file = cache_dir / f"features_{cache_key}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass  # Cache miss or corrupted, reload
    
    # Load features
    if feat_path.suffix == '.npy':
        X = np.load(feat_path)
    else:
        X = np.loadtxt(feat_path, delimiter=",", skiprows=1)
    
    # Save to cache if requested
    if cache_dir:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(X, f)
        except Exception:
            pass  # Cache write failed, not critical
    
    return X


def compute_local_density(Y, n_neighbors=20, metric='euclidean'):
    """
    Efficient local density computation using k-NN.
    
    Args:
        Y: Feature space coordinates (T x d)
        n_neighbors: Number of neighbors to consider
        metric: Distance metric to use
    
    Returns:
        Negative mean distance to k nearest neighbors (higher = denser)
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_neighbors = min(n_neighbors, len(Y) - 1)
    if n_neighbors < 1:
        return np.zeros(len(Y))
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    nbrs.fit(Y)
    distances, _ = nbrs.kneighbors(Y)
    
    # Return negative distance (so higher values = higher density)
    return -distances[:, -1]


def compute_frame_scores(rarity, surprise, density):
    """
    Combine multiple anomaly signals into final frame scores.
    Uses median of normalized signals for robustness.
    
    Args:
        rarity: State rarity scores
        surprise: Transition surprise scores
        density: Local density scores
    
    Returns:
        Combined frame scores in [0, 100] range
    """
    # Normalize each component
    r_norm = minmax_normalize(rarity, clip=True)
    s_norm = minmax_normalize(surprise, clip=True)
    d_norm = minmax_normalize(-density, clip=True)  # Lower density = higher anomaly
    
    # Stack and take median (robust to outliers)
    stacked = np.vstack([r_norm, s_norm, d_norm])
    median_score = np.median(stacked, axis=0)
    
    # Scale to [0, 100]
    return 100.0 * median_score


class ProgressBar:
    """Simple progress bar for long-running operations."""
    
    def __init__(self, total, desc="Progress", width=40):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
    
    def update(self, n=1):
        """Update progress by n steps."""
        self.current = min(self.current + n, self.total)
        self._render()
    
    def _render(self):
        """Render progress bar."""
        if self.total == 0:
            return
        
        frac = self.current / self.total
        filled = int(self.width * frac)
        bar = '=' * filled + '-' * (self.width - filled)
        percent = 100 * frac
        
        print(f'\r{self.desc}: [{bar}] {percent:.1f}% ({self.current}/{self.total})', 
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete

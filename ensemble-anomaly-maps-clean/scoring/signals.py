"""
===============================================================
Module: scoring/signals.py
Description: RMSF, tICA-importance, normalization, and residue aggregation logic.
Author: Siya Jethliya
Project: Ensemble Anomaly Maps
===============================================================
"""

import numpy as np


# ------------------------------
# Compute RMSF Scores
# ------------------------------
def compute_rmsf_scores(topology_path, trajectory_path, selection="name CA", align_selection="name CA"):
    import mdtraj as md

    traj = md.load(str(trajectory_path), top=str(topology_path))
    atoms = traj.topology.select(selection)
    if len(atoms) == 0:
        raise ValueError(f"No atoms found with selection '{selection}'")

    align_atoms = traj.topology.select(align_selection)
    traj.superpose(traj, frame=0, atom_indices=align_atoms)

    positions = traj.xyz[:, atoms, :]
    mean_position = positions.mean(axis=0)
    squared_dev = np.sum((positions - mean_position) ** 2, axis=2)
    rmsf_atoms = np.sqrt(squared_dev.mean(axis=0)) * 10.0

    residue_indices = [traj.topology.atom(i).residue.index for i in atoms]
    n_residues = max(residue_indices) + 1
    residue_totals = np.zeros(n_residues)
    residue_counts = np.zeros(n_residues)
    for atom_index, residue_index in enumerate(residue_indices):
        residue_totals[residue_index] += rmsf_atoms[atom_index]
        residue_counts[residue_index] += 1

    return np.divide(
        residue_totals,
        residue_counts,
        out=np.zeros_like(residue_totals),
        where=residue_counts > 0,
    )


# ------------------------------
# Compute TICA Importance Scores
# ------------------------------
def compute_tica_importance_scores(tica_model):
    eigenvectors = None
    if hasattr(tica_model, "singular_vectors_right"):
        eigenvectors = np.asarray(tica_model.singular_vectors_right)
    elif hasattr(tica_model, "eigenvectors"):
        eigenvectors = np.asarray(tica_model.eigenvectors)

    if eigenvectors is None or eigenvectors.size == 0:
        return {}

    loadings = np.abs(eigenvectors).sum(axis=1)
    if np.max(loadings) > 0:
        loadings = loadings / np.max(loadings)

    return {int(idx): float(value) for idx, value in enumerate(loadings)}


# ------------------------------
# Normalize Scores
# ------------------------------
def normalize_scores(values, method="percentile", low_percentile=0.05, high_percentile=0.95):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values

    if method == "rank":
        ranks = np.argsort(np.argsort(values))
        return ranks / max(len(values) - 1, 1)

    if method == "zscore":
        mean_value = np.mean(values)
        std_value = np.std(values)
        if std_value == 0:
            return np.zeros_like(values)
        z_values = (values - mean_value) / std_value
        return 1.0 / (1.0 + np.exp(-z_values))

    q_low = np.quantile(values, low_percentile)
    q_high = np.quantile(values, high_percentile)
    if q_high <= q_low:
        return np.zeros_like(values)
    return np.clip((values - q_low) / (q_high - q_low), 0, 1)


# ------------------------------
# Aggregate Frame Scores To Residues
# ------------------------------
def aggregate_frame_to_residue(frame_scores, rmsf_scores):
    if len(rmsf_scores) == 0:
        return {}

    frame_scores = np.asarray(frame_scores, dtype=np.float64)
    mean_frame_score = frame_scores.mean() if len(frame_scores) > 0 else 0.0
    normalized_rmsf = normalize_scores(rmsf_scores, method="rank")

    residue_scores = {}
    for residue_index in range(len(normalized_rmsf)):
        combined = 0.5 * normalized_rmsf[residue_index] + 0.5 * mean_frame_score
        residue_scores[residue_index] = float(combined)

    return residue_scores

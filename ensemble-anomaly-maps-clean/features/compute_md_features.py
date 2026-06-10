"""
===============================================================
Module: features/compute_md_features.py
Description: Trajectory parsing and molecular feature extraction for tICA/MSM.
Author: Siya Jethliya
Project: Ensemble Anomaly Maps
===============================================================
"""

import numpy as np
import mdtraj as md
import warnings


# ------------------------------
# Compute MD Features
# ------------------------------
def compute_features(topology_path, trajectory_path, stride=1, reference_frame=0):
    traj = md.load(trajectory_path, top=topology_path, stride=stride)
    n_frames = len(traj)
    features = {}

    ref = traj[reference_frame]
    features["rmsd"] = md.rmsd(traj, ref)
    features["rg"] = md.compute_rg(traj)

    ca_atoms = traj.topology.select("name CA")
    if len(ca_atoms) > 1:
        ii, jj = np.triu_indices(len(ca_atoms), k=1)
        pairs = np.stack([ca_atoms[ii], ca_atoms[jj]], axis=1)
        distances = md.compute_distances(traj, pairs)
        features["contacts"] = (distances < 0.8).sum(axis=1).astype(float)
    else:
        features["contacts"] = np.zeros(n_frames)

    try:
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
    except (ValueError, RuntimeError, KeyError):
        warnings.warn(
            "Backbone dihedrals could not be computed (possible causes: missing protein backbone atoms, "
            "insufficient residues, or malformed topology); substituting zero-valued phi/psi features. "
            "This can reduce downstream tICA/MSM signal quality.",
            RuntimeWarning,
            stacklevel=2,
        )
        phi = np.zeros((n_frames, 1))
        psi = np.zeros((n_frames, 1))

    if phi.size > 0:
        features["phi_sin"] = np.sin(phi).mean(axis=1)
        features["phi_cos"] = np.cos(phi).mean(axis=1)
    else:
        features["phi_sin"] = np.zeros(n_frames)
        features["phi_cos"] = np.zeros(n_frames)

    if psi.size > 0:
        features["psi_sin"] = np.sin(psi).mean(axis=1)
        features["psi_cos"] = np.cos(psi).mean(axis=1)
    else:
        features["psi_sin"] = np.zeros(n_frames)
        features["psi_cos"] = np.zeros(n_frames)

    return features, traj


# ------------------------------
# Convert Features To Matrix
# ------------------------------
def features_to_matrix(features, keys=None):
    if keys is None:
        keys = list(features.keys())
    return np.column_stack([features[k] for k in keys]), keys

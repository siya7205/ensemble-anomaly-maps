# analysis/if_ops.py
import os
import pandas as pd
import numpy as np
import mdtraj as md
from sklearn.ensemble import IsolationForest

from .local_ops import compute_local_ops, summarize_per_residue

def score_pdb_with_if(pdb_path: str, out_dir: str, contamination=0.1, random_state=42):
    """
    Run Isolation Forest across residues for a single PDB.
    Uses per-residue features from local_ops as input to IF.
    Writes out_dir/rollup.csv with columns: resid, mean_if, pct_disallowed, n_frames
    Returns the dataframe.
    """
    os.makedirs(out_dir, exist_ok=True)

    traj = md.load(pdb_path)
    per_frame = compute_local_ops(traj)            # one frame, many residues
    per_res   = summarize_per_residue(per_frame)   # columns: resid, resname, local_score_med, rama_disallowed_pct, n_frames

    # Build a feature matrix for IF (safe, numeric)
    feat_cols = []
    for c in ("local_score_med", "rama_disallowed_pct"):
        if c in per_res.columns:
            feat_cols.append(c)
    # Backstop: if rama_disallowed_pct missing for some reason
    if not feat_cols:
        per_res["local_score_med"] = per_res.get("local_score_med", 0.0)
        feat_cols = ["local_score_med"]

    X = per_res[feat_cols].to_numpy(dtype=float)

    # Fit IsolationForest on residues as samples
    clf = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    ).fit(X)

    # Higher = more normal in sklearn; we invert so higher = more anomalous
    s = clf.score_samples(X)
    per_res["mean_if"] = (-s - (-s).min()) / ( (-s).max() - (-s).min() + 1e-12)

    # Standardize column names expected by /api/top_anomalies
    out = per_res.rename(columns={"rama_disallowed_pct": "pct_disallowed"})
    out["pct_disallowed"] = out["pct_disallowed"].fillna(0.0)
    out["n_frames"] = out.get("n_frames", 1)

    out_path = os.path.join(out_dir, "rollup.csv")
    out[["resid","mean_if","pct_disallowed","n_frames"]].to_csv(out_path, index=False)
    return out

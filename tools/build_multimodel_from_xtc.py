#!/usr/bin/env python3
"""
Build a consistent multi-model PDB straight from a single XTC + topology,
and write per-frame anomaly to the B-factor column.

Inputs
- --top: topology PDB (e.g., data/raw_trajectory/align_topol.pdb)
- --xtc: trajectory file (e.g., data/raw_trajectory/trajectory_0.xtc)
- --scores: optional CSV with columns [frame, b_factor]; if absent we
            set a simple 0..1 gradient so you can at least see coloring.
- --stride: keep every Nth frame (default 10)
- --max_frames: hard cap for demo
Outputs
- data/multi_model_anomaly.pdb
- data/anomaly_timeseries.json  (frame -> b_factor for the viewer)
"""
import argparse, os, json
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter

def load_scores(path, n_frames):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        # must have columns frame, b_factor
        if {"frame","b_factor"}.issubset(df.columns):
            # make sure covers our frames; pad/clamp if needed
            s = np.zeros(n_frames, dtype=float)
            for i in range(n_frames):
                row = df.loc[df["frame"] == i]
                s[i] = float(row["b_factor"].values[0]) if len(row) else 0.0
            # normalize to 0..1 for nicer coloring
            lo, hi = np.min(s), np.max(s)
            if hi > lo:
                s = (s - lo) / (hi - lo)
            return s
    # fallback gradient
    return np.linspace(0.0, 1.0, n_frames, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--xtc", required=True)
    ap.add_argument("--scores", default="")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--out_pdb", default="data/multi_model_anomaly.pdb")
    ap.add_argument("--out_json", default="data/anomaly_timeseries.json")
    args = ap.parse_args()

    u = mda.Universe(args.top, args.xtc)
    # stride by slicing the trajectory (MDAnalysis keeps atoms consistent)
    frames_idx = list(range(0, len(u.trajectory), args.stride))[:args.max_frames]
    n = len(frames_idx)
    print(f"[build] frames in xtc: {len(u.trajectory)}  -> keeping {n} frames (stride={args.stride})")

    # per-frame anomaly scores (0..1) to write into tempfactors (PDB B-factor)
    scores = load_scores(args.scores, n)

    # write multi-model PDB
    os.makedirs(os.path.dirname(args.out_pdb), exist_ok=True)
    with PDBWriter(args.out_pdb, multiframe=True) as w:
        for j, i in enumerate(frames_idx):
            u.trajectory[i]
            # set same per-atom B for this frame (you can refine to per-residue later)
            tf = np.full(u.atoms.n_atoms, scores[j], dtype=float)
            u.atoms.tempfactors = tf
            w.write(u.atoms)
    print(f"[build] wrote {args.out_pdb} with {n} models")

    # save timeseries for the viewer UI
    rows = [{"frame": j, "b_factor": float(scores[j])} for j in range(n)]
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rows, f)
    print(f"[build] wrote {args.out_json}")

if __name__ == "__main__":
    main()

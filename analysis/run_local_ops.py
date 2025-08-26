# analysis/run_local_ops.py
# ------------------------------------------------------------
# CLI runner for analysis/local_ops.py
# Usage:
#   python -m analysis.run_local_ops <pdb_or_list> [--out outputs/run-local-<stamp>]
# Notes:
# - Accepts either a single .pdb OR a text file containing one path per line
#   (each line: a .pdb;.frames from all will be concatenated)
# - Writes:
#     outdir/local_ops.csv                (per-frame, per-residue features)
#     outdir/local_ops_summary.csv        (per-residue rollup across frames)
# ------------------------------------------------------------

from __future__ import annotations
import sys, os, argparse, time
from pathlib import Path
import pandas as pd
import numpy as np
import mdtraj as md

from .local_ops import compute_local_ops

def _is_list_file(path: Path) -> bool:
    # treat as list file if it's not .pdb and exists
    return path.suffix.lower() != ".pdb" and path.exists()

def _load_traj_from_list(list_path: Path) -> md.Trajectory:
    """Load multiple PDBs (one per line) as concatenated frames."""
    pdbs = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
    if not pdbs:
        raise ValueError(f"No paths in list file: {list_path}")
    trajs = []
    for p in pdbs:
        print(f"[local_ops] loading {p}")
        t = md.load(p)
        trajs.append(t)
    return md.join(trajs)

def _rollup(df: pd.DataFrame) -> pd.DataFrame:
    """Per-residue summary across frames."""
    if df.empty:
        return df
    # robust stats helpers
    def med(x): return np.median(x)
    def mad(x): return np.median(np.abs(x - np.median(x)))

    grp = df.groupby(["resid", "resname"], sort=True)
    out = pd.DataFrame({
        "n_frames": grp.size(),
        "phi_med": grp["phi_deg"].median(),
        "psi_med": grp["psi_deg"].median(),
        "omega_med": grp["omega_deg"].median(),
        "rama_disallowed_pct": 100 * (1 - grp["rama_allowed"].mean()),
        "rama_dist_med": grp["rama_distance"].median(),
        "omega_pen_med": grp["omega_penalty"].median(),
        "bondlen_zmax_med": grp["bondlen_zmax"].median(),
        "bondang_zmax_med": grp["bondang_zmax"].median(),
        "chi_rotamer_dev_med": grp["chi_rotamer_dev"].median(),
        "sasa_norm_med": grp["sasa_norm"].median(),
        "local_score_med": grp["local_score"].median(),
    }).reset_index()

    # add MADs for a few key cols
    for c in ["phi_deg", "psi_deg", "local_score"]:
        out[f"{c.split('_')[0]}_mad"] = grp[c].apply(mad).values

    # simple ranking by median local_score (higher => worse)
    out = out.sort_values("local_score_med", ascending=False).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to a .pdb OR a text file listing .pdb paths (one per line)")
    ap.add_argument("--out", help="Output directory (default: outputs/run-local-<timestamp>)")
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.out) if args.out else Path("outputs") / f"run-local-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # load trajectory
    if _is_list_file(inp):
        traj = _load_traj_from_list(inp)
    else:
        print(f"[local_ops] loading {inp.name}")
        traj = md.load(str(inp))

    # compute per-(frame,residue) ops
    df = compute_local_ops(traj)

    # write long table
    long_csv = outdir / "local_ops.csv"
    df.to_csv(long_csv, index=False)
    print(f"[local_ops] wrote: {long_csv}")

    # roll up per residue and write
    summ = _rollup(df)
    summ_csv = outdir / "local_ops_summary.csv"
    summ.to_csv(summ_csv, index=False)
    print(f"[local_ops] wrote: {summ_csv}")

    # small console preview
    preview = summ[["resid","resname","local_score_med","rama_disallowed_pct","n_frames"]].head(12)
    print("\nTop residues by median local_score:")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(preview.to_string(index=False))

if __name__ == "__main__":
    main()

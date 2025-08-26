# analysis/run_trajectory_ops.py
import argparse
import glob
import os
from datetime import datetime

from analysis.trajectory_ops import extract_trajectory_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Glob of PDBs or a single traj file")
    p.add_argument("--top", default=None, help="Topology file if --traj is a DCD/XTC/etc")
    p.add_argument("--win", type=int, default=50, help="Window size (frames)")
    p.add_argument("--step", type=int, default=25, help="Step between windows (frames)")
    p.add_argument("--stride", type=int, default=1, help="Stride for loading frames")
    p.add_argument("--outdir", default=None, help="Output directory (default: outputs/run-traj-<ts>)")
    return p.parse_args()


def main():
    args = parse_args()

    # Expand globs if the user passed PDB patterns;
    # if it's a single non-glob path (e.g., DCD), leave it as is and let mdtraj handle it.
    if any(ch in args.traj for ch in "*?[]"):
        files = sorted(glob.glob(args.traj))
        if not files:
            raise SystemExit("No input files matched your --traj pattern.")
        traj_input = files
        print(f"[traj] Using {len(files)} file(s). First: {files[0]}")
    else:
        traj_input = args.traj
        print(f"[traj] loading: {traj_input} (top={args.top})")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or os.path.join("outputs", f"run-traj-{ts}")
    os.makedirs(outdir, exist_ok=True)

    # ✅ Ask for per-residue table too
    per_win, per_res, summary = extract_trajectory_features(
        traj_files=traj_input,
        top=args.top,
        win=args.win,
        step=args.step,
        stride=args.stride,
        return_residue=True,
    )

    per_win_csv = os.path.join(outdir, "per_window.csv")
    per_res_csv = os.path.join(outdir, "per_window_residue.csv")
    summary_csv = os.path.join(outdir, "summary.csv")

    per_win.to_csv(per_win_csv, index=False)
    per_res.to_csv(per_res_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print("[traj] Wrote:")
    print("  ", per_win_csv)
    print("  ", per_res_csv)
    print("  ", summary_csv)


if __name__ == "__main__":
    main()

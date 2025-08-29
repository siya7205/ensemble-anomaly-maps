# analysis/run_3d_backend.py
import argparse, os, sys, glob, subprocess, shutil, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sh(cmd, check=True):
    print(">", " ".join(cmd))
    return subprocess.run(cmd, check=check)

def latest_run_traj():
    runs = sorted((ROOT/"outputs").glob("run-traj-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

def have(path): 
    p = Path(path)
    return p.exists()

def require(path, msg):
    if not have(path):
        raise SystemExit(f"[ERR] Missing {path} ({msg})")

def step_trajectory(args) -> Path:
    """
    If --run_dir provided and looks complete, reuse it. Otherwise run trajectory extraction.
    """
    if args.run_dir:
        run = Path(args.run_dir)
        print(f"[traj] using provided run: {run}")
        require(run/"per_window.csv", "from run_trajectory_ops")
        require(run/"per_window_residue.csv", "per-residue window table")
        return run

    # Try latest complete run
    run = latest_run_traj()
    if run and have(run/"per_window.csv") and have(run/"per_window_residue.csv"):
        print(f"[traj] reusing latest: {run}")
        return run

    # Build a new run from frames_glob
    files = sorted(glob.glob(args.frames_glob))
    if not files:
        raise SystemExit(f"[ERR] frames_glob matched 0 files: {args.frames_glob}")
    print(f"[traj] {len(files)} frames, first: {files[0]}")

    cmd = [
        sys.executable, "-m", "analysis.run_trajectory_ops",
        "--traj", args.frames_glob,
        "--win", str(args.win),
        "--step", str(args.step),
        "--stride", str(args.stride),
    ]
    sh(cmd)
    run = latest_run_traj()
    require(run/"per_window.csv", "trajectory features")
    require(run/"per_window_residue.csv", "per-residue windows")
    print(f"[traj] ready: {run}")
    return run

def step_deep(args, run: Path):
    """
    Train AE + write per_window_deep.csv if missing or --force_deep.
    """
    out_csv = run/"per_window_deep.csv"
    if have(out_csv) and not args.force_deep:
        print(f"[deep] reuse {out_csv}")
        return
    cmd = [sys.executable, "-m", "analysis.run_deep_ops", "--run", str(run)]
    if args.latent_dim:
        cmd += ["--latent-dim", str(args.latent_dim)]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    sh(cmd)
    require(out_csv, "AE output per_window_deep.csv")
    print(f"[deep] ready: {out_csv}")

def step_cluster(args, run: Path):
    """
    Run clustering on latent + recon_error → anomalies + hybrid + plots.
    """
    deep_dir = run/"deep"
    deep_dir.mkdir(parents=True, exist_ok=True)
    out_latent = deep_dir/"latent_clusters.csv"
    out_anoms = deep_dir/"anomalies.csv"
    out_hybrid = deep_dir/"hybrid_scores.csv"

    need = (not have(out_latent) or not have(out_anoms) or not have(out_hybrid) or args.force_cluster)
    if not need:
        print(f"[cluster] reuse {deep_dir}")
        return

    cmd = [
        sys.executable, "-m", "analysis.run_cluster_ops",
        "--run", str(run),
        "--algo", args.algo,
        "--recon-q", str(args.recon_q),
        "--small-cluster-size", str(args.small_cluster_size),
    ]
    if args.algo == "dbscan":
        cmd += ["--eps", str(args.eps), "--min-samples", str(args.min_samples)]
    elif args.algo == "hdbscan":
        if args.min_cluster_size:
            cmd += ["--min-cluster-size", str(args.min_cluster_size)]
    sh(cmd)

    require(out_latent, "latent_clusters.csv")
    require(out_anoms, "anomalies.csv")
    require(out_hybrid, "hybrid_scores.csv")
    print(f"[cluster] ready: {deep_dir}")

def step_bfactor(args, run: Path):
    """
    Join hybrid + per_window_residue and write B-factors into frames; build multi-model PDB.
    """
    mm_out = Path(args.multi_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(ROOT/"analysis"/"anomaly_frames_to_bfactor.py"),
        "--run", str(run),
        "--frames_glob", args.frames_glob,
        "--out_dir", str(out_dir),
        "--multi_model", str(mm_out),
    ]
    if args.atom_mode:
        cmd += ["--atom_mode", args.atom_mode]  # e.g. "all", "ca"
    sh(cmd)

    require(mm_out, "multi-model anomaly PDB")
    print(f"[bfactor] wrote frames → {out_dir}\n[bfactor] multi-model → {mm_out}")

def main():
    ap = argparse.ArgumentParser(description="End-to-end backend for 3D anomaly coloring (no UI).")
    # Inputs
    ap.add_argument("--frames_glob", default="data/frames_clean/frame_*.pdb",
                    help="Glob of single-frame PDBs (one model each).")
    ap.add_argument("--run_dir", default=None, help="Existing outputs/run-traj-*/ dir to reuse.")
    ap.add_argument("--win", type=int, default=100)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)

    # Deep / clustering
    ap.add_argument("--latent_dim", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force_deep", action="store_true")
    ap.add_argument("--algo", choices=["dbscan","hdbscan"], default="dbscan")
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--min_samples", type=int, default=4)
    ap.add_argument("--min_cluster_size", type=int, default=10)
    ap.add_argument("--recon_q", type=float, default=0.95, help="reconstruction error quantile threshold")
    ap.add_argument("--small_cluster_size", type=int, default=5, help="size ≤ N → anomalous cluster")
    ap.add_argument("--force_cluster", action="store_true")

    # B-factor coloring
    ap.add_argument("--out_dir", default="data/colored_frames")
    ap.add_argument("--multi_model", default="data/multi_model_anomaly.pdb")
    ap.add_argument("--atom_mode", default=None, help='optional: "ca" to color only Cα atoms')

    args = ap.parse_args()

    run = step_trajectory(args)
    step_deep(args, run)
    step_cluster(args, run)
    step_bfactor(args, run)

    print("\n[OK] Pipeline complete.")
    print("     You can now open data/multi_model_anomaly.pdb in any 3D viewer and color by B-factor.\n")

if __name__ == "__main__":
    main()

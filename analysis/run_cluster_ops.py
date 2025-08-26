# analysis/run_cluster_ops.py
import argparse
import glob
from pathlib import Path
from analysis.cluster_ops import cluster_windows

def pick_latest_run():
    runs = sorted(glob.glob("outputs/run-traj-*"))
    if not runs:
        raise SystemExit("No outputs/run-traj-* found.")
    return Path(runs[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None, help="Path to outputs/run-traj-*/ (default: latest)")
    ap.add_argument("--algo", type=str, default="dbscan", choices=["dbscan", "hdbscan"])
    ap.add_argument("--eps", type=float, default=0.8)
    ap.add_argument("--min-samples", type=int, default=4, dest="min_samples")
    ap.add_argument("--small-cluster-size", type=int, default=5, dest="small_cluster_size")
    ap.add_argument("--recon-q", type=float, default=0.95, dest="recon_q")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else pick_latest_run()
    per_window_deep = run_dir / "per_window_deep.csv"
    out_dir = run_dir / "deep"

    print(f"[cluster] run_dir={run_dir}")
    print(f"[cluster] reading {per_window_deep}")
    res = cluster_windows(
        per_window_deep_csv=per_window_deep,
        out_dir=out_dir,
        algo=args.algo,
        eps=args.eps,
        min_samples=args.min_samples,
        small_cluster_size=args.small_cluster_size,
        recon_q=args.recon_q,
    )
    print("[cluster] done:", res)

if __name__ == "__main__":
    main()

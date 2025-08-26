# analysis/run_deep_ops.py
# CLI wrapper to train AE on latest run-traj-* and save outputs

import argparse
import glob
import os
from .deep_ops import AEConfig, run_deep

def latest_traj_run(root: str = "outputs") -> str:
    runs = sorted(glob.glob(os.path.join(root, "run-traj-*")), reverse=True)
    if not runs:
        raise SystemExit("No outputs/run-traj-* found. Run analysis/run_trajectory_ops.py first.")
    return runs[0]

def main():
    ap = argparse.ArgumentParser(description="Train AE on per_window trajectory features")
    ap.add_argument("--run", type=str, default=None, help="Path to outputs/run-traj-YYYYMMDD-HHMMSS")
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--hidden", type=str, default="128,64", help="comma-separated sizes, e.g. 128,64")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--context", type=int, default=0, help="temporal context radius k (0=no stacking)")
    args = ap.parse_args()

    run_dir = args.run or latest_traj_run()
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    cfg = AEConfig(
        latent_dim=args.latent,
        hidden=hidden,
        lr=args.lr,
        batch_size=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        context=args.context,
    )

    print(f"[run_deep_ops] Using run dir: {run_dir}")
    print(f"[run_deep_ops] Config: {cfg}")
    run_deep(run_dir, cfg)

if __name__ == "__main__":
    main()

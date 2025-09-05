# run_tadfm.py
import os, argparse, csv
import numpy as np
from pathlib import Path

from features.compute_md_features import compute_features
from segments.segmenter import segment_by_turning_angle, fixed_windows
from models.autoencoder import fit_autoencoder
from detect.tadfm import segment_shallow_stats, dbscan_cosine

def _to_matrix(feat_dict, keys=None):
    keys = keys or list(feat_dict.keys())
    X = np.column_stack([feat_dict[k] for k in keys])
    return X, keys

def main():
    ap = argparse.ArgumentParser(description="TAD-FM style anomaly detection for MD trajectories")
    ap.add_argument("--top", required=True, help="Topology (PDB/PRMTOP/etc.)")
    ap.add_argument("--traj", required=True, help="Trajectory (XTC/DCD/etc.)")
    ap.add_argument("--out_dir", default="outputs/tadfm_run")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--angle_thresh", type=float, default=135.0)
    ap.add_argument("--min_seg_len", type=int, default=50)
    ap.add_argument("--max_seg_len", type=int, default=2000)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--min_samples", type=int, default=10)
    ap.add_argument("--call_bfactor", action="store_true", help="Attempt to color frames via your script")
    ap.add_argument("--bfactor_script", default="anomaly_frames_to_bfactor.py")
    ap.add_argument("--multi_model_pdb", default="data/multi_model_anomaly.pdb")
    ap.add_argument("--frames_glob", default='data/frames_clean/frame_*.pdb')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Features
    feats, traj = compute_features(args.top, args.traj, stride=args.stride)
    X, keys = _to_matrix(feats)

    # 2) Segments (turning-angle segmentation; fallback to fixed windows if tiny)
    segs = segment_by_turning_angle(X, angle_thresh_deg=args.angle_thresh,
                                    min_len=args.min_seg_len, max_len=args.max_seg_len)
    if len(segs) == 1 and (segs[0][1] - segs[0][0] + 1) < args.min_seg_len:
        segs = fixed_windows(X.shape[0], win=max(args.min_seg_len, 250), overlap=50)

    # 3) Shallow stats per segment
    shallow = segment_shallow_stats(X, segs)

    # 4) Autoencoder latent + mixed features
    _, Z, _ = fit_autoencoder(shallow, latent=args.latent)
    mixed = np.hstack([shallow, Z])

    # 5) DBSCAN (cosine) → labels & scores
    labels, seg_scores = dbscan_cosine(mixed, eps=args.eps, min_samples=args.min_samples)

    # 6) Save per-segment results
    seg_csv = os.path.join(args.out_dir, "segments.csv")
    with open(seg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id","start","end","label","score"])
        for i, ((s,e), lab, sc) in enumerate(zip(segs, labels, seg_scores)):
            w.writerow([i, s, e, int(lab), float(sc)])

    # 7) Upsample segment scores to per-frame scores (piecewise constant)
    frame_scores = np.zeros(X.shape[0], dtype=float)
    for sc, (s,e) in zip(seg_scores, segs):
        frame_scores[s:e+1] = sc
    fs_csv = os.path.join(args.out_dir, "frame_scores.csv")
    with open(fs_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","score"])
        for i, sc in enumerate(frame_scores):
            w.writerow([i, float(sc)])

    print(f"[ok] Wrote: {seg_csv}")
    print(f"[ok] Wrote: {fs_csv}")

    # 8) Optional: handoff to your B-factor coloring script
    if args.call_bfactor:
        try:
            import subprocess, sys
            cmd = [
                sys.executable, args.bfactor_script,
                "--multi_model", args.multi_model_pdb,
                "--frames_glob", args.frames_glob,
                "--run", args.out_dir,
                "--out_dir", os.path.join(args.out_dir, "colored_frames")
            ]
            print("[info] Calling:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        except Exception as e:
            print("[warn] Skipped B-factor coloring:", e)

if __name__ == "__main__":
    main()

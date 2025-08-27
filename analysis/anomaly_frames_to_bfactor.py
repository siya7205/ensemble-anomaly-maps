# analysis/anomaly_frames_to_bfactor.py
import argparse, glob, os, re
import numpy as np
import pandas as pd

def load_frames(frames_glob):
    files = sorted(glob.glob(frames_glob))
    if not files:
        raise SystemExit(f"No frames matched: {frames_glob}")
    return files

def load_window_tables(run_dir):
    pw = pd.read_csv(os.path.join(run_dir, "per_window_deep.csv"))
    h_path = os.path.join(run_dir, "deep", "hybrid_scores.csv")
    if os.path.exists(h_path):
        hy = pd.read_csv(h_path)
        key = ["start", "end"]
        pw = pd.merge(pw, hy[key + ["A_hybrid"]], on=key, how="left")
    else:
        m = pw["deep_recon_error"]
        pw["A_hybrid"] = (m - m.min()) / (m.max() - m.min() + 1e-12)
    pr = pd.read_csv(os.path.join(run_dir, "per_window_residue.csv"))
    return pw, pr

def pick_window_for_frame(i, pw):
    centers = (pw["start"] + pw["end"]) * 0.5
    ix = int(np.argmin(np.abs(centers - i)))
    return pw.iloc[ix]

def residue_scores(win_row, pr):
    sub = pr[(pr["start"] == win_row["start"]) & (pr["end"] == win_row["end"])].copy()
    if "pct_disallowed" in sub.columns:
        sub["res_contrib"] = sub["pct_disallowed"]
    elif "mean_rama_distance" in sub.columns:
        sub["res_contrib"] = sub["mean_rama_distance"]
    else:
        sub["res_contrib"] = 1.0
    sub["bf"] = win_row["A_hybrid"] * sub["res_contrib"]
    if len(sub) and sub["bf"].max() > 0:
        sub["bf"] = sub["bf"] / (sub["bf"].max() + 1e-12) * 100.0
    return sub[["resid", "bf"]]

def write_model(pdb_in, resid_to_b, out, model_number):
    out.write(f"MODEL     {model_number}\n")
    with open(pdb_in, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    resid = int(line[22:26])
                except Exception:
                    resid = None
                bf = float(resid_to_b.get(resid, 0.0))
                line = line[:60] + f"{bf:6.2f}" + line[66:]
            out.write(line)
    out.write("ENDMDL\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frames_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--multi_model", required=True)
    args = ap.parse_args()

    frames = load_frames(args.frames_glob)
    pw, pr = load_window_tables(args.run)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.multi_model, "w") as mm:
        for i, pdb in enumerate(frames):
            win = pick_window_for_frame(i, pw)
            res_b = residue_scores(win, pr)
            resid_to_b = {int(r.resid): float(r.bf) for _, r in res_b.iterrows()}

            single_out = os.path.join(args.out_dir, f"colored_{os.path.basename(pdb)}")
            with open(single_out, "w") as out:
                write_model(pdb, resid_to_b, out, model_number=1)

            write_model(pdb, resid_to_b, mm, model_number=i + 1)

    print("[OK] wrote:")
    print(" - colored frames:", args.out_dir)
    print(" - multi-model   :", args.multi_model)

if __name__ == "__main__":
    main()

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
    """Return (per_window_df, per_residue_df_or_None).
    Backward-compatible: if per_window_deep.csv is absent, use frame_scores.csv.
    """
    import os, pandas as pd
    pw_path = os.path.join(run_dir, "per_window_deep.csv")
    pr_path = os.path.join(run_dir, "per_residue_deep.csv")
    if os.path.exists(pw_path):
        pw = pd.read_csv(pw_path)
    else:
        fs_path = os.path.join(run_dir, "frame_scores.csv")
        if not os.path.exists(fs_path):
            raise FileNotFoundError(f"Neither {pw_path} nor {fs_path} found")
        fs = pd.read_csv(fs_path)  # columns: frame, score
        # Emulate per-window schema: window_id,start,end,score
        pw = fs.rename(columns={"frame": "start", "score": "score"}).copy()
        pw["end"] = pw["start"]
        pw["window_id"] = pw.index
        pw = pw[["window_id", "start", "end", "score"]]
    pr = pd.read_csv(pr_path) if os.path.exists(pr_path) else None
    return pw, pr

def pick_window_for_frame(i, pw):
    centers = (pw["start"] + pw["end"]) * 0.5
    ix = int(np.argmin(np.abs(centers - i)))
    return pw.iloc[ix]

def residue_scores(win_row, pr):
    """Return {residue_index: score} for this window, or None if no per-residue table."""
    import pandas as pd
    if pr is None:
        return None
    # guard in case columns missing or empty
    if not set(['start','end']).issubset(pr.columns):
        return None
    sub = pr[(pr['start'] == win_row['start']) & (pr['end'] == win_row['end'])]
    if sub is None or len(sub) == 0:
        return None
    # Expect columns: residue, score
    if not set(['residue','score']).issubset(sub.columns):
        return None
    return dict(zip(sub['residue'].astype(int), sub['score'].astype(float)))

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

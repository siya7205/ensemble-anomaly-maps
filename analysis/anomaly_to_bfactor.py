# analysis/anomaly_to_bfactor.py
import argparse, os, pathlib
import pandas as pd

def load_scores(path):
    df = pd.read_csv(path)
    # try common columns
    resid_col = next((c for c in df.columns if c.lower() in {"resid", "residue", "res_idx"}), None)
    score_col = next((c for c in df.columns if c.lower() in {"hotspot_score", "instability_index", "score"}), None)
    if resid_col is None or score_col is None:
        raise SystemExit(f"Could not find resid/score columns in {path}. Have: {list(df.columns)}")
    df = df[[resid_col, score_col]].rename(columns={resid_col: "resid", score_col: "score"})
    # normalize to 0..100 (avoid zeros collapsing the color scale)
    s = df["score"].astype(float)
    if s.max() == s.min():
        df["bfac"] = 0.0
    else:
        df["bfac"] = (s - s.min()) / (s.max() - s.min()) * 100.0
    return df[["resid", "bfac"]]

def paint_bfactor(pdb_in, scores_df, pdb_out):
    """
    Write pdb_out where all ATOM/HETATM lines have B-factor set to the
    per-residue anomaly value. Residue numbering is assumed to match your CSV.
    """
    with open(pdb_in, "r") as f:
        lines = f.readlines()

    score_map = dict(zip(scores_df["resid"].astype(int), scores_df["bfac"].astype(float)))
    out = []
    for ln in lines:
        if ln.startswith(("ATOM  ", "HETATM")) and len(ln) >= 66:
            resid = ln[22:26].strip()
            try:
                resid_i = int(resid)
            except:
                resid_i = None
            b = score_map.get(resid_i, 0.0)
            # columns: B-factor is 61-66 in PDB fixed-width
            # format float with 2 decimals and right aligned to width 6
            b_str = f"{b:6.2f}"
            # temperature factor columns are 61-66 (0-based 60:66)
            ln = ln[:60] + b_str + ln[66:]
        out.append(ln)

    with open(pdb_out, "w") as f:
        f.writelines(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="outputs/run-traj-YYYYMMDD-HHMMSS (if omitted, use latest)")
    ap.add_argument("--scores", default=None, help="Path to residue_hotspots.csv or residue_instability.csv")
    ap.add_argument("--pdb", required=True, help="A representative PDB frame to color (e.g., data/frames_clean/frame_00000.pdb)")
    ap.add_argument("--out", default=None, help="Output PDB path (default: colored_<input>.pdb)")
    args = ap.parse_args()

    root = pathlib.Path(".").resolve()
    if args.scores is None:
        if args.run is None:
            # pick latest run-traj-*
            runs = sorted((root / "outputs").glob("run-traj-*"))
            if not runs:
                raise SystemExit("No outputs/run-traj-* found; run trajectory & deep pipeline first.")
            run = runs[-1]
        else:
            run = pathlib.Path(args.run)
        deep = run / "deep"
        # prefer instability; else hotspots
        cand = None
        for fname in ("residue_instability.csv", "residue_hotspots.csv"):
            p = deep / fname
            if p.exists():
                cand = p
                break
        if cand is None:
            raise SystemExit(f"No residue_instability.csv or residue_hotspots.csv under {deep}")
        scores_path = cand
    else:
        scores_path = pathlib.Path(args.scores)

    scores = load_scores(scores_path)
    pdb_in = pathlib.Path(args.pdb)
    if not pdb_in.exists():
        raise SystemExit(f"PDB not found: {pdb_in}")
    out = pathlib.Path(args.out) if args.out else pdb_in.with_name(f"colored_{pdb_in.name}")
    out.parent.mkdir(parents=True, exist_ok=True)

    paint_bfactor(pdb_in, scores, out)
    print(f"[OK] Wrote colored PDB with anomaly-as-Bfactor: {out}")
    print("     Open in PyMOL/NGL/VMD and color by B-factor to see anomalies.")

if __name__ == "__main__":
    main()

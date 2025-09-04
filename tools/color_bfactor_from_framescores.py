import os, glob, csv
from pathlib import Path

def read_frame_scores(run_dir):
    # expects columns: frame,score
    scores = {}
    with open(Path(run_dir) / "frame_scores.csv") as f:
        for i, (frame, score) in enumerate(csv.reader(f)):
            if i == 0 and frame.lower() == "frame":
                continue
            scores[int(frame)] = float(score)
    return scores

def set_bfactor_line(line, b):
    # PDB fixed width: tempFactor columns 61-66 (1-indexed), i.e. [60:66] 0-indexed
    # Occupancy (55-60) we leave untouched.
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return line
    # Make sure line is at least 66 chars long
    if len(line) < 66:
        line = line.rstrip("\n")
        line = line + " " * (66 - len(line)) + "\n"
    bf_str = f"{b:6.2f}"
    # keep trailing newline if present
    nl = "\n" if line.endswith("\n") else ""
    core = line[:-1] if nl else line
    core = core[:60] + bf_str + core[66:]
    return core + nl

def main(multi_model_pdb, frames_glob, run_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    scores = read_frame_scores(run_dir)
    frames = sorted(glob.glob(frames_glob))
    if not frames:
        raise FileNotFoundError(f"No frames matched: {frames_glob}")

    # map frame index from filename .../frame_00042.pdb -> 42
    def frame_idx(p):
        stem = Path(p).stem
        # accept colored_frame_XXXXX or frame_XXXXX
        for key in ("colored_frame_", "frame_"):
            if stem.startswith(key):
                s = stem[len(key):]
                if s.isdigit():
                    return int(s)
                # zero-padded -> int
                try:
                    return int(s)
                except:
                    pass
        # fallback: any trailing digits
        import re
        m = re.search(r"(\d+)$", stem)
        return int(m.group(1)) if m else 0

    for fp in frames:
        idx = frame_idx(fp)
        score = scores.get(idx)
        if score is None:
            # if not found, skip or set zero; choose to set zero
            score = 0.0
        with open(fp) as f:
            lines = f.readlines()
        colored = [set_bfactor_line(ln, score) for ln in lines]
        out_fp = out / f"colored_frame_{idx:05d}.pdb"
        with open(out_fp, "w") as g:
            g.writelines(colored)
        print(f"[ok] wrote {out_fp} (B={score:.3f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi_model", default="data/multi_model_anomaly.pdb")  # kept for compatibility, unused here
    ap.add_argument("--frames_glob", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.multi_model, args.frames_glob, args.run, args.out_dir)

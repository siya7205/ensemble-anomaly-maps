import sys, os, pandas as pd
from analysis.angles import compute_phi_psi
from analysis.anomaly import score_if
from analysis.aggregate import rollup

def main(frames_list_path: str):
    with open(frames_list_path) as f:
        frames = [ln.strip() for ln in f if ln.strip()]

    os.makedirs("data", exist_ok=True)
    tmp_tables = []

    N = min(25, len(frames))  # start with 25 samples
    for i, pdb in enumerate(frames[:N], 1):
        out = f"data/tmp_angles_{i}.parquet"
        print(f"[{i}/{N}] computing φ/ψ for {pdb}")
        compute_phi_psi(pdb, out)
        tmp_tables.append(out)

    if not tmp_tables:
        print("No inputs found. Check data/bioemu/frames.txt")
        sys.exit(1)

    df = pd.concat([pd.read_parquet(p) for p in tmp_tables], ignore_index=True)
    df.to_parquet("data/angles.parquet")
    print("WROTE: data/angles.parquet", len(df), "rows")

    score_if("data/angles.parquet", "data/angles_scored.parquet", contamination=0.05)
    print("WROTE: data/angles_scored.parquet")

    rollup("data/angles_scored.parquet", "data/rollup.csv")
    print("WROTE: data/rollup.csv")

    for p in tmp_tables:
        try: os.remove(p)
        except: pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python analysis/run_ensemble.py data/bioemu/frames.txt")
        sys.exit(1)
    main(sys.argv[1])

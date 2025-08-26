# analysis/run_local_ops.py
import os, sys, time
import pandas as pd
import mdtraj as md

from .local_ops import compute_local_ops

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    gb = df.groupby(["resid","resname"])
    out = gb.agg(
        local_score_med=("local_score","median"),
        rama_disallowed_pct=("rama_allowed", lambda x: 100.0*(1.0 - x.mean())),
        rama_dist_med=("rama_distance","median"),
        omega_pen_med=("omega_penalty","median"),
        bondlen_zmax_med=("bondlen_zmax","median"),
        bondang_zmax_med=("bondang_zmax","median"),
        chi_rotamer_dev_med=("chi_rotamer_dev","median"),
        sasa_norm_med=("sasa_norm","median"),
        n_frames=("frame","count"),
    ).reset_index()

    # DSSP percentages if 'ss' column exists (H/E/C)
    if "ss" in df.columns:
        def pct(df_sub, label):
            return 100.0 * (df_sub["ss"] == label).mean()
        ss_pct = gb.apply(lambda g: pd.Series({
            "ss_H_pct": pct(g, "H"),
            "ss_E_pct": pct(g, "E"),
            "ss_C_pct": pct(g, "C"),
        })).reset_index(drop=True)
        out = pd.concat([out, ss_pct], axis=1)

    return out.sort_values("local_score_med", ascending=False)

def main(pdb_path: str, out_dir: str | None = None):
    name = os.path.basename(pdb_path)
    print(f"[local_ops] loading {name}")
    traj = md.load(pdb_path)

    df = compute_local_ops(traj)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir or f"outputs/run-local-{stamp}"
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "local_ops.csv")
    df.to_csv(csv_path, index=False)
    print(f"[local_ops] wrote: {csv_path}")

    summ = summarize(df)
    csv_sum = os.path.join(run_dir, "local_ops_summary.csv")
    summ.to_csv(csv_sum, index=False)
    print(f"[local_ops] wrote: {csv_sum}\n")

    print("Top residues by median local_score:")
    cols = ["resid","resname","local_score_med","rama_disallowed_pct","n_frames"]
    # show DSSP if present
    for c in ("ss_H_pct","ss_E_pct","ss_C_pct"):
        if c in summ.columns and c not in cols:
            cols.append(c)
    print(summ[cols].head(12).to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m analysis.run_local_ops <protein.pdb> [out_dir]")
        sys.exit(1)
    pdb = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else None
    main(pdb, outd)

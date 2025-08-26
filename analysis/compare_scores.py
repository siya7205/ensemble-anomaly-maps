#!/usr/bin/env python3
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

def latest(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {pattern}")
    return files[-1]

def main():
    # 1) find inputs
    rollup_csv = "outputs/latest/rollup.csv"  # IF rollup written by /api/score_pdb
    local_csv  = latest("outputs/run-local-*/local_ops_summary.csv")

    print(f"[compare] IF rollup:  {rollup_csv}")
    print(f"[compare] Local-ops:  {local_csv}")

    # 2) load
    if_df = pd.read_csv(rollup_csv)
    lo_df = pd.read_csv(local_csv)

    # 3) keep relevant columns
    if_keep = ["resid", "mean_if", "pct_disallowed", "n_frames"]
    lo_keep = [
        "resid", "resname", "local_score_med", "rama_disallowed_pct",
        "rama_dist_med", "omega_pen_med", "bondlen_zmax_med",
        "bondang_zmax_med", "chi_rotamer_dev_med", "sasa_norm_med",
    ]
    if_df = if_df[ [c for c in if_keep if c in if_df.columns] ].copy()
    lo_df = lo_df[ [c for c in lo_keep if c in lo_df.columns] ].copy()

    # 4) join
    df = pd.merge(if_df, lo_df, on="resid", how="outer")

    # 5) normalize & combined score
    # - IF: more negative means more anomalous → invert sign
    df["if_inv"] = -df["mean_if"]
    # simple min-max on non-null
    for c in ("if_inv", "local_score_med"):
        v = df[c]
        vmin, vmax = v.min(skipna=True), v.max(skipna=True)
        if pd.notnull(vmin) and pd.notnull(vmax) and vmax > vmin:
            df[c + "_z01"] = (v - vmin) / (vmax - vmin)
        else:
            df[c + "_z01"] = 0.0

    # weighted combo (tweak weights later)
    df["combined_score"] = 0.5 * df["if_inv_z01"] + 0.5 * df["local_score_med_z01"]

    # 6) rankings
    df = df.sort_values("combined_score", ascending=False)
    out_dir = "outputs/compare"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "if_vs_local.csv")
    df.to_csv(out_csv, index=False)
    print(f"[compare] wrote: {out_csv}")

    # 7) quick plots
    plt.figure()
    df.plot.scatter(x="local_score_med", y="mean_if",
                    alpha=0.7, title="Local (rules) vs IF (note: IF lower=worse)")
    plt.xlabel("local_score_med (higher = more abnormal)")
    plt.ylabel("mean_if (lower = more abnormal)")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "scatter_local_vs_if.png")
    plt.savefig(p1, dpi=150)
    print(f"[compare] wrote: {p1}")

    plt.figure()
    df["combined_score"].hist(bins=30)
    plt.title("Combined anomaly score")
    plt.xlabel("combined_score")
    plt.ylabel("count")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "combined_hist.png")
    plt.savefig(p2, dpi=150)
    print(f"[compare] wrote: {p2}")

    # 8) print top table
    cols_show = [
        "resid","resname","combined_score",
        "local_score_med","mean_if","pct_disallowed","rama_disallowed_pct",
        "sasa_norm_med","chi_rotamer_dev_med"
    ]
    print("\nTop 15 residues (combined):")
    print(df[ [c for c in cols_show if c in df.columns] ].head(15).to_string(index=False))

if __name__ == "__main__":
    main()

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
    out_dir = "outputs/compare"
    os.makedirs(out_dir, exist_ok=True)

    # inputs
    if_rollup = "outputs/latest/rollup.csv"                    # IsolationForest summary
    lo_sum    = latest("outputs/run-local-*/local_ops_summary.csv")  # local_ops summary

    print(f"[enrich] IF:  {if_rollup}")
    print(f"[enrich] OPS: {lo_sum}")

    IF = pd.read_csv(if_rollup)
    LO = pd.read_csv(lo_sum)

    # merge
    keep_if = ["resid","mean_if","pct_disallowed","n_frames"]
    IF = IF[[c for c in keep_if if c in IF.columns]].copy()
    df = pd.merge(LO, IF, on="resid", how="left")

    # a) SASA vs anomaly (local + IF)
    plt.figure()
    df.plot.scatter(x="sasa_norm_med", y="local_score_med", alpha=0.7,
                    title="Local anomaly vs SASA (median)")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "scatter_local_vs_sasa.png")
    plt.savefig(p1, dpi=150)
    print(f"[enrich] wrote: {p1}")

    if "mean_if" in df.columns:
        plt.figure()
        df.plot.scatter(x="sasa_norm_med", y="mean_if", alpha=0.7,
                        title="IF mean_if vs SASA (lower mean_if = more anomalous)")
        plt.tight_layout()
        p2 = os.path.join(out_dir, "scatter_if_vs_sasa.png")
        plt.savefig(p2, dpi=150)
        print(f"[enrich] wrote: {p2}")

    # b) Side-chain rotamer deviation vs anomaly
    if "chi_rotamer_dev_med" in df.columns:
        plt.figure()
        df.plot.scatter(x="chi_rotamer_dev_med", y="local_score_med", alpha=0.7,
                        title="Local anomaly vs rotamer deviation (χ, median)")
        plt.tight_layout()
        p3 = os.path.join(out_dir, "scatter_local_vs_chi.png")
        plt.savefig(p3, dpi=150)
        print(f"[enrich] wrote: {p3}")

    # c) Compare top decile vs rest on SASA
    # define combined score = normalized local + normalized (-mean_if)
    df["if_inv"] = -df["mean_if"] if "mean_if" in df.columns else 0.0
    for c in ("local_score_med","if_inv"):
        v = df[c]
        vmin, vmax = v.min(skipna=True), v.max(skipna=True)
        df[c+"_z01"] = (v - vmin) / (vmax - vmin) if pd.notnull(vmin) and pd.notnull(vmax) and vmax > vmin else 0.0
    df["combined_score"] = 0.5*df["local_score_med_z01"] + 0.5*df["if_inv_z01"]

    cutoff = df["combined_score"].quantile(0.90)
    df["is_top_decile"] = df["combined_score"] >= cutoff

    summary = df.groupby("is_top_decile")["sasa_norm_med"].describe()[["mean","std","25%","50%","75%","count"]]
    csv_out = os.path.join(out_dir, "sasa_by_top_decile.csv")
    summary.to_csv(csv_out)
    print(f"[enrich] wrote: {csv_out}")

    # boxplot SASA by top/others
    plt.figure()
    df.boxplot(column="sasa_norm_med", by="is_top_decile")
    plt.suptitle("")
    plt.title("SASA (median) — top decile vs others")
    plt.xlabel("is_top_decile")
    plt.ylabel("sasa_norm_med")
    plt.tight_layout()
    p4 = os.path.join(out_dir, "box_sasa_top_decile.png")
    plt.savefig(p4, dpi=150)
    print(f"[enrich] wrote: {p4}")

    # d) DSSP composition among top anomalies (if available)
    if set(["ss_H_pct","ss_E_pct","ss_C_pct"]).issubset(df.columns):
        top = df[df["is_top_decile"]]
        rest = df[~df["is_top_decile"]]
        comp = pd.DataFrame({
            "top_H_mean": [top["ss_H_pct"].mean()],
            "top_E_mean": [top["ss_E_pct"].mean()],
            "top_C_mean": [top["ss_C_pct"].mean()],
            "rest_H_mean": [rest["ss_H_pct"].mean()],
            "rest_E_mean": [rest["ss_E_pct"].mean()],
            "rest_C_mean": [rest["ss_C_pct"].mean()],
        })
        comp_csv = os.path.join(out_dir, "dssp_composition_top_vs_rest.csv")
        comp.to_csv(comp_csv, index=False)
        print(f"[enrich] wrote: {comp_csv}")

        # simple bar plot
        plt.figure()
        labels = ["H","E","C"]
        top_vals  = [comp["top_H_mean"][0],  comp["top_E_mean"][0],  comp["top_C_mean"][0]]
        rest_vals = [comp["rest_H_mean"][0], comp["rest_E_mean"][0], comp["rest_C_mean"][0]]
        x = range(len(labels))
        plt.bar([i-0.2 for i in x], top_vals, width=0.4, label="top")
        plt.bar([i+0.2 for i in x], rest_vals, width=0.4, label="rest")
        plt.xticks(list(x), labels)
        plt.ylabel("% frames")
        plt.title("DSSP composition (mean % over residues)")
        plt.legend()
        plt.tight_layout()
        p5 = os.path.join(out_dir, "dssp_composition_bar.png")
        plt.savefig(p5, dpi=150)
        print(f"[enrich] wrote: {p5}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, os, pandas as pd
from datetime import datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path like outputs/run-traj-*/")
    args = ap.parse_args()
    deep = os.path.join(args.run, "deep")
    out_md = os.path.join(deep, "report.md")

    pw = pd.read_csv(os.path.join(deep, "per_window_deep.csv"))
    anom = pd.read_csv(os.path.join(deep, "anomalies.csv")) if os.path.exists(os.path.join(deep,"anomalies.csv")) else pd.DataFrame()
    hot = pd.read_csv(os.path.join(deep, "residue_hotspots.csv")) if os.path.exists(os.path.join(deep,"residue_hotspots.csv")) else pd.DataFrame()
    inst = pd.read_csv(os.path.join(deep, "residue_instability.csv")) if os.path.exists(os.path.join(deep,"residue_instability.csv")) else pd.DataFrame()

    lines = []
    lines.append(f"# Deep Anomaly Summary — {os.path.basename(args.run)}")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")

    # Windows
    if not anom.empty:
        lines.append("## Top anomalous windows")
        lines.append("")
        lines.append(anom.head(10).to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## Top anomalous windows\nNo anomalies.csv (maybe re-run clustering).")

    # Residues
    if not hot.empty:
        lines.append("## Residue hotspots (weighted by anomaly)")
        lines.append("")
        lines.append(hot.head(15).to_markdown(index=False))
        lines.append("")
    if not inst.empty:
        lines.append("## Residue instability index")
        lines.append("")
        lines.append(inst.head(15).to_markdown(index=False))
        lines.append("")

    # Plots
    for png in ("timeline_clusters.png","latent_scatter.png"):
        p = os.path.join(deep, png)
        if os.path.exists(p):
            lines.append(f"## {png.replace('_',' ').replace('.png','').title()}\n")
            lines.append(f"![{png}]({png})\n")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print("WROTE:", out_md)

if __name__ == "__main__":
    main()

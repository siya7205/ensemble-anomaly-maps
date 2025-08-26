import pandas as pd
import matplotlib.pyplot as plt
import sys

outdir = sys.argv[1] if len(sys.argv) > 1 else "outputs/run-traj-20250826-233350"
pw = pd.read_csv(f"{outdir}/per_window.csv")
sm = pd.read_csv(f"{outdir}/summary.csv")

print("Summary:\n", sm.to_string(index=False))

# Top anomalous windows (by max % disallowed)
top = pw.sort_values("max_pct_disallowed", ascending=False).head(15)
print("\nTop 15 windows by max_pct_disallowed:\n", top.to_string(index=False))

# Simple timeline plot (mean % disallowed per window)
plt.figure()
plt.plot(pw["start"], pw["mean_pct_disallowed"])
plt.xlabel("Window start (frame)")
plt.ylabel("Mean % disallowed")
plt.title("Trajectory anomaly timeline")
plt.tight_layout()
plt.savefig(f"{outdir}/timeline_mean_pct_disallowed.png")
print(f"\nSaved {outdir}/timeline_mean_pct_disallowed.png")


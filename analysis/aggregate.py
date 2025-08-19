import numpy as np, pandas as pd
from pathlib import Path

def allowed_mask(phi, psi):
    if np.isnan(phi) or np.isnan(psi):
        return False
    # coarse Rama mask (refine later by residue class)
    return ((-180 <= phi <= -30 and -90 <= psi <= 45) or
            (-160 <= phi <= -30 and  45 <= psi <= 180))

def rollup(scored="data/angles_scored.parquet", out_csv="data/rollup.csv"):
    df = pd.read_parquet(scored)
    df["allowed"] = df.apply(lambda r: allowed_mask(r["phi"], r["psi"]), axis=1)
    agg = (df.groupby("resid")
             .agg(mean_if=("if_score","mean"),
                  pct_disallowed=("allowed", lambda s: 100*(1.0 - s.mean())),
                  n_frames=("frame","nunique"))
             .reset_index()
             .sort_values(["pct_disallowed","mean_if"], ascending=[False, False]))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    return out_csv

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

def score_if(angles="data/angles.parquet", out="data/angles_scored.parquet",
             contamination=0.05, random_state=0):
    df = pd.read_parquet(angles)
    X = df[["cphi","sphi","cpsi","spsi"]].fillna(0.0).values

    # Guard: if we have too few rows to fit, just write zeros
    if X.shape[0] < 5:
        df["if_score"] = 0.0
    else:
        # For tiny datasets, cap contamination so it doesn't complain
        eff_contam = min(contamination, max(0.001, 1.0 / X.shape[0]))
        clf = IsolationForest(n_estimators=200, contamination=eff_contam, random_state=random_state)
        clf.fit(X)  # <-- the missing line
        df["if_score"] = -clf.decision_function(X)  # higher = more anomalous

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out

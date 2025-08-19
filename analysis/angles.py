from pathlib import Path
import numpy as np, pandas as pd, mdtraj as md

def compute_phi_psi(pdb="data/bioemu/sample.pdb", out="data/angles.parquet"):
    traj = md.load(pdb)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    phi_deg, psi_deg = np.degrees(phi), np.degrees(psi)
    rows=[]
    for f in range(phi_deg.shape[0]):
        for r in range(phi_deg.shape[1]):
            ph, ps = phi_deg[f, r], psi_deg[f, r]
            rows.append({
                "frame": f, "resid": r, "phi": ph, "psi": ps,
                "cphi": np.cos(np.radians(ph)) if not np.isnan(ph) else np.nan,
                "sphi": np.sin(np.radians(ph)) if not np.isnan(ph) else np.nan,
                "cpsi": np.cos(np.radians(ps)) if not np.isnan(ps) else np.nan,
                "spsi": np.sin(np.radians(ps)) if not np.isnan(ps) else np.nan,
            })
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out

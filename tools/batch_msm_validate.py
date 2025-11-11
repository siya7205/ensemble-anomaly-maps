import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mdtraj as md
from sklearn.neighbors import NearestNeighbors
from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import optimized utilities
from utils import (
    minmax_normalize,
    compute_local_density,
    compute_frame_scores,
    ProgressBar
)

TOP = "data/raw_trajectory/align_topol.pdb"
TRAJ_DIR = "data/raw_trajectory"
OUT_ROOT = Path("outputs/batch")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---- Feature extraction (optimized) ----
def make_features(traj):
    """Extract features from trajectory with optimized operations."""
    ref = traj[0]
    
    # Vectorized RMSD computation
    rmsd = md.rmsd(traj, ref)
    rg = md.compute_rg(traj)

    # CA-CA contact computation (vectorized)
    ca = traj.topology.select("name CA")
    if len(ca) > 1:
        ii, jj = np.triu_indices(len(ca), k=1)
        pairs = np.stack([ca[ii], ca[jj]], axis=1)
        d = md.compute_distances(traj, pairs)
        contacts = (d < 0.8).sum(axis=1).astype(np.float64)
    else:
        contacts = np.zeros(len(traj))

    # Dihedral angles with error handling
    try:
        _, phi = md.compute_phi(traj)
    except Exception:
        phi = np.array([])
    
    try:
        _, psi = md.compute_psi(traj)
    except Exception:
        psi = np.array([])

    def sincos_mean(A):
        """Compute mean sine and cosine of angles (circular statistics)."""
        if A.size == 0:
            return np.zeros((len(traj), 2))
        return np.column_stack([
            np.sin(A).mean(axis=1),
            np.cos(A).mean(axis=1)
        ])

    # Combine all features
    X = np.column_stack([rmsd, rg, contacts, sincos_mean(phi), sincos_mean(psi)])
    return X, phi, psi

# ---- Event detection (optimized) ----
def label_events(phi, psi, dssp):
    """Detect structural transition events with vectorized operations."""
    def unwrap(a):
        return np.unwrap(a, axis=0) if a.size > 0 else a
    
    ev = {}
    
    # Phi angle events (vectorized)
    if phi.size > 0:
        dphi = np.abs(np.diff(unwrap(phi), axis=0)).mean(axis=1)
        ev["phi"] = (np.where(dphi > 1.2)[0] + 1).tolist()
    else:
        ev["phi"] = []
    
    # Psi angle events (vectorized)
    if psi.size > 0:
        dpsi = np.abs(np.diff(unwrap(psi), axis=0)).mean(axis=1)
        ev["psi"] = (np.where(dpsi > 1.2)[0] + 1).tolist()
    else:
        ev["psi"] = []
    
    # DSSP secondary structure changes (vectorized)
    if dssp is None:
        ev["dssp"] = []
    else:
        changes = (dssp[1:] != dssp[:-1]).mean(axis=1)
        ev["dssp"] = (np.where(changes > 0.15)[0] + 1).astype(int).tolist()
    
    return ev

# ---- Anomaly scoring using Deeptime (optimized) ----
def msm_anomaly(X, lag_tica=10, lag_msm=30, n_clusters=25):
    """
    Compute anomaly scores using MSM+TICA with optimized operations.
    
    Args:
        X: Feature matrix (T x F)
        lag_tica: TICA lag time
        lag_msm: MSM lag time
        n_clusters: Number of clusters
    
    Returns:
        scores: Anomaly scores [0-100]
        dtraj: Discrete trajectory
        msm: Fitted MSM model
    """
    # Fit TICA
    tica = TICA(lagtime=lag_tica, dim=5).fit(X).fetch_model()
    Y = tica.transform(X)

    # Cluster in TICA space
    kmeans = KMeans(n_clusters=n_clusters).fit(Y).fetch_model()
    dtraj = kmeans.transform(Y).astype(np.int64)

    # Fit MSM
    msm = MaximumLikelihoodMSM(lagtime=lag_msm).fit(dtraj).fetch_model()
    pi = msm.stationary_distribution
    n_states = len(pi)

    # Compute rarity (vectorized)
    rarity = np.array([
        1.0 - pi[s] if s < n_states else 1.0 
        for s in dtraj
    ], dtype=np.float64)

    # Compute transition surprise (optimized)
    P = msm.transition_matrix
    surprise = np.zeros_like(rarity)
    
    valid_indices = []
    for i in range(len(rarity) - lag_msm):
        s1, s2 = dtraj[i], dtraj[i + lag_msm]
        if s1 < n_states and s2 < n_states:
            p = max(P[s1, s2], 1e-12)
            surprise[i] = -np.log(p)

    # Local density (optimized with parallel computation)
    lden_scores = -compute_local_density(Y, n_neighbors=min(10, len(Y) - 1))

    # Combine scores using optimized function
    score = compute_frame_scores(rarity, surprise, lden_scores)
    
    return score, dtraj, msm
# ---- Process single trajectory (for parallel execution) ----
def process_trajectory(xtc_path, top_path, out_root):
    """
    Process a single trajectory file.
    
    Args:
        xtc_path: Path to XTC file
        top_path: Path to topology file
        out_root: Output root directory
    
    Returns:
        Summary dict with processing results
    """
    name = Path(xtc_path).stem
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load trajectory
        t = md.load(xtc_path, top=top_path)
        
        # Compute DSSP (optional)
        try:
            dssp = md.compute_dssp(t, simplified=True)
        except Exception:
            dssp = None

        # Extract features
        X, phi, psi = make_features(t)
        
        # Compute anomaly scores
        scores, dtraj, msm = msm_anomaly(X)

        # Save outputs
        pd.DataFrame(X).to_csv(out / "features.csv", index=False)
        pd.DataFrame({
            "frame": np.arange(len(scores)),
            "score": scores
        }).to_csv(out / "frame_scores.csv", index=False)
        np.save(out / "dtraj.npy", dtraj)
        if dssp is not None:
            np.save(out / "dssp.npy", dssp)

        # Label events
        events = label_events(phi, psi, dssp)
        with open(out / "events.json", "w") as f:
            json.dump(events, f)

        # Compute agreement metrics
        y_true = np.zeros(len(scores), dtype=int)
        for arr in events.values():
            for fr in arr:
                for k in (-1, 0, 1):
                    if 0 <= fr + k < len(y_true):
                        y_true[fr + k] = 1

        order = np.argsort(scores)[::-1]
        prec = {}
        for k in [5, 10, 20, 50]:
            k_actual = min(k, len(order))
            prec[f"P@{k}"] = float(y_true[order[:k_actual]].mean())

        with open(out / "agreement.json", "w") as f:
            json.dump(prec, f)
        
        # Write summary
        summary_text = (
            f"{name}: frames={len(scores)}, msm_states={msm.n_states}, "
            f"events phi={len(events['phi'])}, psi={len(events['psi'])}, "
            f"dssp={'NA' if dssp is None else len(events['dssp'])}; "
            f"agreement {prec}\n"
        )
        with open(out / "SUMMARY.txt", "w") as f:
            f.write(summary_text)
        
        return {
            "name": name,
            "success": True,
            "frames": len(scores),
            "msm_states": msm.n_states,
            "summary": summary_text
        }
    
    except Exception as e:
        error_msg = f"Error processing {name}: {str(e)}\n"
        with open(out / "ERROR.txt", "w") as f:
            f.write(error_msg)
        return {
            "name": name,
            "success": False,
            "error": str(e)
        }


# ---- Main loop (with optional parallel processing) ----
def main(parallel=False, max_workers=None):
    """
    Main batch processing function.
    
    Args:
        parallel: Whether to use parallel processing
        max_workers: Max number of parallel workers (None = auto)
    """
    xtcs = sorted(glob.glob(os.path.join(TRAJ_DIR, "trajectory_*.xtc")))
    
    if not Path(TOP).exists():
        raise FileNotFoundError(f"Missing topology {TOP}")
    
    print(f"[info] Found {len(xtcs)} trajectories")
    print(f"[info] Parallel processing: {parallel}")
    
    if parallel and len(xtcs) > 1:
        # Parallel processing
        print(f"[info] Using {max_workers or 'auto'} workers")
        progress = ProgressBar(len(xtcs), desc="Processing trajectories")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_trajectory, xtc, TOP, OUT_ROOT): xtc
                for xtc in xtcs
            }
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.update(1)
                
                if result["success"]:
                    print(f"  ✓ {result['name']}: {result['frames']} frames, {result['msm_states']} states")
                else:
                    print(f"  ✗ {result['name']}: {result['error']}")
    else:
        # Sequential processing
        progress = ProgressBar(len(xtcs), desc="Processing trajectories")
        results = []
        
        for xtc in xtcs:
            result = process_trajectory(xtc, TOP, OUT_ROOT)
            results.append(result)
            progress.update(1)
            
            if result["success"]:
                print(f"  ✓ {result['name']}")
            else:
                print(f"  ✗ {result['name']}: {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n[done] Processed {successful}/{len(xtcs)} trajectories successfully")
    print(f"[done] Results in {OUT_ROOT}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Batch MSM validation with optimization")
    ap.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    ap.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = ap.parse_args()
    
    main(parallel=args.parallel, max_workers=args.workers)

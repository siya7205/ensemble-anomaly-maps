import os, glob, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import mdtraj as md
from sklearn.neighbors import NearestNeighbors
from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM
import logging

# Silence deeptime MSM warnings
logging.getLogger('deeptime').setLevel(logging.ERROR)

TOP = "data/raw_trajectory/align_topol.pdb"
TRAJ_DIR = "data/raw_trajectory"
OUT_ROOT = Path("outputs/batch")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---- Feature extraction ----
def make_features(traj):
    ref = traj[0]
    rmsd = md.rmsd(traj, ref)
    rg = md.compute_rg(traj)
    ca = traj.topology.select("name CA")
    ii, jj = np.triu_indices(len(ca), k=1)
    pairs = np.stack([ca[ii], ca[jj]], axis=1)
    d = md.compute_distances(traj, pairs)
    contacts = (d < 0.8).sum(axis=1).astype(float)
    _, phi = md.compute_phi(traj); _, psi = md.compute_psi(traj)
    def sc(A):
        return np.zeros((len(traj), 2)) if A.size == 0 else np.column_stack([np.sin(A).mean(axis=1), np.cos(A).mean(axis=1)])
    return np.column_stack([rmsd, rg, contacts, sc(phi), sc(psi)]), phi, psi

# ---- Event detection ----
def label_events(phi, psi, dssp):
    def unwrap(a): return np.unwrap(a, axis=0) if a.size else a
    ev = {}
    dphi = np.abs(np.diff(unwrap(phi), axis=0)).mean(axis=1) if phi.size else np.zeros(1)
    dpsi = np.abs(np.diff(unwrap(psi), axis=0)).mean(axis=1) if psi.size else np.zeros(1)
    ev["phi"] = (np.where(dphi > 1.2)[0] + 1).tolist()
    ev["psi"] = (np.where(dpsi > 1.2)[0] + 1).tolist()
    if dssp is None:
        ev["dssp"] = []
    else:
        changes = (dssp[1:] != dssp[:-1]).mean(axis=1)
        ev["dssp"] = (np.where(changes > 0.15)[0] + 1).astype(int).tolist()
    return ev

# ---- MSM anomaly scoring ----
def msm_anomaly(X, lag_tica=10, lag_msm=30, n_clusters=25):
    tica = TICA(lagtime=lag_tica, dim=5).fit(X).fetch_model()
    Y = tica.transform(X)
    kmeans = KMeans(n_clusters=n_clusters).fit(Y).fetch_model()
    dtraj = kmeans.transform(Y).astype(int)
    msm = MaximumLikelihoodMSM(lagtime=lag_msm).fit(dtraj).fetch_model()
    pi = msm.stationary_distribution

    rarity = np.array([1.0 - pi[s] if s < len(pi) else 1.0 for s in dtraj])
    P = msm.transition_matrix
    surprise = np.zeros_like(rarity)
    for i in range(len(rarity) - lag_msm):
        s1, s2 = dtraj[i], dtraj[i + lag_msm]
        if s1 < len(pi) and s2 < len(pi):
            surprise[i] = -np.log(max(P[s1, s2], 1e-12))
        else:
            surprise[i] = -np.log(1e-12)

    nbrs = NearestNeighbors(n_neighbors=min(10, len(Y) - 1)).fit(Y)
    dists, _ = nbrs.kneighbors(Y)
    lden = dists.mean(axis=1)

    def norm01(a):
        q1, q99 = np.quantile(a, [0.01, 0.99])
        return np.clip((a - q1) / (q99 - q1 + 1e-12), 0, 1)

    score = 100.0 * np.median(
        np.vstack([norm01(rarity), norm01(surprise), norm01(lden)]),
        axis=0
    )
    return score, dtraj, msm

# ---- Main loop ----
xtcs = sorted(glob.glob(os.path.join(TRAJ_DIR, "trajectory_*.xtc")))
assert Path(TOP).exists(), f"Missing topology {TOP}"
print(f"[info] Found {len(xtcs)} trajectories")

for xtc in xtcs:
    name = Path(xtc).stem
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run] {name}")

    # Try loading trajectory safely
    try:
        t = md.load(xtc, top=TOP)
    except Exception as e:
        print(f"[skip] Could not load {xtc}: {e}")
        continue

    try:
        dssp = md.compute_dssp(t, simplified=True)
    except Exception:
        dssp = None

    try:
        X, phi, psi = make_features(t)
        scores, dtraj, msm = msm_anomaly(X)
    except Exception as e:
        print(f"[skip] Could not process {name}: {e}")
        continue

    # Save results
    pd.DataFrame(X).to_csv(out / "features.csv", index=False)
    pd.DataFrame({"frame": np.arange(len(scores)), "score": scores}).to_csv(out / "frame_scores.csv", index=False)
    np.save(out / "dtraj.npy", dtraj)
    if dssp is not None: np.save(out / "dssp.npy", dssp)

    events = label_events(phi, psi, dssp)
    json.dump(events, open(out / "events.json", "w"))

    y_true = np.zeros(len(scores), int)
    for arr in events.values():
        for fr in arr:
            for k in (-1, 0, 1):
                if 0 <= fr + k < len(y_true):
                    y_true[fr + k] = 1

    order = np.argsort(scores)[::-1]
    prec = {}
    for k in [5, 10, 20, 50]:
        k = min(k, len(order))
        prec[f"P@{k}"] = float(y_true[order[:k]].mean())
    json.dump(prec, open(out / "agreement.json", "w"))

    with open(out / "SUMMARY.txt", "w") as f:
        f.write(f"{name}: frames={len(scores)}, msm_states={msm.n_states}, "
                f"events phi={len(events['phi'])}, psi={len(events['psi'])}, "
                f"dssp={'NA' if dssp is None else len(events['dssp'])}; "
                f"agreement {prec}\n")

print("[done] Results in outputs/batch/")

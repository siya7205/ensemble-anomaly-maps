# tools/run_msm_tica.py
import numpy as np, argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pyemma
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from deeptime.decomposition import TICA  # modern TICA

def _minmax01(x):
    x = np.asarray(x, float); mn, mx = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn)

def _map_to_active_set(dtraj, active_set, n_clusters):
    """
    Map original cluster labels (0..n_clusters-1) into [0..len(active_set)-1],
    set labels not in active_set to -1.
    """
    m = -np.ones(n_clusters, dtype=int)
    m[np.asarray(active_set, dtype=int)] = np.arange(len(active_set))
    return m[np.asarray(dtraj, dtype=int)]

def main(feat_path, out_dir, lag_tica, lag_msm, n_clusters):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X = np.load(feat_path) if feat_path.endswith(".npy") else np.loadtxt(feat_path, delimiter=",", skiprows=1)

    # ---- TICA (deeptime)
    tica = TICA(lagtime=lag_tica).fit(X).fetch_model()
    Y = tica.transform(X)                  # [T, d]
    np.save(out / "tica_coords.npy", Y)

    # ---- KMeans clustering in TICA space
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    dtraj = km.fit_predict(Y).astype(int)  # labels 0..n_clusters-1
    np.save(out / "dtraj.npy", dtraj)

    # ---- MSM on largest connected set
    msm = pyemma.msm.estimate_markov_model([dtraj], lag=lag_msm,
                                           reversible=True, connectivity='largest')
    np.save(out / "P.npy", msm.P); np.save(out / "pi.npy", msm.pi)

    # ---- Validation (best-effort plots)
    lags = np.unique(np.linspace(max(1, lag_msm//2),
                                 min(len(dtraj)//2, lag_msm*5), 6, dtype=int))
    try:
        its = pyemma.msm.its([dtraj], lags=lags, errors='bayes')
        plt.figure(); pyemma.plots.plot_implied_timescales(its)
        plt.tight_layout(); plt.savefig(out/"its.png", dpi=180)
    except Exception as e:
        print("[warn] ITS plotting failed:", e)

    try:
        ck = msm.cktest(2)
        pyemma.plots.plot_cktest(ck)
        plt.legend(loc='upper center', ncol=2, frameon=False)
        plt.tight_layout(); plt.savefig(out/"cktest.png", dpi=180)
    except Exception as e:
        print("[warn] CK plotting failed:", e)

    # ---- Map labels to active-set indices (critical!)
    dmap = _map_to_active_set(dtraj, msm.active_set, n_clusters)
    mask = dmap >= 0  # frames that live in the active set

    # ---- Scores: rarity + transition surprise + low density
    rarity = np.ones(len(dtraj))
    rarity[mask] = 1.0 - msm.pi[dmap[mask]]

    trans = np.zeros(len(dtraj))
    if len(dtraj) > lag_msm:
        P = msm.P
        # only compute where both t and t+tau are mapped
        m2 = mask.copy()
        m2[:-lag_msm] &= mask[lag_msm:]
        pair_i = dmap[:-lag_msm][m2[:-lag_msm]]
        pair_j = dmap[lag_msm:][m2[:-lag_msm]]
        tmp = np.zeros(len(dtraj))
        tmp_indices = np.nonzero(m2[:-lag_msm])[0]
        tmp[tmp_indices] = -np.log(np.maximum(P[pair_i, pair_j], 1e-12))
        # smear each transition surprise across the interval
        for t in tmp_indices:
            trans[t:t+lag_msm] += tmp[t] / lag_msm

    nn = NearestNeighbors(n_neighbors=min(20, len(Y)-1)).fit(Y)
    dists, _ = nn.kneighbors(Y); dens = -dists[:, -1]

    r = _minmax01(rarity); s = _minmax01(trans); q = _minmax01(-dens)
    M = np.vstack([r, s, q]).T; M.sort(axis=1)
    score100 = 100.0 * M[:,1]  # median (trimmed mean also fine)

    import pandas as pd
    pd.DataFrame({'frame': np.arange(len(score100)), 'score': score100}).to_csv(out/'frame_scores.csv', index=False)
    print(f"[ok] wrote {out/'frame_scores.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lag_tica", type=int, default=10)
    ap.add_argument("--lag_msm", type=int, default=30)   # a bit shorter for 213 frames
    ap.add_argument("--n_clusters", type=int, default=30) # safer for small T
    args = ap.parse_args()
    main(args.features, args.out_dir, args.lag_tica, args.lag_msm, args.n_clusters)

# analysis/cluster_ops.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def cluster_windows(
    per_window_deep_csv,
    out_dir=None,
    algo="dbscan",
    eps=0.8,
    min_samples=4,
    small_cluster_size=5,
    recon_q=0.95,
):
    """
    Clusters trajectory windows in latent space and flags anomalies.

    Inputs
    ------
    per_window_deep_csv : path to outputs/run-traj-*/per_window_deep.csv
        Must contain: start, end, deep_recon_error, and z* latent columns
    out_dir : directory to write results (default: <csv parent>/deep)
    algo : "dbscan" or "hdbscan" (falls back to dbscan if hdbscan not installed)
    eps, min_samples : DBSCAN params
    small_cluster_size : clusters smaller than this are marked anomalous
    recon_q : quantile for "high reconstruction error" threshold (e.g., 0.95)

    Writes
    ------
    latent_clusters.csv         (start, end, z*, cluster)
    cluster_assignments.csv     (start, end, z*, recon, cluster, flags)
    cluster_summary.csv         (per-cluster size, mean recon, anomaly frac)
    anomaly_windows.csv         (only anomalous windows, sorted)
    cluster_recon_timeline.png  (recon error over window index with anomaly marks)
    """
    per_window_deep_csv = Path(per_window_deep_csv)
    df = pd.read_csv(per_window_deep_csv)

    # Basic column checks
    for c in ("start", "end", "deep_recon_error"):
        if c not in df.columns:
            raise ValueError(f"{per_window_deep_csv} must contain column '{c}'")
    zcols = [c for c in df.columns if c.startswith("z")]
    if not zcols:
        raise ValueError(f"{per_window_deep_csv} missing latent columns (z1, z2, ...).")

    # Standardize latent space
    Z = df[zcols].values
    Zs = StandardScaler().fit_transform(Z)

    # Choose clustering algo
    algo = (algo or "dbscan").lower()
    labels = None
    if algo == "hdbscan":
        try:
            import hdbscan  # type: ignore
            labels = hdbscan.HDBSCAN(min_cluster_size=small_cluster_size).fit_predict(Zs)
        except Exception:
            print("[cluster_ops] hdbscan not available; falling back to dbscan")
            algo = "dbscan"

    if labels is None:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Zs)

    # Assemble outputs
    out_dir = Path(out_dir) if out_dir else per_window_deep_csv.parent / "deep"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = df[["start", "end"] + zcols + ["deep_recon_error"]].copy()
    out["cluster"] = labels

    # Cluster sizes (include noise=-1)
    sizes = pd.Series(labels).value_counts().to_dict()

    def is_small(lbl):
        if lbl == -1:
            return True
        return sizes.get(lbl, 0) < small_cluster_size

    out["small_cluster"] = out["cluster"].apply(is_small)

    # Recon error threshold from quantile
    thr = float(out["deep_recon_error"].quantile(recon_q))
    out["high_recon"] = out["deep_recon_error"] >= thr

    # Final anomaly decision
    out["anomaly"] = (out["cluster"] == -1) | out["small_cluster"] | out["high_recon"]
    out["anomaly_reason"] = np.where(
        out["cluster"] == -1,
        "noise",
        np.where(out["small_cluster"], "small_cluster",
                 np.where(out["high_recon"], "high_recon", "")),
    )

    # Save minimal cluster file required by downstream tools
    out_min = out[["start", "end"] + zcols + ["cluster"]]
    out_min.to_csv(out_dir / "latent_clusters.csv", index=False)

    # Save full assignments
    out.to_csv(out_dir / "cluster_assignments.csv", index=False)

    # Cluster summary
    summary = (
        out.groupby("cluster")
           .agg(n=("cluster", "size"),
                mean_recon=("deep_recon_error", "mean"),
                frac_anom=("anomaly", "mean"))
           .reset_index()
    )
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)

    # Anomaly-only table
    anomalies = out[out["anomaly"]].copy()
    anomalies = anomalies.sort_values(
        ["high_recon", "small_cluster", "deep_recon_error"], ascending=[False, False, False]
    )
    anomalies.to_csv(out_dir / "anomaly_windows.csv", index=False)

    # Quick plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(len(out)), out["deep_recon_error"])
        an_idx = np.where(out["anomaly"])[0]
        plt.scatter(an_idx, out.loc[out["anomaly"], "deep_recon_error"])
        plt.title("Reconstruction error over windows")
        plt.xlabel("window index")
        plt.ylabel("reconstruction error")
        plt.tight_layout()
        plt.savefig(out_dir / "cluster_recon_timeline.png", dpi=150)
        plt.close()
    except Exception as e:
        print("[cluster_ops] plot failed:", e)

    return {
        "out_dir": str(out_dir),
        "algo": algo,
        "recon_threshold": thr,
        "n_windows": int(len(out)),
        "n_anomalies": int(out["anomaly"].sum()),
    }

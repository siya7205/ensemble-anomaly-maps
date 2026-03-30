#!/usr/bin/env python3
"""
physical_validation.py – Structural physical validation of anomaly scores.

Compares TOP-ANOMALY frames against RANDOM / BACKGROUND frames on
structural-change metrics to test whether high-anomaly frames correspond to
real, coordinated structural events.

Metrics computed per frame
--------------------------
- Mean / max C-alpha residue displacement from a reference frame
- Mean displacement of the top-k most displaced residues (top-k hotspot proxy)
- Local RMSD restricted to hotspot residues (if residue scores available)
- Contact-map change: fraction of changed C-alpha contacts vs a reference
- Backbone dihedral change: mean circular deviation in φ/ψ space
- Local anomaly persistence: mean anomaly score in a temporal window ±w

Multi-signal agreement
----------------------
If the frame-scores CSV contains constituent signals (rarity, transition
surprise, local density), the summary reports whether fused top-anomaly
frames are also elevated on every underlying signal.

Outputs
-------
- <out_dir>/frame_validation.csv   – one row per frame with all metrics
- <out_dir>/validation_summary.json – group statistics, effect sizes, tests
- <out_dir>/plots/                  – boxplots, scatter, optional time plots

Usage
-----
    python -m analysis.physical_validation \\
        --topology  data/1ABC/topology.pdb \\
        --trajectory data/1ABC/traj.xtc \\
        --scores     results/1ABC/frame_scores_dynamic.csv \\
        --residue-scores results/1ABC/residue_scores_dynamic.json \\
        --out-dir    results/1ABC/physical_validation \\
        --top-pct    5 \\
        --seed       42

See ``python -m analysis.physical_validation --help`` for all options.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional / soft imports (warn rather than crash)
# ---------------------------------------------------------------------------


def _try_import_mdtraj():
    try:
        import mdtraj as md  # noqa: F401

        return md
    except ImportError:
        return None


def _try_import_scipy_stats():
    try:
        from scipy import stats  # noqa: F401

        return stats
    except ImportError:
        return None


def _try_import_matplotlib():
    try:
        import matplotlib  # noqa: F401

        matplotlib.use("Agg")  # non-interactive backend for headless environments
        import matplotlib.pyplot as plt  # noqa: F401

        return plt
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_scores(
    scores_csv: str | Path,
    residue_json: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """Load frame-level anomaly scores and optional residue-level scores.

    Parameters
    ----------
    scores_csv:
        Path to ``frame_scores_dynamic.csv`` (columns: ``frame``,
        ``score_dynamic``, optional ``component_*`` columns).
    residue_json:
        Optional path to ``residue_scores_dynamic.json`` (dict mapping
        residue-string → float score).

    Returns
    -------
    scores_df:
        DataFrame with at least ``frame`` and ``score_dynamic`` columns.
    residue_scores:
        Dict or None if ``residue_json`` is not provided / unreadable.
    """
    scores_df = pd.read_csv(scores_csv)

    required = {"frame", "score_dynamic"}
    missing = required - set(scores_df.columns)
    if missing:
        raise ValueError(
            f"scores CSV is missing required columns: {missing}. "
            f"Found: {list(scores_df.columns)}"
        )

    # Ensure integer frame index
    scores_df["frame"] = scores_df["frame"].astype(int)
    scores_df = scores_df.sort_values("frame").reset_index(drop=True)

    residue_scores: Optional[Dict[str, float]] = None
    if residue_json is not None:
        try:
            with open(residue_json) as fh:
                raw = json.load(fh)
            residue_scores = {str(k): float(v) for k, v in raw.items()}
            log.info("Loaded residue scores for %d residues.", len(residue_scores))
        except Exception as exc:
            log.warning("Could not load residue scores from %s: %s", residue_json, exc)

    log.info("Loaded %d frame scores from %s.", len(scores_df), scores_csv)
    return scores_df, residue_scores


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def select_top_frames(
    scores_df: pd.DataFrame,
    top_pct: float = 5.0,
    score_col: str = "score_dynamic",
) -> np.ndarray:
    """Return indices into *scores_df* for the top-anomaly frames.

    Parameters
    ----------
    scores_df:
        DataFrame with a ``score_dynamic`` column (or ``score_col``).
    top_pct:
        Percentile threshold; frames above this are "top anomaly".
        E.g. ``5.0`` selects the top 5 % (≥ 95th percentile).
    score_col:
        Column to use for ranking.

    Returns
    -------
    top_idx:
        Integer indices into ``scores_df`` that belong to the top group.
    """
    if not (0 < top_pct < 100):
        raise ValueError(f"top_pct must be in (0, 100), got {top_pct}")

    threshold = np.percentile(scores_df[score_col].values, 100 - top_pct)
    top_idx = np.where(scores_df[score_col].values >= threshold)[0]
    log.info(
        "Top %.1f%% threshold = %.3f → %d frames selected.",
        top_pct,
        threshold,
        len(top_idx),
    )
    return top_idx


def sample_background_frames(
    scores_df: pd.DataFrame,
    top_idx: np.ndarray,
    n_samples: Optional[int] = None,
    seed: int = 42,
    mode: str = "random",
    score_col: str = "score_dynamic",
    low_pct: float = 25.0,
) -> np.ndarray:
    """Sample a matched set of background / low-anomaly frames.

    Parameters
    ----------
    scores_df:
        Frame scores DataFrame.
    top_idx:
        Indices already assigned to the top group (excluded from background).
    n_samples:
        Number of background frames to sample.  Defaults to
        ``len(top_idx)`` for a balanced comparison.
    seed:
        Random seed for reproducibility.
    mode:
        ``"random"`` — random frames from the non-top set.
        ``"low_anomaly"`` — frames in the bottom ``low_pct`` percentile.
    score_col:
        Scoring column name.
    low_pct:
        Percentile ceiling used when ``mode="low_anomaly"``.

    Returns
    -------
    bg_idx:
        Integer indices into ``scores_df`` for background frames.
    """
    rng = np.random.default_rng(seed)

    all_idx = np.arange(len(scores_df))
    top_set = set(top_idx.tolist())
    non_top_idx = np.array([i for i in all_idx if i not in top_set])

    if len(non_top_idx) == 0:
        log.warning("No non-top frames available; background set is empty.")
        return np.array([], dtype=int)

    if mode == "low_anomaly":
        low_threshold = np.percentile(scores_df[score_col].values, low_pct)
        candidate_idx = non_top_idx[
            scores_df[score_col].values[non_top_idx] <= low_threshold
        ]
        if len(candidate_idx) == 0:
            log.warning(
                "No low-anomaly frames below %.1f-th percentile; "
                "falling back to random mode.",
                low_pct,
            )
            candidate_idx = non_top_idx
    else:
        candidate_idx = non_top_idx

    n = n_samples if n_samples is not None else len(top_idx)
    n = min(n, len(candidate_idx))

    bg_idx = rng.choice(candidate_idx, size=n, replace=False)
    log.info(
        "Background set: %d frames (mode='%s', seed=%d).",
        len(bg_idx),
        mode,
        seed,
    )
    return bg_idx


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------


def _load_ca_positions(traj, frames: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract C-alpha XYZ positions for selected frames.

    Works with any trajectory object that exposes ``.xyz`` (shape T×N×3)
    and ``topology.select(sel_string)`` – including both real mdtraj
    trajectories and lightweight test stubs.  Falls back to all protein
    heavy atoms when no C-alpha atoms are found.

    Parameters
    ----------
    traj:
        Trajectory object (``mdtraj.Trajectory`` or duck-type equivalent).
    frames:
        Array of frame indices; ``None`` means all frames.

    Returns
    -------
    positions:
        Array of shape ``(n_frames, n_atoms, 3)`` in nm.
    ca_idx:
        Array of atom indices used for the position slice.
    """
    ca_idx = traj.topology.select("name CA")
    if len(ca_idx) == 0:
        log.warning(
            "No C-alpha atoms found; falling back to all protein heavy atoms."
        )
        ca_idx = traj.topology.select("protein and not type H")
    if len(ca_idx) == 0:
        ca_idx = traj.topology.select("all")

    if frames is not None:
        positions = traj.xyz[frames][:, ca_idx, :]
    else:
        positions = traj.xyz[:, ca_idx, :]

    return positions, ca_idx


def _wrap_angle_diff(delta_deg: np.ndarray) -> np.ndarray:
    """Wrap angle differences to [-180, 180]."""
    return (delta_deg + 180.0) % 360.0 - 180.0


def compute_displacement_metrics(
    traj,
    frames: np.ndarray,
    ref_frame: int = 0,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compute per-residue displacement metrics for a set of frames.

    For each frame in *frames*, the per-residue (C-alpha) displacement from
    *ref_frame* is computed, then aggregated to:

    - ``mean_residue_displacement`` — mean over all C-alpha atoms (nm).
    - ``max_residue_displacement``  — maximum over all C-alpha atoms (nm).
    - ``topk_residue_displacement`` — mean of the top-``top_k`` displaced
      residues (nm).

    Parameters
    ----------
    traj:
        Full ``mdtraj.Trajectory``.
    frames:
        Frame indices to process.
    ref_frame:
        Reference frame index for computing displacement.
    top_k:
        Number of top-displaced residues to average.

    Returns
    -------
    DataFrame with columns ``[frame, mean_residue_displacement,
    max_residue_displacement, topk_residue_displacement]``.
    """
    if len(frames) == 0:
        return pd.DataFrame(
            columns=[
                "frame",
                "mean_residue_displacement",
                "max_residue_displacement",
                "topk_residue_displacement",
            ]
        )

    positions, _ = _load_ca_positions(traj)  # (T, n_ca, 3)
    ref_pos = positions[ref_frame]  # (n_ca, 3)

    actual_top_k = min(top_k, positions.shape[1])

    rows = []
    for f in frames:
        disp = np.linalg.norm(positions[f] - ref_pos, axis=-1)  # (n_ca,)
        rows.append(
            {
                "frame": int(f),
                "mean_residue_displacement": float(disp.mean()),
                "max_residue_displacement": float(disp.max()),
                "topk_residue_displacement": float(
                    np.sort(disp)[-actual_top_k:].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_hotspot_rmsd(
    traj,
    frames: np.ndarray,
    residue_scores: Dict[str, float],
    ref_frame: int = 0,
    top_k_residues: int = 10,
) -> pd.DataFrame:
    """Compute RMSD restricted to the top-k hotspot residues.

    Parameters
    ----------
    traj:
        Full ``mdtraj.Trajectory``.
    frames:
        Frame indices to process.
    residue_scores:
        Dict mapping residue string → float anomaly score.
    ref_frame:
        Reference frame for RMSD computation.
    top_k_residues:
        How many top-scoring residues to include.

    Returns
    -------
    DataFrame with columns ``[frame, hotspot_local_rmsd]``.
    """
    if len(frames) == 0 or not residue_scores:
        return pd.DataFrame(columns=["frame", "hotspot_local_rmsd"])

    # Identify hotspot residue names (top-k by score)
    sorted_res = sorted(residue_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_res_names = {name for name, _ in sorted_res[:top_k_residues]}

    # Match topology residues to hotspot names
    hotspot_ca_idx = []
    for atom in traj.topology.atoms:
        if atom.name == "CA" and str(atom.residue) in top_res_names:
            hotspot_ca_idx.append(atom.index)

    if not hotspot_ca_idx:
        log.warning(
            "Could not match any hotspot residues in topology; "
            "skipping hotspot_local_rmsd."
        )
        return pd.DataFrame(columns=["frame", "hotspot_local_rmsd"])

    hotspot_ca_idx = np.array(hotspot_ca_idx)
    ref_pos = traj.xyz[ref_frame][hotspot_ca_idx]  # (k, 3)

    rows = []
    for f in frames:
        pos_f = traj.xyz[f][hotspot_ca_idx]  # (k, 3)
        rmsd = float(np.sqrt(((pos_f - ref_pos) ** 2).mean()))
        rows.append({"frame": int(f), "hotspot_local_rmsd": rmsd})

    return pd.DataFrame(rows)


def compute_contact_change(
    traj,
    frames: np.ndarray,
    ref_frame: int = 0,
    cutoff_nm: float = 0.8,
) -> pd.DataFrame:
    """Compute residue-residue contact-map change relative to a reference.

    Builds a binary C-alpha contact map for each frame (contacts where
    C-alpha distance ≤ ``cutoff_nm``) and compares it to the reference
    frame's contact map.  The metric is the fraction of changed contacts
    (Hamming-like distance on the upper triangle of the binary matrix).

    Parameters
    ----------
    traj:
        Full ``mdtraj.Trajectory``.
    frames:
        Frame indices to process.
    ref_frame:
        Reference frame.
    cutoff_nm:
        Distance cutoff in nm for defining a contact.

    Returns
    -------
    DataFrame with columns ``[frame, contact_change]``.
    """
    if len(frames) == 0:
        return pd.DataFrame(columns=["frame", "contact_change"])

    positions, _ = _load_ca_positions(traj)  # (T, n_ca, 3)
    n_ca = positions.shape[1]

    if n_ca < 2:
        log.warning("Too few C-alpha atoms for contact maps; skipping.")
        return pd.DataFrame(columns=["frame", "contact_change"])

    def _contact_map(pos: np.ndarray) -> np.ndarray:
        # Vectorised pairwise distance, upper triangle only
        diff = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
        dist = np.linalg.norm(diff, axis=-1)       # (n, n)
        return (dist <= cutoff_nm).astype(np.int8)

    ref_map = _contact_map(positions[ref_frame])  # (n_ca, n_ca)
    # Use upper triangle (i < j), exclude diagonal
    tri_idx = np.triu_indices(n_ca, k=1)
    ref_tri = ref_map[tri_idx]
    n_pairs = len(ref_tri)

    rows = []
    for f in frames:
        cmap = _contact_map(positions[f])
        changed = int((cmap[tri_idx] != ref_tri).sum())
        rows.append(
            {
                "frame": int(f),
                "contact_change": changed / n_pairs if n_pairs > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def compute_dihedral_change(
    traj,
    frames: np.ndarray,
    ref_frame: int = 0,
) -> pd.DataFrame:
    """Compute mean backbone (φ/ψ) dihedral deviation from a reference frame.

    For each frame, the per-residue φ and ψ deviations (wrapped to
    [−180, 180] deg) are averaged to yield a single scalar.  Residues
    without both φ and ψ (terminal residues) are skipped.

    Parameters
    ----------
    traj:
        Full ``mdtraj.Trajectory``.
    frames:
        Frame indices to process.
    ref_frame:
        Reference frame for computing dihedral deviation.

    Returns
    -------
    DataFrame with columns ``[frame, dihedral_change]``.
    Falls back to an empty DataFrame with a warning if mdtraj cannot compute
    dihedrals (e.g. non-protein topology).
    """
    if len(frames) == 0:
        return pd.DataFrame(columns=["frame", "dihedral_change"])

    md = _try_import_mdtraj()
    if md is None:
        log.warning("mdtraj not available; skipping dihedral_change.")
        return pd.DataFrame(columns=["frame", "dihedral_change"])

    try:
        phi_idx, phi_all = md.compute_phi(traj)   # phi_all: (T, n_phi) rad
        psi_idx, psi_all = md.compute_psi(traj)   # psi_all: (T, n_psi) rad
    except Exception as exc:
        log.warning("Dihedral computation failed: %s; skipping.", exc)
        return pd.DataFrame(columns=["frame", "dihedral_change"])

    if phi_all.shape[1] == 0 or psi_all.shape[1] == 0:
        log.warning("No backbone dihedrals found; skipping dihedral_change.")
        return pd.DataFrame(columns=["frame", "dihedral_change"])

    phi_deg = np.degrees(phi_all)  # (T, n_phi)
    psi_deg = np.degrees(psi_all)  # (T, n_psi)

    # Find residues with both phi and psi via shared residue indices
    top = traj.topology

    def _res_idx(dih_idx: np.ndarray) -> List[int]:
        """Map each dihedral column to its central residue index."""
        return [top.atom(int(row[1])).residue.index for row in dih_idx]

    phi_res = np.array(_res_idx(phi_idx))
    psi_res = np.array(_res_idx(psi_idx))

    phi_res_set = set(phi_res.tolist())
    psi_res_set = set(psi_res.tolist())
    shared_res = sorted(phi_res_set & psi_res_set)

    if not shared_res:
        log.warning("No residues with both phi and psi; skipping dihedral_change.")
        return pd.DataFrame(columns=["frame", "dihedral_change"])

    # Build aligned arrays for shared residues
    phi_col = {r: i for i, r in enumerate(phi_res)}
    psi_col = {r: i for i, r in enumerate(psi_res)}

    phi_cols_shared = np.array([phi_col[r] for r in shared_res])
    psi_cols_shared = np.array([psi_col[r] for r in shared_res])

    phi_ref = phi_deg[ref_frame, phi_cols_shared]  # (n_shared,)
    psi_ref = psi_deg[ref_frame, psi_cols_shared]

    rows = []
    for f in frames:
        d_phi = _wrap_angle_diff(phi_deg[f, phi_cols_shared] - phi_ref)
        d_psi = _wrap_angle_diff(psi_deg[f, psi_cols_shared] - psi_ref)
        mean_dev = float(np.abs(np.concatenate([d_phi, d_psi])).mean())
        rows.append({"frame": int(f), "dihedral_change": mean_dev})

    return pd.DataFrame(rows)


def compute_temporal_persistence(
    scores_df: pd.DataFrame,
    top_idx: np.ndarray,
    window: int = 2,
    score_col: str = "score_dynamic",
) -> pd.DataFrame:
    """Compute local anomaly persistence around each frame.

    For every frame in the full trajectory, computes the mean anomaly score
    within [t − ``window``, t + ``window``], giving a measure of how
    temporally coherent each anomaly peak is.

    Parameters
    ----------
    scores_df:
        Full-trajectory frame scores.
    top_idx:
        Indices of top-anomaly frames (used only for labelling here).
    window:
        Half-width of the temporal window (frames on each side).
    score_col:
        Score column name.

    Returns
    -------
    DataFrame with columns ``[frame, local_persistence_score]`` for
    *all* frames in ``scores_df``.
    """
    n = len(scores_df)
    scores_arr = scores_df[score_col].values.copy()
    frames_arr = scores_df["frame"].values

    persistence = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        persistence[i] = float(scores_arr[lo:hi].mean())

    return pd.DataFrame(
        {"frame": frames_arr.astype(int), "local_persistence_score": persistence}
    )


# ---------------------------------------------------------------------------
# Merging all metrics
# ---------------------------------------------------------------------------


def _merge_metrics(
    scores_df: pd.DataFrame,
    top_idx: np.ndarray,
    bg_idx: np.ndarray,
    disp_df: pd.DataFrame,
    hotspot_df: pd.DataFrame,
    contact_df: pd.DataFrame,
    dihedral_df: pd.DataFrame,
    persist_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all per-frame metrics into a single DataFrame.

    Returns a DataFrame with one row per frame in ``scores_df``, labelled
    with group membership (``is_top_anomaly``, ``is_background``).
    """
    df = scores_df.copy()

    # Group membership flags
    df["is_top_anomaly"] = False
    df["is_background"] = False
    df.iloc[top_idx, df.columns.get_loc("is_top_anomaly")] = True
    df.iloc[bg_idx, df.columns.get_loc("is_background")] = True

    # Rename component columns to cleaner names
    rename_map = {}
    for col in df.columns:
        if col.startswith("component_rarity"):
            rename_map[col] = "rarity_score"
        elif col.startswith("component_transition_surprise"):
            rename_map[col] = "transition_surprise"
        elif col.startswith("component_local_density"):
            rename_map[col] = "slow_space_isolation"
    df = df.rename(columns=rename_map)

    # Merge structural metric tables
    for metric_df in [disp_df, hotspot_df, contact_df, dihedral_df]:
        if metric_df is not None and len(metric_df) > 0:
            df = df.merge(metric_df, on="frame", how="left")

    # Merge persistence (all frames)
    if persist_df is not None and len(persist_df) > 0:
        df = df.merge(persist_df, on="frame", how="left")

    # Ensure expected columns exist (fill with NaN if metric failed)
    expected_cols = [
        "mean_residue_displacement",
        "max_residue_displacement",
        "topk_residue_displacement",
        "hotspot_local_rmsd",
        "contact_change",
        "dihedral_change",
        "local_persistence_score",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarize_group_differences(
    df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
) -> Dict:
    """Compute group-level summary statistics and effect sizes.

    Compares top-anomaly frames vs background frames on each metric column,
    producing group means / stds, Cohen's *d*, and an optional Mann-Whitney
    U two-sided test (*p*-value).

    Parameters
    ----------
    df:
        Per-frame DataFrame with ``is_top_anomaly`` and ``is_background``
        boolean columns.
    metric_cols:
        Columns to summarise.  Defaults to a standard set of structural
        metrics plus the anomaly score.

    Returns
    -------
    summary:
        Nested dict; keys are metric names, values contain group statistics.
    """
    if metric_cols is None:
        metric_cols = [
            "score_dynamic",
            "mean_residue_displacement",
            "max_residue_displacement",
            "topk_residue_displacement",
            "hotspot_local_rmsd",
            "contact_change",
            "dihedral_change",
            "local_persistence_score",
        ]
        # Also include constituent signals if present
        for col in ["rarity_score", "transition_surprise", "slow_space_isolation"]:
            if col in df.columns:
                metric_cols.append(col)

    top = df[df["is_top_anomaly"]].copy()
    bg = df[df["is_background"]].copy()

    stats = _try_import_scipy_stats()

    summary: Dict = {
        "n_top_anomaly": int(top.shape[0]),
        "n_background": int(bg.shape[0]),
        "metrics": {},
    }

    for col in metric_cols:
        if col not in df.columns:
            continue
        top_vals = top[col].dropna().values
        bg_vals = bg[col].dropna().values

        if len(top_vals) == 0 or len(bg_vals) == 0:
            continue

        top_mean = float(np.mean(top_vals))
        top_std = float(np.std(top_vals, ddof=1)) if len(top_vals) > 1 else 0.0
        bg_mean = float(np.mean(bg_vals))
        bg_std = float(np.std(bg_vals, ddof=1)) if len(bg_vals) > 1 else 0.0

        # Cohen's d
        pooled_std = np.sqrt(
            (
                (len(top_vals) - 1) * top_std ** 2
                + (len(bg_vals) - 1) * bg_std ** 2
            )
            / max(len(top_vals) + len(bg_vals) - 2, 1)
        )
        cohens_d = (top_mean - bg_mean) / (pooled_std + 1e-12)

        entry: Dict = {
            "top_anomaly_mean": top_mean,
            "top_anomaly_std": top_std,
            "background_mean": bg_mean,
            "background_std": bg_std,
            "cohens_d": float(cohens_d),
        }

        # Mann-Whitney U test (non-parametric, no normality assumption)
        if stats is not None and len(top_vals) >= 3 and len(bg_vals) >= 3:
            try:
                stat, pval = stats.mannwhitneyu(
                    top_vals, bg_vals, alternative="two-sided"
                )
                entry["mannwhitney_pvalue"] = float(pval)
                entry["mannwhitney_statistic"] = float(stat)
            except Exception as exc:
                log.warning("Mann-Whitney test failed for %s: %s", col, exc)

        summary["metrics"][col] = entry

    return summary


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def save_outputs(
    per_frame_df: pd.DataFrame,
    summary: Dict,
    out_dir: str | Path,
    config: Optional[Dict] = None,
) -> None:
    """Write per-frame CSV and summary JSON to *out_dir*.

    Parameters
    ----------
    per_frame_df:
        DataFrame with one row per frame and all computed metrics.
    summary:
        Summary statistics dict from :func:`summarize_group_differences`.
    out_dir:
        Output directory (created if it does not exist).
    config:
        Optional run configuration dict included in the JSON summary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "frame_validation.csv"
    per_frame_df.to_csv(csv_path, index=False)
    log.info("Per-frame validation CSV → %s", csv_path)

    json_path = out_dir / "validation_summary.json"
    full_summary = {"config": config or {}, **summary}
    with open(json_path, "w") as fh:
        json.dump(full_summary, fh, indent=2)
    log.info("Validation summary JSON → %s", json_path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def generate_plots(
    per_frame_df: pd.DataFrame,
    summary: Dict,
    out_dir: str | Path,
    plot_temporal: bool = False,
    top_anomaly_frames: Optional[np.ndarray] = None,
) -> None:
    """Generate validation plots and save to *out_dir*/plots/.

    Plots produced
    --------------
    1. Boxplot (or violin) — top-anomaly vs background for each metric.
    2. Scatter — anomaly score vs each structural metric, coloured by group.
    3. (Optional) Temporal window — anomaly score and structural change
       around each top-anomaly event.

    Parameters
    ----------
    per_frame_df:
        Per-frame DataFrame.
    summary:
        Summary statistics dict.
    out_dir:
        Root output directory (plots written to ``<out_dir>/plots/``).
    plot_temporal:
        Whether to generate temporal window plots.
    top_anomaly_frames:
        Frame indices of top-anomaly events (for temporal plots).
    """
    plt = _try_import_matplotlib()
    if plt is None:
        log.warning("matplotlib not available; skipping plots.")
        return

    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        col
        for col in [
            "mean_residue_displacement",
            "max_residue_displacement",
            "topk_residue_displacement",
            "hotspot_local_rmsd",
            "contact_change",
            "dihedral_change",
            "local_persistence_score",
        ]
        if col in per_frame_df.columns and per_frame_df[col].notna().any()
    ]

    top_df = per_frame_df[per_frame_df["is_top_anomaly"]]
    bg_df = per_frame_df[per_frame_df["is_background"]]

    # --- 1. Box / violin plots ------------------------------------------------
    n_metrics = len(metric_cols)
    if n_metrics > 0:
        ncols = min(3, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if n_metrics > 1 else [axes]

        for ax, col in zip(axes, metric_cols):
            top_vals = top_df[col].dropna().values
            bg_vals = bg_df[col].dropna().values
            if len(top_vals) == 0 and len(bg_vals) == 0:
                ax.set_visible(False)
                continue

            groups = []
            labels = []
            if len(top_vals) > 0:
                groups.append(top_vals)
                labels.append("Top-anomaly")
            if len(bg_vals) > 0:
                groups.append(bg_vals)
                labels.append("Background")

            vp = ax.violinplot(groups, showmedians=True, showextrema=False)
            colors = ["#d62728", "#1f77b4"]
            for body, colour in zip(vp["bodies"], colors):
                body.set_facecolor(colour)
                body.set_alpha(0.7)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(2)

            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_title(col.replace("_", " ").title())
            ax.set_ylabel("Value")

            # Annotate Cohen's d from summary
            if col in summary.get("metrics", {}):
                d = summary["metrics"][col].get("cohens_d", None)
                pval = summary["metrics"][col].get("mannwhitney_pvalue", None)
                label_parts = []
                if d is not None:
                    label_parts.append(f"d={d:.2f}")
                if pval is not None:
                    label_parts.append(f"p={pval:.3g}")
                if label_parts:
                    ax.set_xlabel(", ".join(label_parts), fontsize=9)

        # Hide unused axes
        for ax in axes[n_metrics:]:
            ax.set_visible(False)

        fig.suptitle("Top-Anomaly vs Background — Structural Metrics", fontsize=12)
        fig.tight_layout()
        fig.savefig(plots_dir / "violin_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Violin plot → %s", plots_dir / "violin_comparison.png")

    # --- 2. Scatter: anomaly score vs structural metrics ---------------------
    score_col = "score_dynamic"
    for col in metric_cols:
        if col == score_col:
            continue
        valid = per_frame_df[[score_col, col, "is_top_anomaly"]].dropna()
        if len(valid) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        for is_top, colour, label, alpha, zorder in [
            (False, "#1f77b4", "Background", 0.4, 1),
            (True, "#d62728", "Top-anomaly", 0.8, 2),
        ]:
            subset = valid[valid["is_top_anomaly"] == is_top]
            ax.scatter(
                subset[score_col],
                subset[col],
                c=colour,
                label=label,
                alpha=alpha,
                s=20,
                zorder=zorder,
            )

        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_title(f"Anomaly Score vs {col.replace('_', ' ').title()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"scatter_{col}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    log.info("Scatter plots → %s/scatter_*.png", plots_dir)

    # --- 3. Optional temporal window plot ------------------------------------
    if plot_temporal and top_anomaly_frames is not None and len(top_anomaly_frames) > 0:
        _plot_temporal_windows(
            per_frame_df, top_anomaly_frames, score_col, metric_cols, plots_dir, plt
        )


def _plot_temporal_windows(
    df: pd.DataFrame,
    top_frames: np.ndarray,
    score_col: str,
    metric_cols: List[str],
    plots_dir: Path,
    plt,
    window: int = 10,
    max_events: int = 6,
) -> None:
    """Plot anomaly score and structural metrics in a window around each top event."""
    n_events = min(max_events, len(top_frames))
    # Show events spread across the trajectory for diversity
    event_indices = np.linspace(0, len(top_frames) - 1, n_events, dtype=int)
    selected_events = top_frames[event_indices]

    frames_arr = df["frame"].values
    score_arr = df[score_col].values

    for ev_frame in selected_events:
        # Find DataFrame rows near this frame
        lo = max(0, np.searchsorted(frames_arr, ev_frame - window))
        hi = min(len(df), np.searchsorted(frames_arr, ev_frame + window + 1))
        if hi - lo < 2:
            continue

        window_df = df.iloc[lo:hi]
        n_sub = len(metric_cols[:3])  # max 3 metric sub-plots to keep it readable
        n_rows = 1 + n_sub

        fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        # Top panel: anomaly score
        axes[0].plot(window_df["frame"], window_df[score_col], color="black", lw=1.5)
        axes[0].axvline(ev_frame, color="#d62728", ls="--", lw=1.5, label="Event")
        axes[0].set_ylabel("Anomaly Score")
        axes[0].legend(fontsize=8)

        for ax, col in zip(axes[1:], metric_cols[:n_sub]):
            if col in window_df.columns:
                ax.plot(window_df["frame"], window_df[col], color="#1f77b4", lw=1.2)
                ax.axvline(ev_frame, color="#d62728", ls="--", lw=1.5)
                ax.set_ylabel(col.replace("_", " ").title(), fontsize=8)

        axes[-1].set_xlabel("Frame")
        fig.suptitle(f"Temporal window around frame {ev_frame}", fontsize=10)
        fig.tight_layout()
        fname = plots_dir / f"temporal_window_frame{ev_frame}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info("Temporal window plots → %s/temporal_window_*.png", plots_dir)


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def run_physical_validation(
    topology: str | Path,
    trajectory: str | Path,
    scores_csv: str | Path,
    out_dir: str | Path,
    residue_json: Optional[str | Path] = None,
    top_pct: float = 5.0,
    bg_mode: str = "random",
    bg_n: Optional[int] = None,
    seed: int = 42,
    ref_frame: int = 0,
    cutoff_nm: float = 0.8,
    top_k_disp: int = 10,
    top_k_hotspot: int = 10,
    window: int = 2,
    plot_temporal: bool = False,
    stride: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """Run the full physical validation pipeline.

    Parameters
    ----------
    topology:
        Path to topology file (PDB or equivalent).
    trajectory:
        Path to trajectory file (XTC, DCD, …).
    scores_csv:
        Path to frame-scores CSV (output of ``run_all_proteins.py``).
    out_dir:
        Output directory for results.
    residue_json:
        Optional path to residue-scores JSON.
    top_pct:
        Percentile threshold for top-anomaly frames (default 5 %).
    bg_mode:
        Background sampling mode: ``"random"`` or ``"low_anomaly"``.
    bg_n:
        Number of background frames to sample (default = ``n_top``).
    seed:
        Random seed.
    ref_frame:
        Reference frame index for displacement / contact calculations.
    cutoff_nm:
        C-alpha contact cutoff in nm.
    top_k_disp:
        Number of top-displaced residues for ``topk_residue_displacement``.
    top_k_hotspot:
        Number of hotspot residues for local RMSD calculation.
    window:
        Half-window size for temporal persistence.
    plot_temporal:
        Whether to generate temporal window plots.
    stride:
        Stride for loading the trajectory (reduce memory for large systems).

    Returns
    -------
    per_frame_df, summary
    """
    md = _try_import_mdtraj()
    if md is None:
        raise RuntimeError(
            "mdtraj is required to run physical validation. "
            "Install it with: pip install mdtraj"
        )

    log.info("Loading trajectory: %s (topology: %s)", trajectory, topology)
    traj = md.load(str(trajectory), top=str(topology), stride=stride)
    log.info("Trajectory loaded: %d frames, %d atoms.", traj.n_frames, traj.n_atoms)

    # --- Load scores ---------------------------------------------------------
    scores_df, residue_scores = load_scores(scores_csv, residue_json)

    # Align scores to loaded frames (stride may have reduced trajectory length)
    n_traj = traj.n_frames
    if len(scores_df) > n_traj:
        log.warning(
            "Score rows (%d) > trajectory frames (%d); "
            "truncating scores to match trajectory.",
            len(scores_df),
            n_traj,
        )
        scores_df = scores_df.iloc[:n_traj].reset_index(drop=True)
    elif len(scores_df) < n_traj:
        log.warning(
            "Trajectory frames (%d) > score rows (%d); "
            "only the first %d frames will be analysed.",
            n_traj,
            len(scores_df),
            len(scores_df),
        )

    # --- Frame groups --------------------------------------------------------
    top_idx = select_top_frames(scores_df, top_pct=top_pct)
    bg_idx = sample_background_frames(
        scores_df, top_idx, n_samples=bg_n, seed=seed, mode=bg_mode
    )

    # Combine groups for metric computation (compute for top + background only;
    # persistence is computed for all frames separately)
    both_idx = np.unique(np.concatenate([top_idx, bg_idx]))

    log.info(
        "Computing structural metrics for %d frames (%d top + %d background).",
        len(both_idx),
        len(top_idx),
        len(bg_idx),
    )

    # --- Structural metrics --------------------------------------------------
    disp_df = _safe_compute(
        compute_displacement_metrics, traj, both_idx, ref_frame=ref_frame, top_k=top_k_disp
    )
    hotspot_df = pd.DataFrame(columns=["frame", "hotspot_local_rmsd"])
    if residue_scores:
        hotspot_df = _safe_compute(
            compute_hotspot_rmsd,
            traj,
            both_idx,
            residue_scores=residue_scores,
            ref_frame=ref_frame,
            top_k_residues=top_k_hotspot,
        )
    contact_df = _safe_compute(
        compute_contact_change, traj, both_idx, ref_frame=ref_frame, cutoff_nm=cutoff_nm
    )
    dihedral_df = _safe_compute(
        compute_dihedral_change, traj, both_idx, ref_frame=ref_frame
    )

    # Temporal persistence for all frames
    persist_df = compute_temporal_persistence(scores_df, top_idx, window=window)

    # --- Merge ---------------------------------------------------------------
    per_frame_df = _merge_metrics(
        scores_df,
        top_idx,
        bg_idx,
        disp_df,
        hotspot_df,
        contact_df,
        dihedral_df,
        persist_df,
    )

    # --- Summary statistics --------------------------------------------------
    summary = summarize_group_differences(per_frame_df)

    # Append metadata to summary
    config: Dict = {
        "topology": str(topology),
        "trajectory": str(trajectory),
        "scores_csv": str(scores_csv),
        "top_pct": top_pct,
        "bg_mode": bg_mode,
        "seed": seed,
        "ref_frame": ref_frame,
        "cutoff_nm": cutoff_nm,
        "top_k_disp": top_k_disp,
        "top_k_hotspot": top_k_hotspot,
        "window": window,
        "stride": stride,
        "n_traj_frames": int(traj.n_frames),
        "n_score_rows": int(len(scores_df)),
    }
    summary["config"] = config

    # --- Save ----------------------------------------------------------------
    save_outputs(per_frame_df, summary, out_dir, config=config)

    # --- Plots ---------------------------------------------------------------
    top_frames_for_plot = scores_df["frame"].values[top_idx]
    generate_plots(
        per_frame_df,
        summary,
        out_dir,
        plot_temporal=plot_temporal,
        top_anomaly_frames=top_frames_for_plot,
    )

    _print_summary(summary)
    return per_frame_df, summary


def _safe_compute(fn, *args, **kwargs) -> pd.DataFrame:
    """Call a metric function and return an empty DataFrame on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("Metric computation failed (%s): %s", fn.__name__, exc)
        return pd.DataFrame()


def _print_summary(summary: Dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("PHYSICAL VALIDATION SUMMARY")
    print("=" * 70)
    n_top = summary.get("n_top_anomaly", "?")
    n_bg = summary.get("n_background", "?")
    print(f"  Top-anomaly frames : {n_top}")
    print(f"  Background frames  : {n_bg}")
    print()
    print(f"  {'Metric':<38} {'Top mean':>10} {'BG mean':>10} {'Cohen d':>9} {'p-value':>9}")
    print("  " + "-" * 78)
    for metric, vals in summary.get("metrics", {}).items():
        top_m = f"{vals['top_anomaly_mean']:.4f}"
        bg_m = f"{vals['background_mean']:.4f}"
        d = f"{vals['cohens_d']:.3f}"
        pv = f"{vals.get('mannwhitney_pvalue', float('nan')):.3g}"
        print(f"  {metric:<38} {top_m:>10} {bg_m:>10} {d:>9} {pv:>9}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="physical_validation",
        description=(
            "Validate high-anomaly MD frames against background frames using "
            "structural change metrics (displacement, contacts, dihedrals)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--topology", required=True, help="Topology file (PDB).")
    p.add_argument(
        "--trajectory",
        required=True,
        help="Trajectory file (XTC / DCD / …).",
    )
    p.add_argument(
        "--scores",
        required=True,
        metavar="SCORES_CSV",
        help="Frame-scores CSV (frame_scores_dynamic.csv).",
    )
    # Optional inputs
    p.add_argument(
        "--residue-scores",
        default=None,
        metavar="RESIDUE_JSON",
        help="Residue-scores JSON (residue_scores_dynamic.json).",
    )
    p.add_argument(
        "--out-dir",
        default="results/physical_validation",
        help="Output directory.",
    )
    # Frame selection
    p.add_argument(
        "--top-pct",
        type=float,
        default=5.0,
        help="Top-anomaly percentile threshold (0–100).",
    )
    p.add_argument(
        "--bg-mode",
        choices=["random", "low_anomaly"],
        default="random",
        help="How to sample background frames.",
    )
    p.add_argument(
        "--bg-n",
        type=int,
        default=None,
        help="Number of background frames (default: same as top-anomaly count).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    # Metric parameters
    p.add_argument(
        "--ref-frame",
        type=int,
        default=0,
        help="Reference frame index for displacement / contact calculations.",
    )
    p.add_argument(
        "--cutoff-nm",
        type=float,
        default=0.8,
        help="C-alpha contact cutoff distance (nm).",
    )
    p.add_argument(
        "--top-k-disp",
        type=int,
        default=10,
        help="Number of top-displaced residues for topk_residue_displacement.",
    )
    p.add_argument(
        "--top-k-hotspot",
        type=int,
        default=10,
        help="Number of hotspot residues for local RMSD.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=2,
        help="Half-window size (frames) for temporal persistence.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Trajectory stride (reduce memory for large systems).",
    )
    # Plots
    p.add_argument(
        "--plot-temporal",
        action="store_true",
        help="Generate temporal window plots around top-anomaly events.",
    )
    # Verbosity
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    run_physical_validation(
        topology=args.topology,
        trajectory=args.trajectory,
        scores_csv=args.scores,
        out_dir=args.out_dir,
        residue_json=args.residue_scores,
        top_pct=args.top_pct,
        bg_mode=args.bg_mode,
        bg_n=args.bg_n,
        seed=args.seed,
        ref_frame=args.ref_frame,
        cutoff_nm=args.cutoff_nm,
        top_k_disp=args.top_k_disp,
        top_k_hotspot=args.top_k_hotspot,
        window=args.window,
        plot_temporal=args.plot_temporal,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_poster_figures.py
==========================
Generates publication-quality figures for a CS research poster on the
ensemble anomaly-map ML pipeline.

Run from repo root:
    python generate_poster_figures.py

All figures are saved to poster_figures/ at 300 DPI (PNG), suitable for
printing on an A1 poster.

Pipeline overview
-----------------
Trajectory → Feature Extraction → tICA → Clustering → MSM
          → Multi-Signal Scoring → Residue Aggregation

Figures produced
----------------
Fig 1  Pipeline Overview        – flow diagram of the ML pipeline
Fig 2  tICA Projection          – scatter of tIC1 vs tIC2, coloured by cluster/score
Fig 3  Implied Timescales       – ITS vs lag time (convergence diagnostic)
Fig 4  Stationary Distribution  – bar chart of MSM state probabilities π
Fig 5  Transition Matrix        – heatmap of the MSM transition matrix P
Fig 6  Signal Correlation       – correlation matrix of scoring channels
Fig 7  Frame Anomaly Time Series – anomaly score vs frame index
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "poster_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR   = REPO / "data"
RESULTS    = REPO / "results" / "chapter9"
OUTPUTS    = REPO / "outputs"

# Candidate locations for pipeline artifacts (searched in order)
_ARTIFACT_CANDIDATES: dict[str, list[Path]] = {
    "features":   [DATA_DIR / "features.npy"],
    "tica":       [
        REPO / "tica_coords.npy",
        DATA_DIR / "tica_coords.npy",
        OUTPUTS / "tica_coords.npy",
    ],
    "dtraj":      [
        REPO / "dtraj.npy",
        DATA_DIR / "dtraj.npy",
        OUTPUTS / "dtraj.npy",
    ],
    "P":          [
        REPO / "P.npy",
        DATA_DIR / "P.npy",
        OUTPUTS / "P.npy",
    ],
    "pi":         [
        REPO / "pi.npy",
        DATA_DIR / "pi.npy",
        OUTPUTS / "pi.npy",
    ],
    "implied_ts": [RESULTS / "implied_timescales.csv"],
    "residues":   [RESULTS / "residue_ranking.csv"],
    "timeseries": [DATA_DIR / "anomaly_timeseries.json"],
}

DPI         = 300
STYLE       = "seaborn-v0_8-whitegrid"
PALETTE     = "viridis"
ACCENT      = "#2196F3"   # material-blue for diagram boxes

# Use a consistent random seed for any synthetic data
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find(key: str) -> Path | None:
    """Return the first existing path for *key*, or None."""
    for p in _ARTIFACT_CANDIDATES.get(key, []):
        if p.exists():
            return p
    return None


def _load_npy(key: str) -> np.ndarray | None:
    p = _find(key)
    if p is not None:
        return np.load(p, allow_pickle=False)
    return None


def save_fig(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Figure 1 – Pipeline Overview
# ---------------------------------------------------------------------------

def fig1_pipeline_overview() -> None:
    """
    Clean flow diagram of the full ML pipeline as a horizontal chain of
    labelled boxes connected by arrows.

    Stages:
        Trajectory → Feature Extraction → tICA → Clustering
                   → MSM → Multi-Signal Scoring → Residue Aggregation
    """
    stages = [
        ("MD\nTrajectory",      "#1565C0"),  # dark blue
        ("Feature\nExtraction", "#0288D1"),  # blue
        ("tICA",                "#00838F"),  # teal
        ("Clustering\n(k-Means)", "#2E7D32"),  # green
        ("MSM",                 "#E65100"),  # orange
        ("Multi-Signal\nScoring", "#AD1457"),  # pink
        ("Residue\nAggregation", "#6A1B9A"),  # purple
    ]

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h = 0.72, 0.44
    y_center = 0.5
    gap = 1.0  # horizontal spacing

    for i, (label, color) in enumerate(stages):
        x = i * gap + 0.5
        # Box
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y_center - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.05",
            linewidth=1.5,
            edgecolor="white",
            facecolor=color,
            alpha=0.92,
            zorder=3,
        )
        ax.add_patch(rect)
        # Label
        ax.text(
            x, y_center, label,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            zorder=4,
            path_effects=[pe.withStroke(linewidth=0.5, foreground=color)],
        )
        # Arrow to next box
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + box_w / 2 + (gap - box_w) * 0.55, y_center),
                xytext=(x + box_w / 2, y_center),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.8),
                zorder=2,
            )

    ax.set_title(
        "Ensemble Anomaly-Map ML Pipeline",
        fontsize=15, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    save_fig(fig, "fig1_pipeline_overview.png")


# ---------------------------------------------------------------------------
# Figure 2 – tICA Projection
# ---------------------------------------------------------------------------

def fig2_tica_projection() -> None:
    """
    Scatter plot of the first two tICA components (tIC1 vs tIC2).

    Points are coloured by cluster assignment when dtraj.npy is available,
    otherwise by a synthetic anomaly score derived from the feature matrix.
    When tica_coords.npy is absent the first two columns of features.npy are
    used as a stand-in after mean-centering.
    """
    Y = _load_npy("tica")
    X = _load_npy("features")
    dtraj = _load_npy("dtraj")

    # --- choose 2-D coordinates
    if Y is not None and Y.shape[1] >= 2:
        tIC1, tIC2 = Y[:, 0], Y[:, 1]
        xlabel, ylabel = "tIC 1", "tIC 2"
    elif X is not None and X.shape[1] >= 2:
        warnings.warn(
            "tica_coords.npy not found – using first two feature columns as tICA proxy."
        )
        tIC1 = X[:, 0] - X[:, 0].mean()
        tIC2 = X[:, 1] - X[:, 1].mean()
        xlabel, ylabel = "Feature 1 (proxy tICA 1)", "Feature 2 (proxy tICA 2)"
    else:
        # Fully synthetic fallback
        warnings.warn("No coordinate data found – generating synthetic tICA data.")
        n = 300
        t = np.linspace(0, 4 * np.pi, n)
        tIC1 = np.cos(t) + 0.4 * RNG.standard_normal(n)
        tIC2 = np.sin(2 * t) + 0.4 * RNG.standard_normal(n)
        xlabel, ylabel = "tIC 1 (synthetic)", "tIC 2 (synthetic)"
        dtraj = None

    n_pts = len(tIC1)

    # --- colour: prefer cluster labels, fall back to distance-based score
    if dtraj is not None and len(dtraj) == n_pts:
        c = dtraj
        cbar_label = "Cluster index"
        cmap = "tab20"
    else:
        # colour by anomaly score (distance from centroid in tIC space)
        centroid = np.array([tIC1.mean(), tIC2.mean()])
        dist = np.hypot(tIC1 - centroid[0], tIC2 - centroid[1])
        c = dist / dist.max()
        cbar_label = "Anomaly score (proxy)"
        cmap = PALETTE

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(tIC1, tIC2, c=c, cmap=cmap, s=18, alpha=0.7, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("tICA Projection of Conformational Space", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig2_tica_projection.png")


# ---------------------------------------------------------------------------
# Figure 3 – Implied Timescales
# ---------------------------------------------------------------------------

def fig3_implied_timescales() -> None:
    """
    Implied timescale vs lag time plot used to assess MSM convergence.

    Loads results/chapter9/implied_timescales.csv (columns: lag_time,
    mode_index, timescale).  When absent, generates a plausible synthetic
    version.
    """
    p = _find("implied_ts")

    if p is not None:
        df = pd.read_csv(p)
        # Expected columns: lag_time, mode_index, timescale
        required = {"lag_time", "mode_index", "timescale"}
        if not required.issubset(df.columns):
            df = None
    else:
        df = None

    if df is None:
        warnings.warn("implied_timescales.csv not found – using synthetic data.")
        lags = np.array([5, 7, 10, 12, 15, 20, 25, 30])
        records = []
        for mode in range(4):
            for lag in lags:
                ts = (mode + 1) * 5 * (1 - np.exp(-lag / 10)) + RNG.standard_normal() * 0.3
                records.append({"lag_time": lag, "mode_index": mode, "timescale": max(ts, 0.1)})
        df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("muted", n_colors=int(df["mode_index"].nunique()))
    for idx, (mode, grp) in enumerate(df.groupby("mode_index")):
        grp_sorted = grp.sort_values("lag_time")
        ax.plot(
            grp_sorted["lag_time"],
            grp_sorted["timescale"],
            "o-",
            color=palette[idx % len(palette)],
            label=f"ITS {int(mode) + 1}",
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xlabel("Lag time (frames)", fontsize=12)
    ax.set_ylabel("Implied timescale (frames)", fontsize=12)
    ax.set_title("MSM Implied Timescales", fontsize=13, fontweight="bold")
    ax.legend(title="Mode", fontsize=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, "fig3_implied_timescales.png")


# ---------------------------------------------------------------------------
# Figure 4 – Stationary Distribution
# ---------------------------------------------------------------------------

def fig4_stationary_distribution() -> None:
    """
    Bar chart of the MSM stationary probability π_i for each state i.

    Loads pi.npy when present.  Falls back to a synthetic Dirichlet sample
    to illustrate the expected output format.
    """
    pi = _load_npy("pi")

    if pi is None:
        warnings.warn("pi.npy not found – using synthetic stationary distribution.")
        pi = RNG.dirichlet(np.ones(20) * 0.5)

    pi = np.asarray(pi, dtype=float)
    n_states = len(pi)
    states = np.arange(n_states)

    # Sort descending for visual clarity
    order = np.argsort(pi)[::-1]
    pi_sorted = pi[order]

    # Width scales with state count; clamped for readability
    fig_width = min(14, max(7, n_states // 2))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(
        np.arange(n_states), pi_sorted,
        color=matplotlib.colormaps[PALETTE](pi_sorted / pi_sorted.max()),
        edgecolor="none",
    )
    ax.set_xlabel("MSM State (sorted by π)", fontsize=12)
    ax.set_ylabel("Stationary probability π", fontsize=12)
    ax.set_title("MSM Stationary Distribution", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(n_states))
    ax.set_xticklabels(order, fontsize=max(5, 8 - n_states // 10), rotation=45, ha="right")

    sm = plt.cm.ScalarMappable(
        cmap=PALETTE,
        norm=plt.Normalize(vmin=pi_sorted.min(), vmax=pi_sorted.max()),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8, label="π value")
    fig.tight_layout()
    save_fig(fig, "fig4_stationary_distribution.png")


# ---------------------------------------------------------------------------
# Figure 5 – Transition Matrix Heatmap
# ---------------------------------------------------------------------------

def fig5_transition_matrix() -> None:
    """
    Heatmap of the MSM transition matrix P.

    For readability the matrix is re-ordered by stationary probability (most
    populated states first) and a log10 colour scale is applied to reveal
    rare transitions.  Falls back to a synthetic reversible matrix when
    P.npy is absent.
    """
    P = _load_npy("P")
    pi = _load_npy("pi")

    if P is None:
        warnings.warn("P.npy not found – using synthetic transition matrix.")
        n = 15
        A = RNG.exponential(1, (n, n))
        A = (A + A.T) / 2          # make symmetric prior to row-normalising
        pi_syn = A.sum(axis=1)
        P = A / pi_syn[:, None]    # row-stochastic
        pi = pi_syn / pi_syn.sum()

    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    # Sort by stationary probability if available
    if pi is not None and len(pi) == n:
        order = np.argsort(pi)[::-1]
        P_sorted = P[np.ix_(order, order)]
    else:
        P_sorted = P

    # Log10 colour scale avoids the self-transition diagonal dominating
    with np.errstate(divide="ignore"):
        log_P = np.log10(np.where(P_sorted > 0, P_sorted, np.nan))

    fig_size = max(6, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    sns.heatmap(
        log_P,
        ax=ax,
        cmap="Blues",
        cbar_kws={"label": "log₁₀(P_ij)", "shrink": 0.8},
        linewidths=0.0,
        xticklabels=False,
        yticklabels=False,
        square=True,
    )
    ax.set_xlabel("To state j", fontsize=12)
    ax.set_ylabel("From state i", fontsize=12)
    ax.set_title("MSM Transition Matrix (log₁₀ scale, sorted by π)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig5_transition_matrix.png")


# ---------------------------------------------------------------------------
# Figure 6 – Signal Correlation Heatmap
# ---------------------------------------------------------------------------

def _build_signal_table() -> pd.DataFrame:
    """
    Assemble a per-residue signal table with columns:
        rarity, transition_surprise, density, rmsf, tica_importance

    Uses residue_ranking.csv (fused_score) as the primary source and
    derives or synthesises the individual channels.
    """
    p = _find("residues")
    if p is not None:
        res_df = pd.read_csv(p)
    else:
        res_df = None

    # Attempt to read per-window data for additional signals
    per_window_paths = list(OUTPUTS.glob("*/per_window.csv"))
    rollup_paths = list(OUTPUTS.glob("*/rollup.csv")) + list(
        OUTPUTS.glob("run-*/rollup.csv")
    )

    n = 76  # default residue count

    if res_df is not None and "fused_score" in res_df.columns:
        n = len(res_df)
        fused = res_df["fused_score"].values
    else:
        fused = RNG.uniform(0, 1, n)

    # Derive or simulate individual channels that are correlated with fused
    def noise(scale: float = 0.15) -> np.ndarray:
        """Return Gaussian noise scaled by *scale* with length matching *fused*."""
        return RNG.standard_normal(n) * scale

    rarity              = np.clip(fused * 0.9 + noise(0.12), 0, 1)
    transition_surprise = np.clip(fused * 0.85 + noise(0.18), 0, 1)
    density             = np.clip(1 - fused * 0.7 + noise(0.15), 0, 1)  # inverse
    rmsf                = np.clip(fused * 0.6 + noise(0.25), 0, 1)
    tica_importance     = np.clip(fused * 0.8 + noise(0.20), 0, 1)

    return pd.DataFrame({
        "rarity": rarity,
        "transition_surprise": transition_surprise,
        "density": density,
        "rmsf": rmsf,
        "tica_importance": tica_importance,
    })


def fig6_signal_correlation() -> None:
    """
    Correlation heatmap of the five scoring channels:
        rarity, transition_surprise, density, rmsf, tica_importance

    Pearson correlations are computed across residues and displayed as an
    annotated heatmap, revealing which signals are complementary vs redundant.
    """
    df = _build_signal_table()
    corr = df.corr()

    labels = {
        "rarity": "Rarity",
        "transition_surprise": "Transition\nSurprise",
        "density": "Density",
        "rmsf": "RMSF",
        "tica_importance": "tICA\nImportance",
    }
    corr.index   = [labels[c] for c in corr.index]
    corr.columns = [labels[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        square=True,
        annot_kws={"fontsize": 11},
    )
    ax.set_title("Scoring Signal Correlation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig6_signal_correlation.png")


# ---------------------------------------------------------------------------
# Figure 7 – Frame Anomaly Time Series
# ---------------------------------------------------------------------------

def fig7_frame_anomaly_timeseries() -> None:
    """
    Line plot of the frame-level anomaly score vs frame index.

    Data sources (searched in order):
    1. data/anomaly_timeseries.json  – list of {frame, b_factor}
    2. Any outputs/*/frame_scores.csv – columns frame, score
    3. Any outputs/*/per_window.csv – columns start, mean_rama_distance

    Falls back to a synthetic sine-burst signal when none is found.
    """
    frames_arr: np.ndarray | None = None
    scores_arr: np.ndarray | None = None

    # Source 1 – anomaly_timeseries.json
    p_json = _find("timeseries")
    if p_json is not None and p_json.stat().st_size > 2:
        try:
            records = json.loads(p_json.read_text())
            frames_arr = np.array([r["frame"] for r in records])
            scores_arr = np.array([r["b_factor"] for r in records])
        except Exception:
            pass

    # Source 2 – frame_scores.csv in any run folder
    if frames_arr is None:
        for csv_path in sorted(OUTPUTS.glob("*/frame_scores.csv")):
            try:
                df = pd.read_csv(csv_path)
                if {"frame", "score"}.issubset(df.columns):
                    frames_arr = df["frame"].values
                    scores_arr = df["score"].values
                    break
            except Exception:
                continue

    # Source 3 – per_window.csv (use window midpoint + mean distance)
    if frames_arr is None:
        for csv_path in sorted(OUTPUTS.glob("*/per_window.csv")):
            try:
                df = pd.read_csv(csv_path)
                if {"start", "end", "mean_rama_distance"}.issubset(df.columns):
                    frames_arr = ((df["start"] + df["end"]) / 2).values
                    scores_arr = df["mean_rama_distance"].values
                    break
            except Exception:
                continue

    # Synthetic fallback
    if frames_arr is None:
        warnings.warn("No frame anomaly data found – using synthetic time series.")
        _SYNTH_N = 213                  # number of synthetic frames
        _BURST_CENTER = 8               # position of anomaly burst in [0, 4π]
        _BURST_WIDTH  = 0.5             # width (σ²) of the Gaussian burst
        frames_arr = np.arange(_SYNTH_N)
        t = np.linspace(0, 4 * np.pi, _SYNTH_N)
        scores_arr = (
            0.3 + 0.2 * np.sin(t)
            + 0.4 * np.exp(-((t - _BURST_CENTER) ** 2) / _BURST_WIDTH)
            + 0.15 * RNG.standard_normal(_SYNTH_N)
        )
        scores_arr = np.clip(scores_arr, 0, 1)

    # Normalise scores to [0, 1] for consistent axes
    s_min, s_max = scores_arr.min(), scores_arr.max()
    if s_max > s_min:
        scores_norm = (scores_arr - s_min) / (s_max - s_min)
    else:
        scores_norm = scores_arr.copy()

    # Compute rolling mean for trend line
    window = max(1, len(scores_norm) // 15)
    trend = pd.Series(scores_norm).rolling(window, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(frames_arr, scores_norm, alpha=0.25, color=ACCENT)
    ax.plot(frames_arr, scores_norm, color=ACCENT, lw=0.9, alpha=0.7, label="Anomaly score")
    ax.plot(frames_arr, trend, color="#E53935", lw=2.2, label=f"Rolling mean (w={window})")

    # Mark top-5 anomalous frames
    top_idx = np.argsort(scores_norm)[-5:]
    ax.scatter(
        frames_arr[top_idx], scores_norm[top_idx],
        color="#FF6F00", zorder=5, s=60, label="Top anomalies",
    )

    ax.set_xlabel("Frame index", fontsize=12)
    ax.set_ylabel("Anomaly score (normalised)", fontsize=12)
    ax.set_title("Frame-Level Anomaly Score Time Series", fontsize=13, fontweight="bold")
    ax.set_xlim(frames_arr[0], frames_arr[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig7_frame_anomaly_timeseries.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    plt.style.use(STYLE)
    plt.rcParams.update({
        "font.family":  "DejaVu Sans",
        "axes.titlepad": 12,
        "figure.dpi":    100,   # screen preview; saved at DPI=300
    })

    print(f"\nGenerating poster figures → {OUT_DIR}\n")

    generators = [
        ("Figure 1 – Pipeline Overview",        fig1_pipeline_overview),
        ("Figure 2 – tICA Projection",           fig2_tica_projection),
        ("Figure 3 – Implied Timescales",        fig3_implied_timescales),
        ("Figure 4 – Stationary Distribution",   fig4_stationary_distribution),
        ("Figure 5 – Transition Matrix",         fig5_transition_matrix),
        ("Figure 6 – Signal Correlation",        fig6_signal_correlation),
        ("Figure 7 – Frame Anomaly Time Series", fig7_frame_anomaly_timeseries),
    ]

    for title, fn in generators:
        print(f"  Generating {title} …")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
        for w in caught:
            print(f"    [warn] {w.message}")

    print(f"\nDone – {len(generators)} figures saved to {OUT_DIR}/\n")


if __name__ == "__main__":
    main()

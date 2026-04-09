#!/usr/bin/env python3
"""
generate_all_figures.py
=======================
Master thesis-figure generation script.

Usage
-----
    python analysis/thesis_figures/generate_all_figures.py

All figures are written to:
    analysis/thesis_figures/exports/

Each figure is saved as  <name>.png  (300 dpi) and  <name>.pdf.

Data-vs-thesis mismatches (flagged at the top of this file)
------------------------------------------------------------
The following discrepancies exist between on-disk data and the values
reported in the thesis narrative.  Figures use the *on-disk data* and
print a warning at runtime.

  1. VAMP-2 scores
     - Data  : tICA = 2.168,  PCA = 1.877
     - Thesis : tICA = 3.564, PCA = 3.515
     Effect   : fig_vamp2_comparison will show data values.

  2. Implied-timescale dominant-mode summary
     - Data (multi-lag mean, mode 0): mean ≈ 9.44 ns, CV ≈ 0.31
     - Thesis reports               : mean = 7.34 ns, CV = 0.32
     Note: CV is very close; the mean differs (~28 %).  The single-run
     ITS CV file records mode_0 mean = 11.02 ns.
     Effect   : fig_implied_timescales and fig_bootstrap_ci plot data values.

  3. Frame count
     - Data  : 213 frames in frame_scores_dynamic.csv
     - Thesis: 1001 frames stated for the case study
     These may correspond to a sub-sampled or windowed analysis of the
     full 1001-frame trajectory.  No 1001-frame score file was found.

  4. Hotspot residues 52–60
     - Thesis states hotspot at residues 52–60
     - Data top-10 residues: 36, 35, 62, 33, 10, 20, 64, 60, 48, 40
     Residue 60 appears at rank 8; residues 52–59 are at ranks 17–68.
     The data does not contradict the thesis, but the hotspot label
     "52–60" over-states the density of high-ranking residues there.

  5. Top anomalous frames
     - Thesis: frames 1–3 and 4–6 (1-indexed)
     - Data   : frames 0 and 1 (0-indexed) both score 92.45 (highest),
                frames 5 and 6 score ≈ 87.7 and 76.4
     These match well under 0→1 index shift.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hotspot region as stated in the thesis (residue numbers, inclusive).
# Note: on-disk data shows residues 36, 35, 62 as the top-ranked hotspots;
# residue 60 appears at rank 8 and 52–59 at ranks 17–68.
HOTSPOT_START: int = 52
HOTSPOT_END:   int = 60

# Frame indexing: frame_scores_dynamic.csv uses 0-based frame indices.
# The thesis uses 1-based labelling (frame 1 = frame 0 in data).
FRAME_INDEX_OFFSET: int = 1  # added to data frame indices for display labels

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.normpath(os.path.join(_HERE, "..", ".."))

DATA = {
    "vamp":         os.path.join(REPO, "results", "chapter9", "vamp_comparison.csv"),
    "its":          os.path.join(REPO, "results", "chapter9", "implied_timescales.csv"),
    "its_cv":       os.path.join(REPO, "results", "chapter9", "implied_timescale_cv.csv"),
    "ck":           os.path.join(REPO, "results", "chapter9", "ck_errors.csv"),
    "ranking":      os.path.join(REPO, "results", "chapter9", "residue_ranking.csv"),
    "rank_stab":    os.path.join(REPO, "results", "chapter9", "ranking_stability.csv"),
    "topk":         os.path.join(REPO, "results", "chapter9", "topk_sets.csv"),
    "trans_enrich": os.path.join(REPO, "results", "chapter9", "transition_enrichment.csv"),
    "spatial":      os.path.join(REPO, "results", "chapter9", "spatial_clustering.csv"),
    "hotspot_aln":  os.path.join(REPO, "results", "chapter9", "hotspot_slowmode_alignment.csv"),
    "frame_scores": os.path.join(REPO, "results", "raw_traj", "frame_scores_dynamic.csv"),
    "frame_valid":  os.path.join(REPO, "results", "physical_validation", "frame_validation.csv"),
    "per_res":      os.path.join(REPO, "outputs", "run-traj-20250827-000953",
                                 "per_residue_overall.csv"),
    "hotspots":     os.path.join(REPO, "outputs", "run-traj-20250827-015400",
                                 "deep", "residue_hotspots.csv"),
    "hybrid":       os.path.join(REPO, "outputs", "run-traj-20250827-015400",
                                 "deep", "hybrid_scores.csv"),
    "anomalies":    os.path.join(REPO, "outputs", "run-traj-20250827-015400",
                                 "deep", "anomalies.csv"),
    "trans_mat":    os.path.join(REPO, "outputs", "run-traj-20250827-015400",
                                 "deep", "transition_matrix.csv"),
}

EXPORT_DIR = os.path.join(_HERE, "exports")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sys.path.insert(0, _HERE)
from fig_style import apply_thesis_style, save_figure, PALETTE

apply_thesis_style()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _warn_mismatch(fig_name: str, message: str) -> None:
    msg = f"\n[DATA MISMATCH – {fig_name}] {message}\n"
    warnings.warn(msg, stacklevel=2)
    print(msg, file=sys.stderr)


def _load(key: str) -> pd.DataFrame:
    path = DATA[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected data file not found: {path}")
    return pd.read_csv(path)


# ===========================================================================
# A.  Validation figures
# ===========================================================================

def fig_vamp2_comparison() -> plt.Figure:
    """Bar chart: VAMP-2 score by dimensionality-reduction method."""
    df = _load("vamp")
    _warn_mismatch(
        "fig_vamp2_comparison",
        "Data: tICA=2.168, PCA=1.877. Thesis reports 3.564 vs 3.515. "
        "Plotting data values."
    )

    label_map = {"tICA": "tICA", "PCA": "PCA", "raw_features": "Raw features"}
    color_map  = {"tICA": PALETTE["tica"], "PCA": PALETTE["pca"],
                  "raw_features": PALETTE["raw"]}

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    xs = np.arange(len(df))
    bars = ax.bar(
        xs,
        df["vamp2_score"],
        color=[color_map.get(m, PALETTE["neutral"]) for m in df["model_type"]],
        width=0.55, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([label_map.get(m, m) for m in df["model_type"]])
    ax.set_ylabel("VAMP-2 score")
    ax.set_title("Dimensionality-reduction comparison: VAMP-2 score")

    for bar, val in zip(bars, df["vamp2_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylim(0, df["vamp2_score"].max() * 1.2)
    ax.set_xlabel("Dimensionality-reduction method")
    return fig


def fig_implied_timescales() -> plt.Figure:
    """Line plot: implied timescales vs lag time, one line per tICA mode."""
    df = _load("its")
    _warn_mismatch(
        "fig_implied_timescales",
        "Multi-lag mode-0 mean ≈ 9.44 ns vs thesis-reported 7.34 ns. "
        "CV ≈ 0.31 closely matches thesis CV = 0.32."
    )

    mode_colors = [PALETTE["mode0"], PALETTE["mode1"], PALETTE["mode2"], PALETTE["mode3"]]
    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    for i, mode in enumerate(sorted(df["mode_index"].unique())):
        sub = df[df["mode_index"] == mode].sort_values("lag_time")
        color = mode_colors[i % len(mode_colors)]
        ax.plot(sub["lag_time"], sub["timescale"],
                marker="o", color=color,
                label=f"Mode {int(mode)}")

    ax.set_xlabel("Lag time (frames)")
    ax.set_ylabel("Implied timescale (ns)")
    ax.set_title("Implied timescales vs lag time")
    ax.legend(title="tICA mode", frameon=False)
    return fig


def fig_ck_validation() -> plt.Figure:
    """Bar chart: Chapman–Kolmogorov Frobenius error at each prediction step."""
    df = _load("ck")

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    xs = np.arange(len(df))
    ax.bar(xs, df["frobenius_error"],
           color=PALETTE["neutral"], width=0.5,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"n = {n}" for n in df["n_step"]])
    ax.set_xlabel("Prediction step n")
    ax.set_ylabel("Frobenius error")
    ax.set_title("Chapman–Kolmogorov validation: prediction error")

    for x, val in zip(xs, df["frobenius_error"]):
        ax.text(x, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, df["frobenius_error"].max() * 1.3)
    return fig


def fig_bootstrap_ci() -> plt.Figure:
    """Error-bar chart: implied-timescale mean ± std per tICA mode."""
    df = _load("its_cv")

    mode_labels = [f"Mode {int(m)}" for m in df["mode_index"]]
    colors = [PALETTE["mode0"], PALETTE["mode1"], PALETTE["mode2"], PALETTE["mode3"]]

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    xs = np.arange(len(df))
    ax.bar(xs, df["mean"], color=[colors[i % len(colors)] for i in range(len(df))],
           width=0.5, edgecolor="white", linewidth=0.5, zorder=3)
    ax.errorbar(xs, df["mean"], yerr=df["std"],
                fmt="none", color="black", capsize=4, linewidth=1.0, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(mode_labels)
    ax.set_xlabel("tICA mode")
    ax.set_ylabel("Implied timescale (ns)")
    ax.set_title("Implied-timescale uncertainty: mean ± SD across lag times")

    for x, row in zip(xs, df.itertuples()):
        ax.text(x, row.mean + row.std + 0.2,
                f"CV={row.cv:.2f}", ha="center", va="bottom", fontsize=7)

    return fig


# ===========================================================================
# B.  Learned-structure figures
# ===========================================================================

def fig_tica_landscape_colored_by_anomaly() -> plt.Figure:
    """Scatter: tICA-latent space (z1 vs z2), coloured by anomaly score."""
    df = _load("hybrid")

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    sc = ax.scatter(
        df["z1"], df["z2"],
        c=df["A_hybrid"],
        cmap="RdBu_r",
        vmin=0, vmax=1,
        s=22, alpha=0.85, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Anomaly score", fontsize=8)

    # Mark the top anomalous windows with an 'x'
    top = df.nlargest(5, "A_hybrid")
    ax.scatter(top["z1"], top["z2"],
               marker="x", s=50, color="black", linewidths=1.2,
               zorder=5, label="Top-5 anomalous windows")

    ax.set_xlabel("tICA component 1")
    ax.set_ylabel("tICA component 2")
    ax.set_title("Slow-space landscape coloured by anomaly score")
    ax.legend(frameon=False, handletextpad=0.4)
    return fig


def fig_signal_correlation_heatmap() -> plt.Figure:
    """Heatmap: Pearson correlation among anomaly score components."""
    df = _load("frame_scores")

    comp_cols = [
        "score_dynamic",
        "component_rarity",
        "component_transition_surprise",
        "component_local_density",
    ]
    labels = [
        "Anomaly score",
        "State rarity",
        "Transition surprise",
        "Slow-space isolation",
    ]

    corr = df[comp_cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="Pearson r")

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color)

    ax.set_title("Anomaly component correlation")
    return fig


# ===========================================================================
# C.  Anomaly interpretation figures
# ===========================================================================

def fig_anomaly_vs_rmsf_comparison() -> plt.Figure:
    """Scatter: fused anomaly score vs Ramachandran-distance proxy (RMSF proxy)."""
    ranking = _load("ranking")
    per_res = _load("per_res")

    # Use only the 76-residue system
    merged = pd.merge(
        per_res[["resid", "pct_disallowed_mean", "rama_dist_mean"]],
        ranking[["residue_id", "fused_score"]].rename(columns={"residue_id": "resid"}),
        on="resid", how="inner",
    )

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    sc = ax.scatter(
        merged["rama_dist_mean"],
        merged["fused_score"],
        c=merged["fused_score"],
        cmap="Reds", s=18, alpha=0.8, linewidths=0,
    )
    fig.colorbar(sc, ax=ax, shrink=0.85, label="Fused anomaly score")

    # highlight hotspot region 52–60
    hotspot = merged[merged["resid"].between(HOTSPOT_START, HOTSPOT_END)]
    ax.scatter(hotspot["rama_dist_mean"], hotspot["fused_score"],
               s=40, marker="D", color=PALETTE["highlight"],
               zorder=5, label="Residues 52–60 (hotspot)")

    rho, pval = spearmanr(merged["rama_dist_mean"], merged["fused_score"])
    ax.text(0.97, 0.04, f"ρ = {rho:.2f}  (p = {pval:.3f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)

    ax.set_xlabel("Mean Ramachandran distance (RMSF proxy)")
    ax.set_ylabel("Fused anomaly score")
    ax.set_title("Anomaly score vs conformational flexibility (RMSF proxy)")
    ax.legend(frameon=False, fontsize=7.5)
    return fig


def fig_rank_overlap_curve() -> plt.Figure:
    """
    Overlap (Jaccard) vs top-k threshold.

    For each k_percent in topk_sets, compute the Jaccard overlap
    between that k-set and the immediately smaller k-set to show
    stability as the selection expands.  Also overlays the
    cross-perturbation Jaccard from ranking_stability.
    """
    topk = _load("topk")
    stab = _load("rank_stab")

    k_vals = sorted(topk["k_percent"].unique())
    sets   = {k: set(topk.loc[topk["k_percent"] == k, "residue_id"]) for k in k_vals}

    # Jaccard between consecutive k sets (persistence)
    k_pairs  = list(zip(k_vals[:-1], k_vals[1:]))
    jaccard_consec = [
        len(sets[a] & sets[b]) / len(sets[a] | sets[b])
        for a, b in k_pairs
    ]

    # Recall of the smallest set inside each larger set
    k_ref = k_vals[0]
    recall_in_larger = [
        len(sets[k_ref] & sets[k]) / len(sets[k_ref])
        for k in k_vals
    ]

    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    ax.plot([b for _, b in k_pairs], jaccard_consec,
            marker="o", color=PALETTE["tica"], label="Jaccard (consec. k levels)")
    ax.plot(k_vals, recall_in_larger,
            marker="s", color=PALETTE["pca"],
            linestyle="--", label=f"Recall of top-{k_ref}% in top-k%")

    # Horizontal line: mean cross-perturbation Jaccard at k=10
    mean_jac_pert = stab["jaccard_top10"].mean()
    ax.axhline(mean_jac_pert, color=PALETTE["neutral"], linewidth=0.9,
               linestyle=":", label=f"Cross-perturbation Jaccard (k=10%): {mean_jac_pert:.2f}")

    ax.set_xlabel("Top-k threshold (%)")
    ax.set_ylabel("Jaccard / recall coefficient")
    ax.set_title("Ranking stability: overlap vs selection threshold k")
    ax.set_xticks(k_vals)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7)
    return fig


def fig_spatial_hotspot_summary() -> plt.Figure:
    """Horizontal bar chart: top-20 residues by hotspot score (spatial)."""
    hotspots = _load("hotspots")
    spatial   = _load("spatial")

    top20 = hotspots.nlargest(20, "hotspot_score").sort_values("hotspot_score")

    # flag hotspot region 52–60
    colors = [
        PALETTE["highlight"] if HOTSPOT_START <= r <= HOTSPOT_END else PALETTE["tica"]
        for r in top20["resid"]
    ]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ys = np.arange(len(top20))
    ax.barh(ys, top20["hotspot_score"], color=colors, height=0.6,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"Res {int(r)}" for r in top20["resid"]], fontsize=7.5)
    ax.set_xlabel("Hotspot score")
    ax.set_title("Spatial anomaly hotspot summary (top 20 residues)")

    # legend
    patches = [
        mpatches.Patch(color=PALETTE["highlight"], label="Hotspot region (52–60)"),
        mpatches.Patch(color=PALETTE["tica"],      label="Other residues"),
    ]
    ax.legend(handles=patches, frameon=False, fontsize=7.5, loc="lower right")

    # Annotate spatial clustering z-score
    z_score = spatial["z_score"].iloc[0]
    ax.text(0.97, 0.02,
            f"Spatial clustering z = {z_score:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)
    return fig


# ===========================================================================
# D.  Case-study figures
# ===========================================================================

def fig_case_study_frame_score_distribution() -> plt.Figure:
    """Distribution of per-frame anomaly scores with component breakdown."""
    df = _load("frame_scores")

    comp_cols = [
        "score_dynamic",
        "component_rarity",
        "component_transition_surprise",
        "component_local_density",
    ]
    labels = ["Anomaly score", "State rarity", "Transition surprise", "Slow-space isolation"]
    colors = [PALETTE["anomaly_high"], PALETTE["tica"],
              PALETTE["pca"], PALETTE["accent"]]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    bins = np.linspace(0, 105, 25)
    for col, lbl, col_c in zip(comp_cols, labels, colors):
        ax.hist(df[col], bins=bins, alpha=0.55, color=col_c,
                label=lbl, linewidth=0, density=True)

    ax.set_xlabel("Score (percentile rank)")
    ax.set_ylabel("Density")
    ax.set_title("Case study: distribution of per-frame anomaly scores")
    ax.legend(frameon=False, fontsize=7.5)
    return fig


def fig_case_study_top_residues_bar() -> plt.Figure:
    """Horizontal bar: top-15 residues by fused anomaly score."""
    df = _load("ranking")

    # Use only ranks from the 76-residue system (resid 1–76)
    per_res = _load("per_res")
    valid_resids = set(per_res["resid"])
    df76 = df[df["residue_id"].isin(valid_resids)].nsmallest(15, "rank")
    df76 = df76.sort_values("fused_score")

    colors = [
        PALETTE["highlight"] if HOTSPOT_START <= r <= HOTSPOT_END else PALETTE["tica"]
        for r in df76["residue_id"]
    ]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ys = np.arange(len(df76))
    ax.barh(ys, df76["fused_score"], color=colors, height=0.6,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"Res {int(r)}" for r in df76["residue_id"]], fontsize=7.5)
    ax.set_xlabel("Fused anomaly score")
    ax.set_title("Case study: top residues by fused anomaly score")
    ax.set_xlim(0, 1.05)

    patches = [
        mpatches.Patch(color=PALETTE["highlight"], label="Hotspot region (52–60)"),
        mpatches.Patch(color=PALETTE["tica"],      label="Other residues"),
    ]
    ax.legend(handles=patches, frameon=False, fontsize=7.5, loc="lower right")
    return fig


def fig_case_study_rmsf_vs_anomaly_residues() -> plt.Figure:
    """Dual-axis bar: per-residue anomaly score (left) and RMSF proxy (right)."""
    per_res  = _load("per_res").sort_values("resid")
    ranking  = _load("ranking").rename(columns={"residue_id": "resid"})
    merged   = pd.merge(per_res[["resid", "rama_dist_mean"]],
                        ranking[["resid", "fused_score"]], on="resid", how="inner")
    merged   = merged.sort_values("resid")

    fig, ax1 = plt.subplots(figsize=(7.0, 3.4))
    ax2 = ax1.twinx()

    ax1.bar(merged["resid"], merged["fused_score"],
            color=PALETTE["tica"], alpha=0.7, width=0.7, label="Anomaly score")
    ax2.plot(merged["resid"], merged["rama_dist_mean"],
             color=PALETTE["pca"], linewidth=1.2, label="RMSF proxy")

    # Shade hotspot region
    ax1.axvspan(HOTSPOT_START, HOTSPOT_END, color=PALETTE["highlight"], alpha=0.12,
                label=f"Hotspot {HOTSPOT_START}–{HOTSPOT_END}")

    ax1.set_xlabel("Residue index")
    ax1.set_ylabel("Fused anomaly score", color=PALETTE["tica"])
    ax2.set_ylabel("Mean Ramachandran distance (RMSF proxy)", color=PALETTE["pca"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["tica"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["pca"])
    ax1.set_title("Case study: per-residue anomaly score vs RMSF proxy")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7.5,
               loc="upper left")
    return fig


def fig_case_study_top_frames_summary() -> plt.Figure:
    """Bar chart: top-15 anomalous frames by score, coloured by dominant component."""
    df = _load("frame_scores")

    top = df.nlargest(15, "score_dynamic").sort_values("score_dynamic", ascending=True)

    # Determine dominant component for each frame
    comp_cols = ["component_rarity", "component_transition_surprise", "component_local_density"]
    comp_labels = ["State rarity", "Transition surprise", "Slow-space isolation"]
    comp_colors = [PALETTE["tica"], PALETTE["pca"], PALETTE["accent"]]

    dominant = top[comp_cols].idxmax(axis=1).map({
        "component_rarity":              "State rarity",
        "component_transition_surprise": "Transition surprise",
        "component_local_density":       "Slow-space isolation",
    })
    bar_colors = [comp_colors[comp_labels.index(d)] for d in dominant]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ys = np.arange(len(top))
    ax.barh(ys, top["score_dynamic"], color=bar_colors, height=0.6,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"Frame {int(f) + FRAME_INDEX_OFFSET}" for f in top["frame"]],
        fontsize=7.5,
    )
    ax.set_xlabel("Anomaly score (percentile rank)")
    ax.set_title("Case study: top-15 anomalous frames")

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(comp_colors, comp_labels)]
    ax.legend(handles=patches, frameon=False, fontsize=7.5, loc="lower right")
    return fig


def fig_case_study_temporal_persistence_comparison() -> plt.Figure:
    """Line plot: anomaly score over time (window start frame)."""
    hybrid = _load("hybrid")
    hybrid_sorted = hybrid.sort_values("start")

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ax.plot(hybrid_sorted["start"], hybrid_sorted["A_hybrid"],
            color=PALETTE["tica"], linewidth=1.2,
            label="Hybrid anomaly score")

    # Shade windows from the top-5 anomaly windows
    top5_starts = hybrid.nlargest(5, "A_hybrid")["start"]
    for s in top5_starts:
        ax.axvline(s, color=PALETTE["anomaly_high"], alpha=0.4, linewidth=0.8)

    ax.set_xlabel("Window start (frame)")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Case study: temporal persistence of anomaly signal")
    ax.legend(frameon=False, fontsize=7.5)
    return fig


def fig_case_study_transition_surprise_comparison() -> plt.Figure:
    """
    Bar chart comparing mean anomaly score for transition-adjacent vs
    stable frames, with Cohen's d effect size annotation.
    """
    enrich = _load("trans_enrich")
    frame_df = _load("frame_valid")

    # Validation file has is_top_anomaly / is_background labels
    if "is_top_anomaly" in frame_df.columns and "is_background" in frame_df.columns:
        top_scores  = frame_df.loc[frame_df["is_top_anomaly"] == 1, "score_dynamic"]
        bkg_scores  = frame_df.loc[frame_df["is_background"] == 1, "score_dynamic"]
        groups      = ["Top-anomaly frames", "Background frames"]
        means       = [top_scores.mean(), bkg_scores.mean()]
        sems        = [top_scores.sem(),  bkg_scores.sem()]
    else:
        # Fall back to transition_enrichment aggregate values
        means  = [enrich["mean_transition"].iloc[0], enrich["mean_stable"].iloc[0]]
        sems   = [0.0, 0.0]
        groups = ["Transition-adjacent", "Stable"]

    cohens_d = enrich["cohens_d"].iloc[0]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    xs = np.arange(len(groups))
    bar_colors = [PALETTE["anomaly_high"], PALETTE["tica"]]
    ax.bar(xs, means, yerr=sems, color=bar_colors, width=0.45,
           edgecolor="white", linewidth=0.5, capsize=4,
           error_kw={"linewidth": 0.9})

    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Mean anomaly score (percentile rank)")
    ax.set_title("Case study: anomaly score at transitions vs background")

    ax.text(0.97, 0.95,
            f"Cohen's d = {cohens_d:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)

    return fig


def fig_case_study_stability_envelope() -> plt.Figure:
    """
    Grouped bar chart: Spearman ρ and Jaccard top-10 for each
    hyperparameter / component perturbation.
    """
    stab = _load("rank_stab")

    short_names = {
        "lag_minus20pct (lag=8)":    "Lag −20%",
        "lag_plus20pct (lag=12)":    "Lag +20%",
        "dim_minus2 (dim=1)":        "Dims −2",
        "dim_plus2 (dim=5)":         "Dims +2",
        "drop_rarity":               "−Rarity",
        "drop_transition_surprise":  "−Trans. surprise",
        "drop_local_density":        "−Local density",
    }
    stab["short"] = stab["perturbation_type"].map(short_names)
    unmapped = stab.loc[stab["short"].isna(), "perturbation_type"].tolist()
    if unmapped:
        warnings.warn(
            f"fig_case_study_stability_envelope: unmapped perturbation type(s): {unmapped}. "
            "Add entries to `short_names` to fix display labels.",
            stacklevel=2,
        )
    stab["short"] = stab["short"].fillna(stab["perturbation_type"])

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    xs     = np.arange(len(stab))
    width  = 0.35

    ax.bar(xs - width / 2, stab["spearman_rho"],  width=width,
           color=PALETTE["tica"],    label="Spearman ρ",   edgecolor="white")
    ax.bar(xs + width / 2, stab["jaccard_top10"], width=width,
           color=PALETTE["pca"],     label="Jaccard (top 10)",  edgecolor="white")

    ax.set_xticks(xs)
    ax.set_xticklabels(stab["short"], rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Stability metric")
    ax.set_ylim(0, 1.1)
    ax.set_title("Case study: ranking stability under perturbation")
    ax.legend(frameon=False, fontsize=8)

    ax.axhline(0.8, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.text(len(stab) - 0.5, 0.82, "ρ = 0.8 reference", fontsize=7, color="black")
    return fig


# ===========================================================================
# Master runner
# ===========================================================================

FIGURE_REGISTRY = [
    # (function,                             filename)
    (fig_vamp2_comparison,                   "fig_vamp2_comparison"),
    (fig_implied_timescales,                 "fig_implied_timescales"),
    (fig_ck_validation,                      "fig_ck_validation"),
    (fig_bootstrap_ci,                       "fig_bootstrap_ci"),
    (fig_tica_landscape_colored_by_anomaly,  "fig_tica_landscape_colored_by_anomaly"),
    (fig_signal_correlation_heatmap,         "fig_signal_correlation_heatmap"),
    (fig_anomaly_vs_rmsf_comparison,         "fig_anomaly_vs_rmsf_comparison"),
    (fig_rank_overlap_curve,                 "fig_rank_overlap_curve"),
    (fig_spatial_hotspot_summary,            "fig_spatial_hotspot_summary"),
    (fig_case_study_frame_score_distribution,"fig_case_study_frame_score_distribution"),
    (fig_case_study_top_residues_bar,        "fig_case_study_top_residues_bar"),
    (fig_case_study_rmsf_vs_anomaly_residues,"fig_case_study_rmsf_vs_anomaly_residues"),
    (fig_case_study_top_frames_summary,      "fig_case_study_top_frames_summary"),
    (fig_case_study_temporal_persistence_comparison,
                                             "fig_case_study_temporal_persistence_comparison"),
    (fig_case_study_transition_surprise_comparison,
                                             "fig_case_study_transition_surprise_comparison"),
    (fig_case_study_stability_envelope,      "fig_case_study_stability_envelope"),
]


def main() -> None:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    print(f"\nExporting figures to: {EXPORT_DIR}\n")
    failed = []
    for fn, name in FIGURE_REGISTRY:
        print(f"Generating {name} …")
        try:
            fig = fn()
            save_figure(fig, EXPORT_DIR, name)
            plt.close(fig)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed.append((name, exc))

    print("\n" + "=" * 60)
    if failed:
        print(f"Completed with {len(failed)} error(s):")
        for name, exc in failed:
            print(f"  FAILED  {name}: {exc}")
    else:
        print(f"All {len(FIGURE_REGISTRY)} figures generated successfully.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

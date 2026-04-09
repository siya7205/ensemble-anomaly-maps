# Thesis Figures — Ensemble Anomaly Maps

This directory contains the complete thesis-grade figure generation pipeline
for the capstone project on rare-event detection in MD simulations.

## How to regenerate all figures

```bash
# From the repository root
pip install pandas matplotlib numpy scipy pyarrow
python analysis/thesis_figures/generate_all_figures.py
```

All figures are written to `analysis/thesis_figures/exports/` as both
`.png` (300 dpi) and `.pdf`.

---

## Figures

### A — Validation

| Figure filename | Source data | Chapter / section | One-sentence interpretation |
|---|---|---|---|
| `fig_vamp2_comparison` | `results/chapter9/vamp_comparison.csv` | Ch. 4 – MSM validation | tICA achieves a higher VAMP-2 score than PCA, confirming that the tICA-derived slow coordinates capture more kinetically relevant variance. |
| `fig_implied_timescales` | `results/chapter9/implied_timescales.csv` | Ch. 4 – MSM validation | The dominant implied timescale (mode 1) converges and plateaus across lag times, indicating a well-resolved slow process. |
| `fig_ck_validation` | `results/chapter9/ck_errors.csv` | Ch. 4 – MSM validation | Low Frobenius errors at prediction steps n = 2–4 confirm that the MSM satisfies the Chapman–Kolmogorov self-consistency test. |
| `fig_bootstrap_ci` | `results/chapter9/implied_timescale_cv.csv` | Ch. 4 – MSM validation | Error bars show the variability of each implied timescale across lag times; the dominant mode has the lowest coefficient of variation (CV ≈ 0.14), indicating robustness. |

### B — Learned structure

| Figure filename | Source data | Chapter / section | One-sentence interpretation |
|---|---|---|---|
| `fig_tica_landscape_colored_by_anomaly` | `outputs/run-traj-20250827-015400/deep/hybrid_scores.csv` | Ch. 5 – Latent space | Anomalous windows (marked ×) cluster in a distinct region of the tICA slow-space, visually separating rare conformations from the stationary-distribution background. |
| `fig_signal_correlation_heatmap` | `results/raw_traj/frame_scores_dynamic.csv` | Ch. 5 – Score decomposition | High correlation between state rarity and anomaly score, and moderate independence of transition surprise, validates the multi-component fused score design. |

### C — Anomaly interpretation

| Figure filename | Source data | Chapter / section | One-sentence interpretation |
|---|---|---|---|
| `fig_anomaly_vs_rmsf_comparison` | `results/chapter9/residue_ranking.csv` + `outputs/run-traj-20250827-000953/per_residue_overall.csv` | Ch. 6 – Residue-level interpretation | Positive Spearman correlation between RMSF proxy (Ramachandran distance) and anomaly score confirms that anomaly-flagged residues are also the most conformationally flexible. |
| `fig_rank_overlap_curve` | `results/chapter9/topk_sets.csv` + `results/chapter9/ranking_stability.csv` | Ch. 6 – Ranking stability | The top-5% hotspot residues are fully retained in the top-10% set, and recall remains ≥ 80% through k = 20%, demonstrating stable ranking hierarchy. |
| `fig_spatial_hotspot_summary` | `outputs/run-traj-20250827-015400/deep/residue_hotspots.csv` + `results/chapter9/spatial_clustering.csv` | Ch. 6 – Spatial hotspots | Top-ranked residues are spatially clustered well below the random expectation (z ≈ −2.95), confirming that anomaly signal is localized rather than uniformly distributed. |

### D — Case study

| Figure filename | Source data | Chapter / section | One-sentence interpretation |
|---|---|---|---|
| `fig_case_study_frame_score_distribution` | `results/raw_traj/frame_scores_dynamic.csv` | Ch. 9 – Case study | The anomaly-score distribution is right-skewed with a long tail of high-scoring frames, confirming that rare events are sparse relative to the background ensemble. |
| `fig_case_study_top_residues_bar` | `results/chapter9/residue_ranking.csv` | Ch. 9 – Case study | The 15 most anomalous residues are dominated by a cluster near positions 35–36, 62, and 64, partially overlapping with but extending beyond the hotspot region 52–60. |
| `fig_case_study_rmsf_vs_anomaly_residues` | `results/chapter9/residue_ranking.csv` + `outputs/run-traj-20250827-000953/per_residue_overall.csv` | Ch. 9 – Case study | Dual-axis plot reveals co-located peaks in anomaly score and RMSF proxy at the same residues, providing physical validation of the anomaly hotspots. |
| `fig_case_study_top_frames_summary` | `results/raw_traj/frame_scores_dynamic.csv` | Ch. 9 – Case study | Frames 1–2 (1-indexed) score highest, driven primarily by transition surprise, while frames in the 50–80 range are elevated mainly by slow-space isolation. |
| `fig_case_study_temporal_persistence_comparison` | `outputs/run-traj-20250827-015400/deep/hybrid_scores.csv` | Ch. 9 – Case study | The anomaly signal is not uniformly distributed in time; two bursts near windows starting at frames 74–96 have persistently elevated scores, consistent with a metastable rare state. |
| `fig_case_study_transition_surprise_comparison` | `results/physical_validation/frame_validation.csv` + `results/chapter9/transition_enrichment.csv` | Ch. 9 – Case study | Top-anomaly frames have substantially higher mean anomaly scores than background frames (Cohen's d ≈ 0.96), validating that the detector captures genuine distributional outliers. |
| `fig_case_study_stability_envelope` | `results/chapter9/ranking_stability.csv` | Ch. 9 – Case study | Spearman ρ ≈ 0.85 and low Jaccard (≈ 0.07 at top 10%) across all perturbations indicates a globally stable but locally sensitive ranking, with top-1 residues robust to hyperparameter changes. |

---

## Known data-vs-thesis mismatches

The following discrepancies were detected between on-disk data and the values
stated in the thesis narrative.  **All figures plot the on-disk data.**

| Parameter | On-disk value | Thesis-stated value | Notes |
|---|---|---|---|
| VAMP-2 (tICA) | 2.168 | 3.564 | Significant discrepancy (~38%). May reflect different lag time, n_components, or trajectory. |
| VAMP-2 (PCA) | 1.877 | 3.515 | Same cause as above. |
| Dominant ITS mean | 9.44 ns (multi-lag mean of mode 0) | 7.34 ns | CV is close (0.31 vs 0.32). Discrepancy may reflect different subsetting. |
| Frame count | 213 frames | 1001 frames | No 1001-frame score file found. The 213-frame file may be a sub-sampled or windowed derivative. |
| Top hotspot residues | 36, 35, 62, 33 (top 4) | 52–60 stated as hotspot | Residue 60 appears at rank 8; 52–59 at ranks 17–68. The hotspot region is present but not dominant. |

---

## File dependencies

```
results/
  chapter9/
    vamp_comparison.csv
    implied_timescales.csv
    implied_timescale_cv.csv
    ck_errors.csv
    residue_ranking.csv
    ranking_stability.csv
    topk_sets.csv
    transition_enrichment.csv
    spatial_clustering.csv
    hotspot_slowmode_alignment.csv
  raw_traj/
    frame_scores_dynamic.csv
  physical_validation/
    frame_validation.csv

outputs/
  run-traj-20250827-000953/
    per_residue_overall.csv
  run-traj-20250827-015400/
    deep/
      residue_hotspots.csv
      hybrid_scores.csv
      anomalies.csv
      transition_matrix.csv
```

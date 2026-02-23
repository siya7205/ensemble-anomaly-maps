# Test Results

**Run date:** 2026-02-23 17:44 UTC  
**Python:** 3.12.3 (main, Jan  8 2026, 11:30:50) [GCC 13.3.0]  
**Repository:** siya7205/ensemble-anomaly-maps  

## Summary

| Metric | Value |
|--------|-------|
| Test files run | 13 |
| ✅ Passed | 13 |
| ❌ Failed | 0 |
| Total time | 32.0s |

## Per-File Results

| Status | Test File | Time (s) | Notes |
|--------|-----------|----------|-------|
| ✅ PASS | `test_chapter9_evaluation.py` | 1.7 |  |
| ✅ PASS | `test_compute_presentation_metrics.py` | 11.7 |  |
| ✅ PASS | `test_dataset_validation.py` | 0.1 |  |
| ✅ PASS | `test_integration.py` | 1.2 |  |
| ✅ PASS | `test_optimizations.py` | 1.2 |  |
| ✅ PASS | `test_phase1.py` | 1.3 |  |
| ✅ PASS | `test_phase2.py` | 1.2 |  |
| ✅ PASS | `test_phase3.py` | 1.2 |  |
| ✅ PASS | `test_pipeline_edge_cases.py` | 8.8 |  |
| ✅ PASS | `test_reproducibility.py` | 0.2 |  |
| ✅ PASS | `test_scientific_validation.py` | 1.6 |  |
| ✅ PASS | `test_signals.py` | 1.2 |  |
| ✅ PASS | `test_statistical_validation.py` | 0.8 |  |

---

## Full Output Per Test File

### ✅ PASS `test_chapter9_evaluation.py` (1.7s)

```
======================================================================
TESTING CHAPTER 9 EVALUATION MODULE
======================================================================

[TEST] _fit_pipeline returns correct types...
  ✓ Pipeline returns correct types
[TEST] _fused_frame_scores shape...
  ✓ Frame scores shape: (150,), range [0.007, 0.779]
[TEST] _residue_fused_scores no NaN...
  ✓ Residue scores: 30 values, all finite
[TEST] Jaccard top10...
  ✓ Jaccard identical=1.0, flipped=0.000
[TEST] compute_implied_timescales...
  ✓ ITS: 2 rows; CV table: 1 modes
[TEST] compute_ck_errors...
  ✓ CK errors: 3 rows
[TEST] compute_vamp_comparison...
  ✓ VAMP-2: [{'model_type': 'tICA', 'vamp2_score': 3.6506828527610424}, {'model_type': 'PCA', 'vamp2_score': 3.5150398699515506}, {'model_type': 'raw_features', 'vamp2_score': 3.6506828527610424}]
[TEST] compute_residue_ranking...
  ✓ Rankings: 20 residues; top-k subsets present
[TEST] compute_transition_enrichment...
  ✓ Transition enrichment: Cohen's d = 0.1077
[TEST] compute_spatial_clustering...
  ✓ Spatial clustering Z = -0.2166
[TEST] hotspot_slowmode_alignment_no_tica...
  ✓ old_ρ=1.0000  new_ρ=0.2886  circularity=True
[TEST] compute_vamp_comparison_corrected...
  ✓ VAMP corrected: tICA=3.5644  raw=4.5983
[TEST] compute_transition_enrichment_window_sweep...
  ✓ Window sweep rows: 3, windows: [3, 5, 10]
[TEST] compute_ranking_stability_extended...
  ✓ Extended stability rows: 21, k% values: [10, 20, 30]
[TEST] Full end-to-end pipeline...

[ISSUE 5] No second trajectory available: No second trajectory or features_traj2.npy found.
  ✓ All 10 output files present
[TEST] RQ1: frame scores have non-zero variance...
  ✓ Frame score std=0.2057, range=[0.0067, 0.7785]
[TEST] RQ1: ITS plateau CV is finite...
  ✓ ITS CV: 1 mode(s), all finite and non-negative
[TEST] RQ1: transition enrichment Cohen's d is finite...
  ✓ Cohen's d = 5.7677 (positive: transition frames more anomalous)
[TEST] RQ1: corrected VAMP-2 scores are positive...
  ✓ VAMP-2 scores: tICA=3.5644, raw=4.5983, PCA=3.5150
[TEST] RQ2: top-k sets are nested...
  ✓ Nested sets: |top-5%|=2, |top-10%|=3, |top-20%|=6
[TEST] RQ2: residue ranking has visualization columns...
  ✓ Visualization columns present; rank range [1, 25], residue IDs [0, 24]
[TEST] RQ2: frame score length matches trajectory...
  ✓ Frame score length matches trajectory for n_frames in [80, 150, 200]
[TEST] RQ2: window sweep columns for visualization...
  ✓ Window sweep has 3 rows with required visualization columns
[TEST] RQ3: stability metrics bounded correctly...
  ✓ All 21 Jaccard values in [0, 1] for k% ∈ {10, 20, 30}
[TEST] RQ3: median vs mean fusion differ...
  ✓ Median vs mean fusion differ: mean |Δ| = 0.110097
[TEST] RQ3: lag perturbation changes frame scores...
  ✓ Lag 5 vs lag 3 frame scores differ: mean |Δ| = 0.096600
[TEST] RQ3: window sweep Cohen's d varies across windows...
  ✓ Cohen's d values: {3: -0.2874, 5: -0.278, 10: -0.264}

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

**stderr:**
```
[INFO] === 1. Implied Timescales ===
[INFO]   lag times: [2, 3, 5, 6, 7]
[INFO]   lag=2 → 1 timescales
[INFO]   lag=3 → 1 timescales
[INFO]   lag=5 → 0 timescales
[INFO]   lag=6 → 0 timescales
[INFO]   lag=7 → 0 timescales
[INFO]   Saved implied_timescales.csv
[INFO]   Saved implied_timescale_cv.csv
[INFO] === 2. CK Errors ===
[INFO]   n=2: Frobenius error = 0.0000
[WARNING] Skipping state set [5] due to error in estimation: Some row and corresponding column of the count matrix C have zero counts..
[INFO]   n=3: Frobenius error = 0.0000
[WARNING] Skipping state set [4] due to error in estimation: Some row and corresponding column of the count matrix C have zero counts..
[WARNING] Skipping state set [5] due to error in estimation: Some row and corresponding column of the count matrix C have zero counts..
[INFO]   n=4: Frobenius error = 0.0000
[INFO]   Saved ck_errors.csv
[INFO] === 3. VAMP-2 Comparison ===
[INFO]   tICA VAMP-2 score: 3.6507
[INFO]   PCA  VAMP-2 score: 3.5150
[INFO]   Raw  VAMP-2 score: 3.6507
[INFO]   Saved vamp_comparison.csv
[INFO] === 4. Per-Residue Ranking ===
[INFO]   Saved residue_ranking.csv
[INFO]   Saved topk_sets.csv
[INFO] === 6. Transition Enrichment ===
[INFO]   mean_transition=0.5740  mean_stable=0.5410  Cohen's d=0.1077
[INFO]   Saved transition_enrichment.csv
[INFO] === 7. Spatial Clustering ===
[INFO]   Parsed 20 Cα atoms
[INFO]   Observed mean Cα distance (top 10%): 22.8000 Å
[INFO]   random_mean=25.8400  random_std=14.0354  Z=-0.2166
[INFO]   Saved spatial_clustering.csv
[INFO] === ISSUE 1: Hotspot–Slow-Mode (no tICA signal) ===
[INFO]   Old Spearman ρ = 1.0000
[INFO]   New Spearman ρ = 0.2886 (frame-only fused, no tICA signal)
[INFO]   Circularity confirmed: True
[INFO]   Saved hotspot_slowmode_alignment_no_tica.csv
[INFO] === ISSUE 2: Corrected VAMP-2 Comparison ===
[INFO]   tICA VAMP-2 (on projected Y): 3.5644
[INFO]   PCA  VAMP-2 (on projected X_pca): 3.5150
[INFO]   Raw  VAMP-2 (all 7 features, no reduction): 4.5983
[INFO]   Saved vamp_comparison_corrected.csv
[INFO] === ISSUE 3: Transition Enrichment Window Sweep ===
[INFO]   window=± 3: mean_tr=0.6153  mean_st=0.5374  d=0.2557
[INFO]   window=± 5: mean_tr=0.5740  mean_st=0.5410  d=0.1077
[INFO]   window=±10: mean_tr=0.5549  mean_st=0.5435  d=0.0374
[INFO]   Saved transition_enrichment_window_sweep.csv
[INFO] 
  Window sweep summary:
[INFO]   window_size  mean_transition  mean_stable      cohens_d    
[INFO]   3            0.6153           0.5374           0.2557      
[INFO]   5            0.5740           0.5410           0.1077      
[INFO]   10           0.5549           0.5435           0.0374      
[INFO] === 4. Per-Residue Ranking ===
[INFO]   Saved residue_ranking.csv
[INFO]   Saved topk_sets.csv
[INFO] === ISSUE 4: Ranking Stability Extended ===
[INFO]   Baseline fused scores: tied=1  variance=0.039744
[INFO]   lag_minus20pct (lag=4)              J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   lag_plus20pct (lag=6)               J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   dim_minus2 (dim=1)                  J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   dim_plus2 (dim=5)                   J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   drop_rarity                         J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   drop_transition_surprise            J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   drop_local_density                  J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   Saved ranking_stability_extended.csv
[INFO]   n_tied=1  score_variance=0.039744
[INFO] Loading features from /tmp/tmp7m4dnwqp/e2e/features.npy
[INFO]   Feature matrix shape: (150, 7)
[INFO]   Residues (Cα atoms): 25
[INFO] === 1. Implied Timescales ===
[INFO]   lag times: [2, 3, 5, 6, 7]
[INFO]   lag=2 → 1 timescales
[INFO]   lag=3 → 1 timescales
[INFO]   lag=5 → 0 timescales
[INFO]   lag=6 → 0 timescales
[INFO]   lag=7 → 0 timescales
[INFO]   Saved implied_timescales.csv
[INFO]   Saved implied_timescale_cv.csv
[INFO] === 2. CK Errors ===
[INFO]   n=2: Frobenius error = 0.0000
[WARNING] Skipping state set [1] due to error in estimation: Some row and corresponding column of the count matrix C have zero counts..
[INFO]   n=3: Frobenius error = 0.0000
[WARNING] Skipping state set [1] due to error in estimation: Some row and corresponding column of the count matrix C have zero counts..
[WARNING] Skipping state set [2] due to error in estimation: Some row and corresponding column of the count matrix C have zero counts..
[INFO]   n=4: Frobenius error = 0.0000
[INFO]   Saved ck_errors.csv
[INFO] === 3. VAMP-2 Comparison ===
[INFO]   tICA VAMP-2 score: 3.6507
[INFO]   PCA  VAMP-2 score: 3.5150
[INFO]   Raw  VAMP-2 score: 3.6507
[INFO]   Saved vamp_comparison.csv
[INFO] Fitting baseline MSM pipeline ...
[INFO]   MSM: 2 states
[INFO] === 4. Per-Residue Ranking ===
[INFO]   Saved residue_ranking.csv
[INFO]   Saved topk_sets.csv
[INFO] === 5. Hotspot–Slow-Mode Alignment ===
[INFO]   Spearman ρ = 1.0000, p = 1.586e-181
[INFO]   Saved hotspot_slowmode_alignment.csv
[INFO] === 6. Transition Enrichment ===
[INFO]   mean_transition=0.5891  mean_stable=0.4200  Cohen's d=1.0169
[INFO]   Saved transition_enrichment.csv
[INFO] === 7. Spatial Clustering ===
[INFO]   Parsed 25 Cα atoms
[INFO]   Observed mean Cα distance (top 10%): 35.4667 Å
[INFO]   random_mean=35.1373  random_std=12.8099  Z=0.0257
[INFO]   Saved spatial_clustering.csv
[INFO] === 8. Ranking Stability (RQ3) ===
[INFO]   lag_minus20pct (lag=4)              ρ=0.093  J=0.000  med=4.0  p90=17.2
[INFO]   lag_plus20pct (lag=6)               ρ=0.093  J=0.000  med=4.0  p90=17.2
[INFO]   dim_minus2 (dim=1)                  ρ=0.093  J=0.000  med=4.0  p90=17.2
[INFO]   dim_plus2 (dim=5)                   ρ=0.093  J=0.000  med=4.0  p90=17.2
[INFO]   drop_rarity                         ρ=0.093  J=0.000  med=4.0  p90=17.2
[INFO]   drop_transition_surprise            ρ=0.093  J=0.000  med=4.0  p90=17.2
[INFO]   drop_local_density                  ρ=0.093  J=0.000  med=4.0  p90=17.2
[INFO]   Saved ranking_stability.csv
[INFO] === ISSUE 1: Hotspot–Slow-Mode (no tICA signal) ===
[INFO]   Old Spearman ρ = 1.0000
[INFO]   New Spearman ρ = 0.1723 (frame-only fused, no tICA signal)
[INFO]   Circularity confirmed: True
[INFO]   Saved hotspot_slowmode_alignment_no_tica.csv
[INFO] === ISSUE 2: Corrected VAMP-2 Comparison ===
[INFO]   tICA VAMP-2 (on projected Y): 3.5644
[INFO]   PCA  VAMP-2 (on projected X_pca): 3.5150
[INFO]   Raw  VAMP-2 (all 7 features, no reduction): 4.5983
[INFO]   Saved vamp_comparison_corrected.csv
[INFO] === ISSUE 3: Transition Enrichment Window Sweep ===
[INFO]   window=± 3: mean_tr=0.5802  mean_st=0.4609  d=0.6687
[INFO]   window=± 5: mean_tr=0.5891  mean_st=0.4200  d=1.0169
[INFO]   window=±10: mean_tr=0.5715  mean_st=0.3423  d=1.5069
[INFO]   Saved transition_enrichment_window_sweep.csv
[INFO] 
  Window sweep summary:
[INFO]   window_size  mean_transition  mean_stable      cohens_d    
[INFO]   3            0.5802           0.4609           0.6687      
[INFO]   5            0.5891           0.4200           1.0169      
[INFO]   10           0.5715           0.3423           1.5069      
[INFO] === ISSUE 4: Ranking Stability Extended ===
[INFO]   Baseline fused scores: tied=1  variance=0.039744
[INFO]   lag_minus20pct (lag=4)              J@10%=0.000  J@20%=0.000  J@30%=0.231
[INFO]   lag_plus20pct (lag=6)               J@10%=0.000  J@20%=0.000  J@30%=0.231
[INFO]   dim_minus2 (dim=1)                  J@10%=0.000  J@20%=0.000  J@30%=0.231
[INFO]   dim_plus2 (dim=5)                   J@10%=0.000  J@20%=0.000  J@30%=0.231
[INFO]   drop_rarity                         J@10%=0.000  J@20%=0.000  J@30%=0.231
[INFO]   drop_transition_surprise            J@10%=0.000  J@20%=0.000  J@30%=0.231
[INFO]   drop_local_density                  J@10%=0.000  J@20%=0.000  J@30%=0.231
[INFO]   Saved ranking_stability_extended.csv
[INFO]   n_tied=1  score_variance=0.039744
[INFO] === ISSUE 5: Second Trajectory Evaluation ===
[INFO]   No second trajectory available: No second trajectory or features_traj2.npy found.
[INFO] 
======================================================================
[INFO] CHAPTER 9 EVALUATION — SUMMARY
[INFO] ======================================================================
[INFO] 
--- ITS Coefficient of Variation (top modes, plateau region) ---
[INFO]   Mode 0: mean=7.34  std=2.35  CV=0.3200
[INFO] 
--- Mean CK Frobenius Error ---
[INFO]   Mean across n∈{2,3,4}: 0.0000
[INFO]   n=2: 0.0000
[INFO]   n=3: 0.0000
[INFO]   n=4: 0.0000
[INFO] 
--- VAMP-2 Comparison ---
[INFO]   tICA             VAMP-2 = 3.6507
[INFO]   PCA              VAMP-2 = 3.5150
[INFO]   raw_features     VAMP-2 = 3.6507
[INFO] 
--- Hotspot–Slow-Mode Spearman ---
[INFO]   ρ = 1.0000  p = 1.586e-181
[INFO] 
--- Transition Enrichment ---
[INFO]   mean_transition=0.5891  mean_stable=0.4200  Cohen's d=1.0169
[INFO] 
--- Spatial Clustering Z-Score ---
[INFO]   obs=35.4667  rand_mean=35.1373  Z=0.0257
[INFO] 
--- Ranking Stability ---
[INFO]   Perturbation                             ρ   J@10%   med_Δ   p90_Δ
[INFO]   lag_minus20pct (lag=4)               0.093   0.000     4.0    17.2
[INFO]   lag_plus20pct (lag=6)                0.093   0.000     4.0    17.2
[INFO]   dim_minus2 (dim=1)                   0.093   0.000     4.0    17.2
[INFO]   dim_plus2 (dim=5)                    0.093   0.000     4.0    17.2
[INFO]   drop_rarity                          0.093   0.000     4.0    17.2
[INFO]   drop_transition_surprise             0.093   0.000     4.0    17.2
[INFO]   drop_local_density                   0.093   0.000     4.0    17.2
[INFO] 
======================================================================
[INFO] 
======================================================================
[INFO] ISSUE INVESTIGATION SUMMARY
[INFO] ======================================================================
[INFO] 
[ISSUE 1] Circular Hotspot–Slow-Mode Correlation
[INFO]   Old Spearman ρ = 1.0000
[INFO]   New Spearman ρ = 0.1723 (frame-only scores, no tICA signal)
[INFO]   Circularity confirmed: True
[INFO] 
[ISSUE 2] Corrected VAMP-2 Scores
[INFO]   tICA             VAMP-2 = 3.5644
[INFO]   PCA              VAMP-2 = 3.5150
[INFO]   raw_features     VAMP-2 = 4.5983
[INFO] 
[ISSUE 3] Transition Window Sweep
[INFO]   window=± 3: mean_tr=0.5802  mean_st=0.4609  d=0.6687
[INFO]   window=± 5: mean_tr=0.5891  mean_st=0.4200  d=1.0169
[INFO]   window=±10: mean_tr=0.5715  mean_st=0.3423  d=1.5069
[INFO] 
[ISSUE 4] Ranking Stability (extended top-k)
[INFO]   lag_minus20pct (lag=4)              top-10%  J=0.000
[INFO]   lag_minus20pct (lag=4)              top-20%  J=0.000
[INFO]   lag_minus20pct (lag=4)              top-30%  J=0.231
[INFO]   lag_plus20pct (lag=6)               top-10%  J=0.000
[INFO]   lag_plus20pct (lag=6)               top-20%  J=0.000
[INFO]   lag_plus20pct (lag=6)               top-30%  J=0.231
[INFO]   dim_minus2 (dim=1)                  top-10%  J=0.000
[INFO]   dim_minus2 (dim=1)                  top-20%  J=0.000
[INFO]   dim_minus2 (dim=1)                  top-30%  J=0.231
[INFO]   dim_plus2 (dim=5)                   top-10%  J=0.000
[INFO]   dim_plus2 (dim=5)                   top-20%  J=0.000
[INFO]   dim_plus2 (dim=5)                   top-30%  J=0.231
[INFO]   drop_rarity                         top-10%  J=0.000
[INFO]   drop_rarity                         top-20%  J=0.000
[INFO]   drop_rarity                         top-30%  J=0.231
[INFO]   drop_transition_surprise            top-10%  J=0.000
[INFO]   drop_transition_surprise            top-20%  J=0.000
[INFO]   drop_transition_surprise            top-30%  J=0.231
[INFO]   drop_local_density                  top-10%  J=0.000
[INFO]   drop_local_density                  top-20%  J=0.000
[INFO]   drop_local_density                  top-30%  J=0.231
[INFO] 
======================================================================
[INFO] All outputs saved to /tmp/tmp7m4dnwqp/e2e/results/chapter9
[INFO] === 1. Implied Timescales ===
[INFO]   lag times: [2, 3, 5, 6, 7]
[INFO]   lag=2 → 1 timescales
[INFO]   lag=3 → 0 timescales
[INFO]   lag=5 → 0 timescales
[INFO]   lag=6 → 0 timescales
[INFO]   lag=7 → 0 timescales
[INFO]   Saved implied_timescales.csv
[INFO]   Saved implied_timescale_cv.csv
[INFO] === 6. Transition Enrichment ===
[INFO]   mean_transition=0.7677  mean_stable=0.4005  Cohen's d=5.7677
[INFO]   Saved transition_enrichment.csv
[INFO] === ISSUE 2: Corrected VAMP-2 Comparison ===
[INFO]   tICA VAMP-2 (on projected Y): 3.5644
[INFO]   PCA  VAMP-2 (on projected X_pca): 3.5150
[INFO]   Raw  VAMP-2 (all 7 features, no reduction): 4.5983
[INFO]   Saved vamp_comparison_corrected.csv
[INFO] === 4. Per-Residue Ranking ===
[INFO]   Saved residue_ranking.csv
[INFO]   Saved topk_sets.csv
[INFO] === 4. Per-Residue Ranking ===
[INFO]   Saved residue_ranking.csv
[INFO]   Saved topk_sets.csv
[INFO] === ISSUE 3: Transition Enrichment Window Sweep ===
[INFO]   window=± 3: mean_tr=0.5061  mean_st=0.5219  d=-0.0528
[INFO]   window=± 5: mean_tr=0.5269  mean_st=0.5176  d=0.0314
[INFO]   window=±10: mean_tr=0.5231  mean_st=0.5172  d=0.0198
[INFO]   Saved transition_enrichment_window_sweep.csv
[INFO] 
  Window sweep summary:
[INFO]   window_size  mean_transition  mean_stable      cohens_d    
[INFO]   3            0.5061           0.5219           -0.0528     
[INFO]   5            0.5269           0.5176           0.0314      
[INFO]   10           0.5231           0.5172           0.0198      
[INFO] === 4. Per-Residue Ranking ===
[INFO]   Saved residue_ranking.csv
[INFO]   Saved topk_sets.csv
[INFO] === ISSUE 4: Ranking Stability Extended ===
[INFO]   Baseline fused scores: tied=1  variance=0.030137
[INFO]   lag_minus20pct (lag=4)              J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   lag_plus20pct (lag=6)               J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   dim_minus2 (dim=1)                  J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   dim_plus2 (dim=5)                   J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   drop_rarity                         J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   drop_transition_surprise            J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   drop_local_density                  J@10%=1.000  J@20%=1.000  J@30%=1.000
[INFO]   Saved ranking_stability_extended.csv
[INFO]   n_tied=1  score_variance=0.030137
[INFO] === ISSUE 3: Transition Enrichment Window Sweep ===
[INFO]   window=± 3: mean_tr=0.3799  mean_st=0.4667  d=-0.2874
[INFO]   window=± 5: mean_tr=0.3892  mean_st=0.4730  d=-0.2780
[INFO]   window=±10: mean_tr=0.4085  mean_st=0.4879  d=-0.2640
[INFO]   Saved transition_enrichment_window_sweep.csv
[INFO] 
  Window sweep summary:
[INFO]   window_size  mean_transition  mean_stable      cohens_d    
[INFO]   3            0.3799           0.4667           -0.2874     
[INFO]   5            0.3892           0.4730           -0.2780     
[INFO]   10           0.4085           0.4879           -0.2640
```

### ✅ PASS `test_compute_presentation_metrics.py` (11.7s)

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps
collecting ... collected 9 items

tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_sample_predictions_exists PASSED [ 11%]
tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_sample_predictions_format PASSED [ 22%]
tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_script_runs_successfully FAILED [ 33%]
tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_output_files_created FAILED [ 44%]
tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_metrics_summary_content FAILED [ 55%]
tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_per_run_metrics FAILED [ 66%]
tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_positional_argument FAILED [ 77%]
tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_dry_run FAILED [ 88%]
tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_reproducibility FAILED [100%]

=================================== FAILURES ===================================
_________ TestComputePresentationMetrics.test_script_runs_successfully _________

self = <test_compute_presentation_metrics.TestComputePresentationMetrics object at 0x7f3358e858b0>
sample_predictions_path = PosixPath('/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv')
output_dir = PosixPath('/tmp/pytest-of-runner/pytest-1/test_script_runs_successfully0/metrics_output')

    def test_script_runs_successfully(self, sample_predictions_path, output_dir):
        """Test that the script runs and produces expected outputs."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
    
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--out-dir', str(output_dir),
                '--bootstrap', '200',  # Fewer resamples for fast CI
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
    
        # Print output for debugging
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    
>       assert result.returncode == 0, f"Script failed with: {result.stderr}"
E       AssertionError: Script failed with: Traceback (most recent call last):
E           File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>
E             import seaborn as sns
E         ModuleNotFoundError: No module named 'seaborn'
E         
E       assert 1 == 0
E        +  where 1 = CompletedProcess(args=['/usr/bin/python3', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py', '--predictions', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv', '--out-dir', '/tmp/pytest-of-runner/pytest-1/test_script_runs_successfully0/metrics_output', '--bootstrap', '200', '--seed', '42'], returncode=1, stdout='', stderr='Traceback (most recent call last):\n  File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>\n    import seaborn as sns\nModuleNotFoundError: No module named \'seaborn\'\n').returncode

tests/test_compute_presentation_metrics.py:73: AssertionError
----------------------------- Captured stdout call -----------------------------
STDOUT: 
STDERR: Traceback (most recent call last):
  File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>
    import seaborn as sns
ModuleNotFoundError: No module named 'seaborn'

___________ TestComputePresentationMetrics.test_output_files_created ___________

self = <test_compute_presentation_metrics.TestComputePresentationMetrics object at 0x7f3358e85bb0>
sample_predictions_path = PosixPath('/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv')
output_dir = PosixPath('/tmp/pytest-of-runner/pytest-1/test_output_files_created0/metrics_output')

    def test_output_files_created(self, sample_predictions_path, output_dir):
        """Test that expected output files are created."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
    
        subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--out-dir', str(output_dir),
                '--bootstrap', '200',
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
    
        # Check that key output files exist
>       assert (output_dir / 'metrics_summary.csv').exists(), "metrics_summary.csv not created"
E       AssertionError: metrics_summary.csv not created
E       assert False
E        +  where False = exists()
E        +    where exists = (PosixPath('/tmp/pytest-of-runner/pytest-1/test_output_files_created0/metrics_output') / 'metrics_summary.csv').exists

tests/test_compute_presentation_metrics.py:93: AssertionError
_________ TestComputePresentationMetrics.test_metrics_summary_content __________

self = <test_compute_presentation_metrics.TestComputePresentationMetrics object at 0x7f3358e85eb0>
sample_predictions_path = PosixPath('/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv')
output_dir = PosixPath('/tmp/pytest-of-runner/pytest-1/test_metrics_summary_content0/metrics_output')

    def test_metrics_summary_content(self, sample_predictions_path, output_dir):
        """Test that metrics_summary.csv has expected content."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
    
        subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--out-dir', str(output_dir),
                '--bootstrap', '200',
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
    
        # Load and verify metrics
>       metrics_df = pd.read_csv(output_dir / 'metrics_summary.csv')
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_compute_presentation_metrics.py:116: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../../../.local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:873: in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:300: in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1645: in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1904: in _make_engine
    self.handles = get_handle(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

path_or_buf = PosixPath('/tmp/pytest-of-runner/pytest-1/test_metrics_summary_content0/metrics_output/metrics_summary.csv')
mode = 'r'

    def get_handle(
        path_or_buf: FilePath | BaseBuffer,
        mode: str,
        *,
        encoding: str | None = None,
        compression: CompressionOptions | None = None,
        memory_map: bool = False,
        is_text: bool = True,
        errors: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> IOHandles[str] | IOHandles[bytes]:
        """
        Get file handle for given path/buffer and mode.
    
        Parameters
        ----------
        path_or_buf : str or file handle
            File path or object.
        mode : str
            Mode to open path_or_buf with.
        encoding : str or None
            Encoding to use.
        compression : str or dict, default 'infer'
            For on-the-fly compression of the output data. If 'infer' and 'path_or_buf'
            is path-like, then detect compression from the following extensions: '.gz',
            '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
            (otherwise no compression).
            Set to ``None`` for no compression.
            Can also be a dict with key ``'method'`` set
            to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``}
            and other key-value pairs are forwarded to
            ``zipfile.ZipFile``, ``gzip.GzipFile``,
            ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
            ``tarfile.TarFile``, respectively.
            As an example, the following could be passed for faster compression and to
            create a reproducible gzip archive:
            ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.
    
               May be a dict with key 'method' as compression mode
               and other keys as compression options if compression
               mode is 'zip'.
    
               Passing compression options as keys in dict is
               supported for compression modes 'gzip', 'bz2', 'zstd' and 'zip'.
    
        memory_map : bool, default False
            See parsers._parser_params for more information. Only used by read_csv.
        is_text : bool, default True
            Whether the type of the content passed to the file/buffer is string or
            bytes. This is not the same as `"b" not in mode`. If a string content is
            passed to a binary file/buffer, a wrapper is inserted.
        errors : str, default 'strict'
            Specifies how encoding and decoding errors are to be handled.
            See the errors argument for :func:`open` for a full list
            of options.
        storage_options: StorageOptions = None
            Passed to _get_filepath_or_buffer
    
        Returns the dataclass IOHandles
        """
        # Windows does not default to utf-8. Set to utf-8 for a consistent behavior
        encoding = encoding or "utf-8"
    
        errors = errors or "strict"
    
        # read_csv does not know whether the buffer is opened in binary/text mode
        if _is_binary_mode(path_or_buf, mode) and "b" not in mode:
            mode += "b"
    
        # validate encoding and errors
        codecs.lookup(encoding)
        if isinstance(errors, str):
            codecs.lookup_error(errors)
    
        # open URLs
        ioargs = _get_filepath_or_buffer(
            path_or_buf,
            encoding=encoding,
            compression=compression,
            mode=mode,
            storage_options=storage_options,
        )
    
        handle = ioargs.filepath_or_buffer
        handles: list[BaseBuffer]
    
        # memory mapping needs to be the first step
        # only used for read_csv
        handle, memory_map, handles = _maybe_memory_map(handle, memory_map)
    
        is_path = isinstance(handle, str)
        compression_args = dict(ioargs.compression)
        compression = compression_args.pop("method")
    
        # Only for write methods
        if "r" not in mode and is_path:
            check_parent_directory(str(handle))
    
        if compression:
            if compression != "zstd":
                # compression libraries do not like an explicit text-mode
                ioargs.mode = ioargs.mode.replace("t", "")
            elif compression == "zstd" and "b" not in ioargs.mode:
                # python-zstandard defaults to text mode, but we always expect
                # compression libraries to use binary mode.
                ioargs.mode += "b"
    
            # GZ Compression
            if compression == "gzip":
                if isinstance(handle, str):
                    # error: Incompatible types in assignment (expression has type
                    # "GzipFile", variable has type "Union[str, BaseBuffer]")
                    handle = gzip.GzipFile(  # type: ignore[assignment]
                        filename=handle,
                        mode=ioargs.mode,
                        **compression_args,
                    )
                else:
                    handle = gzip.GzipFile(
                        # No overload variant of "GzipFile" matches argument types
                        # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
                        fileobj=handle,  # type: ignore[call-overload]
                        mode=ioargs.mode,
                        **compression_args,
                    )
    
            # BZ Compression
            elif compression == "bz2":
                import bz2
    
                # Overload of "BZ2File" to handle pickle protocol 5
                # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
                handle = bz2.BZ2File(  # type: ignore[call-overload]
                    handle,
                    mode=ioargs.mode,
                    **compression_args,
                )
    
            # ZIP Compression
            elif compression == "zip":
                # error: Argument 1 to "_BytesZipFile" has incompatible type
                # "Union[str, BaseBuffer]"; expected "Union[Union[str, PathLike[str]],
                # ReadBuffer[bytes], WriteBuffer[bytes]]"
                handle = _BytesZipFile(
                    handle,  # type: ignore[arg-type]
                    ioargs.mode,
                    **compression_args,
                )
                if handle.buffer.mode == "r":
                    handles.append(handle)
                    zip_names = handle.buffer.namelist()
                    if len(zip_names) == 1:
                        handle = handle.buffer.open(zip_names.pop())
                    elif not zip_names:
                        raise ValueError(f"Zero files found in ZIP file {path_or_buf}")
                    else:
                        raise ValueError(
                            "Multiple files found in ZIP file. "
                            f"Only one file per ZIP: {zip_names}"
                        )
    
            # TAR Encoding
            elif compression == "tar":
                compression_args.setdefault("mode", ioargs.mode)
                if isinstance(handle, str):
                    handle = _BytesTarFile(name=handle, **compression_args)
                else:
                    # error: Argument "fileobj" to "_BytesTarFile" has incompatible
                    # type "BaseBuffer"; expected "Union[ReadBuffer[bytes],
                    # WriteBuffer[bytes], None]"
                    handle = _BytesTarFile(
                        fileobj=handle,  # type: ignore[arg-type]
                        **compression_args,
                    )
                assert isinstance(handle, _BytesTarFile)
                if "r" in handle.buffer.mode:
                    handles.append(handle)
                    files = handle.buffer.getnames()
                    if len(files) == 1:
                        file = handle.buffer.extractfile(files[0])
                        assert file is not None
                        handle = file
                    elif not files:
                        raise ValueError(f"Zero files found in TAR archive {path_or_buf}")
                    else:
                        raise ValueError(
                            "Multiple files found in TAR archive. "
                            f"Only one file per TAR archive: {files}"
                        )
    
            # XZ Compression
            elif compression == "xz":
                # error: Argument 1 to "LZMAFile" has incompatible type "Union[str,
                # BaseBuffer]"; expected "Optional[Union[Union[str, bytes, PathLike[str],
                # PathLike[bytes]], IO[bytes]], None]"
                import lzma
    
                handle = lzma.LZMAFile(
                    handle,  # type: ignore[arg-type]
                    ioargs.mode,
                    **compression_args,
                )
    
            # Zstd Compression
            elif compression == "zstd":
                zstd = import_optional_dependency("zstandard")
                if "r" in ioargs.mode:
                    open_args = {"dctx": zstd.ZstdDecompressor(**compression_args)}
                else:
                    open_args = {"cctx": zstd.ZstdCompressor(**compression_args)}
                handle = zstd.open(
                    handle,
                    mode=ioargs.mode,
                    **open_args,
                )
    
            # Unrecognized Compression
            else:
                msg = f"Unrecognized compression type: {compression}"
                raise ValueError(msg)
    
            assert not isinstance(handle, str)
            handles.append(handle)
    
        elif isinstance(handle, str):
            # Check whether the filename is to be opened in binary mode.
            # Binary mode does not support 'encoding' and 'newline'.
            if ioargs.encoding and "b" not in ioargs.mode:
                # Encoding
>               handle = open(
                    handle,
                    ioargs.mode,
                    encoding=ioargs.encoding,
                    errors=errors,
                    newline="",
                )
E               FileNotFoundError: [Errno 2] No such file or directory: '/tmp/pytest-of-runner/pytest-1/test_metrics_summary_content0/metrics_output/metrics_summary.csv'

../../../.local/lib/python3.12/site-packages/pandas/io/common.py:926: FileNotFoundError
_____________ TestComputePresentationMetrics.test_per_run_metrics ______________

self = <test_compute_presentation_metrics.TestComputePresentationMetrics object at 0x7f3358e861b0>
sample_predictions_path = PosixPath('/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv')
output_dir = PosixPath('/tmp/pytest-of-runner/pytest-1/test_per_run_metrics0/metrics_output')

    def test_per_run_metrics(self, sample_predictions_path, output_dir):
        """Test that per-run metrics are computed when run_id is present."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
    
        # Check if sample data has run_id
        df = pd.read_csv(sample_predictions_path)
        has_run_id = 'run_id' in df.columns and df['run_id'].nunique() > 1
    
        subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--out-dir', str(output_dir),
                '--bootstrap', '200',
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
    
        if has_run_id:
>           assert (output_dir / 'metrics_summary_per_run.csv').exists(), "Per-run metrics not created"
E           AssertionError: Per-run metrics not created
E           assert False
E            +  where False = exists()
E            +    where exists = (PosixPath('/tmp/pytest-of-runner/pytest-1/test_per_run_metrics0/metrics_output') / 'metrics_summary_per_run.csv').exists

tests/test_compute_presentation_metrics.py:151: AssertionError
___________ TestComputePresentationMetrics.test_positional_argument ____________

self = <test_compute_presentation_metrics.TestComputePresentationMetrics object at 0x7f3358e864b0>
sample_predictions_path = PosixPath('/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv')
output_dir = PosixPath('/tmp/pytest-of-runner/pytest-1/test_positional_argument0/metrics_output')

    def test_positional_argument(self, sample_predictions_path, output_dir):
        """Test that positional argument works for predictions path."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
    
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                str(sample_predictions_path),  # positional argument
                '--out-dir', str(output_dir),
                '--bootstrap', '100',
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
    
        # Print output for debugging
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    
>       assert result.returncode == 0, f"Script failed with: {result.stderr}"
E       AssertionError: Script failed with: Traceback (most recent call last):
E           File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>
E             import seaborn as sns
E         ModuleNotFoundError: No module named 'seaborn'
E         
E       assert 1 == 0
E        +  where 1 = CompletedProcess(args=['/usr/bin/python3', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv', '--out-dir', '/tmp/pytest-of-runner/pytest-1/test_positional_argument0/metrics_output', '--bootstrap', '100', '--seed', '42'], returncode=1, stdout='', stderr='Traceback (most recent call last):\n  File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>\n    import seaborn as sns\nModuleNotFoundError: No module named \'seaborn\'\n').returncode

tests/test_compute_presentation_metrics.py:177: AssertionError
----------------------------- Captured stdout call -----------------------------
STDOUT: 
STDERR: Traceback (most recent call last):
  File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>
    import seaborn as sns
ModuleNotFoundError: No module named 'seaborn'

_________________ TestComputePresentationMetrics.test_dry_run __________________

self = <test_compute_presentation_metrics.TestComputePresentationMetrics object at 0x7f3358e867e0>
sample_predictions_path = PosixPath('/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv')

    def test_dry_run(self, sample_predictions_path):
        """Test dry run mode doesn't create output files."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
    
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--dry-run'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
    
>       assert result.returncode == 0, f"Dry run failed: {result.stderr}"
E       AssertionError: Dry run failed: Traceback (most recent call last):
E           File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>
E             import seaborn as sns
E         ModuleNotFoundError: No module named 'seaborn'
E         
E       assert 1 == 0
E        +  where 1 = CompletedProcess(args=['/usr/bin/python3', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py', '--predictions', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv', '--dry-run'], returncode=1, stdout='', stderr='Traceback (most recent call last):\n  File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>\n    import seaborn as sns\nModuleNotFoundError: No module named \'seaborn\'\n').returncode

tests/test_compute_presentation_metrics.py:195: AssertionError
_____________ TestComputePresentationMetrics.test_reproducibility ______________

self = <test_compute_presentation_metrics.TestComputePresentationMetrics object at 0x7f3358e86a80>
sample_predictions_path = PosixPath('/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv')
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-1/test_reproducibility0')

    def test_reproducibility(self, sample_predictions_path, tmp_path):
        """Test that results are reproducible with same seed."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
    
        out_dir1 = tmp_path / 'run1'
        out_dir2 = tmp_path / 'run2'
    
        # Run twice with same seed
        for out_dir in [out_dir1, out_dir2]:
            subprocess.run(
                [
                    sys.executable, str(script_path),
                    '--predictions', str(sample_predictions_path),
                    '--out-dir', str(out_dir),
                    '--bootstrap', '200',
                    '--seed', '42'
                ],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)
            )
    
        # Compare metrics
>       metrics1 = pd.read_csv(out_dir1 / 'metrics_summary.csv')
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_compute_presentation_metrics.py:221: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../../../.local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:873: in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:300: in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1645: in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../.local/lib/python3.12/site-packages/pandas/io/parsers/readers.py:1904: in _make_engine
    self.handles = get_handle(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

path_or_buf = PosixPath('/tmp/pytest-of-runner/pytest-1/test_reproducibility0/run1/metrics_summary.csv')
mode = 'r'

    def get_handle(
        path_or_buf: FilePath | BaseBuffer,
        mode: str,
        *,
        encoding: str | None = None,
        compression: CompressionOptions | None = None,
        memory_map: bool = False,
        is_text: bool = True,
        errors: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> IOHandles[str] | IOHandles[bytes]:
        """
        Get file handle for given path/buffer and mode.
    
        Parameters
        ----------
        path_or_buf : str or file handle
            File path or object.
        mode : str
            Mode to open path_or_buf with.
        encoding : str or None
            Encoding to use.
        compression : str or dict, default 'infer'
            For on-the-fly compression of the output data. If 'infer' and 'path_or_buf'
            is path-like, then detect compression from the following extensions: '.gz',
            '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
            (otherwise no compression).
            Set to ``None`` for no compression.
            Can also be a dict with key ``'method'`` set
            to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``}
            and other key-value pairs are forwarded to
            ``zipfile.ZipFile``, ``gzip.GzipFile``,
            ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
            ``tarfile.TarFile``, respectively.
            As an example, the following could be passed for faster compression and to
            create a reproducible gzip archive:
            ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.
    
               May be a dict with key 'method' as compression mode
               and other keys as compression options if compression
               mode is 'zip'.
    
               Passing compression options as keys in dict is
               supported for compression modes 'gzip', 'bz2', 'zstd' and 'zip'.
    
        memory_map : bool, default False
            See parsers._parser_params for more information. Only used by read_csv.
        is_text : bool, default True
            Whether the type of the content passed to the file/buffer is string or
            bytes. This is not the same as `"b" not in mode`. If a string content is
            passed to a binary file/buffer, a wrapper is inserted.
        errors : str, default 'strict'
            Specifies how encoding and decoding errors are to be handled.
            See the errors argument for :func:`open` for a full list
            of options.
        storage_options: StorageOptions = None
            Passed to _get_filepath_or_buffer
    
        Returns the dataclass IOHandles
        """
        # Windows does not default to utf-8. Set to utf-8 for a consistent behavior
        encoding = encoding or "utf-8"
    
        errors = errors or "strict"
    
        # read_csv does not know whether the buffer is opened in binary/text mode
        if _is_binary_mode(path_or_buf, mode) and "b" not in mode:
            mode += "b"
    
        # validate encoding and errors
        codecs.lookup(encoding)
        if isinstance(errors, str):
            codecs.lookup_error(errors)
    
        # open URLs
        ioargs = _get_filepath_or_buffer(
            path_or_buf,
            encoding=encoding,
            compression=compression,
            mode=mode,
            storage_options=storage_options,
        )
    
        handle = ioargs.filepath_or_buffer
        handles: list[BaseBuffer]
    
        # memory mapping needs to be the first step
        # only used for read_csv
        handle, memory_map, handles = _maybe_memory_map(handle, memory_map)
    
        is_path = isinstance(handle, str)
        compression_args = dict(ioargs.compression)
        compression = compression_args.pop("method")
    
        # Only for write methods
        if "r" not in mode and is_path:
            check_parent_directory(str(handle))
    
        if compression:
            if compression != "zstd":
                # compression libraries do not like an explicit text-mode
                ioargs.mode = ioargs.mode.replace("t", "")
            elif compression == "zstd" and "b" not in ioargs.mode:
                # python-zstandard defaults to text mode, but we always expect
                # compression libraries to use binary mode.
                ioargs.mode += "b"
    
            # GZ Compression
            if compression == "gzip":
                if isinstance(handle, str):
                    # error: Incompatible types in assignment (expression has type
                    # "GzipFile", variable has type "Union[str, BaseBuffer]")
                    handle = gzip.GzipFile(  # type: ignore[assignment]
                        filename=handle,
                        mode=ioargs.mode,
                        **compression_args,
                    )
                else:
                    handle = gzip.GzipFile(
                        # No overload variant of "GzipFile" matches argument types
                        # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
                        fileobj=handle,  # type: ignore[call-overload]
                        mode=ioargs.mode,
                        **compression_args,
                    )
    
            # BZ Compression
            elif compression == "bz2":
                import bz2
    
                # Overload of "BZ2File" to handle pickle protocol 5
                # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
                handle = bz2.BZ2File(  # type: ignore[call-overload]
                    handle,
                    mode=ioargs.mode,
                    **compression_args,
                )
    
            # ZIP Compression
            elif compression == "zip":
                # error: Argument 1 to "_BytesZipFile" has incompatible type
                # "Union[str, BaseBuffer]"; expected "Union[Union[str, PathLike[str]],
                # ReadBuffer[bytes], WriteBuffer[bytes]]"
                handle = _BytesZipFile(
                    handle,  # type: ignore[arg-type]
                    ioargs.mode,
                    **compression_args,
                )
                if handle.buffer.mode == "r":
                    handles.append(handle)
                    zip_names = handle.buffer.namelist()
                    if len(zip_names) == 1:
                        handle = handle.buffer.open(zip_names.pop())
                    elif not zip_names:
                        raise ValueError(f"Zero files found in ZIP file {path_or_buf}")
                    else:
                        raise ValueError(
                            "Multiple files found in ZIP file. "
                            f"Only one file per ZIP: {zip_names}"
                        )
    
            # TAR Encoding
            elif compression == "tar":
                compression_args.setdefault("mode", ioargs.mode)
                if isinstance(handle, str):
                    handle = _BytesTarFile(name=handle, **compression_args)
                else:
                    # error: Argument "fileobj" to "_BytesTarFile" has incompatible
                    # type "BaseBuffer"; expected "Union[ReadBuffer[bytes],
                    # WriteBuffer[bytes], None]"
                    handle = _BytesTarFile(
                        fileobj=handle,  # type: ignore[arg-type]
                        **compression_args,
                    )
                assert isinstance(handle, _BytesTarFile)
                if "r" in handle.buffer.mode:
                    handles.append(handle)
                    files = handle.buffer.getnames()
                    if len(files) == 1:
                        file = handle.buffer.extractfile(files[0])
                        assert file is not None
                        handle = file
                    elif not files:
                        raise ValueError(f"Zero files found in TAR archive {path_or_buf}")
                    else:
                        raise ValueError(
                            "Multiple files found in TAR archive. "
                            f"Only one file per TAR archive: {files}"
                        )
    
            # XZ Compression
            elif compression == "xz":
                # error: Argument 1 to "LZMAFile" has incompatible type "Union[str,
                # BaseBuffer]"; expected "Optional[Union[Union[str, bytes, PathLike[str],
                # PathLike[bytes]], IO[bytes]], None]"
                import lzma
    
                handle = lzma.LZMAFile(
                    handle,  # type: ignore[arg-type]
                    ioargs.mode,
                    **compression_args,
                )
    
            # Zstd Compression
            elif compression == "zstd":
                zstd = import_optional_dependency("zstandard")
                if "r" in ioargs.mode:
                    open_args = {"dctx": zstd.ZstdDecompressor(**compression_args)}
                else:
                    open_args = {"cctx": zstd.ZstdCompressor(**compression_args)}
                handle = zstd.open(
                    handle,
                    mode=ioargs.mode,
                    **open_args,
                )
    
            # Unrecognized Compression
            else:
                msg = f"Unrecognized compression type: {compression}"
                raise ValueError(msg)
    
            assert not isinstance(handle, str)
            handles.append(handle)
    
        elif isinstance(handle, str):
            # Check whether the filename is to be opened in binary mode.
            # Binary mode does not support 'encoding' and 'newline'.
            if ioargs.encoding and "b" not in ioargs.mode:
                # Encoding
>               handle = open(
                    handle,
                    ioargs.mode,
                    encoding=ioargs.encoding,
                    errors=errors,
                    newline="",
                )
E               FileNotFoundError: [Errno 2] No such file or directory: '/tmp/pytest-of-runner/pytest-1/test_reproducibility0/run1/metrics_summary.csv'

../../../.local/lib/python3.12/site-packages/pandas/io/common.py:926: FileNotFoundError
=========================== short test summary info ============================
FAILED tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_script_runs_successfully - AssertionError: Script failed with: Traceback (most recent call last):
    File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>
      import seaborn as sns
  ModuleNotFoundError: No module named 'seaborn'
  
assert 1 == 0
 +  where 1 = CompletedProcess(args=['/usr/bin/python3', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py', '--predictions', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv', '--out-dir', '/tmp/pytest-of-runner/pytest-1/test_script_runs_successfully0/metrics_output', '--bootstrap', '200', '--seed', '42'], returncode=1, stdout='', stderr='Traceback (most recent call last):\n  File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>\n    import seaborn as sns\nModuleNotFoundError: No module named \'seaborn\'\n').returncode
FAILED tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_output_files_created - AssertionError: metrics_summary.csv not created
assert False
 +  where False = exists()
 +    where exists = (PosixPath('/tmp/pytest-of-runner/pytest-1/test_output_files_created0/metrics_output') / 'metrics_summary.csv').exists
FAILED tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_metrics_summary_content - FileNotFoundError: [Errno 2] No such file or directory: '/tmp/pytest-of-runner/pytest-1/test_metrics_summary_content0/metrics_output/metrics_summary.csv'
FAILED tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_per_run_metrics - AssertionError: Per-run metrics not created
assert False
 +  where False = exists()
 +    where exists = (PosixPath('/tmp/pytest-of-runner/pytest-1/test_per_run_metrics0/metrics_output') / 'metrics_summary_per_run.csv').exists
FAILED tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_positional_argument - AssertionError: Script failed with: Traceback (most recent call last):
    File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>
      import seaborn as sns
  ModuleNotFoundError: No module named 'seaborn'
  
assert 1 == 0
 +  where 1 = CompletedProcess(args=['/usr/bin/python3', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv', '--out-dir', '/tmp/pytest-of-runner/pytest-1/test_positional_argument0/metrics_output', '--bootstrap', '100', '--seed', '42'], returncode=1, stdout='', stderr='Traceback (most recent call last):\n  File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>\n    import seaborn as sns\nModuleNotFoundError: No module named \'seaborn\'\n').returncode
FAILED tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_dry_run - AssertionError: Dry run failed: Traceback (most recent call last):
    File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>
      import seaborn as sns
  ModuleNotFoundError: No module named 'seaborn'
  
assert 1 == 0
 +  where 1 = CompletedProcess(args=['/usr/bin/python3', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py', '--predictions', '/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tests/sample_predictions.csv', '--dry-run'], returncode=1, stdout='', stderr='Traceback (most recent call last):\n  File "/home/runner/work/ensemble-anomaly-maps/ensemble-anomaly-maps/tools/compute_presentation_metrics.py", line 47, in <module>\n    import seaborn as sns\nModuleNotFoundError: No module named \'seaborn\'\n').returncode
FAILED tests/test_compute_presentation_metrics.py::TestComputePresentationMetrics::test_reproducibility - FileNotFoundError: [Errno 2] No such file or directory: '/tmp/pytest-of-runner/pytest-1/test_reproducibility0/run1/metrics_summary.csv'
========================= 7 failed, 2 passed in 11.19s =========================
```

### ✅ PASS `test_dataset_validation.py` (0.1s)

```
======================================================================
DATASET VALIDATION TESTS
======================================================================

[TEST] Trajectory completeness
  Frames: 1500
  Timestep: 0.002 ps
  Total time: 3.00 ps
  ✓ Trajectory is complete and continuous

[TEST] Topology consistency
  Atoms: 1500
  Residues: 100
  Atoms/residue: 15.0
  ✓ Topology is consistent

[TEST] Coordinate validity
  Atoms: 1000
  Range: [-1.65, 12.78] nm
  Box size: [12.60842006 12.22012254 13.96185378] nm
  ✓ Coordinates are valid

[TEST] Trajectory RMSD sanity check
  Mean RMSD: 1.99 ± 0.49 Å
  Range: [0.55, 3.39] Å
  ✓ RMSD values are reasonable

[TEST] Feature quality for ML
  Features: 50
  Frames: 1000
  Mean variance: 1.007
  ✓ Features suitable for ML

[TEST] Dataset citation metadata
  DOI: 10.17617/3.8O
  Source: Edmond (Max Planck Digital Library)
  ✓ Citation metadata present

[TEST] Reproducibility metadata
  Random seed: 42
  tICA lag: 10
  MSM lag: 30
  ✓ Reproducibility metadata complete

======================================================================
RESULTS: 7 passed, 0 failed
======================================================================
```

### ✅ PASS `test_integration.py` (1.2s)

```
======================================================================
INTEGRATION TEST: Metrics Computation Pipeline
======================================================================

Setup:
  Frames: 500
  Residues: 50
  States: 10

[1/5] Creating synthetic MSM...
  ✓ MSM with 10 states
  ✓ Stationary distribution: min=0.006, max=0.293

[2/5] Creating synthetic tICA coordinates...
  ✓ Coordinates shape: (500, 3)
  ✓ Mean: [ 0.03797094 -0.07307855  0.04473311]
  ✓ Std: [0.78835699 0.76841358 0.8641682 ]

[3/5] Computing dynamic anomaly signals...
  ✓ rarity: range=[0.000, 1.000], mean=0.500
  ✓ transition_surprise: range=[0.000, 1.000], mean=0.500
  ✓ local_density: range=[0.000, 1.000], mean=0.500

[4/5] Testing normalization strategies...
  ✓ Rank: range=[0.000, 1.000]
  ✓ Percentile: range=[0.000, 1.000]
  ✓ Global 2D: shape=(10, 20)
  ✓ Per-frame 2D: shape=(10, 20)

[5/5] Creating synthetic residue metrics...
  ✓ RMSF: 50 residues
  ✓ tICA importance: 50 residues

[6/6] Creating unified output...
  ✓ Unified output with 3 metrics

[7/7] Validating output format...
  ✓ Output format valid
  ✓ All scores in [0, 1] range

======================================================================
SUMMARY
======================================================================

Dynamic Anomaly (per frame):
  Range: [0.036, 0.968]
  Mean: 0.493 ± 0.216
  High anomaly frames (>90th percentile): 50

Per-Residue Scores:
  dynamic_anomaly     : range=[0.000, 1.000], mean=0.278
  rmsf                : range=[0.000, 1.000], mean=0.278
  tica_importance     : range=[0.000, 1.000], mean=0.280

Top 5 Dynamic Hotspots:
  Res 16 : dynamic=1.000, rmsf=1.000, tica_importance=0.294
  Res 23 : dynamic=1.000, rmsf=1.000, tica_importance=0.341
  Res 39 : dynamic=1.000, rmsf=1.000, tica_importance=0.069
  Res 42 : dynamic=0.802, rmsf=0.802, tica_importance=0.006
  Res 40 : dynamic=0.657, rmsf=0.657, tica_importance=0.204

======================================================================
✓ INTEGRATION TEST PASSED
======================================================================
```

### ✅ PASS `test_optimizations.py` (1.2s)

```
======================================================================
TESTING ML PIPELINE OPTIMIZATIONS
======================================================================

[TEST] minmax_normalize...
  ✓ minmax_normalize works correctly
[TEST] map_to_active_set...
  ✓ map_to_active_set works correctly
[TEST] compute_transition_surprise...
  ✓ compute_transition_surprise works correctly
[TEST] compute_local_density...
  ✓ compute_local_density works correctly
[TEST] compute_frame_scores...
  ✓ compute_frame_scores works correctly
[TEST] ProgressBar...

Testing: [====------------------------------------] 10.0% (1/10)
Testing: [========--------------------------------] 20.0% (2/10)
Testing: [============----------------------------] 30.0% (3/10)
Testing: [================------------------------] 40.0% (4/10)
Testing: [====================--------------------] 50.0% (5/10)
Testing: [========================----------------] 60.0% (6/10)
Testing: [============================------------] 70.0% (7/10)
Testing: [================================--------] 80.0% (8/10)
Testing: [====================================----] 90.0% (9/10)
Testing: [========================================] 100.0% (10/10)

  ✓ ProgressBar works correctly

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

### ✅ PASS `test_phase1.py` (1.3s)

```
======================================================================
TESTING PHASE 1: MODEL SELECTION & BOOTSTRAP
======================================================================

[TEST] VAMP-2 score computation...
  ✓ VAMP-2 score: 1.5937
[TEST] VAMP-2 reproducibility...
  ✓ Reproducible: 0.145853 == 0.145853
[TEST] Bootstrap resampling shape...
  ✓ Shapes preserved
[TEST] Bootstrap reproducibility...
  ✓ Bootstrap is reproducible
[TEST] MSM pipeline fitting...
  ✓ MSM fitted: 10 states
[TEST] Seed sequence generation...
  ✓ Seed sequence generation works
[TEST] Global seed setting...
  ✓ Global seed setting works
[TEST] Edge cases...
  ✓ Edge cases handled

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

### ✅ PASS `test_phase2.py` (1.2s)

```
======================================================================
TESTING PHASE 2: FEATURE EXTENSIONS
======================================================================

[TEST] Contact energy - attractive...
  ✓ Attractive energy: -0.500 kcal/mol
[TEST] Contact energy - repulsive...
  ✓ Repulsive energy: 1190.400 kcal/mol
[TEST] Contact energy - far distance...
  ✓ Far energy: 0.000000 kcal/mol
[TEST] Energy symmetry...
  ✓ Symmetric: -0.500000 == -0.500000
[TEST] Energy distance dependence...
  ✓ Energy distance dependence verified
[TEST] Energy value ranges...
  ✓ All energy values in reasonable range
[TEST] Residue classification...
  ✓ Residue classifications correct
[TEST] Grid creation...
  ✓ Grid shape: (6, 6, 6), points: 216
[TEST] Pocket detection basics...
  ✓ Grid: (10, 10, 4), protein extent: 2.00
[TEST] Grid spacing...
  ✓ Grid spacing: 0.4 nm

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

### ✅ PASS `test_phase3.py` (1.2s)

```
======================================================================
TESTING PHASE 3: ENHANCED SCORING & SOFT STATES
======================================================================

[TEST] Rank normalization...
  ✓ Input: [1 5 3 9 2]
  ✓ Normalized: [0.   0.75 0.5  1.   0.25]
[TEST] Rank normalization - constant...
  ✓ Constant handled: [0. 0. 0. 0.]
[TEST] Quantile normalization...
  ✓ Input: [  1   2   3   4   5 100]
  ✓ Normalized: [0.         0.00980392 0.02941176 0.04901961 0.06862745 1.        ]
[TEST] Normalization preserves order...
  ✓ Ordering preserved
[TEST] Z-score computation...
  ✓ Z-scores: mean=0.000000, std=1.000000
[TEST] Z-score - constant...
  ✓ Constant handled
[TEST] Moving median...
  ✓ Input: [ 1 10  2 11  3 12  4]
  ✓ Smoothed: [ 1  2 10  3 11  4  4]
[TEST] Signal fusion - median...
  ✓ Fused scores: [0.  0.5 1. ]
[TEST] Signal fusion - mean...
  ✓ Mean fusion: [0.25 0.5  0.75]
[TEST] Fusion reproducibility...
  ✓ Reproducible
[TEST] State entropy...
  ✓ Deterministic entropy: 0.000000
  ✓ Uniform entropy: 1.098513
[TEST] Monotone behavior...
  ✓ Monotone under perturbation
[TEST] Edge cases...
  ✓ Edge cases handled

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

### ✅ PASS `test_pipeline_edge_cases.py` (8.8s)

```
======================================================================
TESTING PIPELINE EDGE CASES
======================================================================

[TEST] Very short trajectory (10 frames)...
  ✓ Computed signals for 10 frames successfully
[TEST] Short trajectory (50 frames)...
  ✓ Computed signals for 50 frames successfully
[TEST] VAMP-2 with short trajectory...
  ✓ VAMP-2 returns -inf for short trajectory (expected -inf or finite)
[TEST] Constant features...
  ✓ Constant features handled correctly
[TEST] Near-zero variance features...
  ✓ Low variance features handled correctly
[TEST] Single unique value...
  ✓ Single unique value handled correctly
[TEST] Disconnected MSM states...
  ✓ Disconnected states handled correctly
[TEST] Single-state trajectory...
  ✓ Single-state trajectory handled correctly
[TEST] Extreme outlier coordinates...
  ✓ Outlier detected with density score 1.000
[TEST] Extreme probability values...
  ✓ Extreme probabilities handled correctly
[TEST] Empty array normalization...
  ✓ Empty arrays handled correctly
[TEST] Single element array...
  ✓ Single element arrays handled correctly
[TEST] Two element array...
  ✓ Two element arrays handled correctly
[TEST] Bootstrap preserves length...
  ✓ Bootstrap preserves trajectory length
[TEST] Bootstrap short trajectory...
  ✓ Bootstrap handles short trajectories
[TEST] Log(0) protection...
  ✓ Log(0) protected with epsilon
[TEST] Division by zero protection...
  ✓ Division by zero protected
[TEST] Negative coordinates...
  ✓ Negative coordinates handled correctly
[TEST] Invalid state indices...
  ✓ Invalid state indices handled correctly
[TEST] Adaptive parameter scaling...
  ✓ Adaptive parameters scale correctly
[TEST] Large trajectory optimization...
  [Auto-optimize] n_frames=100,000, k=50, lag=50
  ✓ Processed 100,000 frames in 7.38s
  ✓ All signals computed correctly

======================================================================
RESULTS: 21 passed, 0 failed
======================================================================
```

### ✅ PASS `test_reproducibility.py` (0.2s)

```
======================================================================
REPRODUCIBILITY & ROBUSTNESS TESTS
======================================================================

[TEST] Random seed reproducibility
  Seed 42 (run 1): mean=0.019332
  Seed 42 (run 2): mean=0.019332
  Seed 123: mean=-0.039564
  ✓ Reproducibility verified

[TEST] Noise injection robustness
  Noise level: 0.1
  Relative error: 0.05%
  Correlation: 0.9902
  ✓ Robust to noise

[TEST] Parameter sensitivity analysis
  Baseline output: 0.9750
  Lag sensitivity: 0.0256
  Dim sensitivity: 0.0513
  Cluster sensitivity: 0.0103
  ✓ Sensitivity analysis complete

[TEST] Cross-validation stability
  Runs: 5
  Mean score: 0.8096 ± 0.0334
  CV coefficient: 4.13%
  ✓ Cross-validation stable

[TEST] Data subset consistency
  Full data size: 1000
  Subset size: 500
  Subsets tested: 5
  Mean ± std: 0.0363 ± 0.0116
  ✓ Subsets consistent

[TEST] Computation determinism
  Input shape: (100, 10)
  Output shape: (10,)
  Max difference: 0.00e+00
  ✓ Deterministic computation verified

[TEST] Missing data handling
  Original rows: 100
  Rows with NaN: 5
  Clean rows: 95
  ✓ Missing data handled correctly

[TEST] Extreme parameter values
  Warning: lag 1000 >= half of frames 500
  ✓ Correctly rejected dim=0
  ✓ Correctly rejected negative clusters
  ✓ Extreme values handled appropriately

======================================================================
RESULTS: 8 passed, 0 failed
======================================================================
```

### ✅ PASS `test_scientific_validation.py` (1.6s)

```
======================================================================
TESTING SCIENTIFIC VALIDATION TOOLS
======================================================================

[TEST] Chapman-Kolmogorov test - basic functionality
  Lags tested: [20 40 60]
  Mean absolute error: 0.0600
  ✓ Test completed

[TEST] Implied timescales convergence
  Lags tested: [ 5 10 15 20 30]
  Timescales shape: (5, 3)
  ✓ Test completed

[TEST] VAMP-2 cross-validation
  Mean VAMP-2 score: 1.8471 ± 0.0321
  ✓ Test completed

[TEST] VAMP-2 CV reproducibility
  Score 1: 1.830699
  Score 2: 1.830699
  ✓ Reproducible

[TEST] Signal correlation analysis
  Correlation matrix:
          signal1   signal2   signal3
signal1  1.000000  0.950185  0.019612
signal2  0.950185  1.000000  0.011177
signal3  0.019612  0.011177  1.000000
  ✓ Test completed

[TEST] Stationary distribution validation - passing case
  Max relative error: 0.1675
  Mean relative error: 0.0758
  Validation: ✓ PASSED

[TEST] Stationary distribution validation - failing case
  Max relative error: 2.6364
  Validation: ✗ FAILED

[TEST] Validation report generation

✓ Validation report saved to /tmp/tmp0qo1o01u/validation_report.json
  Overall status: PASSED
  Tests performed: chapman_kolmogorov, stationary_distribution
  Report saved to: /tmp/tmp0qo1o01u/validation_report.json
  Overall status: PASSED
  ✓ Test completed

[TEST] Edge case - short trajectory
  Handled short trajectory: 2 lags computed
  ✓ No crash

[TEST] Edge case - disconnected states
  Handled disconnected states
  ✓ No crash

======================================================================
RESULTS: 10 passed, 0 failed
======================================================================
```

### ✅ PASS `test_signals.py` (1.2s)

```
======================================================================
TESTING SIGNALS MODULE
======================================================================

[TEST] Rank normalization...
  ✓ Input: [1 5 3 9 2]
  ✓ Normalized: [0.   0.75 0.5  1.   0.25]
[TEST] Percentile normalization...
  ✓ Input: [  1   2   3   4   5 100]
  ✓ Normalized: [0.         0.00980392 0.02941176 0.04901961 0.06862745 1.        ]
[TEST] Normalization consistency...
  ✓ Original order preserved
[TEST] Normalization ranges...
  ✓ All methods respect [0,1] range across distributions
[TEST] Global normalization...
  ✓ Rank range: [0.000, 1.000]
  ✓ Percentile range: [0.000, 1.000]
[TEST] Per-frame normalization...
  ✓ Normalized 10 frames independently
[TEST] Dynamic anomaly signals...
  ✓ Computed 3 signals
  ✓ Rarity range: [0.000, 1.000]
  ✓ Surprise range: [0.000, 1.000]
  ✓ Density range: [0.000, 1.000]
[TEST] Signal properties...
  ✓ Rare state (2) rarity: 0.900
  ✓ Common state (0) rarity: 0.550
[TEST] Frame-to-residue aggregation...
  ✓ Aggregated 50 frames to 20 residues
  ✓ Range: [0.468, 0.548]
[TEST] Edge cases...
  ✓ Edge cases handled correctly

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

### ✅ PASS `test_statistical_validation.py` (0.8s)

```
======================================================================
STATISTICAL VALIDATION TESTS
======================================================================

[TEST] Statistical power of anomaly detection
  Normal frames: 900
  Anomaly frames: 100
  t-statistic: 9.310
  p-value: 7.8915e-20
  Cohen's d: 0.98 (Large effect)
  ✓ Sufficient statistical power

[TEST] Multiple testing correction (FDR)
  Tests performed: 100
  FDR level: 0.05
  Discoveries: 17
  ✓ Multiple testing properly controlled

[TEST] Bootstrap confidence intervals
  Sample size: 200
  Bootstrap iterations: 1000
  Observed mean: 1.891
  95% CI: [1.644, 2.159]
  CI width: 0.515
  ✓ Bootstrap CI computed successfully

[TEST] Normality testing (Shapiro-Wilk)
  Normal data: W=0.9885, p=0.5423 (PASS)
  Non-normal data: W=0.8444, p=0.0000 (FAIL)
  ✓ Normality testing works correctly

[TEST] Correlation significance testing
  Correlated: r=0.924, p=1.3181e-42
  Uncorrelated: r=0.007, p=0.9484
  ✓ Correlation testing works

[TEST] Distribution comparison (KS test)
  Same distribution: KS=0.065, p=0.7934
  Different distributions: KS=0.695, p=1.6336e-46
  ✓ KS test works correctly

[TEST] Variance homogeneity (Levene's test)
  Equal variance: F=1.475, p=0.2260
  Unequal variance: F=76.249, p=1.0448e-15
  ✓ Levene's test works correctly

[TEST] Outlier detection (MAD method)
  Sample size: 103
  MAD: 0.653
  Outliers detected: 3
  ✓ Outlier detection working

======================================================================
RESULTS: 8 passed, 0 failed
======================================================================
```

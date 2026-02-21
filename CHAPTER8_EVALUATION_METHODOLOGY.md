# Chapter 8: Evaluation Methodology

## Thesis: ML-Based Detection and Visualization of Dynamic Hotspots in Molecular Dynamics Simulations

> **Research Questions addressed:**
> - **RQ1**: Can a multi-signal ML pipeline detect residue-level dynamic hotspots that align with plausible structural transitions?
> - **RQ2**: Do dual visualization modalities (Trame/VTK and NGL/three.js) complement each other in validating hotspot signals?
> - **RQ3**: How sensitive are detected hotspots to MSM/tICA hyperparameters and signal fusion choices?

---

## 1. Dataset and Trajectory Information

### Trajectory Files

| Property | Value | Source |
|---|---|---|
| **Topology file** | `data/raw_trajectory/align_topol.pdb` | `configs/pipeline.yaml` (paths.topology) |
| **Trajectory files** | `data/raw_trajectory/trajectory_0.xtc` through `trajectory_19.xtc` (20 files) | `configs/pipeline.yaml` (paths.trajectory: `trajectory_*.xtc`) |
| **Format** | XTC (GROMACS compressed trajectory) + PDB topology | Directory listing |
| **Number of trajectory files** | 20 independent replicas | `ls data/raw_trajectory/` |

### Frame and Atom Information

- **Approximate frames per trajectory**: Not embedded in source code; derived at runtime via MDTraj (`md.load(trajectory_path, top=topology_path, stride=stride)`). The `stride` parameter defaults to 1 (every frame loaded). The `features.npy` artifact in `data/` encodes the total frame count as its first dimension.
- **Number of atoms/residues**: Loaded from the PDB topology at runtime via `traj.n_atoms` and `traj.n_residues` (MDTraj). Not hard-coded.
- **Simulation length**: Not specified in source code; depends on the MD production run associated with each XTC file.

### Preprocessing Steps

Implemented in `features/compute_md_features.py`:

1. **Loading**: `md.load(trajectory_path, top=topology_path, stride=stride)` — every `stride`-th frame is loaded (default `stride=1`, i.e., all frames).
2. **RMSD reference frame**: Set by `reference_frame` argument (default: frame 0, the first frame).
3. **Alignment for RMSF**: `traj.superpose(traj, frame=0, atom_indices=align_atoms)` — global rotation/translation removed using Cα atoms (`scoring/signals.py`, `compute_rmsf_scores()`).
4. **No explicit frame trimming or filtering**: All loaded frames are used unless the `stride` argument is set.
5. **Native contact cutoff**: Cα–Cα pairs within **0.8 nm (8 Å)** in the reference frame (`features/compute_md_features.py`, line: `(distances < 0.8).sum(axis=1)`).
6. **Feature caching**: Optional disk caching in `.cache/` via `tools/utils.py:load_features_cached()`.

### Configuration Reference

- `configs/pipeline.yaml`:
  - `paths.topology: data/raw_trajectory/align_topol.pdb`
  - `paths.trajectory: data/raw_trajectory/trajectory_*.xtc`
  - `features.rmsd_reference: 'first'`
  - `features.contact_cutoff: 0.8` (nm)

---

## 2. Feature Engineering Details

### Computed Features

All features are implemented in `features/compute_md_features.py` → `compute_features()`:

| Feature | Description | Dimensionality | Formula/Method |
|---|---|---|---|
| `rmsd` | RMSD from reference frame | 1 (scalar per frame) | `md.rmsd(traj, ref)` using MDTraj |
| `rg` | Radius of gyration | 1 (scalar per frame) | `md.compute_rg(traj)` using MDTraj |
| `contacts` | Count of native Cα–Cα contacts | 1 (scalar per frame) | Pairs within 0.8 nm; `(distances < 0.8).sum(axis=1)` |
| `phi_sin` | Mean sine of backbone φ angles | 1 (mean over all residues) | `np.sin(phi).mean(axis=1)` |
| `phi_cos` | Mean cosine of backbone φ angles | 1 (mean over all residues) | `np.cos(phi).mean(axis=1)` |
| `psi_sin` | Mean sine of backbone ψ angles | 1 (mean over all residues) | `np.sin(psi).mean(axis=1)` |
| `psi_cos` | Mean cosine of backbone ψ angles | 1 (mean over all residues) | `np.cos(psi).mean(axis=1)` |

**Total feature dimensionality**: **d = 7** scalar features per frame (from the base implementation). Extended pipelines in `tools/generate_energy.py` (residue energies, stored as `residue_energy.parquet`) and `tools/generate_pockets.py` (pocket volumes, stored as `pockets.parquet`) add optional signals.

**Sin/cos dihedral encoding rationale**: Resolves angular periodicity — angles of −179° and +179° are numerically proximate (differ by 2°), not 358°.

**Dihedral fallback**: If topology lacks standard backbone atoms (termini, non-standard residues), `except (ValueError, RuntimeError, KeyError)` returns zero-valued placeholders (not excluded).

### Feature Normalization

No explicit pre-normalization is applied to the raw feature matrix before tICA. tICA itself is a linear projection that implicitly handles scale differences by finding variance-preserving directions. Downstream anomaly scoring applies normalization per signal (see Section 5).

### Feature Storage

| Artifact | Location | Format |
|---|---|---|
| Raw feature matrix | `data/features.npy` | NumPy array `(n_frames, 7)` |
| Energy features (optional) | `data/derived/residue_energy.parquet` | Parquet (columns: `frame`, `res_id`, `energy`) |
| Pocket features (optional) | `data/derived/pockets.parquet` | Parquet (columns: `frame`, pocket metrics) |
| Feature loading utility | `tools/utils.py:load_features_cached()` | Disk-cached via `.cache/` |

### Where Defined

- `features/compute_md_features.py`: `compute_features()`, `features_to_matrix()`
- `tools/extract_features.py`: CLI wrapper
- `configs/pipeline.yaml`: `features.rmsd_reference`, `features.contact_cutoff`, `features.cache_features`

---

## 3. tICA Configuration

### Lag Times and Dimensionalities Tested

Implemented in `msm/select_lag_and_dim.py` → `select_lag_and_dim()`:

| Parameter | Candidates Tested | Source |
|---|---|---|
| **Lag times (τ)** | `[5, 10, 15, 20, 30, 50]` frames | `configs/pipeline.yaml` → `tica.lag_candidates` |
| **Dimensions (d)** | `[2, 3, 4, 5, 6, 8, 10]` | `configs/pipeline.yaml` → `tica.dim_candidates` |
| **Default lag** | 10 frames | `configs/pipeline.yaml` → `tica.default_lag` |
| **Default dim** | 5 | `configs/pipeline.yaml` → `tica.default_dim` |

**Grid size**: 6 lags × 7 dims = **42 combinations** evaluated.

### VAMP-2 Model Selection Logic

Implemented in `msm/select_lag_and_dim.py:compute_vamp2_score()`:

1. Split data into train (80%) and validation (20%) sets (`validation_fraction=0.2`).
2. Fit `deeptime.decomposition.VAMP(lagtime=lag, dim=dim)` on training set.
3. Transform validation set through the fitted model.
4. Compute time-lagged covariance matrices: **C₀**, **C₁**, **C₀₁**.
5. Regularize diagonals: `C += 1e-6 * I` for numerical stability.
6. Compute VAMP-2 score: `Σᵢ σᵢ²` (sum of squared singular values of `C₀⁻¹/² · C₀₁ · C₁⁻¹/²`).
7. Select `(lag*, dim*)` with maximum VAMP-2 score across all 42 combinations.

**5-fold cross-validation**: Also implemented in `msm/validation.py:vamp2_cross_validation()` (default `n_folds=5`, seed=42). Returns mean and std of VAMP-2 scores.

**Random seed for VAMP selection**: `seeds.vamp: 456` (`configs/pipeline.yaml`).

### Stored Outputs

| Artifact | Location | Schema |
|---|---|---|
| Best parameters | `outputs/phase1/reports/vamp2_best.json` | `{lag, dim, vamp2_score, n_candidates, features_shape, validation_fraction, seed}` |
| Grid search results | `outputs/phase1/reports/vamp2_grid.csv` | Columns: `lag, dim, vamp2_score` (sorted descending) |
| TICA coordinates | `outputs/msm/tica_coords.npy` | NumPy array `(n_frames, dim*)` |

### Implied Timescale Plots

Generated in `tools/run_msm_tica.py` (step 5) and `msm/validation.py:implied_timescales_convergence()`:

- **Method**: For each lag in auto-determined range (`np.logspace(log10(5), log10(min(T//10, 100)), num=10)`), fit MSM and extract top eigenvalues.
- **Timescale formula**: `τᵢ = −τ_lag / ln(λᵢ)` where `λᵢ` is the i-th eigenvalue.
- **Plot saved**: `outputs/msm/its.png` (from `run_msm_tica.py`) and `outputs/validation/implied_timescales.png` (from `validation.py`).
- **Convergence criterion**: Coefficient of variation `(std/mean) < 0.2` for the slowest timescale.

### Where Defined

- `msm/select_lag_and_dim.py`: VAMP-2 grid search and model selection
- `msm/validation.py`: `implied_timescales_convergence()`, `vamp2_cross_validation()`
- `tools/run_msm_tica.py`: Main pipeline with ITS plotting
- `configs/pipeline.yaml`: `tica.*`, `model_selection.*`

---

## 4. MSM Configuration and Validation

### Clustering and MSM Parameters

| Parameter | Value | Source |
|---|---|---|
| **Clustering algorithm** | K-means (`sklearn.cluster.KMeans`) | `tools/run_msm_tica.py` |
| **Number of clusters** | `n_clusters = 30` | `configs/pipeline.yaml` → `msm.n_clusters` |
| **Clustering space** | tICA coordinates `(n_frames, dim*)` | `tools/run_msm_tica.py` |
| **K-means seed** | `random_state=42` (`seeds.kmeans`) | `configs/pipeline.yaml` |
| **MSM lag time** | `lag_msm = 30` frames | `configs/pipeline.yaml` → `msm.lag` |
| **Reversibility** | `reversible=True` (detailed balance enforced) | `configs/pipeline.yaml` → `msm.reversible` |
| **Connectivity** | Largest connected set (`largest_connected_set`) | `configs/pipeline.yaml` → `msm.connectivity` |
| **Estimator** | `deeptime.markov.msm.MaximumLikelihoodMSM` | `tools/run_msm_tica.py` |

### Chapman–Kolmogorov Test

Implemented in `msm/validation.py:chapman_kolmogorov_test()`:

- **Method**: Compare predicted `P^k` (matrix power) vs. empirically estimated `P̂(k·τ)` at `n_lags=5` multiples of the MSM lag time.
- **Formula**: Chapman-Kolmogorov equation `P(k·τ) = P(τ)^k`.
- **Reference**: Prinz et al. (2011), *J. Chem. Phys.* 134: 174105.
- **Plot saved**: `outputs/validation/chapman_kolmogorov.png` (2×2 grid for top 4 states).
- **Test states**: First 4 most-populated states (transition probability > 0.05).

### Bootstrap Confidence Intervals

Implemented in `msm/bootstrap_msm.py:bootstrap_msm()`:

| Parameter | Value | Source |
|---|---|---|
| **Bootstrap iterations** | 100 | `configs/pipeline.yaml` → `bootstrap.n_iterations` |
| **Resampling method** | `'frames'` (frame-level with replacement) | `configs/pipeline.yaml` → `bootstrap.method` |
| **Block size** (if block bootstrap) | 10 frames | `configs/pipeline.yaml` → `bootstrap.block_size` |
| **Bootstrap seed** | 123 | `configs/pipeline.yaml` → `seeds.bootstrap` |
| **Confidence level** | 95% | `configs/pipeline.yaml` → `bootstrap.confidence_level` |
| **CI method** | Percentile bootstrap (2.5th and 97.5th percentiles) | `msm/bootstrap_msm.py` |

**Bootstrap pipeline for each sample**: TICA → K-means → MaximumLikelihoodMSM (same parameters as reference MSM).

**Outputs**:

| Artifact | Location | Content |
|---|---|---|
| `pi_ci.parquet` | `outputs/models/msm_bootstrap/` | Columns: `state, mean, std, lower, upper, reference` |
| `P_ci.npz` | `outputs/models/msm_bootstrap/` | Arrays: `mean, lower, upper, reference, n_states` |
| `mfpt_ci.parquet` | `outputs/models/msm_bootstrap/` | Columns: `from_state, to_state, mean, lower, upper` |
| `bootstrap_metadata.json` | `outputs/models/msm_bootstrap/` | Run parameters and success count |

### Stationary Distribution Validation

Implemented in `msm/validation.py:validate_stationary_distribution()`:

- **Method**: Compare MSM stationary distribution π against empirical frame counts.
- **Tolerance**: Maximum relative error < 0.1 (10%) for states with ≥ 10 frame counts.
- **Metric**: `|π[s] − f̂[s]| / (f̂[s] + 1e-10)` per state.

### Where Stationary Distribution is Saved

| Artifact | Location |
|---|---|
| `pi.npy` | `outputs/msm/` (NumPy array, shape `(n_active_states,)`) |
| `P.npy` | `outputs/msm/` (Transition matrix, shape `(n_active_states, n_active_states)`) |
| `dtraj.npy` | `outputs/msm/` (Discrete trajectory, shape `(n_frames,)`) |

### Additional Validation Metrics

- **Signal correlation analysis** (`msm/validation.py:signal_correlation_analysis()`): Spearman correlation matrix across signal channels. High correlation (> 0.7) flags redundancy.
- **Validation report** (`msm/validation.py:generate_validation_report()`): JSON report at `outputs/validation/validation_report.json` with overall status `'PASSED'`/`'NEEDS_REVIEW'`.
- **Input validation** (`msm/input_validation.py:check_trajectory_quality()`): Pre-flight checks for frame count, atomic clashes, and unfolding events.

---

## 5. Signal Construction

### Signal Formulas

#### 5.1 State Rarity

**Location**: `scoring/anomaly_v2.py:compute_kinetic_signals()` (lines 154–189); also `tools/run_msm_tica.py` (step 6).

```
rarity[t] = 1.0 - π[s_t]
```

- `π` = MSM stationary distribution (loaded from `pi.npy`)
- `s_t` = discrete state at frame `t` (from `dtraj.npy`)
- Frames in the MSM inactive set receive `rarity = 1.0` (maximum rarity).

#### 5.2 Transition Surprise

**Location**: `scoring/anomaly_v2.py:compute_kinetic_signals()` and `tools/utils.py:compute_transition_surprise()`.

```
surprise[t] = -log(max(P[s_t, s_{t+lag}], ε))
```

- `P` = MSM transition matrix (loaded from `P.npy`)
- `lag` = MSM lag time (default 30 frames)
- `ε = 1e-12` (numerical floor for forbidden transitions)
- In the optimized version (`tools/utils.py`), the cost is **smeared** uniformly over the lag interval: `surprise[t:t+lag] += cost / lag`.

#### 5.3 Local Density

**Location**: `scoring/anomaly_v2.py:compute_local_density_signal()` and `tools/utils.py:compute_local_density()`.

```
density[t] = -mean_distance(t, k-NN in tICA space)
```

- `k = 20` neighbors (default; `configs/pipeline.yaml` → `scoring` or CLI `--k_neighbors`)
- Uses `sklearn.neighbors.NearestNeighbors(n_neighbors=k, n_jobs=-1)` in tICA coordinate space.
- **Sign convention**: Returned as negative mean distance so that lower density (more isolated) → higher anomaly score.
- **Final fusion uses**: `-density` (invert again), so isolated frames (large k-NN distance) get high anomaly.

#### 5.4 RMSF

**Location**: `scoring/signals.py:compute_rmsf_scores()`.

```
RMSF[i] = sqrt(mean_t((r_i(t) - <r_i>)²))  [Angstroms]
```

- `i` = atom index (Cα atoms selected via `"name CA"`)
- `r_i(t)` = 3D position of atom `i` at frame `t` (MDTraj, units: nm → converted to Å × 10)
- Trajectory aligned before RMSF calculation: `traj.superpose(traj, frame=0, atom_indices=align_atoms)`
- Per-residue RMSF: average over all Cα atoms in each residue.

#### 5.5 tICA Importance

**Location**: `scoring/signals.py:compute_tica_importance_scores()`.

```
importance[i] = L2_norm(loadings[i, :n_top_components])
```

- `loadings` = tICA eigenvectors `(n_features, n_components)`
- `n_top = min(5, n_components)` slowest components used
- Per-residue aggregation: sum of feature loading magnitudes for all features belonging to each residue.
- Features attributed to residues via name parsing (e.g., `phi_10` → residue 10) or sequential ordering (2 features/residue assumed: φ, ψ).
- Final normalization: divided by `max_contrib` → `[0, 1]`.

### Normalization Methods

Implemented in `scoring/signals.py:normalize_scores()`, `scoring/anomaly_v2.py:rank_normalize()`, and `tools/utils.py:minmax_normalize()`:

| Method | Formula | Location | Default? |
|---|---|---|---|
| **Rank** | `rank(x) / (n-1)` → uniform [0,1] | `scoring/signals.py:_rank_normalize()` | Yes (Phase 3 scoring) |
| **Percentile** | Clip to [p_low, p_high] quantiles then min-max | `scoring/signals.py:_percentile_normalize()` | Yes (unified metrics) |
| **z-score** | `(x - mean) / std` | `scoring/signals.py:_compute_zscore()` | Optional |
| **Robust min-max** | `clip((x - q₀.₀₁) / (q₀.₉₉ − q₀.₀₁), 0, 1)` | `tools/utils.py:minmax_normalize(clip=True)` | Used in `run_msm_tica.py` |

**Default percentile range** (unified metrics, `tools/compute_all_metrics.py`):
- `--low-percentile 0.05`, `--high-percentile 0.95`
- Robust mode overrides: `low=0.10`, `high=0.90`

**Scope options** (unified metrics):
- Global normalization (default): normalize across all frames.
- Per-frame normalization: `--per-frame-norm` flag.

### Fusion Strategy

Implemented in `scoring/anomaly_v2.py:fuse_signals()`:

1. Each signal independently normalized (rank or quantile, default rank).
2. Signals stacked into matrix `(n_signals, n_frames)`.
3. Fusion via element-wise **median** (default) or **mean** across signals:
   - `median`: robust to single-signal failures.
   - `mean`: equal weighting, more sensitive to outliers.
4. Final scale: `× 100` (maps to [0, 100] range for `run_msm_tica.py`; back to [0,1] for unified metrics).

**Configuration**:
- `configs/pipeline.yaml` → `scoring.fusion_method: 'median'`
- CLI: `--fusion median|mean` in `tools/compute_all_metrics.py` and `tools/score_v2.py`

### Smoothing

Implemented in `scoring/anomaly_v2.py:moving_median()`:

```
score_smoothed[t] = median(score[t-W//2 : t+W//2+1])
```

- **Window size**: `W = 5` frames (default)
- **Method**: Rolling median (edge-preserving, suppresses single-frame spikes)
- **Configuration**: `configs/pipeline.yaml` → `scoring.window_size: 5`; CLI `--window`

### Per-Frame and Per-Residue Outputs

| Artifact | Location | Content |
|---|---|---|
| `frame_scores.csv` | `outputs/msm/` | Columns: `frame, score` (raw combined scores 0–100) |
| `frame_scores_dynamic.csv` | `outputs/metrics/` | Columns: `frame, score, component_rarity, component_transition_surprise, component_local_density` |
| `hotspots_unified.json` | `outputs/metrics/` | All 3 metric channels per residue (see Section 8) |
| `residue_scores_dynamic.json` | `outputs/metrics/` | Per-residue dynamic anomaly scores `{residue_id: score}` |
| `residue_scores_rmsf.json` | `outputs/metrics/` | Per-residue RMSF scores `{residue_id: score}` |
| `residue_scores_tica_importance.json` | `outputs/metrics/` | Per-residue tICA importance scores `{residue_id: score}` |
| `hotspots_residue.json` | `outputs/metrics/` | Legacy format: `{"scores": [{"label": "Res N", "score": float}]}` |
| `anomaly_timeseries.json` | `data/` | Frame-wise B-factor proxy: `[{"frame": int, "b_factor": float}]` (22 entries in the stored version) |

---

## 6. Sensitivity Analysis and Ablations

### What Is Implemented

#### Parameter Sensitivity Test (Simplified Mock)
**Location**: `tests/test_reproducibility.py:test_parameter_sensitivity_analysis()`

A simplified simulation tests relative sensitivity of three hyperparameters:
- `lag`: 50% increase from baseline (10 → 15)
- `dim`: 50% increase from baseline (5 → 7.5)
- `n_clusters`: 50% increase from baseline (50 → 75)

Sensitivity is measured as fractional change in model output: `|output_varied − output_base| / output_base`. This is a **mock sensitivity test** (not a real grid search on actual trajectory data).

#### Noise Injection Robustness Test
**Location**: `tests/test_reproducibility.py:test_noise_injection_robustness()`

- Adds Gaussian noise at SNR ~20 dB (noise_level=0.1) to a clean sinusoidal signal.
- Asserts: relative error < 50%, Pearson correlation > 0.9.

#### Cross-Validation Stability Test
**Location**: `tests/test_reproducibility.py:test_cross_validation_stability()`

- Runs 5 simulated CV folds across 5 independent random seeds.
- Asserts: coefficient of variation of mean fold scores < 10%.

#### Edge Case Robustness
**Location**: `tests/test_pipeline_edge_cases.py`

Tests pipeline resilience for:
- **Very short trajectories**: 10 frames and 50 frames.
- **Low/zero-variance features**.
- **Disconnected MSM states**.
- **Extreme outliers**.
- **Missing or invalid data** (NaN handling).

### What Is NOT Implemented

The following are **absent** from the codebase:

| Missing Component | Status |
|---|---|
| Formal hyperparameter grid search on real trajectory data (RQ3) | **Not implemented** |
| Lag time sweep comparing hotspot rankings across τ ∈ {5,10,…,50} | **Not implemented** |
| Cluster count sweep comparing hotspot rankings for n_clusters ∈ {10,…,100} | **Not implemented** |
| Signal ablation study (removing one signal at a time and comparing output) | **Not implemented** |
| Jaccard similarity comparison between hotspot sets under different hyperparameters | **Not implemented** |
| Spearman correlation of hotspot rankings between conditions | **Not implemented** |
| Comparison of `median` vs. `mean` fusion on same trajectory | **Not implemented** |

> **Implication for RQ3**: The pipeline provides the infrastructure for sensitivity analysis (VAMP-2 grid search generates `vamp2_grid.csv` with 42 parameter combinations and their VAMP-2 scores, which can serve as a proxy for model quality sensitivity). However, full downstream sensitivity analysis of hotspot residue rankings is not automated.

---

## 7. Visualization Validation Hooks

### 7.1 Dual Visualization Modalities

#### Modality 1: NGL/three.js (Primary Interactive Viewer)

**Files**: `templates/three.html`, `static/js/three.js`, `ngl_viewer.html`, `ngl_with_tooltips.html`, `ngl_slider.html`

**How anomaly scores map to residues**:
1. Per-residue anomaly scores are written into PDB B-factor field (columns 61–66) by `anomaly_frames_to_bfactor.py:write_model()`.
   - Formula: `bf = float(resid_to_b.get(resid, default_b))` with `"{bf:6.2f}"` formatting.
   - Uniform fallback: if no per-residue data, the window-level score is used as a constant B-factor for all residues.
2. Multi-model PDB assembled (`data/multi_model_anomaly.pdb`): one MODEL block per trajectory frame.
3. NGL viewer loads the PDB and colors by `colorScheme: "bfactor"` with `colorScale: "RdYlBu"`.

**Color scale**: NGL built-in `"RdYlBu"` diverging scale (blue = low anomaly, red = high anomaly). **No explicit auto-scaling logic**: NGL maps the PDB B-factor field range automatically.

**Frame scrubbing**: Implemented via HTML range input slider (`<input id="modelSlider" type="range">`):
- Slider `min=0`, `max=maxModel` (auto-detected from PDB MODEL count).
- On `input` event: `setModel(m)` re-renders the cartoon with `sele: "@MODEL {m+1}"` (1-based MODEL records).
- `comp.autoView(sel)` re-centers camera on selected model.

**Tooltips** (`ngl_with_tooltips.html`):
- `atom.setTooltip(atom.resname + ' ' + atom.index + ' ' + atom.bfactor)` — residue name, index, and anomaly score shown on hover.

**Color scheme switching** (`static/js/three.js`):
- Dropdown options: `"bfactor"` (Anomaly), `"sstruc"` (Secondary structure), `"chainname"` (Chain).
- Switching re-renders without page reload.

#### Modality 2: Trame/VTK (Flask + Python Backend)

**Files**: `app/app.py`, `hello_trame.py`, `vtk_app.py`, `vtk_visual.py`, `vtk_3d.py`

**How scalar values are served**:
- Flask app at `app/app.py` serves `data/multi_model_anomaly.pdb` via `/file/<path:relpath>`.
- API endpoint `/api/deep_latest`: returns `anomalies.csv`, `residue_hotspots.csv`, `hybrid_scores.csv`, `latent_clusters_annot.csv` (top 25–50 rows).
- API endpoint `/api/local_ops_top`: returns per-residue scores sorted by `local_score_med` (top 20 residues).

**Auto-scaling**: Not explicitly implemented in the Flask app. Score ranges are determined by the normalization step applied upstream (percentile/rank normalization to [0,1]).

**Logging**: No explicit logging of selected residues or thresholds in the viewer code.

### 7.2 Parity Checks Between Trame and NGL

**Status**: **Not implemented.** The two modalities read from the same data sources (`data/multi_model_anomaly.pdb` and `outputs/metrics/*.json`) but there is no automated cross-check comparing their displayed values, no unit tests asserting score parity, and no shared normalization reference that both viewers query simultaneously.

### 7.3 B-Factor Encoding

Implemented in `anomaly_frames_to_bfactor.py:write_model()`:

```python
# B-factor written at PDB columns 61-66 (6 chars, 2 decimal places)
bf = float(resid_to_b.get(resid, default_b))
line = line[:60] + f"{bf:6.2f}" + line[66:]
```

- Anomaly scores are **not re-normalized** when written to B-factor field; they retain whatever scale was produced by the scoring pipeline.
- The NGL viewer maps these values to color using its built-in B-factor normalization (auto min–max across the loaded structure).

---

## 8. Reproducibility Artifacts

### `run.json` Structure

Generated by `tools/run_phase1.py` → `msm/reproducibility.py:save_run_config()`. Saved to `outputs/phase1/run.json`:

```json
{
  "timestamp": "2026-02-21T10:50:33.007Z",
  "config": {
    "features_path": "data/features.npy",
    "output_base": "outputs/phase1",
    "lag_tica": 10,
    "dim_tica": 5,
    "skip_vamp2": false,
    "skip_bootstrap": false,
    "seeds": {
      "global": 42,
      "kmeans": 42,
      "bootstrap": 123,
      "vamp": 456
    },
    "msm_params": {
      "lag": 30,
      "n_clusters": 30,
      "connectivity": "largest",
      "reversible": true
    },
    "bootstrap_params": {
      "n_iterations": 100,
      "method": "frames",
      "block_size": 10,
      "confidence_level": 0.95
    }
  }
}
```

### `vamp2_best.json` Structure

Generated by `msm/select_lag_and_dim.py`. Saved to `outputs/phase1/reports/vamp2_best.json`:

```json
{
  "lag": 10,
  "dim": 5,
  "vamp2_score": 2.3147,
  "n_candidates": 42,
  "features_shape": [1231, 7],
  "validation_fraction": 0.2,
  "seed": 456
}
```

*(Concrete values depend on run; structure is fixed.)*

### Seed Settings

All random seeds are defined in `configs/pipeline.yaml`:

| Seed Purpose | Value | Consumer |
|---|---|---|
| Global master seed | 42 | `msm/reproducibility.py:set_global_seed()` |
| K-means clustering | 42 | `sklearn.cluster.KMeans(random_state=42)` |
| Bootstrap resampling | 123 | `msm/bootstrap_msm.py:bootstrap_resample(seed=seed+i)` |
| VAMP score computation | 456 | `msm/select_lag_and_dim.py:compute_vamp2_score(seed=456)` |
| VAMP cross-validation | 42 | `msm/validation.py:vamp2_cross_validation(seed=42)` |

The function `msm/reproducibility.py:generate_seed_sequence(master_seed, n_seeds)` generates deterministic child seeds from the master seed using `np.random.RandomState(master_seed).randint(0, 2^31-1, n_seeds)`.

### Integration Tests

**Location**: `tests/` (14 test files)

| Test File | Purpose |
|---|---|
| `test_phase1.py` | VAMP-2 score correctness, bootstrap resampling shape, reproducibility |
| `test_phase2.py` | Energy and pocket feature integration |
| `test_phase3.py` | Multi-signal scoring, fusion, smoothing |
| `test_scientific_validation.py` | CK test, ITS convergence, cross-validation |
| `test_integration.py` | End-to-end pipeline smoke tests |
| `test_reproducibility.py` | Seed reproducibility, noise robustness, parameter sensitivity (mock) |
| `test_statistical_validation.py` | Statistical correctness of bootstrap CIs |
| `test_signals.py` | Individual signal unit tests (rarity, surprise, density, RMSF) |
| `test_optimizations.py` | Performance benchmarks for vectorized operations |
| `test_dataset_validation.py` | Trajectory quality checks |
| `test_pipeline_edge_cases.py` | Short trajectory, zero-variance, disconnected states, NaN |
| `test_compute_presentation_metrics.py` | Metric computation correctness |
| `run_all_validation.py` | Orchestrator for all tests |
| `sample_predictions.csv` | Reference predictions for regression testing |

### Validation Reports Saved to Disk

| Artifact | Location | Content |
|---|---|---|
| `validation_report.json` | `outputs/validation/` | Overall PASSED/NEEDS_REVIEW, tests performed, timestamps |
| `bootstrap_metadata.json` | `outputs/models/msm_bootstrap/` | Bootstrap parameters, success count, n_states |
| `vamp2_grid.csv` | `outputs/phase1/reports/` | All 42 (lag, dim, vamp2_score) combinations |
| `vamp2_best.json` | `outputs/phase1/reports/` | Selected optimal parameters |
| `run.json` | `outputs/phase1/` | Full run configuration with seeds and parameters |
| `pi_ci.parquet` | `outputs/models/msm_bootstrap/` | π bootstrap confidence intervals |
| `P_ci.npz` | `outputs/models/msm_bootstrap/` | Transition matrix bootstrap CIs |
| `mfpt_ci.parquet` | `outputs/models/msm_bootstrap/` | Mean first-passage time bootstrap CIs |
| `chapman_kolmogorov.png` | `outputs/validation/` | CK test visualization |
| `implied_timescales.png` | `outputs/validation/` | ITS convergence plot |
| `signal_correlations.png` | `outputs/validation/` | Spearman correlation heatmap across signals |
| `its.png` | `outputs/msm/` | Implied timescales from `run_msm_tica.py` |

---

## 9. Structured Summary

### Datasets

| Item | Value |
|---|---|
| Topology | `data/raw_trajectory/align_topol.pdb` |
| Trajectories | 20 XTC files: `trajectory_0.xtc` … `trajectory_19.xtc` |
| Format | GROMACS XTC (compressed binary) + PDB topology |
| Frame count | Runtime-determined (MDTraj `md.load()`); stored as first dim of `data/features.npy` |
| Preprocessing | Alignment to frame 0 (RMSF); native contact cutoff 0.8 nm; optional stride |

### Feature Space

| Item | Value |
|---|---|
| Features computed | RMSD (1), radius of gyration (1), native contacts (1), φ_sin (1), φ_cos (1), ψ_sin (1), ψ_cos (1) |
| Dimensionality | d = 7 (base); optional energy and pocket features |
| Dihedral encoding | sin/cos to handle angular periodicity |
| Normalization | None at feature level; tICA handles scale implicitly |
| Storage | `data/features.npy` (NumPy, shape `(n_frames, 7)`) |

### tICA Configuration

| Item | Value |
|---|---|
| Lag candidates | [5, 10, 15, 20, 30, 50] frames |
| Dimension candidates | [2, 3, 4, 5, 6, 8, 10] |
| Grid size | 42 combinations |
| Selection criterion | Maximum VAMP-2 score (20% held-out validation) |
| Cross-validation | 5-fold VAMP-2 CV (seed=42) |
| Outputs | `tica_coords.npy`, `vamp2_best.json`, `vamp2_grid.csv` |
| ITS plots | `its.png`, `implied_timescales.png` |

### MSM Configuration

| Item | Value |
|---|---|
| Clusters | 30 (K-means, seed=42, in tICA space) |
| MSM lag | 30 frames |
| Estimator | MaximumLikelihoodMSM (deeptime), reversible |
| Connectivity | Largest connected set |
| Bootstrap | 100 iterations, frame resampling, 95% CI |
| Outputs | `P.npy`, `pi.npy`, `dtraj.npy`, `pi_ci.parquet`, `P_ci.npz` |

### Signals

| Signal | Formula | Type |
|---|---|---|
| State rarity | `1 − π[s_t]` | Kinetic-thermodynamic |
| Transition surprise | `−log P[s_t → s_{t+lag}]` | Kinetic-barrier |
| Local density | `−mean_kNN_distance(tICA)` | Geometric-structural |
| RMSF | `sqrt(mean_t((r_i(t) − <r_i>)²))` in Å | Structural flexibility |
| tICA importance | `L2_norm(loadings[:, :5])` per residue | Slow-mode contribution |

### Fusion Strategy

| Item | Value |
|---|---|
| Pre-fusion normalization | Rank normalization to [0,1] (default) |
| Fusion method | Element-wise median across signals (default) |
| Alternative | Element-wise mean |
| Temporal smoothing | Rolling median, window=5 frames |
| Scale | Mapped to [0,100] (internal), then to [0,1] for JSON outputs |

### Validation Procedures

| Procedure | Implementation | Metric |
|---|---|---|
| VAMP-2 model selection | `msm/select_lag_and_dim.py` | VAMP-2 score (max over 42 grid points) |
| 5-fold VAMP-2 CV | `msm/validation.py:vamp2_cross_validation()` | Mean ± std VAMP-2 score |
| Chapman-Kolmogorov test | `msm/validation.py:chapman_kolmogorov_test()` | Predicted vs. estimated P^k |
| Implied timescales | `msm/validation.py:implied_timescales_convergence()` | Plateau check (CV < 0.2) |
| Stationary distribution | `msm/validation.py:validate_stationary_distribution()` | Max relative error < 0.1 |
| Bootstrap uncertainty | `msm/bootstrap_msm.py` | 95% CI on π, P, MFPTs |
| Signal correlation | `msm/validation.py:signal_correlation_analysis()` | Spearman correlation matrix |

### Sensitivity Analysis

| Test | Status |
|---|---|
| VAMP-2 grid search (42 combinations, proxy for model sensitivity) | **Implemented** (`msm/select_lag_and_dim.py`) |
| Mock hyperparameter sensitivity (lag, dim, n_clusters) | **Implemented** (`tests/test_reproducibility.py`) |
| Noise injection robustness | **Implemented** (`tests/test_reproducibility.py`) |
| Edge cases (short traj, zero variance, NaN) | **Implemented** (`tests/test_pipeline_edge_cases.py`) |
| Full lag sweep on hotspot rankings (RQ3) | **Not implemented** |
| Signal ablation study | **Not implemented** |
| Jaccard similarity comparison | **Not implemented** |
| Cluster count sweep on hotspot rankings | **Not implemented** |

### Visualization Validation Mechanisms

| Mechanism | Implementation |
|---|---|
| B-factor encoding | `anomaly_frames_to_bfactor.py:write_model()` — PDB columns 61–66 |
| NGL coloring | `colorScheme: "bfactor"`, `colorScale: "RdYlBu"` (NGL auto-scaled) |
| Frame scrubbing | HTML range slider in `templates/three.html`, `static/js/three.js` |
| Residue tooltips | `atom.setTooltip(resname + index + bfactor)` in `ngl_with_tooltips.html` |
| Color scheme toggle | Dropdown for bfactor / secondary structure / chain in `static/js/three.js` |
| Trame/Flask API | `/api/deep_latest`, `/api/local_ops_top` (top 20–50 residues) |
| Parity check (Trame vs. NGL) | **Not implemented** |
| Score range auto-scaling | NGL built-in (not configurable from pipeline); Flask serves raw scores |
| Threshold logging | **Not implemented** |

### Reproducibility Controls

| Control | Value |
|---|---|
| Global seed | 42 (`configs/pipeline.yaml`) |
| K-means seed | 42 |
| Bootstrap seed | 123 |
| VAMP seed | 456 |
| Run config | `run.json` (timestamp + all parameters) |
| Best model params | `vamp2_best.json` |
| Feature caching | `.cache/` directory (hash-based, mtime-keyed) |
| Determinism tests | `tests/test_reproducibility.py:test_computation_determinism()` |
| Seed sequence generator | `msm/reproducibility.py:generate_seed_sequence(master_seed, n)` |

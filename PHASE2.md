# Phase 2: Feature Extensions (Energy & Pockets)

This phase adds per-residue energetic signals and pocket/cavity dynamics to the MD anomaly detection pipeline.

## Overview

Phase 2 implements two critical feature types:
1. **Per-Residue Energetic Features**: Knowledge-based contact potentials and hydrogen bonding
2. **Pocket/Cavity Dynamics**: Time-varying pocket volume, mouth radius, and SASA

## Features

### 1. Per-Residue Energetic Features (`features_energy/`)

Computes energetic proxies for each residue per frame using:
- **Knowledge-based contact potentials** (simplified Miyazawa-Jernigan potential)
- **Hydrogen bond counting** (geometric heuristics)
- **Electrostatic contributions** (charge-charge interactions)

```bash
python tools/generate_energy.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --output data/derived/residue_energy.parquet
```

**Output Schema** (`data/derived/residue_energy.parquet`):
```
frame: int       - Frame index
res_id: int      - Residue ID
chain: str       - Chain identifier
energy: float    - Total energy in kcal/mol (negative = favorable)
hbonds: int      - Number of hydrogen bonds
```

**Algorithm:**
- For each residue i, compute:
  - Contact energy: Σⱼ E(rᵢ, rⱼ, dᵢⱼ) using distance-dependent potential
  - Electrostatic: Coulombic for charged pairs within cutoff
  - H-bonds: Count based on proximity of polar/charged residues
- Distance cutoff: 0.8 nm (configurable)
- Potentials based on statistical analysis of protein structures

**Performance:**
- ~3-5 seconds per 100 frames
- Caching enabled by default (checks file hashes)
- Memory: ~10 MB per 1000 residues × 1000 frames

### 2. Pocket/Cavity Dynamics (`features_pockets/`)

Detects and tracks pockets/cavities over trajectory using grid-based method:

```bash
python tools/generate_pockets.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc \
    --output_pockets data/derived/pockets.parquet \
    --output_rims data/derived/pocket_rims.parquet
```

**Output Schemas:**

`data/derived/pockets.parquet`:
```
frame: int           - Frame index
pocket_id: int       - Pocket identifier (per frame)
volume: float        - Pocket volume in nm³
mouth_radius: float  - Pocket opening radius in nm
sasa_rim: float      - Solvent-accessible surface area of rim in nm²
```

`data/derived/pocket_rims.parquet`:
```
frame: int           - Frame index
pocket_id: int       - Pocket identifier
res_id: int          - Residue ID near pocket
rim_distance: float  - Distance from residue to pocket center in nm
```

**Algorithm:**
1. Create 3D grid around protein (spacing: 0.5 nm, configurable)
2. Identify grid points that:
   - Are > probe_radius from any atom (fit water-sized probe)
   - Are < probe_radius + 0.6 nm from atoms (not in bulk solvent)
3. Label connected components as pockets
4. Filter by volume (0.5 - 50 nm³, configurable)
5. Compute metrics:
   - Volume: voxel count × voxel_volume
   - Mouth radius: 90th percentile distance from center
   - SASA: boundary voxels × grid_spacing²
6. Map nearby residues (within 0.8 nm)

**Performance:**
- ~10-20 seconds per 100 frames (grid_spacing=0.5)
- Faster with coarser grid (0.7-1.0 nm)
- Memory: ~50-100 MB during computation
- Caching enabled by default

## Configuration

Both tools support caching via `.energy_cache.json` and `.pockets_cache.json` in the output directory. Caching checks:
- Topology file hash (MD5)
- Trajectory file hash (MD5)

Disable caching with `--no-cache` flag.

## Usage Examples

### Basic Pipeline

```bash
# 1. Generate energetic features
python tools/generate_energy.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc

# 2. Generate pocket features
python tools/generate_pockets.py \
    --topology data/raw_trajectory/align_topol.pdb \
    --trajectory data/raw_trajectory/trajectory.xtc

# Outputs:
#   data/derived/residue_energy.parquet
#   data/derived/pockets.parquet
#   data/derived/pocket_rims.parquet
```

### Custom Parameters

```bash
# Finer energy computation with tighter cutoff
python tools/generate_energy.py \
    --topology data/top.pdb \
    --trajectory data/traj.xtc \
    --contact_cutoff 0.6 \
    --output data/custom_energy.parquet

# Finer pocket detection with smaller probe
python tools/generate_pockets.py \
    --topology data/top.pdb \
    --trajectory data/traj.xtc \
    --grid_spacing 0.3 \
    --probe_radius 0.12 \
    --min_volume 1.0
```

### With Stride (Subsample Trajectory)

```bash
# Process every 5th frame
python tools/generate_energy.py \
    --topology data/top.pdb \
    --trajectory data/traj.xtc \
    --stride 5

python tools/generate_pockets.py \
    --topology data/top.pdb \
    --trajectory data/traj.xtc \
    --stride 5
```

## Analysis Examples

### Load and Analyze Energies

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load energies
df_energy = pd.read_parquet('data/derived/residue_energy.parquet')

# Energy per frame (sum over residues)
frame_energy = df_energy.groupby('frame')['energy'].sum()

plt.plot(frame_energy)
plt.xlabel('Frame')
plt.ylabel('Total Energy (kcal/mol)')
plt.show()

# Most energetically stressed residues
stressed = df_energy.groupby('res_id')['energy'].mean().sort_values()
print("Most unfavorable residues:")
print(stressed.tail(10))
```

### Load and Analyze Pockets

```python
import pandas as pd

# Load pocket data
df_pockets = pd.read_parquet('data/derived/pockets.parquet')
df_rims = pd.read_parquet('data/derived/pocket_rims.parquet')

# Pocket count per frame
pocket_counts = df_pockets.groupby('frame').size()
print(f"Average pockets per frame: {pocket_counts.mean():.1f}")

# Volume volatility (frame-to-frame changes)
volume_changes = df_pockets.groupby('frame')['volume'].sum().diff().abs()
print(f"Mean volume change: {volume_changes.mean():.2f} nm³")

# Residues most often near pockets
rim_frequency = df_rims.groupby('res_id').size().sort_values(ascending=False)
print("Residues most often at pocket rims:")
print(rim_frequency.head(10))
```

## Integration with Existing Pipeline

Phase 2 features integrate seamlessly:

```python
import pandas as pd

# Load Phase 2 features
energy_df = pd.read_parquet('data/derived/residue_energy.parquet')
pockets_df = pd.read_parquet('data/derived/pockets.parquet')

# Merge with existing features (from Phase 1)
# Example: Add to anomaly scoring (Phase 3)

# Energy stress per frame
energy_stress = energy_df.groupby('frame')['energy'].agg(['mean', 'std'])

# Pocket volatility per frame
pocket_vol = pockets_df.groupby('frame')['volume'].sum()
pocket_volatility = pocket_vol.diff().abs()

# These can be used as additional signals in anomaly_v2.py (Phase 3)
```

## Testing

Run unit tests:

```bash
python tests/test_phase2.py
```

Tests cover:
- Contact energy computation (attractive/repulsive regimes)
- Energy symmetry and distance dependence
- Grid creation and spacing
- Residue classification
- Value ranges

## Scientific Justification

### Why Energetic Features?

Residue energetics capture:
- **Strain**: High-energy residues indicate local frustration
- **Stability**: Favorable contacts stabilize conformations
- **Allosteric signals**: Energy changes propagate through structure
- **Druggability**: High-energy pockets are potential binding sites

**References:**
- Miyazawa & Jernigan (1996). "Residue-residue potentials..."
- Moal & Fernández-Recio (2013). "SKEMPI: Database of kinetic and energetic mutations"

### Why Pocket Dynamics?

Pockets are functionally critical:
- **Binding sites**: Most drugs/ligands bind in pockets
- **Allosteric sites**: Regulation often involves cryptic pockets
- **Transient pockets**: MD reveals pockets invisible in static structures
- **Druggability**: Pocket volume/shape predicts binding affinity

**References:**
- Schmidtke et al. (2011). "MDpocket: open-source cavity detection..."
- Kokh et al. (2018). "Estimation of drug-target residence times..."

## Performance Considerations

### Energy Computation
- **Bottleneck**: Distance matrix (O(N²) per frame)
- **Optimization**: Sparse contacts within cutoff
- **Speedup**: Use stride to skip frames
- **Memory**: Grows with N_residues × N_frames

### Pocket Detection
- **Bottleneck**: Grid creation and labeling
- **Optimization**: Coarser grid (0.7-1.0 nm) for screening
- **Speedup**: Parallel frame processing (future)
- **Memory**: Grid size ~ (protein_size / grid_spacing)³

### Caching
Both tools cache results indexed by file hashes:
- First run: Full computation
- Subsequent runs: Instant (<1 sec) if files unchanged
- Cache invalidation: Automatic on file modification

## Limitations & Future Work

### Current Limitations
1. **Energy Model**: Simplified knowledge-based potential
   - Not as accurate as full MM/PBSA
   - Ignores solvation details
   - No quantum effects

2. **Pocket Detection**: Grid-based approximation
   - May miss very small pockets (< grid_spacing)
   - Mouth radius is heuristic
   - No distinction between pockets/tunnels

### Future Enhancements
1. **Energy**: 
   - Add MM/PBSA decomposition (expensive)
   - Include solvation free energy
   - GPU acceleration

2. **Pockets**:
   - Use Alpha shapes for better geometry
   - Distinguish pocket types (buried, surface, tunnel)
   - Track pocket persistence across frames

## Next Steps

Phase 3 will integrate these features into enhanced anomaly scoring:
- Energy stress: z-score of per-frame total energy
- Pocket volatility: |Δ volume| frame-to-frame
- Multi-signal fusion with kinetic features

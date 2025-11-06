# Interactive Molecular Visualization

## Overview

This document describes the new interactive molecular visualization features added to the ensemble-anomaly-maps pipeline.

## Features

### 1. Interactive 3D Viewer
- **URL**: `http://localhost:5051/interactive`
- Three.js-based molecular rendering
- Smooth camera controls (drag to rotate, scroll to zoom)
- Real-time atom and residue selection

### 2. Molecular Slicing
- Toggle clipping plane to "cut" through the molecule
- Adjust plane position along X, Y, or Z axis
- Reveals internal structure and binding sites

### 3. Atom/Residue Selection
- Click on any atom to view detailed information
- Sidebar shows:
  - Residue name and number
  - Chain ID
  - Hotspot score (from ML analysis)
  - RMSF (flexibility metric)
  - SASA (solvent accessible surface area)

### 4. Dynamic Color Mapping
Choose from multiple coloring modes:
- **Hotspot Score**: Blue (low) → White (medium) → Red (high)
- **RMSF**: Visualize residue flexibility
- **Residue Type**: Color by residue sequence
- **Chain**: Color by molecular chain

### 5. Frame Navigation
- Slider to navigate through trajectory frames
- View molecular dynamics over time
- See how hotspots change across frames

## API Endpoints

### Trajectory Data
- `GET /api/trajectory/meta` - Get metadata (frames, atoms, residues)
- `GET /api/trajectory/frame/<frame_id>` - Get atom coordinates for frame
- `GET /api/trajectory/residue_map` - Map atoms to residues

### Analysis Data
- `GET /api/hotspots` - Get all residue hotspot scores
- `GET /api/rmsf` - Get RMSF data for all residues
- `GET /api/residue/<resid>` - Get detailed info for specific residue
- `GET /api/atom/<atom_idx>` - Get information about specific atom

### Other
- `GET /api/residues` - Get table of all residues
- `GET /` - Status page
- `GET /three` - Original NGL viewer

## Running the Application

### 1. Install Dependencies
```bash
pip install flask pandas numpy MDAnalysis
```

### 2. Start the Flask Server
```bash
cd /path/to/ensemble-anomaly-maps
python app/app.py
```

The server will start on `http://localhost:5051`

### 3. Access the Interactive Viewer
Open your browser and navigate to:
```
http://localhost:5051/interactive
```

## Controls

### Mouse Controls
- **Left Click**: Select atom/residue
- **Left Drag**: Rotate camera
- **Right Drag**: Pan camera
- **Scroll**: Zoom in/out
- **Hover**: Show atom tooltip

### UI Controls
- **Color Mode**: Select coloring scheme
- **Frame Slider**: Navigate trajectory frames
- **Toggle Clipping**: Enable/disable molecular slicing
- **Plane Position**: Adjust clipping plane position
- **Plane Axis**: Choose X, Y, or Z axis for clipping

## Data Requirements

The viewer automatically loads data from:
- `data/multi_model_anomaly.pdb` - Molecular structure
- `outputs/run-traj-*/deep/residue_hotspots.csv` - Hotspot scores
- `outputs/run-traj-*/per_residue_overall.csv` - RMSF data (if available)

## Technical Details

### Frontend Stack
- Three.js r128 for 3D rendering
- Vanilla JavaScript (no framework dependencies)
- Modern CSS with dark theme
- Responsive design

### Backend Stack
- Flask web framework
- MDAnalysis for molecular data
- Pandas for CSV processing
- NumPy for numerical operations

### Performance
- Atom meshes: ~2400 spheres for typical protein
- Bonds: Distance-based heuristic
- Clipping: GPU-accelerated via Three.js
- Data caching: In-memory for fast access

## Future Enhancements

Potential improvements:
- [ ] Multi-atom selection for interaction visualization
- [ ] Contact network visualization
- [ ] Hydrogen bond display
- [ ] State transition animations (VAMPnet integration)
- [ ] Export screenshots and 3D models
- [ ] Measurement tools (distances, angles)
- [ ] Secondary structure visualization
- [ ] Ligand highlighting

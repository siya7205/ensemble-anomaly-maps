# Running the Interactive Molecular Viewer

## Quick Start

### 1. Install Dependencies

```bash
pip install flask pandas numpy MDAnalysis requests
```

### 2. Start the Flask Server

**Important**: Run the server from the repository root directory, not from the `app/` directory.

```bash
# Make sure you're in the repository root
cd /path/to/ensemble-anomaly-maps

# Start the server
python app/app.py
```

The server will start on `http://localhost:5051`

### 3. Access the Viewer

Open your browser and navigate to:
```
http://localhost:5051/interactive
```

## Troubleshooting

### 404 Error on /interactive

If you get a 404 error, make sure you:

1. **Run from repository root**: The app must be started from the main repository directory, not from `app/`
   ```bash
   # Wrong:
   cd app && python app.py
   
   # Correct:
   python app/app.py
   ```

2. **Check Flask output**: Look for the line showing routes are registered:
   ```
   * Running on http://127.0.0.1:5051
   ```

3. **Verify template exists**:
   ```bash
   ls templates/interactive_viewer.html
   ```

### Module Import Errors

If you see import errors:

```bash
# Install missing dependencies
pip install flask pandas numpy MDAnalysis

# If still having issues, try:
pip install --upgrade flask pandas numpy MDAnalysis
```

## Features

### Visualization Modes

The viewer now supports three rendering modes:

1. **Atoms (Spheres)**: Standard ball representation
   - Each atom shown as a colored sphere
   - Good for seeing all atoms clearly

2. **Ball and Stick**: Molecular structure with bonds
   - Smaller spheres for atoms
   - Cylindrical bonds connecting nearby atoms
   - Classic molecular visualization

3. **Ribbon (Cartoon)**: Protein backbone visualization
   - Smooth ribbon through CA atoms
   - Shows overall protein structure
   - Good for seeing secondary structure

### Color Modes

- **Hotspot Score**: Shows ML-detected anomalies (Blue → White → Red)
- **RMSF**: Visualizes residue flexibility
- **Residue Type**: Colors by sequence position
- **Chain**: Colors by molecular chain

### Controls

- **Left Click + Drag**: Rotate molecule
- **Scroll**: Zoom in/out
- **Click Atom**: View residue details
- **Toggle Clipping**: Enable molecular slicing
- **Frame Slider**: Navigate trajectory frames

## API Endpoints

The backend provides these REST APIs:

- `GET /interactive` - Interactive viewer page
- `GET /api/trajectory/meta` - Trajectory metadata
- `GET /api/trajectory/frame/<id>` - Atom coordinates
- `GET /api/residue/<id>` - Residue details
- `GET /api/hotspots` - Hotspot scores
- `GET /api/rmsf` - RMSF data

## Testing

Run the automated test suite:

```bash
python test_interactive_viewer.py
```

This validates:
- Data loading (hotspots, RMSF)
- API endpoints
- File structure

For full documentation, see `INTERACTIVE_VIEWER.md`.

import vtk
import json
from trame.app import get_server
from trame.ui import html
from trame.widgets.vtk import VtkViewer

# Initialize Trame server
server = get_server()

# Load the anomaly timeseries
with open('../data/anomaly_timeseries.json') as f:
    anomaly_data = json.load(f)

# Load the PDB file and apply anomalies
def load_pdb_with_anomalies(file_path, anomaly_data):
    pdb_reader = vtk.vtkPDBReader()
    pdb_reader.SetFileName(file_path)
    pdb_reader.Update()

    # Apply anomaly scores (B-factor)
    apply_anomaly_scores(pdb_reader, anomaly_data)

    # Color the atoms by anomaly (B-factor)
    actor = color_by_anomaly(pdb_reader)

    return actor

@server.app.route("/")
def index():
    # Load the PDB and color by anomaly
    pdb_actor = load_pdb_with_anomalies('../data/multi_model_anomaly.pdb', anomaly_data)

    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pdb_actor)

    # Return the VTK viewer in the Trame application
    return html.Div([VtkViewer(renderer=renderer)])

# Start the server
if __name__ == "__main__":
    server.start()

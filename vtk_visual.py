import vtk
from trame.app import get_server
from trame.ui import html
from trame.widgets.vtk import VtkViewer

# Initialize Trame server
server = get_server()

# Load the PDB file
def load_pdb(file_path):
    pdb_reader = vtk.vtkPDBReader()
    pdb_reader.SetFileName(file_path)
    pdb_reader.Update()

    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(pdb_reader.GetOutputPort())

    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

# Route to load the PDB and visualize
@server.app.route("/")
def index():
    # Path to your PDB file
    pdb_actor = load_pdb('../data/multi_model_anomaly.pdb')  # Correct the path as needed
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pdb_actor)

    # Return the VTK viewer in the Trame application
    return html.Div([VtkViewer(renderer=renderer)])

# Start the server
if __name__ == "__main__":
    server.start()

# filename: vtk_demo_app.py

from trame.app import get_server
from trame.widgets import vtk as vtk_widgets
from trame.ui.vuetify import SinglePageLayout
import vtkmodules.all as vtk

# Load your PDB file
reader = vtk.vtkPDBReader()
reader.SetFileName("data/multi_model_anomaly.pdb")
reader.Update()

# Create mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.SetScalarModeToUsePointFieldData()
mapper.SelectColorArray("B-factor")  # This is where you embed anomalies
mapper.SetScalarRange(0, 1)  # Set according to your anomalies

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create renderer and render window
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Setup Trame
server = get_server()
server.client_type = "vue2"

with SinglePageLayout(server) as layout:
    layout.title.set_text("Anomaly Visualization - MD Trajectory")
    with layout.content:
        vtk_widgets.VtkRemoteView(render_window)

if __name__ == "__main__":
    server.start()

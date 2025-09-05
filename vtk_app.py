from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk as vtk_widgets
import vtkmodules.all as vtk

# Load your PDB file
reader = vtk.vtkPDBReader()
reader.SetFileName("data/colored_frames/frame_0010.pdb")
reader.Update()

# Mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.SetScalarModeToUsePointFieldData()
mapper.SelectColorArray("BFactor")
mapper.SetColorModeToMapScalars()
mapper.SetScalarRange(reader.GetOutput().GetPointData().GetScalars().GetRange())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Renderer setup
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.ResetCamera()

# Render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Trame setup
server = get_server()
state, ctrl = server.state, server.controller

with SinglePageLayout(server) as layout:
    layout.title.set_text("Anomaly Viewer")
    with layout.content:
        vtk_widgets.VtkRemoteView(render_window)

server.start()

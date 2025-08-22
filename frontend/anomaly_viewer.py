# frontend/anomaly_viewer.py
import os
import threading
import requests
import numpy as np

# --- VTK pipeline ---
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

# --- Trame / UI ---
from trame.app import get_server
from trame.widgets import vuetify3 as v3
from trame.ui.vuetify3 import VAppLayout
from trame_vtk.widgets import vtk as vtkw


# --------------------------
# Data helpers
# --------------------------
def fetch_points_from_api(url="http://127.0.0.1:5051/api/points"):
    """Return (N, 3) xyz and (N,) score arrays, or None on failure."""
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        js = r.json()
        pts = js.get("points", [])
        if not pts:
            return None
        xyz = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=np.float32)
        score = np.array([p.get("score", 0.0) for p in pts], dtype=np.float32)
        return xyz, score
    except Exception:
        return None


def make_synthetic_cloud(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0, 0], 0.6, size=(n // 2, 3))
    b = rng.normal([3, 3, 3], 0.6, size=(n // 3, 3))
    c = rng.uniform(-5, 5, size=(n - len(a) - len(b), 3))
    xyz = np.vstack([a, b, c]).astype(np.float32)
    score = np.linalg.norm(xyz, axis=1).astype(np.float32)
    return xyz, score


# --------------------------
# Build VTK pipeline
# --------------------------
def build_pipeline():
    points = vtk.vtkPoints()
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    verts = vtk.vtkCellArray()
    poly.SetVerts(verts)

    score_da = vtk.vtkFloatArray()
    score_da.SetName("score")
    poly.GetPointData().AddArray(score_da)
    poly.GetPointData().SetActiveScalars("score")

    glyph = vtk.vtkVertexGlyphFilter()
    glyph.SetInputData(poly)
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("score")
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(0.0, 1.0)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(3)

    axes = vtk.vtkCubeAxesActor()
    axes.SetBounds(-5, 5, -5, 5, -5, 5)
    axes.SetXTitle("X")
    axes.SetYTitle("Y")
    axes.SetZTitle("Z")

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.AddActor(actor)
    ren.AddActor(axes)

    rw = vtk.vtkRenderWindow()
    rw.AddRenderer(ren)
    rw.SetOffScreenRendering(1)  # avoid macOS Cocoa window creation on non-main thread
    rw.SetSize(1100, 800)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)

    axes.SetCamera(ren.GetActiveCamera())

    return dict(
        poly=poly,
        points=points,
        verts=verts,
        score_da=score_da,
        glyph=glyph,
        mapper=mapper,
        actor=actor,
        axes=axes,
        renderer=ren,
        render_window=rw,
        interactor=iren,
    )


def update_polydata_from_numpy(p, xyz, score):
    N = xyz.shape[0]
    p["points"].SetData(numpy_to_vtk(xyz, deep=1))
    p["poly"].SetPoints(p["points"])

    ids = np.arange(N, dtype=np.int64)
    cells = np.empty((N, 2), dtype=np.int64)
    cells[:, 0] = 1
    cells[:, 1] = ids
    cell_arr = numpy_to_vtkIdTypeArray(cells.reshape(-1), deep=1)
    p["verts"].SetCells(N, cell_arr)
    p["poly"].SetVerts(p["verts"])

    p["score_da"].SetNumberOfValues(N)
    for i, v in enumerate(score):
        p["score_da"].SetValue(i, float(v))
    p["score_da"].Modified()

    p["poly"].GetPointData().SetScalars(p["score_da"])
    p["poly"].Modified()
    p["glyph"].Update()


# --------------------------
# Trame app
# --------------------------
server = get_server(name="Anomaly3D", client_type="vue3")
state, ctrl = server.state, server.controller

state.threshold = 0.0
state.status = "Idle"
state.n_points = 0

P = build_pipeline()

def noop():
    pass


ctrl.view_update = noop


def load_data():
    data = fetch_points_from_api()
    if data is None:
        xyz, score = make_synthetic_cloud()
        state.status = "Demo data (API not found)."
    else:
        xyz, score = data
        state.status = "Loaded from /api/points"

    lo, hi = float(np.min(score)), float(np.max(score))
    rng = max(hi - lo, 1e-9)
    score01 = (score - lo) / rng

    update_polydata_from_numpy(P, xyz, score01)
    P["mapper"].SetScalarRange(0.0, 1.0)
    P["render_window"].Render()

    state.n_points = int(xyz.shape[0])
    ctrl.view_update()


@ctrl.trigger("refresh_data")
def refresh_data():
    state.status = "Refreshing..."
    threading.Thread(target=lambda: (load_data(), None), daemon=True).start()


# --------------------------
# UI (no layout helpers)
# --------------------------
with VAppLayout(server) as layout:
    # Top app bar
    with v3.VAppBar(density="comfortable", flat=True):
        v3.VToolbarTitle("3D Anomaly Viewer")
        v3.VSpacer()
        v3.VBtn("Refresh from /api/points", variant="tonal", click=ctrl.refresh_data)
        v3.VDivider(vertical=True, classes="mx-2")
        v3.VSlider(
            v_model=("threshold", 0.0),
            min=0,
            max=1,
            step=0.01,
            label="Score threshold",
            style="max-width: 300px;",
        )
        v3.VDivider(vertical=True, classes="mx-2")
        v3.VChip(("status",), color="primary", variant="tonal")
        v3.VChip(("`N=` + n_points",), color="secondary", variant="tonal", classes="ml-2")

    # Main content
    with v3.VMain():
        view = vtkw.VtkRemoteView(P["render_window"], interactive_ratio=1, server=server)
        ctrl.view_update = view.update

        v3.VAlert(
            type="info",
            text="Tip: drag to rotate, wheel/trackpad to zoom. Colors map to anomaly score.",
            variant="tonal",
            density="comfortable",
            classes="ma-3",
        )


@state.change("threshold")
def on_threshold(threshold, **_):
    # Hook to implement masking later if you want. For now, just re-render.
    P["render_window"].Render()
    ctrl.view_update()


# --- READY HOOK / STARTUP ---
# --- READY HOOK / STARTUP ---
def _on_ready(**_):
    try:
        view.update()
    except Exception as e:
        print("Initial view.update() failed:", e)

ctrl.on_server_ready.add(_on_ready)

if __name__ == "__main__":
    server.start(port=int(os.environ.get("PORT", 8090)), open_browser=True)

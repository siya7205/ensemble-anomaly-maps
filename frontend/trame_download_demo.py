# frontend/trame_download_demo.py
# run: python frontend/trame_download_demo.py
# it shows two buttons:
#  - "Download CSV (from memory)" -> creates a CSV on the fly and downloads it
#  - "Download last_score.json (from disk)" -> downloads your latest results if present

import os, io, csv, json
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify as v

server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller
state.setdefault("msg", "Click a button below")

@ctrl.trigger("download_memory")
def download_memory():
    # build a small CSV in memory
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["resid", "if_score"])
    for i in range(1, 6):
        w.writerow([i, round(-0.2 + i * 0.01, 6)])
    content = buf.getvalue().encode("utf-8")

    # register file with trame and trigger browser download
    key = server.file.content(
        name="demo_scores.csv",
        content=content,
        mime_type="text/csv",
    )
    ctrl.download(key)
    state.msg = "Downloaded demo_scores.csv ✅"
    server.flush_state()

@ctrl.trigger("download_last_json")
def download_last_json():
    # point to your pipeline summary (created when you hit /api/score_pdb)
    path = os.path.expanduser("~/capstone/ensemble-clean/results/last_score.json")
    if not os.path.exists(path):
        state.msg = "No results/last_score.json yet — upload/score a PDB first."
        server.flush_state()
        return

    key = server.file.path(
        path,
        name="last_score.json",
        mime_type="application/json",
    )
    ctrl.download(key)
    state.msg = "Downloaded last_score.json ✅"
    server.flush_state()

with SinglePageLayout(server) as layout:
    layout.title.set_text("trame download tutorial — demo")

    with layout.toolbar:
        v.Spacer()
        v.Btn(
            "Download CSV (from memory)",
            click=ctrl.download_memory,
            outlined=True,
        )
        v.Btn(
            "Download last_score.json (from disk)",
            click=ctrl.download_last_json,
            outlined=True,
            classes="ml-2",
        )

    with layout.content:
        with v.Container(fluid=True, classes="pa-4"):
            v.TextField(
                label="Status",
                v_model=("msg",),
                readonly=True,
                hide_details=True,
            )

if __name__ == "__main__":
    server.start(port=5052, open_browser=True)

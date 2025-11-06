# app/app.py (clean minimal)
import os, pathlib, glob
from flask import Flask, jsonify, send_file, render_template, request
import pandas as pd

try:
    from app.data_loader import get_loader
except ImportError:
    from data_loader import get_loader

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STATIC = ROOT / "static"
TEMPL = ROOT / "templates"
DATA = ROOT / "data"

app = Flask(__name__, static_folder=str(STATIC), template_folder=str(TEMPL))

# Data loader for trajectory and analysis data
loader = get_loader()

# ---------- helpers ----------
def _latest(pattern):
    paths = sorted(OUT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ---------- pages ----------
@app.route("/")
def home():
    # simple status page so you can see what’s wired
    return render_template("status.html")

@app.route("/three")
def three():
    # 3D NGL viewer for data/multi_model_anomaly.pdb
    return render_template("three.html")

@app.route("/interactive")
def interactive():
    # New interactive molecular viewer with slicing and selection
    return render_template("interactive_viewer.html")

# ---------- files ----------
@app.route("/file/<path:relpath>")
def file_serve(relpath):
    p = ROOT / relpath
    if not p.exists():
        return "Not found", 404
    # basic mime
    mt = "text/plain"
    s = p.suffix.lower()
    if s in {".png", ".jpg", ".jpeg", ".gif"}:
        mt = "image/png"
    elif s in {".html", ".htm"}:
        mt = "text/html"
    elif s == ".csv":
        mt = "text/csv"
    elif s == ".pdb":
        mt = "chemical/x-pdb"
    return send_file(str(p), mimetype=mt)

# ---------- APIs ----------
@app.route("/api/health")
def api_health():
    has_mm = (DATA / "multi_model_anomaly.pdb").exists()
    latest_traj = _latest("run-traj-*")
    latest_local = _latest("run-local-*")
    return jsonify(ok=True,
                   multi_model_exists=has_mm,
                   latest_traj=str(latest_traj.relative_to(ROOT)) if latest_traj else None,
                   latest_local=str(latest_local.relative_to(ROOT)) if latest_local else None)

@app.route("/api/deep_latest")
def api_deep_latest():
    run = _latest("run-traj-*")
    if not run:
        return jsonify(error="No trajectory runs found"), 404
    deep = run / "deep"

    def head_csv(name, n=25):
        p = deep / name
        if p.exists():
            try:
                return pd.read_csv(p).head(n).to_dict(orient="records")
            except Exception:
                return []
        return []

    imgs = []
    for nm in ("timeline_clusters.png", "latent_scatter.png"):
        p = deep / nm
        if p.exists():
            imgs.append(f"/file/{p.relative_to(ROOT)}")

    return jsonify(
        run_dir=str(run.relative_to(ROOT)),
        images=imgs,
        anomalies=head_csv("anomalies.csv", 50),
        hotspots=head_csv("residue_hotspots.csv", 50),
        hybrid=head_csv("hybrid_scores.csv", 50),
        latent=head_csv("latent_clusters_annot.csv", 50),
    )

@app.route("/api/local_ops_top")
def api_local_ops_top():
    run = _latest("run-local-*")
    if not run:
        return jsonify(error="No local-ops runs found"), 404

    # try common filenames
    for cand in ("per_residue.csv", "per_residue_local.csv", "rollup_local.csv"):
        src = run / cand
        if src.exists():
            df = pd.read_csv(src)
            # normalize some common columns
            if "local_score_med" not in df.columns and "local_score_mean" in df.columns:
                df = df.rename(columns={"local_score_mean": "local_score_med"})
            for c in ("rama_disallowed_pct", "%disallowed"):
                if c in df.columns and c != "rama_disallowed_pct":
                    df = df.rename(columns={c: "rama_disallowed_pct"})
                    break
            for c in ("n_frames", "frames"):
                if c in df.columns and c != "n_frames":
                    df = df.rename(columns={c: "n_frames"})
                    break

            cols = [c for c in ("resid","resname","local_score_med","rama_disallowed_pct","n_frames") if c in df.columns]
            if "local_score_med" in df.columns:
                df = df.sort_values("local_score_med", ascending=False)
            out = df[cols].head(20).to_dict(orient="records")
            return jsonify(path=str(src.relative_to(ROOT)), rows=out)

    return jsonify(error=f"No per-residue CSV found under {run}"), 404

# ---------- NEW: Interactive Viewer APIs ----------
@app.route("/api/trajectory/meta")
def api_trajectory_meta():
    """Get trajectory metadata (frames, atoms, residues)."""
    try:
        meta = loader.get_trajectory_meta()
        return jsonify(meta)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/trajectory/frame/<int:frame_idx>")
def api_trajectory_frame(frame_idx):
    """Get atom coordinates for a specific frame."""
    try:
        data = loader.get_frame_coordinates(frame_idx)
        return jsonify(data)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/trajectory/residue_map")
def api_residue_map():
    """Get mapping from atom index to residue number."""
    try:
        resnums = loader.get_residue_map()
        return jsonify({"resnums": resnums})
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/residues")
def api_residues():
    """Get table of all residues with basic info."""
    try:
        residues = loader.get_residue_table()
        return jsonify({"residues": residues})
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/residue/<int:resid>")
def api_residue_details(resid):
    """Get detailed information for a specific residue including hotspot and RMSF data."""
    try:
        details = loader.get_residue_details(resid)
        if details is None:
            return jsonify(error=f"Residue {resid} not found"), 404
        return jsonify(details)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/atom/<int:atom_idx>")
def api_atom_info(atom_idx):
    """Get information about a specific atom."""
    try:
        info = loader.get_atom_info(atom_idx)
        if info is None:
            return jsonify(error=f"Atom {atom_idx} not found"), 404
        return jsonify(info)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/hotspots")
def api_hotspots():
    """Get all residue hotspot scores."""
    try:
        hotspots = loader.load_hotspots()
        return jsonify({"hotspots": hotspots})
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/rmsf")
def api_rmsf():
    """Get RMSF data for all residues."""
    try:
        rmsf = loader.load_rmsf()
        return jsonify({"rmsf": rmsf})
    except Exception as e:
        return jsonify(error=str(e)), 500

# ---------- main ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5051"))
    app.run(host="0.0.0.0", port=port, debug=False)

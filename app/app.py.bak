# app/app.py
import os
import tempfile
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ----- project paths -----
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATIC = ROOT / "static"
TEMPLATES = ROOT / "templates"

# Flask app
app = Flask(__name__)
CORS(app)  # allow calls from your frontend / trame

# ---------- pages ----------
@app.get("/")
def home():
    """Serve the main HTML page."""
    return send_from_directory(str(TEMPLATES), "index.html")

@app.get("/rama")
def rama_html():
    """Serve the latest Ramachandran plot, if present."""
    rama = STATIC / "rama_view.html"
    if not rama.exists():
        return "Ramachandran plot not generated yet. Upload a PDB or run analysis.", 404
    return send_from_directory(str(STATIC), "rama_view.html")

# (optional) static passthrough, if you link to other files
@app.get("/static/<path:p>")
def static_files(p: str):
    return send_from_directory(str(STATIC), p)

# ---------- API ----------
@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.get("/api/top_anomalies")
def api_top_anomalies():
    """Return top anomalies from the rollup CSV."""
    csv_path = DATA / "rollup.csv"
    if not csv_path.exists():
        return jsonify({"error": "rollup.csv not found — upload a PDB or run analysis first"}), 404
    df = pd.read_csv(csv_path)
    top = df.sort_values("mean_if", ascending=False).head(50).to_dict(orient="records")
    return jsonify(top)

# analysis helpers
from analysis.angles import compute_phi_psi
from analysis.anomaly import score_if
from analysis.aggregate import rollup

@app.post("/api/score_pdb")
def score_pdb():
    """
    Upload a .pdb, compute φ/ψ, score residues (IsolationForest),
    roll up per-residue, regenerate the Ramachandran HTML, and return top10.
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "Upload a .pdb file with form field name 'file'"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".pdb"):
        return jsonify({"error": "Not a .pdb file"}), 400

    # ensure folders
    DATA.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)

    # save upload to a temp file under data/
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", dir=DATA) as tf:
        pdb_path = tf.name
        f.save(pdb_path)

    angles_parquet = DATA / "angles.parquet"
    angles_scored  = DATA / "angles_scored.parquet"
    rollup_csv     = DATA / "rollup.csv"
    rama_html      = STATIC / "rama_view.html"

    # --- compute phi/psi ---
    # compute_phi_psi may return a DataFrame OR a path to a parquet/table.
    result = compute_phi_psi(pdb_path)
    if isinstance(result, (str, Path)):
        # helper already wrote parquet/table and returned a path
        angles_parquet = Path(result)
    else:
        # got a DataFrame — write it ourselves
        result.to_parquet(angles_parquet)

    # --- anomaly scoring + rollup ---
    score_if(angles_parquet, angles_scored, contamination=0.05)
    rollup(angles_scored, rollup_csv)

    # --- build Ramachandran HTML ---
    import plotly.express as px
    df_scored = pd.read_parquet(angles_scored)
    fig = px.scatter(
        df_scored,
        x="phi", y="psi", color="if_score",
        range_x=[-180, 180], range_y=[-180, 180],
        height=700,
        title="Ramachandran (φ/ψ) — IsolationForest anomaly score",
    )
    fig.update_layout(xaxis_title="phi (°)", yaxis_title="psi (°)")
    fig.write_html(str(rama_html), include_plotlyjs="cdn")

    # --- response payload ---
    df_roll = pd.read_csv(rollup_csv)
    top10 = (
        df_roll.sort_values("mean_if", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    return jsonify({
        "ok": True,
        "angles_parquet": str(angles_parquet),
        "angles_scored_parquet": str(angles_scored),
        "rollup_csv": str(rollup_csv),
        "rama_html": "/rama",
        "top10": top10,
    })

# ----- entrypoint -----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5051))
    # host=0.0.0.0 lets you open from another device on the LAN too
    app.run(host="0.0.0.0", port=port, debug=False)

import os, sys, tempfile, pandas as pd
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

# Project root (one level above app/)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATIC = ROOT / "static"

# Make sure Python can import analysis/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# analysis helpers
from analysis.angles import compute_phi_psi
from analysis.anomaly import score_if
from analysis.aggregate import rollup

# Tell Flask where the real static folder is
app = Flask(
    __name__,
    static_url_path="/static",
    static_folder=str(STATIC),
)

@app.get("/")
def health():
    return "Ensemble Anomaly Maps API is running. Try /api/top_anomalies"

@app.get("/api/top_anomalies")
def top_anomalies():
    df = pd.read_csv(DATA / "rollup.csv")
    return jsonify(df.sort_values("mean_if", ascending=False).head(10).to_dict(orient="records"))

# (Optional explicit static route—kept for clarity)
@app.get("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC, p)

@app.post("/api/score_pdb")
def score_pdb():
    """
    Upload a .pdb (form field name 'file'), run φ/ψ -> IsolationForest,
    write outputs into ROOT/data and ROOT/static, and return top residues.
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "Upload a .pdb file with form field name 'file'"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".pdb"):
        return jsonify({"error": "Not a .pdb file"}), 400

    DATA.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)

    # save upload into data/tmp
    tmpdir = DATA / "tmp"; tmpdir.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", dir=tmpdir) as tf:
        f.save(tf.name)
        pdb_path = tf.name

    # run pipeline
    angles_path = DATA / "angles.parquet"
    scored_path = DATA / "angles_scored.parquet"
    rollup_path = DATA / "rollup.csv"
    html_path = STATIC / "rama_view.html"

    a = compute_phi_psi(pdb_path, str(angles_path))
    s = score_if(str(angles_path), str(scored_path), contamination=0.05)
    r = rollup(str(scored_path), str(rollup_path))

    # make plot
    import plotly.express as px
    df = pd.read_parquet(scored_path)
    fig = px.scatter(df, x="phi", y="psi", color="if_score",
                     range_x=[-180,180], range_y=[-180,180], height=700,
                     title="Ramachandran (φ/ψ) — IsolationForest score")
    fig.write_html(html_path, include_plotlyjs="cdn")

    # cleanup tmp
    try: os.remove(pdb_path)
    except: pass

    return jsonify({
        "ok": True,
        "angles_parquet": str(angles_path),
        "angles_scored_parquet": str(scored_path),
        "rollup_csv": str(rollup_path),
        "rama_html": f"/static/{html_path.name}",
        "top10": pd.read_csv(rollup_path).sort_values("mean_if", ascending=False).head(10).to_dict(orient="records"),
    })
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5051))
    app.run(host="0.0.0.0", port=port, debug=False)

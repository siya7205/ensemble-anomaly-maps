# app/app.py
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd

# ---------- repo paths ----------
ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
STATIC    = ROOT / "static"
TEMPLATES = ROOT / "templates"
OUTPUTS   = ROOT / "outputs"           # NEW: where we persist results

# analysis helpers (from analysis/*.py)
from analysis.angles import compute_phi_psi
from analysis.anomaly import score_if
from analysis.aggregate import rollup

app = Flask(__name__)
CORS(app)  # allow front-ends (trame or static html) to call the API


# ---------------- PAGES ----------------
@app.get("/")
def home():
    """Serve the simple UI (if present)."""
    return send_from_directory(TEMPLATES, "index.html")


@app.get("/rama")
def rama_html():
    """Serve the latest Ramachandran HTML if present."""
    rama = STATIC / "rama_view.html"
    if not rama.exists():
        return "Ramachandran plot not generated yet. Upload a PDB or run analysis.", 404
    return send_from_directory(STATIC, "rama_view.html")

@app.get("/static/<path:p>")
def static_files(p):
    """Pass-through for static assets (JS, CSS, HTML)."""
    return send_from_directory(STATIC, p)


# ---------------- API ----------------
@app.get("/api/top_anomalies")
def api_top_anomalies():
    """
    Return top residues by mean_if with the fields expected by the frontend:
      resid, mean_if, pct_disallowed, n_frames
    Reads from data/rollup.csv (latest run written there by /api/score_pdb).
    """
    csv_path = DATA_DIR / "rollup.csv"
    if not csv_path.exists():
        return jsonify({"error": "rollup.csv not found — upload a PDB or run analysis first"}), 404

    df = pd.read_csv(csv_path)
    want_cols = ["resid", "mean_if", "pct_disallowed", "n_frames"]
    for c in want_cols:
        if c not in df.columns:
            df[c] = 0

    df["mean_if"] = pd.to_numeric(df["mean_if"], errors="coerce").fillna(0.0)
    df["pct_disallowed"] = pd.to_numeric(df["pct_disallowed"], errors="coerce").fillna(0.0)
    df["n_frames"] = pd.to_numeric(df["n_frames"], errors="coerce").fillna(0).astype(int)

    top = (
        df.sort_values("mean_if", ascending=False)
          .head(50)[want_cols]
          .to_dict(orient="records")
    )
    return jsonify(top)

# ---------------- Local-ops API ----------------
import glob

@app.get("/api/local_ops_top")
def api_local_ops_top():
    """
    Return top residues from the latest analysis/run_local_ops.py output.
    Looks for outputs/run-local-*/local_ops_summary.csv
    """
    paths = sorted(glob.glob(str(ROOT / "outputs" / "run-local-*" / "local_ops_summary.csv")))
    if not paths:
        return jsonify({"error": "No local_ops_summary.csv found. Run: python -m analysis.run_local_ops <pdb>"}), 404

    csv_path = Path(paths[-1])
    df = pd.read_csv(csv_path)

    # Map columns -> concise names expected by the UI
    if "local_score_med" not in df.columns:
        return jsonify({"error": f"Unexpected columns in {csv_path.name}"}), 500

    out = (
        df.rename(columns={
            "local_score_med": "score_med",
            "rama_disallowed_pct": "pct_disallowed",
            "n_frames": "frames",
        })[["resid", "resname", "score_med", "pct_disallowed", "frames"]]
          .sort_values("score_med", ascending=False)
          .head(50)
          .to_dict(orient="records")
    )
    return jsonify(out)
@app.get("/api/points")
def api_points():
    """
    Points endpoint for the 3D viewer.
    Tries outputs/latest/points.json, else falls back to building from data/rollup.csv.
    Schema: [{label, score, x, y, z}, ...]  (x/y/z are 0 unless you add coords)
    """
    latest_points = OUTPUTS / "latest" / "points.json"
    if latest_points.exists():
        try:
            return send_from_directory(latest_points.parent, latest_points.name)
        except Exception:
            pass

    # Fallback: build minimal points from rollup
    csv_path = DATA_DIR / "rollup.csv"
    if not csv_path.exists():
        return jsonify([])

    df = pd.read_csv(csv_path)
    # Minimal schema without 3D coords
    out = []
    for _, r in df.iterrows():
        out.append({
            "label": f"Res {int(r.get('resid', -1))}",
            "score": float(r.get("mean_if", 0.0)),
            "x": 0.0, "y": 0.0, "z": 0.0,
        })
    return jsonify(out)


@app.post("/api/score_pdb")
def api_score_pdb():
    """
    Accept an uploaded .pdb (form field 'file'), compute φ/ψ, score with IsolationForest,
    roll up per-residue, regenerate Ramachandran HTML, persist artifacts to outputs/, and return a summary.
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "Upload a .pdb file with form field name 'file'"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".pdb"):
        return jsonify({"error": "Not a .pdb file"}), 400

    # Ensure dirs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Timestamped run dir + a rolling 'latest' dir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUTS / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = OUTPUTS / "latest"
    if latest_dir.exists() and latest_dir.is_dir():
        # Clear latest to avoid stale files
        for p in latest_dir.iterdir():
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p)
    else:
        latest_dir.mkdir(parents=True, exist_ok=True)

    # Save upload to a temp pdb (also copy into run_dir)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", dir=DATA_DIR) as tf:
        pdb_path = Path(tf.name)
        f.save(pdb_path)
    shutil.copy2(pdb_path, run_dir / f.filename)

    # Canonical "latest" artifact paths in /data and /static
    angles_parquet        = DATA_DIR / "angles.parquet"
    angles_scored_parquet = DATA_DIR / "angles_scored.parquet"
    rollup_csv            = DATA_DIR / "rollup.csv"
    rama_html             = STATIC / "rama_view.html"

    # --- 1) φ/ψ angles ---
    angles_obj = compute_phi_psi(str(pdb_path))  # some versions expect str
    if isinstance(angles_obj, pd.DataFrame):
        angles_obj.to_parquet(angles_parquet)
    elif isinstance(angles_obj, (str, Path)):
        src = Path(angles_obj)
        if src.exists():
            if src.resolve() != angles_parquet.resolve():
                df_tmp = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
                df_tmp.to_parquet(angles_parquet)
        else:
            return jsonify({"error": f"compute_phi_psi returned non-existent path: {src}"}), 500
    else:
        return jsonify({"error": "compute_phi_psi returned unexpected type"}), 500

    # --- 2) IsolationForest scoring ---
    score_if(str(angles_parquet), str(angles_scored_parquet), contamination=0.05)

    # --- 3) Per-residue rollup ---
    rollup(str(angles_scored_parquet), str(rollup_csv))

    # --- 4) Ramachandran HTML (colored by if_score) ---
    try:
        import plotly.express as px
        df_scored = pd.read_parquet(angles_scored_parquet)
        fig = px.scatter(
            df_scored, x="phi", y="psi", color="if_score",
            range_x=[-180, 180], range_y=[-180, 180], height=700,
            title="Ramachandran (φ/ψ) — colored by IsolationForest anomaly score",
        )
        fig.update_layout(xaxis_title="phi (°)", yaxis_title="psi (°)")
        fig.write_html(rama_html, include_plotlyjs="cdn")
    except Exception as e:
        print("Plot generation failed:", e)

    # --- 5) Build 3D viewer points.json (minimal if no coords available) ---
    points_json = []
    try:
        df_roll = pd.read_csv(rollup_csv)
        for _, r in df_roll.iterrows():
            points_json.append({
                "label": f"Res {int(r.get('resid', -1))}",
                "score": float(pd.to_numeric(r.get("mean_if", 0.0), errors="coerce") or 0.0),
                "x": 0.0, "y": 0.0, "z": 0.0,  # replace with real coords if/when you have them
            })
    except Exception:
        pass

    # --- 6) PERSIST: copy artifacts to run_dir and to outputs/latest ---
    # Save canonical (latest) copies first (they already exist in data/static)
    # Then copy to run_dir and latest_dir for archival / quick access.
    def safe_copy(src: Path, dst: Path):
        if src.exists():
            if not dst.parent.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Save points.json + summary.json
    import json
    (run_dir / "points.json").write_text(json.dumps(points_json, indent=2))
    (latest_dir / "points.json").write_text(json.dumps(points_json))

    # Copy parquet/csv/html into the run dir
    safe_copy(angles_parquet,        run_dir / "angles.parquet")
    safe_copy(angles_scored_parquet, run_dir / "angles_scored.parquet")
    safe_copy(rollup_csv,            run_dir / "rollup.csv")
    safe_copy(rama_html,             run_dir / "rama_view.html")

    # Also mirror into outputs/latest/
    safe_copy(angles_parquet,        latest_dir / "angles.parquet")
    safe_copy(angles_scored_parquet, latest_dir / "angles_scored.parquet")
    safe_copy(rollup_csv,            latest_dir / "rollup.csv")
    safe_copy(rama_html,             latest_dir / "rama_view.html")

    # Write a summary.json for the run
    summary = {
        "timestamp": ts,
        "pdb_filename": f.filename,
        "run_dir": str(run_dir),
        "artifacts": {
            "angles_parquet":        str(run_dir / "angles.parquet"),
            "angles_scored_parquet": str(run_dir / "angles_scored.parquet"),
            "rollup_csv":            str(run_dir / "rollup.csv"),
            "points_json":           str(run_dir / "points.json"),
            "rama_html":             str(run_dir / "rama_view.html"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (latest_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- 7) Prepare top10 for response ---
    df_roll = pd.read_csv(rollup_csv) if rollup_csv.exists() else pd.DataFrame()
    if not df_roll.empty:
        for c in ["mean_if", "pct_disallowed"]:
            if c in df_roll:
                df_roll[c] = pd.to_numeric(df_roll[c], errors="coerce").fillna(0.0)
        if "n_frames" in df_roll:
            df_roll["n_frames"] = pd.to_numeric(df_roll["n_frames"], errors="coerce").fillna(0).astype(int)
        top10 = (
            df_roll.sort_values("mean_if", ascending=False)
                   .head(10)[["resid", "mean_if", "pct_disallowed", "n_frames"]]
                   .to_dict(orient="records")
        )
    else:
        top10 = []

    return jsonify({
        "ok": True,
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "angles_parquet": str(angles_parquet),
        "angles_scored_parquet": str(angles_scored_parquet),
        "rollup_csv": str(rollup_csv),
        "rama_html": "/rama",
        "points_json": "/api/points",
        "top10": top10,
    })


if __name__ == "__main__":
    # Default 5051 unless you set PORT in your shell.
    port = int(os.environ.get("PORT", 5051))
    app.run(host="0.0.0.0", port=port, debug=False)

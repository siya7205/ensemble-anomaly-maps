# app/app.py
import os, json, pathlib
from flask import Flask, jsonify, render_template, send_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data"
STATIC = ROOT / "static"
TEMPL = ROOT / "templates"

app = Flask(__name__, static_folder=str(STATIC), template_folder=str(TEMPL))

# ---------- helpers ----------
def _latest_dir(glob_pat):
    matches = sorted(OUT.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ---------- pages ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/local")
def page_local():
    return render_template("local.html")

@app.route("/deep")
def page_deep():
    return render_template("deep.html")

@app.route("/three-d")
def page_three_d():
    # the page fetches this file via /file/...
    return render_template("three_d.html", pdb_rel="data/multi_model_anomaly.pdb")

# ---------- api ----------
@app.route("/api/local_ops_top")
def api_local_ops_top():
    """Return top residues from latest outputs/run-local-*/per_residue.csv (or local_ops_summary.csv)."""
    import pandas as pd
    run = _latest_dir("run-local-*")
    if not run:
        return jsonify(error="No local-ops runs found"), 404

    candidates = [run / "per_residue.csv", run / "local_ops_summary.csv", run / "per_residue_local.csv"]
    src = next((c for c in candidates if c.exists()), None)
    if not src:
        return jsonify(error=f"No per-residue CSV in {run.name}"), 404

    df = pd.read_csv(src)

    # Normalize columns
    rename = {}
    for a, b in (("local_score_mean", "local_score_med"), ("local_score_median", "local_score_med")):
        if a in df.columns and "local_score_med" not in df.columns:
            rename[a] = "local_score_med"
    for a in ("rama_disallowed_pct", "%disallowed"):
        if a in df.columns:
            rename[a] = "rama_disallowed_pct"
            break
    for a in ("n_frames", "frames", "N_frames"):
        if a in df.columns:
            rename[a] = "n_frames"
            break
    df = df.rename(columns=rename)

    need = {"resid", "resname", "local_score_med"}
    if not need.issubset(set(df.columns)):
        # Derive a quick summary from per-frame table if needed
        if "local_score" in df.columns and {"resid", "resname"}.issubset(df.columns):
            g = df.groupby(["resid", "resname"])["local_score"].median().rename("local_score_med")
            if "rama_allowed" in df.columns:
                pct = (1 - df.groupby(["resid","resname"])["rama_allowed"].mean()) * 100
                pct = pct.rename("rama_disallowed_pct")
            else:
                pct = g * 0
            nf = df.groupby(["resid","resname"]).size().rename("n_frames")
            df = g.reset_index().merge(pct.reset_index(), on=["resid","resname"]).merge(nf.reset_index(), on=["resid","resname"])
        else:
            return jsonify(error="per_residue-like columns not found"), 500

    df = df.sort_values("local_score_med", ascending=False).head(30)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "resid": int(r.get("resid", -1)),
            "resname": str(r.get("resname", "UNK")),
            "local_score_med": _safe_float(r.get("local_score_med", 0)),
            "rama_disallowed_pct": _safe_float(r.get("rama_disallowed_pct", 0)),
            "n_frames": int(r.get("n_frames", 0)),
        })
    return jsonify(path=str(src.relative_to(ROOT)), rows=rows)

@app.route("/api/deep_latest")
def api_deep_latest():
    """Summarize latest deep run under outputs/run-traj-*/deep/."""
    import pandas as pd
    run = _latest_dir("run-traj-*")
    if not run:
        return jsonify(error="No trajectory runs yet"), 404
    deep = run / "deep"
    if not deep.exists():
        return jsonify(error="No deep/ folder in latest trajectory run"), 404

    def read_csv_rel(name, limit=None):
        p = deep / name
        if p.exists():
            df = pd.read_csv(p)
            return df if limit is None else df.head(limit)
        return None

    anomalies = read_csv_rel("anomalies.csv", 20)
    hotspots = read_csv_rel("residue_hotspots.csv", 30)
    hybrid = read_csv_rel("hybrid_scores.csv", 30)
    latent = read_csv_rel("latent_clusters_annot.csv", 30)

    imgs = []
    for name in ("timeline_clusters.png", "latent_scatter.png", "deep_vs_shallow.png", "recon_over_time.png"):
        p = deep / name
        if p.exists():
            imgs.append(f"/file/{p.relative_to(ROOT)}")

    return jsonify(
        run_dir=str(run.relative_to(ROOT)),
        images=imgs,
        anomalies=(anomalies or pd.DataFrame()).to_dict(orient="records"),
        hotspots=(hotspots or pd.DataFrame()).to_dict(orient="records"),
        hybrid_head=(hybrid or pd.DataFrame()).to_dict(orient="records"),
        latent_head=(latent or pd.DataFrame()).to_dict(orient="records"),
    )

@app.route("/rama")
def rama():
    """Serve latest Ramachandran HTML."""
    latest = OUT / "latest" / "rama_view.html"
    if latest.exists():
        return send_file(str(latest), mimetype="text/html")
    run = _latest_dir("run-*")
    if run and (run / "rama_view.html").exists():
        return send_file(str(run / "rama_view.html"), mimetype="text/html")
    return "<html><body><p>No Ramachandran plot available yet.</p></body></html>"

@app.route("/file/<path:rel>")
def file_serve(rel):
    """Read-only file server for anything under project root (used by images/PDB)."""
    p = ROOT / rel
    if not p.exists():
        return "Not found", 404
    mime = "text/plain"
    suf = p.suffix.lower()
    if suf in {".png", ".jpg", ".jpeg", ".gif"}:
        mime = "image/png"
    elif suf in {".html", ".htm"}:
        mime = "text/html"
    elif suf == ".csv":
        mime = "text/csv"
    elif suf in {".pdb", ".ent", ".cif"}:
        mime = "chemical/x-pdb"
    return send_file(str(p), mimetype=mime)

# ---------- main ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5051"))
    app.run(host="0.0.0.0", port=port, debug=False)

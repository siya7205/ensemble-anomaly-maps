import os, pandas as pd
from flask import Flask, jsonify, send_from_directory
app = Flask(__name__)
@app.get("/api/top_anomalies")
def top_anomalies():
    df = pd.read_csv("data/rollup.csv")
    return jsonify(df.sort_values("mean_if", ascending=False).head(10).to_dict(orient="records"))
@app.get("/static/<path:p>")
def static_files(p): return send_from_directory("static", p)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)

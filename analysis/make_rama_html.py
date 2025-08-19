import os, sys, pandas as pd, plotly.express as px

IN  = "data/angles_scored.parquet"
OUT = os.path.join("static", "rama_view.html")

def main():
    if not os.path.exists(IN):
        print("Missing", IN, "— run analysis/run_ensemble.py first")
        sys.exit(1)
    df = pd.read_parquet(IN)
    fig = px.scatter(
        df, x="phi", y="psi", color="if_score",
        range_x=[-180,180], range_y=[-180,180], height=700,
        title="Ramachandran (φ/ψ) — colored by IsolationForest anomaly score"
    )
    fig.update_layout(xaxis_title="phi (°)", yaxis_title="psi (°)")
    os.makedirs("static", exist_ok=True)
    fig.write_html(OUT, include_plotlyjs="cdn")
    print("Open in browser: http://127.0.0.1:5050/static/rama_view.html")

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_frame_scores(csv_file, out_file=None):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Check columns
    if not {"frame", "score"}.issubset(df.columns):
        raise ValueError("CSV must have columns: frame, score")
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["frame"], df["score"], color="blue", lw=1.5)
    plt.title(f"Anomaly Scores: {csv_file}")
    plt.xlabel("Frame Index")
    plt.ylabel("Anomaly Score")
    plt.grid(alpha=0.3)
    
    # Save or show
    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {out_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot anomaly scores from frame_scores.csv")
    parser.add_argument("csv_file", help="Path to frame_scores.csv")
    parser.add_argument("--out", help="Optional output image file (PNG, JPG, etc.)")
    args = parser.parse_args()

    plot_frame_scores(args.csv_file, args.out)

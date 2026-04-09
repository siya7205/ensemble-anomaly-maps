"""
Shared thesis-grade matplotlib style settings.

All figures in this pipeline import apply_thesis_style() before plotting.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colour palette: neutral, accessible, print-safe
PALETTE = {
    "tica": "#2c7bb6",
    "pca": "#d7191c",
    "raw": "#74add1",
    "mode0": "#1a1a1a",
    "mode1": "#4d4d4d",
    "mode2": "#808080",
    "mode3": "#bfbfbf",
    "anomaly_high": "#d73027",
    "anomaly_low": "#4575b4",
    "neutral": "#636363",
    "highlight": "#e6550d",
    "accent": "#756bb1",
}

FIGURE_WIDTH_SINGLE = 3.5   # inches (single column)
FIGURE_WIDTH_DOUBLE = 7.0   # inches (double column / full width)
FIGURE_HEIGHT_BASE  = 3.0


def apply_thesis_style() -> None:
    """Apply consistent thesis-grade matplotlib rcParams."""
    mpl.rcParams.update({
        # font
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Times New Roman", "Georgia"],
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        # lines & markers
        "lines.linewidth":    1.2,
        "lines.markersize":   4,
        # axes
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.5,
        "grid.linestyle":     ":",
        # ticks
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        # figure / saving
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        # layout
        "figure.constrained_layout.use": True,
    })


def save_figure(fig: plt.Figure, export_dir: str, name: str) -> None:
    """Save figure as both PNG (300 dpi) and PDF."""
    import os
    os.makedirs(export_dir, exist_ok=True)
    png_path = os.path.join(export_dir, f"{name}.png")
    pdf_path = os.path.join(export_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"  Saved  {png_path}")
    print(f"  Saved  {pdf_path}")

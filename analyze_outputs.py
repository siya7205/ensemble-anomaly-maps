#!/usr/bin/env python3
"""
Analyze all run outputs, write summaries/plots to outputs/summary/,
and auto-commit & push those artifacts to your GitHub repo.

Requires: pandas, matplotlib
Run from repo root:  conda activate mdcapstone && python analyze_outputs.py
"""
import os, glob, json, subprocess, sys, datetime as dt
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "outputs"
SUMMARY_DIR = RUNS_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

def find_runs():
    # matches e.g. outputs/run-20250822-141822
    return sorted([p for p in RUNS_DIR.glob("run-*") if p.is_dir()])

def read_rollup_csv(run_dir: Path) -> pd.DataFrame | None:
    csv_path = run_dir / "rollup.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["run_id"] = run_dir.name
    try:
        # parse timestamp from folder name if present: run-YYYYMMDD-HHMMSS
        ts = run_dir.name.split("run-")[-1]
        dt_obj = dt.datetime.strptime(ts, "%Y%m%d-%H%M%S")
    except Exception:
        dt_obj = dt.datetime.fromtimestamp(run_dir.stat().st_mtime)
    df["run_ts"] = dt_obj
    return df

def combine_runs():
    frames = []
    for run in find_runs():
        df = read_rollup_csv(run)
        if df is not None:
            frames.append(df)
    if not frames:
        print("⚠️ No run rollup.csv files found in outputs/run-*/")
        sys.exit(1)
    all_df = pd.concat(frames, ignore_index=True)
    all_csv = SUMMARY_DIR / "all_runs.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"✅ Saved combined CSV -> {all_csv}")
    return all_df

def latest_run_dir():
    runs = find_runs()
    if not runs:
        return None
    # sort by mtime
    return sorted(runs, key=lambda p: p.stat().st_mtime)[-1]

def save_latest_rollup(all_df: pd.DataFrame):
    lr = latest_run_dir()
    if lr is None:
        return
    latest_df = all_df[all_df["run_id"] == lr.name].copy()
    out_csv = SUMMARY_DIR / "latest_rollup.csv"
    latest_df.to_csv(out_csv, index=False)
    print(f"✅ Saved latest run rollup -> {out_csv}")
    # also copy per-run html/parquet if they exist (optional view convenience)
    for leaf in ("rama_view.html", "angles.parquet", "angles_scored.parquet", "points.json"):
        src = lr / leaf
        if src.exists():
            dst = SUMMARY_DIR / f"latest_{leaf}"
            try:
                # small files only; for parquet/html/json this is fine
                dst.write_bytes(src.read_bytes())
            except Exception as e:
                print(f"⚠️ Skipped copying {leaf}: {e}")

def make_plots(all_df: pd.DataFrame):
    png = SUMMARY_DIR / "score_distributions.png"
    plt.figure()
    # distribution of mean_if across runs
    try:
        all_df.boxplot(column="mean_if", by="run_id", rot=90)
        plt.title("mean_if by run")
        plt.suptitle("")
        plt.xlabel("run_id")
        plt.ylabel("mean_if")
        plt.tight_layout()
        plt.savefig(png, dpi=150)
        plt.close()
        print(f"✅ Saved comparison plot -> {png}")
    except Exception as e:
        print(f"⚠️ Plot failed: {e}")

def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()

def run(cmd, cwd=REPO_ROOT) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output

def git_user_configured() -> bool:
    code, name = run(["git", "config", "--get", "user.name"])
    code2, email = run(["git", "config", "--get", "user.email"])
    return (name.strip() != "") and (email.strip() != "")

def git_stage_and_commit():
    if not is_git_repo(REPO_ROOT):
        print("ℹ️ Not a git repository; skipping auto-commit.")
        return
    paths_to_add = [
        "outputs/summary/all_runs.csv",
        "outputs/summary/latest_rollup.csv",
        "outputs/summary/score_distributions.png",
        # optional convenience copies:
        "outputs/summary/latest_rama_view.html",
        "outputs/summary/latest_points.json",
        "outputs/summary/latest_angles.parquet",
        "outputs/summary/latest_angles_scored.parquet",
    ]
    # stage only those that exist
    paths_to_add = [p for p in paths_to_add if (REPO_ROOT / p).exists()]
    if not paths_to_add:
        print("ℹ️ Nothing new to commit.")
        return

    # ensure user.name/email exist (don’t set them for you; just warn)
    if not git_user_configured():
        print("⚠️ git user.name/email not set; commit may be rejected.\n"
              "   Set once with:\n"
              '   git config --global user.name "Your Name"\n'
              '   git config --global user.email "you@example.com"')

    code, out = run(["git", "add"] + paths_to_add)
    if code != 0:
        print("⚠️ git add failed:\n" + out)
        return

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"Analyze outputs: summaries & plots ({stamp})"
    code, out = run(["git", "commit", "-m", msg])
    if code != 0:
        if "nothing to commit" in out.lower():
            print("ℹ️ No changes to commit.")
            return
        print("⚠️ git commit failed:\n" + out)
        return

    print("✅ Committed analysis artifacts.")
    code, out = run(["git", "push"])
    if code != 0:
        print("⚠️ git push failed:\n" + out)
    else:
        print("✅ Pushed to remote.")

def main():
    # 1) Combine everything
    all_df = combine_runs()
    # 2) Save latest-only slice
    save_latest_rollup(all_df)
    # 3) Make a quick comparison plot
    make_plots(all_df)
    # 4) Auto-commit & push the new/updated files
    git_stage_and_commit()

if __name__ == "__main__":
    main()

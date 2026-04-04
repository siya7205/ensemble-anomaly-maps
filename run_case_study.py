#!/usr/bin/env python3
"""
Run the full ensemble-anomaly-maps pipeline on the case study trajectory.

Case study : case_study_10ns_1001frames
Input      : data/case_study_10ns_1001frames/input/md.{gro,tpr,xtc}

Outputs follow the repo's standard layout:
    artifacts/case_study_10ns_1001frames/  ← NumPy artifacts
    results/case_study_10ns_1001frames/    ← CSV / JSON scores
    exports/case_study_10ns_1001frames/    ← ASVS-compatible JSON
    outputs/case_study_10ns_1001frames/    ← run metadata & summary

Usage
-----
    python run_case_study.py
    python run_case_study.py --lag_tica 10 --n_clusters 20 --seed 42
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Case study constants ─────────────────────────────────────────────────────
CASE_STUDY_ID   = "case_study_10ns_1001frames"
INPUT_DIR       = Path("data") / CASE_STUDY_ID / "input"
TOPOLOGY_PATH   = INPUT_DIR / "md.gro"
TRAJECTORY_PATH = INPUT_DIR / "md.xtc"


# ─── ASVS export helper ───────────────────────────────────────────────────────

def _export_for_asvs(
    topology_path: Path,
    trajectory_path: Path,
    results_dir: Path,
    exports_dir: Path,
) -> None:
    """Build ASVS-compatible JSON files from core pipeline results.

    This calls the existing export functions from tools/export_for_asvs.py
    directly, so no subprocess spawning is needed.  The one adaptation is
    column-name normalisation: run_all_proteins.py writes *score_dynamic*
    but the exporter expects *score* (normalised to [0, 1]).
    """
    import pandas as pd
    from tools.export_for_asvs import (
        create_anomaly_residue_json,
        create_hotspots_residue_json,
        create_rmsf_json,
        create_tica_importance_json,
        get_n_residues,
    )

    exports_dir.mkdir(parents=True, exist_ok=True)

    # ── Residue count ──────────────────────────────────────────────────
    n_residues = get_n_residues(str(topology_path))
    log.info("[export] Detected %d residues", n_residues)

    # ── Frame scores ───────────────────────────────────────────────────
    csv_path = results_dir / "frame_scores_dynamic.csv"
    frame_scores_df = None
    if csv_path.exists():
        frame_scores_df = pd.read_csv(csv_path)
        # Normalise column name: the pipeline writes 'score_dynamic' (scaled to
        # [0, 100]); the exporter weighs residues by 'score' expected in [0, 1].
        if "score_dynamic" in frame_scores_df.columns and "score" not in frame_scores_df.columns:
            frame_scores_df = frame_scores_df.copy()
            frame_scores_df["score"] = frame_scores_df["score_dynamic"] / 100.0
        log.info("[export] Loaded %d frame scores", len(frame_scores_df))
    else:
        log.warning("[export] frame_scores_dynamic.csv not found — exporting placeholders")

    # ── Residue scores ─────────────────────────────────────────────────
    residue_scores: dict = {}
    json_path = results_dir / "residue_scores_dynamic.json"
    if json_path.exists():
        with open(json_path) as fh:
            raw = json.load(fh)
        # Keys may be MDTraj residue labels (e.g. "ALA1"); re-index as
        # consecutive integers (str) for the ASVS format.
        residue_scores["dynamic"] = {str(i): v for i, (_, v) in enumerate(raw.items())}

    # ── Write ASVS JSON files ─────────────────────────────────────────
    hotspots_data = create_hotspots_residue_json(frame_scores_df, residue_scores, n_residues)
    with open(exports_dir / "hotspots_residue.json", "w") as fh:
        json.dump(hotspots_data, fh)
    log.info("[export] hotspots_residue.json  (%d frames)", len(hotspots_data))

    anomaly_data = create_anomaly_residue_json(frame_scores_df, residue_scores, n_residues)
    with open(exports_dir / "anomaly_residue.json", "w") as fh:
        json.dump(anomaly_data, fh)
    log.info("[export] anomaly_residue.json")

    rmsf_data = create_rmsf_json(residue_scores)
    with open(exports_dir / "rmsf_residue.json", "w") as fh:
        json.dump(rmsf_data, fh)
    log.info("[export] rmsf_residue.json")

    tica_data = create_tica_importance_json(residue_scores)
    with open(exports_dir / "tica_importance.json", "w") as fh:
        json.dump(tica_data, fh)
    log.info("[export] tica_importance.json")

    # ── Copy topology / trajectory into the exports directory ─────────
    shutil.copy2(topology_path, exports_dir / "topology.gro")
    shutil.copy2(trajectory_path, exports_dir / "trajectory.xtc")
    log.info("[export] Copied topology.gro and trajectory.xtc")


# ─── Metadata / summary helpers ───────────────────────────────────────────────

def _write_metadata(
    output_dir: Path,
    topology_path: Path,
    trajectory_path: Path,
    artifacts_dir: Path,
    results_dir: Path,
    exports_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Write run_metadata.json and pipeline_summary.txt to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    params = {
        "stride":      args.stride,
        "lag_tica":    args.lag_tica,
        "dim_tica":    args.dim_tica,
        "n_clusters":  args.n_clusters,
        "lag_msm":     args.lag_msm,
        "k_neighbors": args.k_neighbors,
        "window":      args.window,
        "seed":        args.seed,
    }

    meta = {
        "case_study":    CASE_STUDY_ID,
        "run_timestamp": timestamp,
        "inputs": {
            "topology":   str(topology_path),
            "trajectory": str(trajectory_path),
        },
        "parameters": params,
        "outputs": {
            "artifacts": str(artifacts_dir),
            "results":   str(results_dir),
            "exports":   str(exports_dir),
            "metadata":  str(output_dir),
        },
    }

    with open(output_dir / "run_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    summary_lines = [
        f"Case study : {CASE_STUDY_ID}",
        f"Run time   : {timestamp}",
        "",
        "── Artifacts (NumPy) ──────────────────────────────────────",
        f"  {artifacts_dir}/tica_coords.npy   (tICA coordinates, T×dim)",
        f"  {artifacts_dir}/dtraj.npy          (discrete state trajectory, T)",
        f"  {artifacts_dir}/P.npy              (MSM transition matrix, S×S)",
        f"  {artifacts_dir}/pi.npy             (stationary distribution, S)",
        "",
        "── Results (CSV / JSON) ────────────────────────────────────",
        f"  {results_dir}/frame_scores_dynamic.csv",
        f"  {results_dir}/residue_scores_dynamic.json",
        "",
        "── Exports (ASVS-compatible JSON) ─────────────────────────",
        f"  {exports_dir}/hotspots_residue.json   ← visualizer",
        f"  {exports_dir}/anomaly_residue.json    ← visualizer",
        f"  {exports_dir}/rmsf_residue.json       ← visualizer",
        f"  {exports_dir}/tica_importance.json    ← visualizer",
        f"  {exports_dir}/topology.gro",
        f"  {exports_dir}/trajectory.xtc",
        "",
        "── Parameters ──────────────────────────────────────────────",
    ]
    for k, v in params.items():
        summary_lines.append(f"  {k}: {v}")

    with open(output_dir / "pipeline_summary.txt", "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    log.info("Summary → %s", output_dir / "pipeline_summary.txt")
    log.info("Metadata → %s", output_dir / "run_metadata.json")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Run the full pipeline on {CASE_STUDY_ID}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stride",      type=int, default=1,  help="Trajectory stride")
    parser.add_argument("--lag_tica",    type=int, default=10, help="tICA lag time (frames)")
    parser.add_argument("--dim_tica",    type=int, default=5,  help="Number of tICA components")
    parser.add_argument("--n_clusters",  type=int, default=20, help="KMeans clusters")
    parser.add_argument("--lag_msm",     type=int, default=10, help="MSM lag time (frames)")
    parser.add_argument("--k_neighbors", type=int, default=10, help="k for local-density signal")
    parser.add_argument("--window",      type=int, default=5,  help="Smoothing window size")
    parser.add_argument("--seed",        type=int, default=42, help="Global random seed")
    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────
    for p in (TOPOLOGY_PATH, TRAJECTORY_PATH):
        if not p.exists():
            log.error("Input not found: %s", p)
            return 1

    # ── Resolve output paths ───────────────────────────────────────────
    # run_pipeline() appends pdb_id internally, so we pass the parent dirs.
    artifacts_base = Path("artifacts")
    results_base   = Path("results")
    artifacts_dir  = artifacts_base / CASE_STUDY_ID
    results_dir    = results_base   / CASE_STUDY_ID
    exports_dir    = Path("exports") / CASE_STUDY_ID
    output_dir     = Path("outputs") / CASE_STUDY_ID

    log.info("=" * 60)
    log.info("Case Study Pipeline — %s", CASE_STUDY_ID)
    log.info("=" * 60)
    log.info("Topology   : %s", TOPOLOGY_PATH)
    log.info("Trajectory : %s", TRAJECTORY_PATH)
    log.info("")

    # ── Steps 1–5: Core ML pipeline ───────────────────────────────────
    from run_all_proteins import run_pipeline

    ok = run_pipeline(
        pdb_id=CASE_STUDY_ID,
        topology_path=TOPOLOGY_PATH,
        trajectory_path=TRAJECTORY_PATH,
        artifacts_dir=artifacts_base,   # run_pipeline appends CASE_STUDY_ID
        results_dir=results_base,        # run_pipeline appends CASE_STUDY_ID
        stride=args.stride,
        lag_tica=args.lag_tica,
        dim_tica=args.dim_tica,
        n_clusters=args.n_clusters,
        lag_msm=args.lag_msm,
        k_neighbors=args.k_neighbors,
        window=args.window,
        seed=args.seed,
    )

    if not ok:
        log.error("Core pipeline failed — aborting.")
        return 1

    # ── Step 6: ASVS export ────────────────────────────────────────────
    log.info("")
    log.info("── ASVS Export")
    try:
        _export_for_asvs(
            topology_path=TOPOLOGY_PATH,
            trajectory_path=TRAJECTORY_PATH,
            results_dir=results_dir,
            exports_dir=exports_dir,
        )
    except Exception as exc:
        log.warning("ASVS export encountered an error (non-fatal): %s", exc)

    # ── Step 7: Write run metadata & summary ──────────────────────────
    log.info("")
    log.info("── Writing run metadata")
    _write_metadata(
        output_dir=output_dir,
        topology_path=TOPOLOGY_PATH,
        trajectory_path=TRAJECTORY_PATH,
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        exports_dir=exports_dir,
        args=args,
    )

    log.info("")
    log.info("=" * 60)
    log.info("✓ Case study pipeline complete.")
    log.info("  Artifacts : %s/", artifacts_dir)
    log.info("  Results   : %s/", results_dir)
    log.info("  Exports   : %s/", exports_dir)
    log.info("  Summary   : %s/", output_dir)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

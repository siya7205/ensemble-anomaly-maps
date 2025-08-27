# analysis/run_local_ops.py
import pathlib
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import mdtraj as md

# use the functions you already tested interactively
from analysis.local_ops import compute_local_ops, summarize_per_residue

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _compute_residue_sasa(traj: md.Trajectory) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns: frame, resid, sasa
    (Å^2). Uses MDTraj Shrake-Rupley in residue mode.
    """
    # per-res SASA, shape = (n_frames, n_residues), in nm^2 -> convert to Å^2 by *100
    sasa_nm2 = md.shrake_rupley(traj, mode="residue")  # (F, R)
    sasa_A2 = sasa_nm2 * 100.0

    frames = np.arange(traj.n_frames, dtype=int)
    resids = [res.index for res in traj.topology.residues]
    # build tidy table
    records = []
    for f in frames:
        vals = sasa_A2[f, :]
        for j, resid in enumerate(resids):
            records.append((f, resid, float(vals[j])))

    return pd.DataFrame(records, columns=["frame", "resid", "sasa"])


def _attach_sasa_and_norm(df: pd.DataFrame, traj: md.Trajectory) -> pd.DataFrame:
    """
    Ensure df has 'sasa' and 'sasa_norm'.
    - If 'sasa' missing, compute it and merge (frame,resid).
    - Compute 'sasa_norm' per residue by scaling each residue's
      median SASA by the 95th percentile of residue medians.
    """
    out = df.copy()

    if "sasa" not in out.columns:
        try:
            sasa_df = _compute_residue_sasa(traj)  # frame,resid,sasa
            out = out.merge(sasa_df, on=["frame", "resid"], how="left")
        except Exception:
            # if SASA fails for any reason, just fill zeros to keep pipeline alive
            out["sasa"] = 0.0

    # make normalized SASA (residue-level median scaled by P95 of medians)
    med = (
        out.groupby(["resid", "resname"], as_index=False)["sasa"]
           .median()
           .rename(columns={"sasa": "sasa_median"})
    )
    if len(med):
        p95 = float(np.percentile(med["sasa_median"].values, 95))
        if p95 <= 1e-8:
            p95 = 1.0
        med["sasa_norm"] = (med["sasa_median"] / p95).clip(0.0, 1.5)
        out = out.merge(med[["resid", "resname", "sasa_norm"]], on=["resid", "resname"], how="left")
    else:
        out["sasa_norm"] = 0.0

    out["sasa_norm"] = out["sasa_norm"].fillna(0.0)
    return out


def main(pdb_path: str, out_dir: Optional[str]) -> None:
    pdb_path = pathlib.Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    # create run dir
    run_dir = (
        _ensure_dir(pathlib.Path(out_dir))
        if out_dir else
        _ensure_dir(OUT / f"run-local-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    )

    print(f"[local_ops] loading {pdb_path.name}")
    traj = md.load(str(pdb_path))

    # 1) per-frame, per-residue local operators
    df = compute_local_ops(traj)

    # 2) ensure SASA + normalized SASA exist
    df = _attach_sasa_and_norm(df, traj)

    # write per-frame table
    out_csv = run_dir / "local_ops.csv"
    df.to_csv(out_csv, index=False)
    print(f"[local_ops] wrote: {out_csv}")

    # 3) per-residue rollup (use your tested helper)
    per_res = summarize_per_residue(df)

    # add a few useful medians back for context
    extras = (
        df.groupby(["resid", "resname"], as_index=False)
          .agg(
              omega_penalty_med=("omega_penalty", "median"),
              bondlen_zmax_med=("bondlen_zmax", "median"),
              bondang_zmax_med=("bondang_zmax", "median"),
              sasa_norm_med=("sasa_norm", "median"),
          )
    )
    per_res = per_res.merge(extras, on=["resid", "resname"], how="left")

    out_roll = run_dir / "per_residue.csv"
    per_res.to_csv(out_roll, index=False)
    print(f"[local_ops] wrote: {out_roll}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb", help="Path to a PDB file (single frame is fine)")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/run-local-<stamp>)")
    args = ap.parse_args()
    main(args.pdb, args.out)

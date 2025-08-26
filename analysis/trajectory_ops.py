# analysis/trajectory_ops.py
import numpy as np
import pandas as pd
import mdtraj as md
from typing import List, Tuple, Union, Optional

# --------------------------
# Small helpers / domain ops
# --------------------------

def _wrap180(x_deg):
    """Wrap degrees to (-180, 180]. Works on scalars or numpy arrays."""
    return (x_deg + 180.0) % 360.0 - 180.0

def _elliptical_distance(phi, psi, cphi, cpsi, a, b):
    """
    Elliptical distance in (phi, psi) space:
    sqrt(((phi-cphi)/a)^2 + ((psi-cpsi)/b)^2).
    Vectorized over phi/psi.
    """
    dphi = _wrap180(phi - cphi)
    dpsi = _wrap180(psi - cpsi)
    return np.sqrt((dphi / a) ** 2 + (dpsi / b) ** 2)

def _rama_centers_and_radii(resnames: np.ndarray, prepro: np.ndarray):
    """
    For each residue produce radii scaling for Ramachandran basins.
    Basins: alpha_R (-63,-42), beta (-135,135), alpha_L (60,45)
    Base radii (Δφ,Δψ) = (25,25) degrees.
    Scale:
      GLY -> 1.5  (broader)
      PRO or pre-Pro -> 0.8 (tighter)
      else -> 1.0
    Returns:
      centers: list of (cphi, cpsi)
      radii_per_res: (n_res, 2) array of (a,b)
    """
    centers = np.array([[-63.0, -42.0],
                        [-135.0, 135.0],
                        [  60.0,  45.0]], dtype=float)
    base = np.array([25.0, 25.0], dtype=float)

    scale = np.ones(len(resnames), dtype=float)
    scale[resnames == "GLY"] = 1.5
    # PRO or pre-Pro (res whose next is PRO)
    scale[(resnames == "PRO") | prepro] = 0.8

    radii = base[None, :] * scale[:, None]
    return centers, radii

def _compute_phi_psi(traj: md.Trajectory):
    """
    Compute phi/psi arrays and maps from residue index -> column index.
    Returns:
      phi: (n_frames, n_phi) angles in degrees
      psi: (n_frames, n_psi) angles in degrees
      phi_res_index: list length n_phi, residue index for that phi column
      psi_res_index: list length n_psi, residue index for that psi column
    """
    phi_idx, phi = md.compute_phi(traj)     # phi_idx shape (n_phi, 4), phi (n_frames, n_phi)
    psi_idx, psi = md.compute_psi(traj)

    # Map each dihedral to its central residue = residue of the 2nd atom (N for phi; CA is 2nd? For phi: [C(i-1), N(i), CA(i), C(i)] -> use atom 1)
    top = traj.topology
    phi_res_index = [top.atom(int(idx[1])).residue.index for idx in phi_idx]
    # For psi: [N(i), CA(i), C(i), N(i+1)] -> central residue i is atom 1 as well
    psi_res_index = [top.atom(int(idx[1])).residue.index for idx in psi_idx]

    # to degrees
    phi = np.degrees(phi)
    psi = np.degrees(psi)
    return phi, psi, phi_res_index, psi_res_index

def _rama_distance_series(traj: md.Trajectory):
    """
    Build per-frame, per-residue arrays:
      allowed: (n_frames, n_res) in {0,1} (nan if phi/psi not defined for that residue)
      rama_d : (n_frames, n_res) = min elliptical distance to any basin (nan where undefined)
    Gly/Pro/pre-Pro scaling applied.
    """
    nF = traj.n_frames
    residues = list(traj.topology.residues)
    nR = len(residues)
    resnames = np.array([r.name for r in residues], dtype=object)

    # pre-Pro flag: residue i is pre-Pro if residue i+1 is PRO
    prepro = np.zeros(nR, dtype=bool)
    for i in range(nR - 1):
        if residues[i + 1].name == "PRO":
            prepro[i] = True

    centers, radii = _rama_centers_and_radii(resnames, prepro)

    # Compute phi/psi and map dihedrals to residue indices
    phi, psi, phi_res_index, psi_res_index = _compute_phi_psi(traj)

    # Build per-residue column maps (some residues have no phi/psi at termini etc.)
    phi_col_for_res = {ri: i for i, ri in enumerate(phi_res_index)}
    psi_col_for_res = {ri: i for i, ri in enumerate(psi_res_index)}

    allowed = np.full((nF, nR), np.nan, dtype=float)
    rama_d  = np.full((nF, nR), np.nan, dtype=float)

    # Vectorized per-residue filling
    for r in range(nR):
        if r not in phi_col_for_res or r not in psi_col_for_res:
            continue
        pcol = phi_col_for_res[r]
        qcol = psi_col_for_res[r]
        phi_r = phi[:, pcol]    # (nF,)
        psi_r = psi[:, qcol]    # (nF,)
        a, b = radii[r]         # radii scaling for this residue

        # Distances to each center
        d0 = _elliptical_distance(phi_r, psi_r, centers[0,0], centers[0,1], a, b)
        d1 = _elliptical_distance(phi_r, psi_r, centers[1,0], centers[1,1], a, b)
        d2 = _elliptical_distance(phi_r, psi_r, centers[2,0], centers[2,1], a, b)

        dmin = np.minimum(d0, np.minimum(d1, d2))   # (nF,)
        rama_d[:, r] = dmin
        allowed[:, r] = (dmin <= 1.0).astype(float)

    return allowed, rama_d

# --------------------------------------
# Main extraction (windowed feature sets)
# --------------------------------------

def _load_trajectory(
    traj_files: Union[str, List[str]],
    top: Optional[str] = None,
    stride: int = 1
) -> md.Trajectory:
    """
    Load a trajectory from:
      - a list of PDB files (multi-PDB "trajectory"), or
      - a single trajectory file (dcd/xtc/etc) with a topology.
    """
    if isinstance(traj_files, (list, tuple)):
        # List of PDBs — md.load handles lists; each PDB carries topology.
        return md.load(list(traj_files), stride=stride)
    # Single path string
    path = traj_files
    if path.lower().endswith((".pdb", ".pdbx", ".cif")):
        return md.load(path, stride=stride)
    # Trajectory file needs topology
    if top is None:
        raise ValueError("Topology file (--top) is required for trajectory formats like DCD/XTC.")
    return md.load(path, top=top, stride=stride)

def extract_trajectory_features(
    traj_files: Union[str, List[str]],
    top: Optional[str],
    win: int,
    step: int,
    stride: int = 1,
    return_residue: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Compute windowed features over a trajectory.

    Returns by default (to keep your existing scripts working):
      per_window, summary

    If return_residue=True, returns:
      per_window, per_window_residue, summary
    """
    traj = _load_trajectory(traj_files, top=top, stride=stride)

    # Ramachandran series
    allowed, rama_d = _rama_distance_series(traj)   # (nF, nR)
    nF, nR = allowed.shape

    # ---------- per-window summary (frames x residues collapsed) ----------
    rows = []
    for s in range(0, nF - win + 1, step):
        e = s + win
        sl = slice(s, e)
        if rama_d[sl].size == 0:
            continue

        # % disallowed PER FRAME within window (then summarize across frames)
        pct_dis_per_frame = (1.0 - np.nanmean(allowed[sl], axis=1)) * 100.0  # (win,)
        mean_d_per_res    = np.nanmean(rama_d[sl], axis=0)                    # (nR,)

        rows.append({
            "start": s,
            "end": e,
            "n_frames": e - s,
            "mean_pct_disallowed": float(np.nanmean(pct_dis_per_frame)),
            "max_pct_disallowed": float(np.nanmax(pct_dis_per_frame)),
            "mean_rama_distance": float(np.nanmean(mean_d_per_res)),
            "max_rama_distance":  float(np.nanmax(mean_d_per_res)),
        })
    per_window = pd.DataFrame(rows)

    # ---------- per-window, per-residue breakdown (optional to return) ----------
    rows_res = []
    for s in range(0, nF - win + 1, step):
        e = s + win
        sl = slice(s, e)
        if rama_d[sl].size == 0:
            continue

        # per-residue summaries inside this window
        pct_dis_res = (1.0 - np.nanmean(allowed[sl], axis=0)) * 100.0  # (nR,)
        mean_d_res  = np.nanmean(rama_d[sl], axis=0)                   # (nR,)

        for resid in range(nR):
            rows_res.append({
                "start": s,
                "end": e,
                "resid": resid,
                "pct_disallowed": float(pct_dis_res[resid]),
                "mean_rama_distance": float(mean_d_res[resid]),
            })
    per_window_residue = pd.DataFrame(rows_res)

    # ---------- overall summary ----------
    # Fraction of disallowed per residue across ALL frames, then average across residues
    overall_dis_per_res = (1.0 - np.nanmean(allowed, axis=0)) * 100.0  # (nR,)
    summary = pd.DataFrame({
        "n_frames_total": [int(nF)],
        "n_residues": [int(nR)],
        "win": [int(win)],
        "step": [int(step)],
        "stride": [int(stride)],
        "overall_mean_pct_disallowed": [float(np.nanmean(overall_dis_per_res))],
        "overall_max_pct_disallowed":  [float(np.nanmax(overall_dis_per_res))],
        "overall_mean_rama_distance":  [float(np.nanmean(rama_d))],
        "overall_max_rama_distance":   [float(np.nanmax(rama_d))],
    })

    if return_residue:
        return per_window, per_window_residue, summary
    return per_window, summary

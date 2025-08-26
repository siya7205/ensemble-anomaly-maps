# analysis/local_ops.py
# ------------------------------------------------------------
# Local, interpretable operators for protein backbone geometry
# (φ/ψ Ramachandran, ω cis/trans penalty, robust bond/angle z)
# with optional χ1/χ2, DSSP, and SASA features.
#
# Returns a per-(frame, residue) DataFrame you can aggregate
# or feed to an anomaly detector.
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import mdtraj as md


# ---------- small helpers ----------

def _wrap180(x_deg: float | np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-180, 180]."""
    x = (np.asarray(x_deg) + 180.0) % 360.0 - 180.0
    return x


def _elliptical_distance(phi, psi, center, radii):
    """Distance to an ellipse centered at `center` with radii (Δφ, Δψ)."""
    dphi = _wrap180(phi - center[0])
    dpsi = _wrap180(psi - center[1])
    a, b = radii
    return np.sqrt((dphi / a) ** 2 + (dpsi / b) ** 2)


def _rama_ops(phi_deg: float, psi_deg: float, resname: str, prepro: bool = False):
    """
    Very light, interpretable Ramachandran operator.

    We approximate allowed basins with 3 ellipses and widen/tighten them
    for Gly/Pro/pre-Pro residues.
    """
    # canonical basin centers: αR, β, αL (deg)
    centers = [(-63.0, -42.0), (-135.0, 135.0), (60.0, 45.0)]
    base_radii = np.array([25.0, 25.0])   # Δφ, Δψ base radii
    # widen for GLY, tighten for PRO or pre-PRO
    scale = 1.5 if resname.upper() == "GLY" else (0.8 if (resname.upper() == "PRO" or prepro) else 1.0)
    radii = base_radii * scale

    dists = np.array([_elliptical_distance(phi_deg, psi_deg, c, radii) for c in centers])
    dmin = float(dists.min())
    allowed = 1 if dmin <= 1.0 else 0  # inside any ellipse ⇒ allowed
    return allowed, dmin  # dmin ~ how far from allowed basin edge (≤1 good, >1 worse)


def _true_mad(x: np.ndarray) -> float:
    """Median absolute deviation (MAD)."""
    x = np.asarray(x)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _robust_z(x: float, med: float, mad: float) -> float:
    """Robust z using MAD; guard for tiny MAD."""
    if mad <= 1e-6:
        return 0.0
    # 1.4826 ≈ MAD→σ under normality
    return abs(x - med) / (1.4826 * mad)


def _resid_from_torsion_quad(top: md.Topology, quad, atom_pos: int = 1) -> int:
    """
    Map a torsion quadruplet (list of 4 atom indices) to residue index.
    Using the 2nd atom (pos=1) gives N(i) for φ/ω and CA(i) for ψ → residue i.
    """
    atom_idx = int(quad[atom_pos])
    return top.atom(atom_idx).residue.index


def _min_rotamer_deviation(deg_value: float, rotamers=(60.0, -60.0, 180.0)) -> float:
    """Smallest wrapped angular distance (deg) to a set of canonical rotamers."""
    return float(min(abs(_wrap180(deg_value - r)) for r in rotamers))


# ---------- main routine ----------

def compute_local_ops(traj: md.Trajectory) -> pd.DataFrame:
    """
    Compute per-(frame, residue) local, interpretable geometry features.

    Columns include:
      - phi_deg, psi_deg, omega_deg
      - rama_allowed (0/1), rama_distance (<=1 good, >1 outside)
      - omega_penalty (cis/trans + near-planarity deviations)
      - bondlen_zmax (robust z across N-CA, CA-C, C-N+1)
      - bondang_zmax (robust z across C-N-CA, N-CA-C, CA-C-N+1)
      - (optional) chi1_deg/chi2_deg, chi_rotamer_dev
      - (optional) ss (H/E/C) and sasa

    Returns
    -------
    DataFrame of shape ≈ (n_frames * n_residues, features)
    """
    top = traj.topology
    res_list = list(top.residues)
    n_res = len(res_list)
    n_frames = traj.n_frames

    # --- Backbone torsions: indices + values (in degrees) ---
    phi_idx,   phi_vals   = md.compute_phi(traj)       # (n_phi, 4), (n_frames, n_phi)
    psi_idx,   psi_vals   = md.compute_psi(traj)       # (n_psi, 4), (n_frames, n_psi)
    omega_idx, omega_vals = md.compute_omega(traj)     # (n_omega, 4), (n_frames, n_omega)

    phi   = np.rad2deg(phi_vals)
    psi   = np.rad2deg(psi_vals)
    omega = np.rad2deg(omega_vals)

    # Map residue -> column in angle matrices using 2nd atom of quadruplet
    phi_map   = { _resid_from_torsion_quad(top, quad, 1): i for i, quad in enumerate(phi_idx) }
    psi_map   = { _resid_from_torsion_quad(top, quad, 1): i for i, quad in enumerate(psi_idx) }
    omega_map = { _resid_from_torsion_quad(top, quad, 1): i for i, quad in enumerate(omega_idx) }

    # --- Backbone atom indices per residue (for bonds/angles) ---
    N_idx  = np.full(n_res, -1, dtype=int)
    CA_idx = np.full(n_res, -1, dtype=int)
    C_idx  = np.full(n_res, -1, dtype=int)

    for r in res_list:
        names = {a.name: a.index for a in r.atoms}
        if "N" in names:   N_idx[r.index]  = names["N"]
        if "CA" in names:  CA_idx[r.index] = names["CA"]
        if "C" in names:   C_idx[r.index]  = names["C"]

    # --- Bond lengths (Å) ---
    bl_pairs = []
    bl_tags  = []  # (name, resid)
    for i in range(n_res):
        if N_idx[i] >= 0 and CA_idx[i] >= 0:
            bl_pairs.append([N_idx[i], CA_idx[i]]); bl_tags.append(("d_N_CA", i))
        if CA_idx[i] >= 0 and C_idx[i] >= 0:
            bl_pairs.append([CA_idx[i], C_idx[i]]); bl_tags.append(("d_CA_C", i))
        if i + 1 < n_res and C_idx[i] >= 0 and N_idx[i + 1] >= 0:
            bl_pairs.append([C_idx[i], N_idx[i + 1]]); bl_tags.append(("d_C_Np1", i))

    if bl_pairs:
        dists_nm = md.compute_distances(traj, bl_pairs)   # (n_frames, n_pairs) in nm
        dists = dists_nm * 10.0                           # Å
        bond_df = []
        for col, (name, ires) in enumerate(bl_tags):
            bond_df.append(pd.DataFrame({"bond": name, "residue": ires, "val": dists[:, col]}))
        bond_df = pd.concat(bond_df, ignore_index=True)
        bl_med = bond_df.groupby("bond")["val"].median().to_dict()
        bl_mad = bond_df.groupby("bond")["val"].apply(_true_mad).to_dict()
    else:
        bl_med, bl_mad, dists = {}, {}, None

    # --- Bond angles (deg) ---
    ang_triplets = []
    ang_tags = []
    for i in range(n_res):
        # angle: C(i-1)-N(i)-CA(i)
        if i - 1 >= 0 and C_idx[i - 1] >= 0 and N_idx[i] >= 0 and CA_idx[i] >= 0:
            ang_triplets.append([C_idx[i - 1], N_idx[i], CA_idx[i]]); ang_tags.append(("ang_Cnm1_N_CA", i))
        # angle: N(i)-CA(i)-C(i)
        if N_idx[i] >= 0 and CA_idx[i] >= 0 and C_idx[i] >= 0:
            ang_triplets.append([N_idx[i], CA_idx[i], C_idx[i]]); ang_tags.append(("ang_N_CA_C", i))
        # angle: CA(i)-C(i)-N(i+1)
        if i + 1 < n_res and CA_idx[i] >= 0 and C_idx[i] >= 0 and N_idx[i + 1] >= 0:
            ang_triplets.append([CA_idx[i], C_idx[i], N_idx[i + 1]]); ang_tags.append(("ang_CA_C_Np1", i))

    if ang_triplets:
        ang_vals = np.rad2deg(md.compute_angles(traj, ang_triplets))
        ang_df = []
        for col, (name, ires) in enumerate(ang_tags):
            ang_df.append(pd.DataFrame({"angle": name, "residue": ires, "val": ang_vals[:, col]}))
        ang_df = pd.concat(ang_df, ignore_index=True)
        ang_med = ang_df.groupby("angle")["val"].median().to_dict()
        ang_mad = ang_df.groupby("angle")["val"].apply(_true_mad).to_dict()
    else:
        ang_med, ang_mad, ang_vals = {}, {}, None

    # --- Optional χ1/χ2 torsions (deg) ---
    chi1_deg, chi1_map = None, {}
    chi2_deg, chi2_map = None, {}
    try:
        chi1_idx, chi1 = md.compute_chi1(traj)  # (n_chi1, 4), (n_frames, n_chi1)
        chi1_deg = np.rad2deg(chi1)
        # map residue index from 2nd atom of the quadruplet
        chi1_map = { _resid_from_torsion_quad(top, quad, 1): i for i, quad in enumerate(chi1_idx) }
    except Exception:
        pass
    try:
        chi2_idx, chi2 = md.compute_chi2(traj)
        chi2_deg = np.rad2deg(chi2)
        chi2_map = { _resid_from_torsion_quad(top, quad, 1): i for i, quad in enumerate(chi2_idx) }
    except Exception:
        pass

    # --- Optional DSSP (secondary structure) ---
    # Simplified alphabet: 'H' (helix), 'E' (strand), 'C' (coil)
    try:
        ss_arr = md.compute_dssp(traj, simplified=True)  # (n_frames, n_res)
        if ss_arr.shape[1] != n_res:
            # fallback if mdtraj/provider mismatched residue count
            ss_arr = np.full((n_frames, n_res), 'C', dtype='<U1')
    except Exception:
        ss_arr = np.full((n_frames, n_res), 'C', dtype='<U1')

    # --- Optional SASA (Å^2) per residue ---
    try:
        sasa = md.shrake_rupley(traj, mode="residue")  # (n_frames, n_res)
        # normalize to [0,1] by global 95th percentile to reduce outlier impact
        denom = np.percentile(sasa[~np.isnan(sasa)], 95)
        if denom <= 1e-6:
            denom = 1.0
        sasa_norm = np.clip(sasa / denom, 0.0, 1.0)
    except Exception:
        sasa, sasa_norm = None, None

    # --- Pre-Proline flags (affects rama class) ---
    prepro = { r.index: (res_list[r.index - 1].name.upper() == "PRO") if r.index > 0 else False
               for r in res_list }

    rows = []

    # For quick lookup from (name, ires) to column in dists / ang_vals
    bl_lookup = { (nm, ires): ci for ci, (nm, ires) in enumerate(bl_tags) }
    ang_lookup = { (nm, ires): ci for ci, (nm, ires) in enumerate(ang_tags) }

    for f in range(n_frames):
        for r in res_list:
            i = r.index

            # Need phi & psi to define Ramachandran features; skip if missing (e.g., terminal)
            if i not in phi_map or i not in psi_map:
                continue

            phi_i = float(phi[f, phi_map[i]])
            psi_i = float(psi[f, psi_map[i]])

            # Rama operators
            rama_allowed, rama_d = _rama_ops(phi_i, psi_i, r.name, prepro=prepro[i])

            # ω: map to residue i if present; otherwise treat as trans (180)
            if i in omega_map:
                omega_i = float(omega[f, omega_map[i]])
            else:
                omega_i = 180.0
            w = _wrap180(omega_i)
            is_cis = abs(w) <= 30.0
            # penalty: cis (non-Pro) worst; cis-Pro somewhat tolerated; near-planar deviations otherwise
            if is_cis and r.name.upper() != "PRO":
                omega_pen = 1.0
            elif is_cis and r.name.upper() == "PRO":
                omega_pen = 0.3
            else:
                # trans: penalize deviation from ±180
                omega_pen = min(1.0, abs(abs(w) - 180.0) / 60.0)

            # Bond-length robust z (take max over 3 backbone bonds when available)
            bl_z = 0.0
            for nm in ("d_N_CA", "d_CA_C", "d_C_Np1"):
                key = (nm, i)
                if dists is not None and key in bl_lookup:
                    col = bl_lookup[key]
                    val = float(dists[f, col])
                    med = bl_med.get(nm, val)
                    mad = bl_mad.get(nm, 0.0)
                    z = _robust_z(val, med, mad)
                    bl_z = max(bl_z, z)
            bl_z = min(bl_z, 6.0)

            # Bond-angle robust z (max over 3 angles)
            ba_z = 0.0
            for nm in ("ang_Cnm1_N_CA", "ang_N_CA_C", "ang_CA_C_Np1"):
                key = (nm, i)
                if ang_vals is not None and key in ang_lookup:
                    col = ang_lookup[key]
                    val = float(ang_vals[f, col])
                    med = ang_med.get(nm, val)
                    mad = ang_mad.get(nm, 0.0)
                    z = _robust_z(val, med, mad)
                    ba_z = max(ba_z, z)
            ba_z = min(ba_z, 6.0)

            # Optional χ rotamer deviations (normalized 0..1 approx)
            chi1_val = None
            chi2_val = None
            chi_dev = 0.0
            if chi1_deg is not None and i in chi1_map:
                chi1_val = float(chi1_deg[f, chi1_map[i]])
                chi1_delta = _min_rotamer_deviation(chi1_val)  # in deg
                chi_dev += min(1.0, chi1_delta / 60.0)  # 60° ~ one rotamer width
            if chi2_deg is not None and i in chi2_map:
                chi2_val = float(chi2_deg[f, chi2_map[i]])
                chi2_delta = _min_rotamer_deviation(chi2_val)
                chi_dev += min(1.0, chi2_delta / 60.0)

            # Secondary structure & SASA (optional)
            ss = ss_arr[f, i] if ss_arr is not None else 'C'
            sasa_i = float(sasa_norm[f, i]) if sasa_norm is not None else np.nan

            # Composite local anomaly score (interpretable, tunable weights)
            score = (
                1.00 * (1 - rama_allowed)              # disallowed basin
                + 0.50 * max(0.0, rama_d - 1.0)        # outside ellipse margin
                + 0.75 * omega_pen                      # cis or non-planar peptide
                + 0.50 * min(1.0, bl_z / 3.0)           # long/short bonds
                + 0.50 * min(1.0, ba_z / 3.0)           # bent/straight angles
                + 0.25 * min(1.0, chi_dev / 2.0)        # side-chain off-rotamer
            )

            rows.append(
                {
                    "frame": f,
                    "resid": i,
                    "resname": r.name,
                    "phi_deg": phi_i,
                    "psi_deg": psi_i,
                    "omega_deg": omega_i,
                    "rama_allowed": int(rama_allowed),
                    "rama_distance": rama_d,
                    "omega_penalty": omega_pen,
                    "bondlen_zmax": bl_z,
                    "bondang_zmax": ba_z,
                    "chi1_deg": chi1_val,
                    "chi2_deg": chi2_val,
                    "chi_rotamer_dev": chi_dev,
                    "ss": ss,
                    "sasa_norm": sasa_i,
                    "local_score": float(score),
                }
            )

    return pd.DataFrame(rows)

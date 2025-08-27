# analysis/local_ops.py
# Local-ops: biophysics-informed per-residue scoring
# - Correct φ/ψ/ω residue mapping from mdtraj
# - Bond length / bond angle robust z-scores (median + MAD)
# - Ramachandran allowed flag + elliptical distance to nearest basin
# - ω penalty (cis/trans and deviations)
# - SASA (Shrake-Rupley)
# - χ rotamer deviation (χ1, χ2 if available)
# - Optional DSSP (simplified) if the system has DSSP installed

from __future__ import annotations
import numpy as np
import pandas as pd
import mdtraj as md
from typing import Dict, Tuple, List


# ---------------------------
# Small math / utility helpers
# ---------------------------

def _wrap180(x_deg: float | np.ndarray) -> np.ndarray:
    x = (np.asarray(x_deg) + 180.0) % 360.0 - 180.0
    return x

def _elliptical_distance(phi, psi, center, radii) -> float:
    # distance in an ellipse metric around a center
    dphi = _wrap180(phi - center[0])
    dpsi = _wrap180(psi - center[1])
    a, b = radii
    return float(np.sqrt((dphi / a) ** 2 + (dpsi / b) ** 2))

def _robust_median_mad(x: np.ndarray) -> Tuple[float, float]:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(med), float(mad)

def _robust_z(x: float, med: float, mad: float) -> float:
    if mad <= 1e-8:
        return 0.0
    # 1.4826 turns MAD into std-like
    return abs(x - med) / (1.4826 * mad)

# ---------------------------
# Ramachandran ops
# ---------------------------

def _rama_ops(phi_deg: float, psi_deg: float, resname: str, prepro: bool) -> Tuple[int, float]:
    """
    A light-weight allowed classifier + distance-to-basin.
    Basins: αR ~ (-63,-42), β ~ (-135,135), αL ~ (60,45).
    Radii scaled for Gly (wider) and Pro / pre-Pro (tighter).
    """
    centers = [(-63.0, -42.0), (-135.0, 135.0), (60.0, 45.0)]   # αR, β, αL
    base_radii = np.array([25.0, 25.0])                         # Δφ, Δψ (deg)

    # Gly is more permissive; Pro and pre-Pro more restricted
    if resname == "GLY":
        scale = 1.5
    elif resname == "PRO" or prepro:
        scale = 0.8
    else:
        scale = 1.0

    radii = base_radii * scale
    dists = [ _elliptical_distance(phi_deg, psi_deg, c, radii) for c in centers ]
    dmin = float(np.min(dists))
    allowed = 1 if dmin <= 1.0 else 0
    return allowed, dmin

# ---------------------------
# χ rotamer deviation
# ---------------------------

def _chi_rotamer_dev(traj: md.Trajectory) -> np.ndarray:
    """
    Compute simple deviation from a coarse rotamer center for χ1 and χ2.
    Returns an array shape (n_frames, n_residues) with a per-residue deviation (deg).
    If residue has no sidechain χ, value is 0.
    """
    top = traj.topology
    nF, nR = traj.n_frames, top.n_residues
    dev = np.zeros((nF, nR), dtype=np.float32)

    # Predefine coarse χ1 centers (deg). This is intentionally coarse.
    chi_centers = {
        # three canonical rotamers for χ1
        'chi1': np.array([-60.0, 60.0, 180.0], dtype=np.float32),
        'chi2': np.array([-60.0, 60.0, 180.0], dtype=np.float32),
    }

    def _min_circ_dist(value_deg: np.ndarray, centers_deg: np.ndarray) -> np.ndarray:
        # circular distance in degrees to nearest center
        diffs = np.abs(_wrap180(value_deg[..., None] - centers_deg[None, None, :]))
        return np.min(diffs, axis=-1)  # shape (n_frames, n_chiX)

    # χ1
    try:
        chi1_idx, chi1 = md.compute_chi1(traj)  # chi1: (n_frames, n_chi1)
        chi1_deg = chi1 * 180.0 / np.pi
        if chi1_idx.shape[0] > 0:
            # chi1_idx rows relate to residues; pick their residue index from any atom
            res_idx_chi1 = [ traj.topology.atom(int(idx4[1])).residue.index for idx4 in chi1_idx ]
            # deviation to nearest coarse center
            d1 = _min_circ_dist(chi1_deg, chi_centers['chi1'])  # (n_frames, n_chi1)
            # For residues with multiple χ1 entries (rare), keep min per residue
            for col, ridx in enumerate(res_idx_chi1):
                dev[:, ridx] = np.maximum(dev[:, ridx], d1[:, col])
    except Exception:
        pass

    # χ2
    try:
        chi2_idx, chi2 = md.compute_chi2(traj)  # chi2: (n_frames, n_chi2)
        chi2_deg = chi2 * 180.0 / np.pi
        if chi2_idx.shape[0] > 0:
            res_idx_chi2 = [ traj.topology.atom(int(idx4[1])).residue.index for idx4 in chi2_idx ]
            d2 = _min_circ_dist(chi2_deg, chi_centers['chi2'])
            for col, ridx in enumerate(res_idx_chi2):
                # combine with χ1 dev (max)
                dev[:, ridx] = np.maximum(dev[:, ridx], d2[:, col])
    except Exception:
        pass

    return dev  # degrees, larger = further from coarse rotamer centers

# ---------------------------
# Main computation
# ---------------------------

def compute_local_ops(traj: md.Trajectory) -> pd.DataFrame:
    """
    Return per-frame, per-residue measurements with a composite local_score.
    Columns include:
      frame, resid, resname, phi_deg, psi_deg, omega_deg,
      rama_allowed, rama_distance, omega_penalty,
      bondlen_zmax, bondang_zmax, sasa, chi_rotamer_dev, dssp(optional),
      local_score
    """
    top = traj.topology
    nF, nR = traj.n_frames, top.n_residues

    # ---------- backbone torsions ----------
    phi_idx, phi = md.compute_phi(traj)     # idx: (n_phi,4), phi: (n_frames,n_phi)
    psi_idx, psi = md.compute_psi(traj)
    omega_idx, omega = md.compute_omega(traj)

    phi = phi * 180.0 / np.pi
    psi = psi * 180.0 / np.pi
    omega = omega * 180.0 / np.pi

    # Map residue index -> column
    phi_map: Dict[int,int] = {}
    for col, idx4 in enumerate(phi_idx):
        # residue = residue of the 2nd atom (N(i))
        res_i = top.atom(int(idx4[1])).residue.index
        phi_map[res_i] = col

    psi_map: Dict[int,int] = {}
    for col, idx4 in enumerate(psi_idx):
        # residue = residue of the 2nd atom (CA(i))
        res_i = top.atom(int(idx4[1])).residue.index
        psi_map[res_i] = col

    omega_map: Dict[int,int] = {}
    for col, idx4 in enumerate(omega_idx):
        # map ω to residue i = residue of the 2nd atom (C(i))
        res_i = top.atom(int(idx4[1])).residue.index
        omega_map[res_i] = col

    # ---------- atom indices for bonds/angles ----------
    # Collect backbone atom indices per residue (or -1)
    N_idx = np.full(nR, -1, dtype=int)
    CA_idx = np.full(nR, -1, dtype=int)
    C_idx = np.full(nR, -1, dtype=int)
    res_list = list(top.residues)

    for r in res_list:
        names = {a.name: a.index for a in r.atoms}
        if "N" in names:   N_idx[r.index] = names["N"]
        if "CA" in names: CA_idx[r.index] = names["CA"]
        if "C" in names:   C_idx[r.index] = names["C"]

    # Bonds: N-CA, CA-C, C-N(+1)
    pairs, tags = [], []
    for i in range(nR):
        if N_idx[i] >= 0 and CA_idx[i] >= 0:
            pairs.append([N_idx[i], CA_idx[i]]); tags.append(("d_N_CA", i))
        if CA_idx[i] >= 0 and C_idx[i] >= 0:
            pairs.append([CA_idx[i], C_idx[i]]); tags.append(("d_CA_C", i))
        if i+1 < nR and C_idx[i] >= 0 and N_idx[i+1] >= 0:
            pairs.append([C_idx[i], N_idx[i+1]]); tags.append(("d_C_Np1", i))

    dists = md.compute_distances(traj, pairs) * 10.0  # Å
    # Aggregate robust stats per bond type (across all residues & frames)
    bond_records = []
    for col, (name, ires) in enumerate(tags):
        bond_records.append(pd.DataFrame({"bond": name, "residue": ires, "val": dists[:, col]}))
    bond_df = pd.concat(bond_records, ignore_index=True) if bond_records else pd.DataFrame(columns=["bond","residue","val"])

    bond_med: Dict[str,float] = {}
    bond_mad: Dict[str,float] = {}
    if not bond_df.empty:
        for bond_name, sub in bond_df.groupby("bond"):
            med, mad = _robust_median_mad(sub["val"].values)
            bond_med[bond_name] = med
            bond_mad[bond_name] = mad

    # Angles: C(i-1)-N(i)-CA(i), N(i)-CA(i)-C(i), CA(i)-C(i)-N(i+1)
    triplets, atags = [], []
    for i in range(nR):
        if i-1 >= 0 and C_idx[i-1] >= 0 and N_idx[i] >= 0 and CA_idx[i] >= 0:
            triplets.append([C_idx[i-1], N_idx[i], CA_idx[i]]); atags.append(("ang_Cnm1_N_CA", i))
        if N_idx[i] >= 0 and CA_idx[i] >= 0 and C_idx[i] >= 0:
            triplets.append([N_idx[i], CA_idx[i], C_idx[i]]); atags.append(("ang_N_CA_C", i))
        if i+1 < nR and CA_idx[i] >= 0 and C_idx[i] >= 0 and N_idx[i+1] >= 0:
            triplets.append([CA_idx[i], C_idx[i], N_idx[i+1]]); atags.append(("ang_CA_C_Np1", i))

    if triplets:
        angs = md.compute_angles(traj, triplets) * 180.0 / np.pi
        ang_records = []
        for col, (name, ires) in enumerate(atags):
            ang_records.append(pd.DataFrame({"angle": name, "residue": ires, "val": angs[:, col]}))
        ang_df = pd.concat(ang_records, ignore_index=True)
        ang_med: Dict[str,float] = {}
        ang_mad: Dict[str,float] = {}
        for ang_name, sub in ang_df.groupby("angle"):
            med, mad = _robust_median_mad(sub["val"].values)
            ang_med[ang_name] = med
            ang_mad[ang_name] = mad
    else:
        angs = np.zeros((nF, 0), dtype=float)
        ang_med, ang_mad = {}, {}

    # ---------- SASA ----------
    try:
        sasa = md.shrake_rupley(traj, mode='residue')  # (n_frames, n_residues), nm^2
        sasa = sasa * 100.0  # convert to Å^2
    except Exception:
        sasa = np.zeros((nF, nR), dtype=float)

    # ---------- χ rotamer deviation ----------
    try:
        chi_dev = _chi_rotamer_dev(traj)  # (n_frames, n_residues), degrees
    except Exception:
        chi_dev = np.zeros((nF, nR), dtype=float)

    # ---------- DSSP (optional) ----------
    dssp_simpl = None
    try:
        # DSSP may need external binary; returns strings or codes per residue
        dssp_simpl = md.compute_dssp(traj, simplified=True)  # (n_frames, n_residues), e.g., H/E/C
    except Exception:
        pass

    # pre-Pro flags (residue preceded by Proline)
    prepro = {r.index: (res_list[r.index-1].name == "PRO") if r.index > 0 else False for r in res_list}

    # ---------- Build per-frame rows ----------
    rows: List[Dict] = []
    for f in range(nF):
        for r in res_list:
            i = r.index
            # φ/ψ if available
            if i not in phi_map or i not in psi_map:
                continue
            phi_i = float(phi[f, phi_map[i]])
            psi_i = float(psi[f, psi_map[i]])

            # Ramachandran allowed + distance
            rama_allowed, rama_d = _rama_ops(phi_i, psi_i, r.name, prepro=prepro[i])

            # ω(i)
            if i in omega_map:
                omega_i = float(omega[f, omega_map[i]])
            else:
                omega_i = 180.0  # default trans
            w = _wrap180(omega_i)
            cis = abs(w) <= 30.0
            if cis and r.name != "PRO":
                omega_pen = 1.0
            elif cis and r.name == "PRO":
                omega_pen = 0.3
            else:
                # distance from trans; penalize large deviations
                omega_pen = min(1.0, abs(abs(w) - 180.0) / 60.0)

            # bond z-max over the three bonds for this residue
            bl_z = 0.0
            for name, want in (("d_N_CA", (N_idx[i], CA_idx[i])),
                               ("d_CA_C", (CA_idx[i], C_idx[i])),
                               ("d_C_Np1", (C_idx[i], N_idx[i+1] if i+1 < nR else -1))):
                # find column for this residue+bond
                try:
                    col = next(ci for ci, (nm, ri) in enumerate(tags) if nm == name and ri == i)
                except StopIteration:
                    continue
                val = float(dists[f, col])
                med = bond_med.get(name, val)
                mad = bond_mad.get(name, 0.0)
                bl_z = max(bl_z, _robust_z(val, med, mad))
            bl_z = min(bl_z, 4.0)

            # angle z-max over the three angles
            ba_z = 0.0
            for name in ("ang_Cnm1_N_CA", "ang_N_CA_C", "ang_CA_C_Np1"):
                try:
                    col = next(ci for ci, (nm, ri) in enumerate(atags) if nm == name and ri == i)
                except StopIteration:
                    continue
                val = float(angs[f, col])
                med = ang_med.get(name, val)
                mad = ang_mad.get(name, 0.0)
                ba_z = max(ba_z, _robust_z(val, med, mad))
            ba_z = min(ba_z, 4.0)

            # SASA / χ dev / DSSP (if available)
            sasa_i = float(sasa[f, i]) if sasa is not None and sasa.size else 0.0
            chi_i = float(chi_dev[f, i]) if chi_dev is not None and chi_dev.size else 0.0
            dssp_i = dssp_simpl[f, i] if dssp_simpl is not None else None

            # Composite local score (weights are heuristic)
            score = (
                1.00 * (1 - rama_allowed) +          # disallowed = +1
                0.50 * max(0.0, rama_d - 1.0) +      # distance beyond the allowed ellipse
                0.75 * omega_pen +                   # cis (non-Pro) or big deviations from trans
                0.50 * min(1.0, bl_z / 3.0) +        # bond strains
                0.50 * min(1.0, ba_z / 3.0) +        # angle strains
                0.10 * min(1.0, chi_i / 120.0)       # side-chain oddity (soft)
            )

            rows.append({
                "frame": f,
                "resid": i,
                "resname": r.name,
                "phi_deg": phi_i,
                "psi_deg": psi_i,
                "omega_deg": omega_i,
                "rama_allowed": int(rama_allowed),
                "rama_distance": float(rama_d),
                "omega_penalty": float(omega_pen),
                "bondlen_zmax": float(bl_z),
                "bondang_zmax": float(ba_z),
                "sasa": float(sasa_i),
                "chi_rotamer_dev": float(chi_i),
                "dssp": dssp_i if dssp_i is not None else "",
                "local_score": float(score),
            })

    return pd.DataFrame(rows)


# ---------------------------
# Optional: per-residue rollup (median score, % disallowed)
# ---------------------------

def summarize_per_residue(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the per-frame DataFrame returned by compute_local_ops(), build a
    per-residue summary: median local score, % frames disallowed, n_frames.
    """
    if df.empty:
        return pd.DataFrame(columns=["resid","resname","local_score_med","rama_disallowed_pct","n_frames"])

    g = df.groupby(["resid","resname"], as_index=False)
    out = g.agg(
        local_score_med=("local_score", "median"),
        rama_disallowed_pct=("rama_allowed", lambda x: 100.0 * (1.0 - np.mean(x.astype(float)))),
        n_frames=("frame", "nunique"),
    ).sort_values("local_score_med", ascending=False).reset_index(drop=True)
    return out

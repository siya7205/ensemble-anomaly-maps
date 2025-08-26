# analysis/local_ops.py
import numpy as np
import mdtraj as md
import pandas as pd

# ---------------- utils ----------------
def _wrap180(x_deg):
    x = (x_deg + 180) % 360 - 180
    return x

def _elliptical_distance(phi, psi, center, radii):
    dphi = _wrap180(phi - center[0])
    dpsi = _wrap180(psi - center[1])
    a, b = radii
    return np.sqrt((dphi / a) ** 2 + (dpsi / b) ** 2)

def _rama_ops(phi_deg, psi_deg, resname, prepro=False):
    # very small analytic model of Ramachandran basins
    centers = [(-63, -42), (-135, 135), (60, 45)]  # αR, β, αL
    base_radii = np.array([25.0, 25.0])
    scale = 1.5 if resname == "GLY" else (0.8 if (resname == "PRO" or prepro) else 1.0)
    radii = base_radii * scale
    dists = np.array([_elliptical_distance(phi_deg, psi_deg, c, radii) for c in centers])
    dmin = float(np.min(dists))
    allowed = 1 if dmin <= 1.0 else 0
    return allowed, dmin

def _robust_z(x, med, mad):
    if mad <= 1e-6:
        return 0.0
    return abs(x - med) / (1.4826 * mad)

def _true_mad(x):
    x = np.asarray(x)
    return float(np.median(np.abs(x - np.median(x))))

# ---------------- core ----------------
def compute_local_ops(traj: md.Trajectory) -> pd.DataFrame:
    """
    Per-frame / per-residue local operators:
      - phi, psi, omega (deg), Ramachandran allowed flag + distance-from-basin
      - backbone bond-length / bond-angle robust Z (max of the 3 around residue)
      - side-chain rotamer deviation proxy (max |delta-chi|)
      - SASA (normalized to [0,1] by per-protein max)
      - DSSP secondary structure (if available; simplified H/E/C)

    Returns a long DataFrame with one row per (frame, resid).
    """
    top = traj.topology

    # --- backbone torsions ---
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    phi = phi * 180/np.pi
    psi = psi * 180/np.pi
    # map residue index -> column in phi/psi arrays
    phi_map = {r.index: i for i, (_, r) in enumerate(phi_idx)}
    psi_map = {r.index: i for i, (_, r) in enumerate(psi_idx)}

    # omega between i and i+1; map column i -> residue i
    omega_idx, omega = md.compute_omega(traj)
    omega = omega * 180/np.pi
    # omega_idx is a list of (atom_indices, residue_i), use residue_i directly
    omega_col_for_res = {res.index: col for col, (_, res) in enumerate(omega_idx)}

    # --- atom indices per residue ---
    N_idx, CA_idx, C_idx = [], [], []
    res_list = list(top.residues)
    for r in res_list:
        names = {a.name: a.index for a in r.atoms}
        N_idx.append(names.get("N", -1))
        CA_idx.append(names.get("CA", -1))
        C_idx.append(names.get("C", -1))
    N_idx = np.array(N_idx)
    CA_idx = np.array(CA_idx)
    C_idx = np.array(C_idx)

    # --- bond lengths (Å) around each residue ---
    pairs, tags = [], []
    for i in range(len(res_list)):
        if N_idx[i] >= 0 and CA_idx[i] >= 0:
            pairs.append([N_idx[i], CA_idx[i]]); tags.append(("d_N_CA", i))
        if CA_idx[i] >= 0 and C_idx[i] >= 0:
            pairs.append([CA_idx[i], C_idx[i]]); tags.append(("d_CA_C", i))
        if i+1 < len(res_list) and C_idx[i] >= 0 and N_idx[i+1] >= 0:
            pairs.append([C_idx[i], N_idx[i+1]]); tags.append(("d_C_Np1", i))
    dists = md.compute_distances(traj, pairs) * 10.0  # nm -> Å

    bond_df = []
    for col, (name, ires) in enumerate(tags):
        bond_df.append(pd.DataFrame({"bond": name, "residue": ires, "val": dists[:, col]}))
    bond_df = pd.concat(bond_df, ignore_index=True) if bond_df else pd.DataFrame(columns=["bond","residue","val"])
    med = bond_df.groupby("bond")["val"].median() if not bond_df.empty else pd.Series(dtype=float)
    mad = bond_df.groupby("bond")["val"].apply(_true_mad) if not bond_df.empty else pd.Series(dtype=float)

    # --- bond angles (deg) around each residue ---
    triplets, atags = [], []
    for i in range(len(res_list)):
        if i-1 >= 0 and C_idx[i-1] >= 0 and N_idx[i] >= 0 and CA_idx[i] >= 0:
            triplets.append([C_idx[i-1], N_idx[i], CA_idx[i]]); atags.append(("ang_Cnm1_N_CA", i))
        if N_idx[i] >= 0 and CA_idx[i] >= 0 and C_idx[i] >= 0:
            triplets.append([N_idx[i], CA_idx[i], C_idx[i]]); atags.append(("ang_N_CA_C", i))
        if i+1 < len(res_list) and CA_idx[i] >= 0 and C_idx[i] >= 0 and N_idx[i+1] >= 0:
            triplets.append([CA_idx[i], C_idx[i], N_idx[i+1]]); atags.append(("ang_CA_C_Np1", i))
    angs = md.compute_angles(traj, triplets) * 180/np.pi if triplets else np.zeros((traj.n_frames, 0))
    ang_df = []
    for col, (name, ires) in enumerate(atags):
        ang_df.append(pd.DataFrame({"angle": name, "residue": ires, "val": angs[:, col]}))
    ang_df = pd.concat(ang_df, ignore_index=True) if ang_df else pd.DataFrame(columns=["angle","residue","val"])
    ang_med = ang_df.groupby("angle")["val"].median() if not ang_df.empty else pd.Series(dtype=float)
    ang_mad = ang_df.groupby("angle")["val"].apply(_true_mad) if not ang_df.empty else pd.Series(dtype=float)

    # --- side-chain χ angles (deg) and rotamer deviation proxy ---
    try:
        chi_list = md.compute_chi(traj)  # list of (idx, values) per chi-order
        # flatten all chis into a single array per frame with NaN for missing residues
        # we’ll compute per-residue max |delta-chi| from nearest canonical (−60, 60, 180)
        def nearest_rotamer_delta(x):
            # wrap to [-180,180], compare to {−60, 60, 180}
            xw = _wrap180(x)
            cands = np.array([-60.0, 60.0, 180.0, -180.0])
            return np.min(np.abs(_wrap180(xw - cands)))
        chi_dev = {r.index: np.zeros(traj.n_frames) for r in res_list}
        for (idx_set, chi_vals) in chi_list:
            # idx_set is list of atom indices; mdtraj also provides residue mapping:
            # find the residue index via the first atom’s residue
            for col in range(chi_vals.shape[1]):
                # which residue?
                try:
                    ridx = top.atom(idx_set[col][1]).residue.index  # use CA atom to grab residue
                except Exception:
                    # fallback: first atom’s residue
                    ridx = top.atom(idx_set[col][0]).residue.index
                dev = np.array([nearest_rotamer_delta(v) for v in chi_vals[:, col]])
                chi_dev[ridx] = np.maximum(chi_dev[ridx], np.abs(dev))
    except Exception:
        chi_dev = {r.index: np.zeros(traj.n_frames) for r in res_list}

    # --- SASA (Å^2) normalized per protein to [0,1] ---
    try:
        sasa = md.shrake_rupley(traj, mode='residue')  # (n_frames, n_res)
        # normalize by per-protein max across frames/residues
        max_sasa = np.nanmax(sasa) if np.isfinite(sasa).any() else 1.0
        if max_sasa <= 0: max_sasa = 1.0
        sasa_norm = sasa / max_sasa
    except Exception:
        sasa_norm = np.zeros((traj.n_frames, len(res_list)))

    # --- DSSP secondary structure (simplified H/E/C) ---
    try:
        dssp = md.compute_dssp(traj, simplified=True)  # (n_frames, n_res) in {'H','E','C'}
        has_dssp = True
    except Exception as e:
        print("[local_ops] DSSP unavailable:", e)
        dssp = None
        has_dssp = False

    # pre-pro flag (prev residue is PRO)
    prepro = {r.index: (res_list[r.index-1].name == "PRO") if r.index > 0 else False for r in res_list}

    # --- build rows ---
    rows = []
    n_frames = traj.n_frames
    for f in range(n_frames):
        for r in res_list:
            i = r.index
            # phi/psi
            if i not in phi_map or i not in psi_map:
                continue
            phi_i = float(phi[f, phi_map[i]])
            psi_i = float(psi[f, psi_map[i]])
            rama_allowed, rama_d = _rama_ops(phi_i, psi_i, r.name, prepro=prepro[i])

            # omega for residue i (peptide i→i+1)
            if i in omega_col_for_res and i < omega.shape[1]:
                omega_i = float(omega[f, omega_col_for_res[i]])
            else:
                omega_i = 180.0
            w = _wrap180(omega_i)
            cis = abs(w) <= 30.0
            if cis and r.name != "PRO":
                omega_pen = 1.0
            elif cis and r.name == "PRO":
                omega_pen = 0.3
            else:
                omega_pen = min(1.0, abs(abs(w) - 180.0) / 60.0)

            # bond-length Zmax across the 3 bonds
            bl_z = 0.0
            for name in ("d_N_CA", "d_CA_C", "d_C_Np1"):
                # find column with matching (name, i)
                col = next((ci for ci, (nm, ri) in enumerate(tags) if nm == name and ri == i), None)
                if col is None or bond_df.empty:
                    continue
                val = dists[f, col]
                if name in med.index and name in mad.index:
                    z = _robust_z(val, med[name], mad[name])
                    bl_z = max(bl_z, z)
            bl_z = min(bl_z, 4.0)

            # bond-angle Zmax
            ba_z = 0.0
            for name in ("ang_Cnm1_N_CA", "ang_N_CA_C", "ang_CA_C_Np1"):
                col = next((ci for ci, (nm, ri) in enumerate(atags) if nm == name and ri == i), None)
                if col is None or ang_df.empty:
                    continue
                val = angs[f, col]
                if name in ang_med.index and name in ang_mad.index:
                    z = _robust_z(val, ang_med[name], ang_mad[name])
                    ba_z = max(ba_z, z)
            ba_z = min(ba_z, 4.0)

            # side-chain rotamer deviation proxy
            chi_dev_i = float(chi_dev[i][f]) if i in chi_dev else 0.0
            # SASA norm
            sasa_i = float(sasa_norm[f, i]) if sasa_norm.shape[1] > i else 0.0
            # DSSP (H/E/C or 'NA')
            ss_i = dssp[f, i] if has_dssp else 'NA'

            # composite local score (you can retune weights later)
            score = (
                1.00 * (1 - rama_allowed)
                + 0.50 * max(0.0, rama_d - 1.0)
                + 0.75 * omega_pen
                + 0.50 * min(1.0, bl_z / 3.0)
                + 0.50 * min(1.0, ba_z / 3.0)
                + 0.25 * min(1.0, chi_dev_i / 60.0)  # χ deviation ~ 60° bins
            )

            rows.append({
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
                "chi_rotamer_dev": chi_dev_i,
                "sasa_norm": sasa_i,
                "ss": ss_i,                      # 'H','E','C' or 'NA'
                "local_score": score,
            })

    return pd.DataFrame(rows)

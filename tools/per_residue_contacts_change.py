import argparse, glob
from pathlib import Path
import numpy as np
import mdtraj as md

def write_bfactors_for_frame(pdb_in, pdb_out, res_scores, resnum_from_line):
    with open(pdb_in) as f:
        lines = f.readlines()
    out = []
    for ln in lines:
        if ln.startswith(("ATOM","HETATM")) and len(ln) >= 66:
            rn = resnum_from_line(ln)
            b  = float(res_scores.get(rn, 0.0))
            nl = "\n" if ln.endswith("\n") else ""
            core = ln[:-1] if nl else ln
            if len(core) < 66: core = core + " "*(66-len(core))
            bf = f"{b:6.2f}"
            core = core[:60] + bf + core[66:]
            ln = core + nl
        out.append(ln)
    Path(pdb_out).write_text("".join(out))

def resnum_from_pdb_line(line):
    return int(line[22:26])  # PDB columns 23-26

def main(top, traj, frames_glob, out_dir, cutoff_nm=0.8):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    t = md.load(traj, top=top)
    ca = t.topology.select("name CA")
    if len(ca) < 2: raise RuntimeError("Not enough CA atoms to compute contacts.")
    ii, jj = np.triu_indices(len(ca), k=1)
    pairs = np.stack([ca[ii], ca[jj]], axis=1)

    d = md.compute_distances(t, pairs)          # [T, P]
    C = (d < cutoff_nm).astype(np.int8)         # contacts

    atom_to_res = {a.index: a.residue.index for a in t.topology.atoms}
    pair_res = np.array([[atom_to_res[pairs[k,0]], atom_to_res[pairs[k,1]]] for k in range(pairs.shape[0])], dtype=int)

    pdb_frames = sorted(glob.glob(frames_glob))
    n = min(C.shape[0], len(pdb_frames))

    for f in range(n):
        if f == 0:
            res_scores = {}
        else:
            flips = np.abs(C[f] - C[f-1])
            res_changes = {}
            for k, flip in enumerate(flips):
                if flip:
                    r1, r2 = pair_res[k]
                    res_changes[r1] = res_changes.get(r1, 0) + 1
                    res_changes[r2] = res_changes.get(r2, 0) + 1
            if res_changes:
                vals = np.array(list(res_changes.values()), float)
                mn, mx = float(vals.min()), float(vals.max())
                if mx > mn:
                    for r in list(res_changes.keys()):
                        res_changes[r] = 100.0 * (res_changes[r] - mn) / (mx - mn)
                else:
                    for r in list(res_changes.keys()):
                        res_changes[r] = 0.0
            res_scores = res_changes

        src = pdb_frames[f]
        dst = Path(out) / Path(src).name.replace("frame_", "colored_res_")
        write_bfactors_for_frame(src, dst, res_scores, resnum_from_pdb_line)
        print(f"[ok] wrote {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-residue B-factors from CA-CA contact change")
    ap.add_argument("--top", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cutoff", type=float, default=0.8)
    args = ap.parse_args()
    main(args.top, args.traj, args.frames_glob, args.out_dir, cutoff_nm=args.cutoff)

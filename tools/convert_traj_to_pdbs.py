# tools/convert_traj_to_pdbs.py
import argparse, os, pathlib
import mdtraj as md

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True, help="Topology file (.pdb/.psf/.prmtop)")
    ap.add_argument("--traj", required=True, nargs="+", help="Trajectory file(s) (.dcd/.xtc/.trr/.nc)")
    ap.add_argument("--stride", type=int, default=10, help="Keep every Nth frame")
    ap.add_argument("--out_dir", default="data/frames_clean", help="Output folder for PDB frames")
    ap.add_argument("--keep_ligand", action="store_true", help="Keep hetero atoms/ligands")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] top={args.top} traj={args.traj} stride={args.stride}")
    traj = md.load(args.traj, top=args.top, stride=args.stride)

    # select atoms
    sel = traj.topology.select("protein") if not args.keep_ligand else traj.topology.select("protein or not water")
    traj = traj.atom_slice(sel)

    # center (optional but keeps boxes reasonable)
    traj.center_coordinates()

    # write frames
    for i in range(traj.n_frames):
        path = out / f"frame_{i:05d}.pdb"
        traj[i].save_pdb(str(path))
    print(f"[done] wrote {traj.n_frames} frames to {out}")

if __name__ == "__main__":
    main()

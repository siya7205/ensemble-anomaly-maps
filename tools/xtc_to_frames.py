#!/usr/bin/env python3
"""
Convert one or more XTC trajectories to PDB frames and an optional multi-model PDB.
Requires: MDAnalysis
"""
import argparse, os
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter

def write_multi_model(universe, stride, out_path):
    from MDAnalysis.coordinates.PDB import PDBWriter
    with PDBWriter(out_path, multiframe=True) as w:
        for ts in universe.trajectory[::stride]:
            w.write(universe.atoms)

def write_frames(universe, stride, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    i = 0
    for ts in universe.trajectory[::stride]:
        wpath = out / f"frame_{i:05d}.pdb"
        with PDBWriter(str(wpath)) as w:
            w.write(universe.atoms)
        i += 1
    return i

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--xtc", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--multi", default=None)
    a = ap.parse_args()

    u = mda.Universe(a.top, a.xtc)
    n = write_frames(u, a.stride, a.out_dir)
    print(f"[xtc] Wrote {n} frames → {a.out_dir}")
    if a.multi:
        write_multi_model(u, a.stride, a.multi)
        print(f"[xtc] Wrote multi-model PDB → {a.multi}")

if __name__ == "__main__":
    main()

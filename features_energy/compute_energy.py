#!/usr/bin/env python3
"""
Per-residue energetic features for MD trajectories.

Implements a fast surrogate model for MM/PBSA-like energetic decomposition:
- Knowledge-based contact potentials
- Hydrogen bond counting
- Electrostatic proxies

Output: data/derived/residue_energy.parquet
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import mdtraj as md
from scipy.spatial.distance import cdist


# Knowledge-based potential parameters (simplified MJ potential)
# Values in kcal/mol for C-alpha contacts
CONTACT_POTENTIAL = {
    ('ALA', 'ALA'): -0.3, ('ALA', 'CYS'): -0.5, ('ALA', 'ASP'): -0.1,
    ('ALA', 'GLU'): -0.1, ('ALA', 'PHE'): -0.7, ('ALA', 'GLY'): -0.2,
    ('ALA', 'HIS'): -0.4, ('ALA', 'ILE'): -0.6, ('ALA', 'LYS'): -0.1,
    ('ALA', 'LEU'): -0.6, ('ALA', 'MET'): -0.5, ('ALA', 'ASN'): -0.2,
    ('ALA', 'PRO'): -0.3, ('ALA', 'GLN'): -0.2, ('ALA', 'ARG'): -0.2,
    ('ALA', 'SER'): -0.2, ('ALA', 'THR'): -0.3, ('ALA', 'VAL'): -0.5,
    ('ALA', 'TRP'): -0.6, ('ALA', 'TYR'): -0.5,
    # Add more pairs... (for brevity, using symmetric defaults)
}

# Hydrophobic residues
HYDROPHOBIC = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}

# Charged residues
CHARGED_POSITIVE = {'LYS', 'ARG', 'HIS'}
CHARGED_NEGATIVE = {'ASP', 'GLU'}

# Polar residues
POLAR = {'SER', 'THR', 'ASN', 'GLN', 'CYS', 'TYR'}


def get_contact_energy(res1, res2, distance):
    """
    Compute contact energy using knowledge-based potential.
    
    Args:
        res1, res2: Residue names
        distance: C-alpha distance in nm
        
    Returns:
        energy: Contact energy in kcal/mol
    """
    # Convert to Angstroms
    dist_A = distance * 10.0
    
    # Contact cutoff (8 Angstroms)
    if dist_A > 8.0:
        return 0.0
    
    # Get potential (symmetric)
    pair = tuple(sorted([res1, res2]))
    base_energy = CONTACT_POTENTIAL.get(pair, -0.3)  # Default weak attractive
    
    # Distance-dependent weighting (6-12 Lennard-Jones like)
    r0 = 6.0  # Optimal distance in Angstroms
    epsilon = np.abs(base_energy)
    
    if dist_A < r0:
        # Repulsive at very short distances
        energy = epsilon * ((r0/dist_A)**12 - 2*(r0/dist_A)**6)
    else:
        # Attractive at moderate distances
        energy = base_energy * np.exp(-(dist_A - r0)**2 / 4.0)
    
    return energy


def count_hbonds_simple(traj, frame_idx, residue_idx, cutoff_dist=0.35, cutoff_angle=120):
    """
    Count hydrogen bonds for a residue using simple geometric criteria.
    
    Args:
        traj: MDTraj trajectory
        frame_idx: Frame index
        residue_idx: Residue index
        cutoff_dist: Donor-acceptor distance cutoff in nm
        cutoff_angle: D-H...A angle cutoff in degrees
        
    Returns:
        n_hbonds: Number of hydrogen bonds
    """
    # Simplified: count based on backbone N-H...O=C geometry
    # In production, use md.baker_hubbard() or wernet_nilsson()
    
    # For now, use a proxy: nearby polar/charged residues
    residue = list(traj.topology.residues)[residue_idx]
    res_name = residue.name
    
    if res_name not in (POLAR | CHARGED_POSITIVE | CHARGED_NEGATIVE):
        return 0
    
    # Get CA position for this residue
    ca_atoms = [a for a in residue.atoms if a.name == 'CA']
    if not ca_atoms:
        return 0
    
    ca_idx = ca_atoms[0].index
    ca_pos = traj.xyz[frame_idx, ca_idx]
    
    # Find nearby polar/charged residues
    hbond_count = 0
    for other_res in traj.topology.residues:
        if other_res.index == residue_idx:
            continue
        
        other_name = other_res.name
        if other_name not in (POLAR | CHARGED_POSITIVE | CHARGED_NEGATIVE):
            continue
        
        # Get other CA position
        other_ca = [a for a in other_res.atoms if a.name == 'CA']
        if not other_ca:
            continue
        
        other_ca_idx = other_ca[0].index
        other_pos = traj.xyz[frame_idx, other_ca_idx]
        
        dist = np.linalg.norm(ca_pos - other_pos)
        
        # Heuristic: if close enough and compatible polarity
        if dist < cutoff_dist:
            # Check if donor-acceptor pair
            if (res_name in CHARGED_POSITIVE and other_name in CHARGED_NEGATIVE) or \
               (res_name in CHARGED_NEGATIVE and other_name in CHARGED_POSITIVE) or \
               (res_name in POLAR and other_name in POLAR):
                hbond_count += 1
    
    return hbond_count


def compute_residue_energies(traj, contact_cutoff=0.8, progress_callback=None):
    """
    Compute per-residue energetic proxies for all frames.
    
    Args:
        traj: MDTraj trajectory
        contact_cutoff: Distance cutoff for contacts in nm
        progress_callback: Optional callback(frame_idx, total_frames)
        
    Returns:
        df: DataFrame with columns [frame, res_id, chain, energy, hbonds]
    """
    n_frames = len(traj)
    residues = list(traj.topology.residues)
    n_residues = len(residues)
    
    # Get CA atoms for all residues
    ca_indices = []
    res_names = []
    res_ids = []
    chains = []
    
    for res in residues:
        ca_atoms = [a for a in res.atoms if a.name == 'CA']
        if ca_atoms:
            ca_indices.append(ca_atoms[0].index)
            res_names.append(res.name)
            res_ids.append(res.resSeq)
            chains.append(res.chain.chain_id)
    
    ca_indices = np.array(ca_indices)
    n_ca = len(ca_indices)
    
    # Storage for results
    results = []
    
    # Process each frame
    for frame_idx in range(n_frames):
        if progress_callback:
            progress_callback(frame_idx, n_frames)
        
        # Get CA positions for this frame
        ca_positions = traj.xyz[frame_idx, ca_indices]  # (n_ca, 3)
        
        # Compute pairwise distances
        distances = cdist(ca_positions, ca_positions)  # (n_ca, n_ca)
        
        # Compute energies for each residue
        for i in range(n_ca):
            total_energy = 0.0
            
            # Sum contact energies with all other residues
            for j in range(n_ca):
                if i == j:
                    continue
                
                dist_nm = distances[i, j]
                
                if dist_nm < contact_cutoff:
                    energy = get_contact_energy(res_names[i], res_names[j], dist_nm)
                    total_energy += energy
            
            # Count hydrogen bonds (simplified)
            n_hbonds = count_hbonds_simple(traj, frame_idx, i)
            
            # Add electrostatic contribution (simplified)
            if res_names[i] in CHARGED_POSITIVE:
                # Attraction to negative charges
                for j in range(n_ca):
                    if res_names[j] in CHARGED_NEGATIVE and distances[i, j] < contact_cutoff:
                        total_energy -= 2.0 * np.exp(-distances[i, j] / 0.3)
            elif res_names[i] in CHARGED_NEGATIVE:
                # Attraction to positive charges
                for j in range(n_ca):
                    if res_names[j] in CHARGED_POSITIVE and distances[i, j] < contact_cutoff:
                        total_energy -= 2.0 * np.exp(-distances[i, j] / 0.3)
            
            results.append({
                'frame': frame_idx,
                'res_id': res_ids[i],
                'chain': chains[i],
                'energy': total_energy,
                'hbonds': n_hbonds
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-residue energetic features'
    )
    parser.add_argument('--topology', required=True,
                       help='Path to topology file (PDB)')
    parser.add_argument('--trajectory', required=True,
                       help='Path to trajectory file (XTC, DCD, etc.)')
    parser.add_argument('--output', default='data/derived/residue_energy.parquet',
                       help='Output parquet file')
    parser.add_argument('--contact_cutoff', type=float, default=0.8,
                       help='Contact cutoff in nm (default: 0.8)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for trajectory (default: 1)')
    
    args = parser.parse_args()
    
    print(f"[1/3] Loading trajectory")
    print(f"  Topology: {args.topology}")
    print(f"  Trajectory: {args.trajectory}")
    
    traj = md.load(args.trajectory, top=args.topology, stride=args.stride)
    print(f"  Loaded: {len(traj)} frames, {traj.n_residues} residues")
    
    print(f"[2/3] Computing per-residue energies")
    
    def progress(frame_idx, total):
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx + 1}/{total}")
    
    df = compute_residue_energies(traj, args.contact_cutoff, progress)
    
    print(f"[3/3] Saving results")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"\nSummary statistics:")
    print(df.groupby('frame')['energy'].agg(['mean', 'std', 'min', 'max']))


if __name__ == '__main__':
    main()

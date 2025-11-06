# app/data_loader.py - Data loading utilities for interactive visualization
import os
import json
import pathlib
import pandas as pd
import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    mda = None


class DataLoader:
    """Loads and caches trajectory, PDB, and analysis data for the viewer."""
    
    def __init__(self, root_dir=None):
        self.root = pathlib.Path(root_dir or pathlib.Path(__file__).resolve().parents[1])
        self.data_dir = self.root / "data"
        self.outputs_dir = self.root / "outputs"
        
        # Cache
        self._universe = None
        self._hotspots = None
        self._rmsf = None
        self._residue_info = None
        
    def _latest_run(self, pattern="run-traj-*"):
        """Find the most recent run directory."""
        paths = sorted(self.outputs_dir.glob(pattern), 
                      key=lambda p: p.stat().st_mtime, reverse=True)
        return paths[0] if paths else None
    
    def get_universe(self, pdb_path=None, traj_path=None):
        """Load MDAnalysis Universe. Cache it after first load."""
        if self._universe is not None:
            return self._universe
            
        if mda is None:
            raise RuntimeError("MDAnalysis not available")
        
        # Default to multi_model_anomaly.pdb
        if pdb_path is None:
            pdb_path = self.data_dir / "multi_model_anomaly.pdb"
        
        if not pathlib.Path(pdb_path).exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
        # Load with or without trajectory
        if traj_path and pathlib.Path(traj_path).exists():
            self._universe = mda.Universe(str(pdb_path), str(traj_path))
        else:
            self._universe = mda.Universe(str(pdb_path))
        
        return self._universe
    
    def get_trajectory_meta(self):
        """Get basic trajectory metadata."""
        u = self.get_universe()
        return {
            "n_frames": len(u.trajectory),
            "n_atoms": len(u.atoms),
            "n_residues": len(u.residues),
            "n_chains": len(set(u.atoms.segids)),
        }
    
    def get_frame_coordinates(self, frame_idx=0):
        """Get XYZ coordinates for all atoms in a specific frame."""
        u = self.get_universe()
        if frame_idx < 0 or frame_idx >= len(u.trajectory):
            raise ValueError(f"Frame {frame_idx} out of range [0, {len(u.trajectory)-1}]")
        
        u.trajectory[frame_idx]
        positions = u.atoms.positions  # shape (n_atoms, 3)
        
        return {
            "frame": frame_idx,
            "n_atoms": len(u.atoms),
            "xyz": positions.tolist()
        }
    
    def get_residue_map(self):
        """Map atom indices to residue numbers."""
        u = self.get_universe()
        # Return resid for each atom
        resnums = []
        for atom in u.atoms:
            resnums.append(int(atom.resid))
        return resnums
    
    def get_residue_table(self):
        """Get detailed residue information."""
        if self._residue_info is not None:
            return self._residue_info
        
        u = self.get_universe()
        residues = []
        for idx, res in enumerate(u.residues):
            residues.append({
                "index": idx,
                "resid": int(res.resid),
                "resname": str(res.resname),
                "chain": str(res.segid) if res.segid else "",
                "n_atoms": len(res.atoms)
            })
        
        self._residue_info = residues
        return residues
    
    def load_hotspots(self):
        """Load residue hotspot scores from latest deep run."""
        if self._hotspots is not None:
            return self._hotspots
        
        # Find a run with deep analysis
        runs = sorted(self.outputs_dir.glob("run-traj-*"), 
                     key=lambda p: p.stat().st_mtime, reverse=True)
        hotspot_file = None
        for run in runs:
            candidate = run / "deep" / "residue_hotspots.csv"
            if candidate.exists():
                hotspot_file = candidate
                break
        
        if not hotspot_file:
            return {}
        
        df = pd.read_csv(hotspot_file)
        # Create dict: resid -> hotspot_score
        hotspots = {}
        if "resid" not in df.columns:
            return {}
        
        # Try different column formats
        score_col = None
        if "delta_err" in df.columns:
            score_col = "delta_err"
        elif "total_contrib" in df.columns:
            score_col = "total_contrib"
        elif "mean_local_score" in df.columns:
            score_col = "mean_local_score"
        
        if not score_col:
            return {}
        
        for _, row in df.iterrows():
            resid = int(row["resid"])
            hotspots[resid] = {
                "score": float(row[score_col]),
                "delta_err": float(row.get("delta_err", row[score_col])),
                "mean_recon_err": float(row.get("overall_mean_recon_err", 0)),
                "n_windows": int(row.get("n_windows_considered", 0)),
                "rank": int(row.get("rank", 0))
            }
        
        self._hotspots = hotspots
        return hotspots
    
    def load_rmsf(self):
        """Load RMSF data if available."""
        if self._rmsf is not None:
            return self._rmsf
        
        # Find a run with per_residue_overall.csv
        runs = sorted(self.outputs_dir.glob("run-traj-*"), 
                     key=lambda p: p.stat().st_mtime, reverse=True)
        rmsf_file = None
        for run in runs:
            candidate = run / "per_residue_overall.csv"
            if candidate.exists():
                rmsf_file = candidate
                break
        
        if not rmsf_file:
            return {}
        
        df = pd.read_csv(rmsf_file)
        rmsf_data = {}
        if "resid" in df.columns:
            for _, row in df.iterrows():
                resid = int(row["resid"])
                rmsf_data[resid] = {
                    "rmsf": float(row.get("rmsf", 0)) if "rmsf" in row else None,
                    "rgyr": float(row.get("rgyr", 0)) if "rgyr" in row else None,
                    "sasa": float(row.get("sasa", 0)) if "sasa" in row else None,
                }
        
        self._rmsf = rmsf_data
        return rmsf_data
    
    def get_residue_details(self, resid):
        """Get comprehensive details for a specific residue."""
        residues = self.get_residue_table()
        hotspots = self.load_hotspots()
        rmsf = self.load_rmsf()
        
        # Find residue info
        res_info = next((r for r in residues if r["resid"] == resid), None)
        if not res_info:
            return None
        
        # Merge with analysis data
        details = res_info.copy()
        details["hotspot"] = hotspots.get(resid, {})
        details["rmsf_data"] = rmsf.get(resid, {})
        
        return details
    
    def get_atom_info(self, atom_idx):
        """Get information about a specific atom."""
        u = self.get_universe()
        if atom_idx < 0 or atom_idx >= len(u.atoms):
            return None
        
        atom = u.atoms[atom_idx]
        return {
            "index": atom_idx,
            "name": str(atom.name),
            "type": str(atom.type) if hasattr(atom, "type") else "",
            "element": str(atom.element) if hasattr(atom, "element") else "",
            "resid": int(atom.resid),
            "resname": str(atom.resname),
            "chain": str(atom.segid) if atom.segid else "",
            "mass": float(atom.mass) if hasattr(atom, "mass") else 0.0,
        }


# Global instance
_loader = None

def get_loader():
    """Get singleton data loader instance."""
    global _loader
    if _loader is None:
        _loader = DataLoader()
    return _loader

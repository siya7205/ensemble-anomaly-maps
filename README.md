# Ensemble-Clean Capstone

This project builds and serves ensemble-based anomaly detection for protein structures, with a Ramachandran (φ/ψ) HTML viewer.

---

## Quickstart (conda)

```bash
# 0) (first time only) create environment
conda create -y -n mdcapstone python=3.10
conda activate mdcapstone

# 1) install dependencies
conda install -y -c conda-forge mdtraj pandas numpy scikit-learn pyarrow flask plotly

# 2) clone the repository (if not already cloned)
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>/ensemble-clean

# 3) run the Flask app
export PORT=5051
PYTHONPATH=$(pwd) python -m app.app

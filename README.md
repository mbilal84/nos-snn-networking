# Network-Optimised Spiking (NOS)

Reference code, notebooks, and interactive demos for **Network-Optimised Spiking (NOS)**.

- **Paper (arXiv):** https://arxiv.org/abs/2509.23516  
- **DOI:** https://doi.org/10.48550/arXiv.2509.23516  
- **Live demos (GitHub Pages):** https://mbilal84.github.io/nos-snn-networking/  
- **Repository:** https://github.com/mbilal84/nos-snn-networking  

## What is NOS

NOS is a compact two-state spiking model for event-driven networking. The state **v** represents normalised queue occupancy, and **u** captures recovery or resource. The model emits discrete events (“spikes”) that can be used for monitoring and control, with graph-local coupling for networked ssystems.

## Repository layout

- `src/`  
  Core Python modules used across experiments (for example: `src/nos.py`, `src/topology.py`, `src/linearisation.py`, `src/metrics.py`).

- `notebooks/`  
  Main Jupyter notebooks (01–11) used to generate paper figures and tables.

- `ablation/`  
  Ablation scripts that mirror notebook pipelines.

- `docs/`  
  GitHub Pages site (landing page + demos).

- `docs/demos/`  
  The three interactive HTML demos.

## Quick start

### Requirements
- Python 3.10+ recommended (older versions may work, but are not tested here).

### Install dependencies

From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
```

If you prefer an editable install (recommended for `src/` layout):

```bash
pip install -e .
```

## Running notebooks (modules live in `src/`)

These notebooks import modules from the repository `src/` directory. To run them **without editing notebooks**, use one of the following.

### Recommended (editable install)

From the repository root:

```bash
pip install -e .
jupyter lab
```

Then open any notebook under `notebooks/`. Imports should work as:

```python
import nos
import topology
```

### Alternative (no install)

Start Jupyter from the repository root with `src/` on `PYTHONPATH`.

**Windows (PowerShell)**

```powershell
$env:PYTHONPATH = "$(Get-Location)\src"
jupyter lab
```

**macOS / Linux (bash/zsh)**

```bash
export PYTHONPATH="$PWD/src"
jupyter lab
```

If imports fail, it usually means Jupyter was started from the wrong directory. Close it and repeat the steps from the repository root.



## Citation

If you use NOS code, models, or results in your work, please cite the paper:

```bibtex
@article{bilal2025nos,
  title   = {Network-Optimised Spiking Neural Network for Event-Driven Networking},
  author  = {Bilal, Muhammad},
  journal = {arXiv preprint arXiv:2509.23516},
  year    = {2025},
  doi     = {10.48550/arXiv.2509.23516},
  url     = {https://arxiv.org/abs/2509.23516}
}
```

This repository also includes:
- `CITATION.cff` (for GitHub’s citation widget)
- `CITATION.bib` (BibTeX file)

## Licence and copyright

- **Licence:** MIT (see `LICENSE`).
- **Copyright:** © 2025 Muhammad Bilal.


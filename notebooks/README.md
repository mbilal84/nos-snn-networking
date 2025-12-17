# Notebooks


## Running notebooks (imports from `src/`)

These notebooks import NOS modules from the repository `src/` directory (for example `src/nos.py`, `src/topology.py`).

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

Inside a notebook, imports should work as:

```python
import nos
import topology
```



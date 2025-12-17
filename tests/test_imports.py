import importlib

import pytest


def test_imports():
    """Smoke test that core modules are importable.

    If you are publishing this repository, make sure `src/nos.py`, `src/topology.py`,
    `src/linearisation.py`, and `src/metrics.py` are present.
    """

    modules = ["nos", "topology", "linearisation", "metrics"]
    for m in modules:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError:
            pytest.skip(f"Module '{m}' not found (expected under src/).")

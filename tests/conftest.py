from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MWE_DATA = Path(__file__).parent.parent / "mwe_data"
EXP_DATA = Path(__file__).parent.parent / "experiment_workflow_aps_28Feb"


def make_vms_csv(path: Path, nx: int = 10, ny: int = 10, seed: int = 0) -> Path:
    """Write a small synthetic von-Mises stress CSV."""
    rng = np.random.default_rng(seed)
    n = nx * ny
    xs = np.tile(np.arange(nx, dtype=float), ny)
    ys = np.repeat(np.arange(ny, dtype=float), nx)
    sxx = rng.normal(100.0, 20.0, n)
    syy = rng.normal(50.0, 10.0, n)
    szz = rng.normal(30.0, 5.0, n)
    sxy = rng.normal(0.0, 5.0, n)
    sxz = rng.normal(0.0, 5.0, n)
    syz = rng.normal(0.0, 5.0, n)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "x": xs,
            "y": ys,
            "z": np.zeros(n),
            "sxx": sxx,
            "syy": syy,
            "szz": szz,
            "sxy": sxy,
            "sxz": sxz,
            "syz": syz,
        }
    )
    df.to_csv(path, index=False)
    return path


def make_grain_csv(path: Path, n_grains: int = 20, seed: int = 42) -> Path:
    """Write a small synthetic grain CSV with positions and Euler angles."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n_grains + 1),
            "X": rng.uniform(0, 500, n_grains),
            "Y": rng.uniform(0, 500, n_grains),
            "Z": rng.uniform(0, 500, n_grains),
            "GrainRadius": rng.uniform(20, 80, n_grains),
            "Eul0": rng.uniform(0, 360, n_grains),
            "Eul1": rng.uniform(0, 180, n_grains),
            "Eul2": rng.uniform(0, 360, n_grains),
        }
    )
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def vms_csv(tmp_path):
    return make_vms_csv(tmp_path / "vms.csv")


@pytest.fixture
def grain_csv(tmp_path):
    return make_grain_csv(tmp_path / "grains.csv")


@pytest.fixture
def mwe_data():
    if not MWE_DATA.exists():
        pytest.skip("mwe_data/ not found")
    return MWE_DATA


@pytest.fixture
def exp_data():
    if not EXP_DATA.exists():
        pytest.skip("experiment_workflow_aps_28Feb/ not found")
    return EXP_DATA

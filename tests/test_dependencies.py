"""Check that optional heavy dependencies are available."""
from __future__ import annotations

import os
import pytest


def test_neml2_importable():
    """neml2 must be importable (built from MOOSE contrib or PyPI)."""
    import neml2  # noqa: F401


def test_neml2_has_tensors():
    """neml2.tensors must expose Rot for orientation math."""
    from neml2 import tensors
    assert hasattr(tensors, "Rot"), "neml2.tensors.Rot not found"
    assert hasattr(tensors, "Vec"), "neml2.tensors.Vec not found"


def test_neml2_has_crystallography():
    """neml2.crystallography must expose symmetry()."""
    from neml2 import crystallography
    assert hasattr(crystallography, "symmetry") or hasattr(
        crystallography, "symmetry_operations_from_orbifold"
    ), "neml2.crystallography.symmetry not found"


def test_cubit_psculpt_exists():
    """psculpt binary must exist and be executable."""
    psculpt = "/home/tranh/Progs/cubit_gov/bin/psculpt"
    assert os.path.isfile(psculpt), f"psculpt not found at {psculpt}"
    assert os.access(psculpt, os.X_OK), f"psculpt not executable: {psculpt}"


def test_cubit_mpiexec_exists():
    """mpiexec for CUBIT must exist."""
    mpiexec = "/home/tranh/Progs/cubit_gov/bin/mpi/bin/mpiexec"
    assert os.path.isfile(mpiexec), f"mpiexec not found at {mpiexec}"
    assert os.access(mpiexec, os.X_OK), f"mpiexec not executable: {mpiexec}"

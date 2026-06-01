# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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

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

"""Per-grain orientation history output of the uniaxial Taylor model."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("neml2")
pytest.importorskip("pyzag")

# pylint: disable=wrong-import-position
import graintrace as _gt
from graintrace.taylor import TaylorModel

_CALIB_I = str(Path(_gt.__file__).parent / "cpfe_base" / "neml2_cpfe_calibration.i")
_N_GRAINS = 5
_NTIME = 25
_MAX_STRAIN = 0.015
_ELASTIC_STRAIN = 1.0e-3  # well below yield (~0.15% axial) for the shipped params
_RATE = 1.0e-4


def _misorientation_deg(r1, r2):
    """Geodesic angle (deg) between two neml2 MRP tensors, shadow-aware."""
    from neml2.types import MRP, dist  # pylint: disable=import-outside-toplevel

    d = dist(MRP(r1.contiguous()), MRP(r2.contiguous()))
    return torch.rad2deg(torch.as_tensor(getattr(d, "data", d)))


@pytest.fixture(scope="module")
def setup():
    """Model, random orientations and strain grid, plus a full-output run."""
    torch.manual_seed(42)
    model = TaylorModel(_CALIB_I, device="cpu", nchunk=5)
    axis = torch.nn.functional.normalize(
        torch.randn(_N_GRAINS, 3, dtype=torch.float64), dim=1
    )
    angle = torch.rand(_N_GRAINS, dtype=torch.float64) * np.pi
    ori = axis * torch.tan(angle / 4.0)[:, None]  # neml2 v3 MRP
    strain = torch.linspace(0.0, _MAX_STRAIN, _NTIME, dtype=torch.float64)
    with torch.no_grad():
        full = model.tmodel(
            ori, strain, _RATE, return_state=True, return_orientation=True
        )
    return model, ori, strain, full


def test_default_and_return_state_unchanged(setup):
    """Default returns a bare tensor; return_state alone returns the same 2-tuple."""
    model, ori, strain, full = setup
    with torch.no_grad():
        stress = model.tmodel(ori, strain, _RATE)
        out = model.tmodel(ori, strain, _RATE, return_state=True)
    assert isinstance(stress, torch.Tensor)
    assert stress.shape == (_NTIME, 6)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (_NTIME, 6)
    assert out[1].shape == (_NTIME, _N_GRAINS, 6)
    # stress identical with and without the extra outputs
    assert torch.allclose(stress, full[0], rtol=1e-10, atol=1e-10)
    assert torch.allclose(out[1], full[1], rtol=1e-10, atol=1e-12)


def test_orientation_only_ordering(setup):
    """return_orientation alone gives (stress, orientation)."""
    model, ori, strain, full = setup
    with torch.no_grad():
        out = model.tmodel(ori, strain, _RATE, return_orientation=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (_NTIME, 6)
    assert out[1].shape == (_NTIME, _N_GRAINS, 3)
    assert torch.allclose(out[0], full[0], rtol=1e-10, atol=1e-10)
    assert torch.allclose(out[1], full[2], rtol=1e-10, atol=1e-12)


def test_orientation_history_shape_and_initial_value(setup):
    """Orientation history is (ntime, n_grains, 3), finite, and starts at the input."""
    _, ori, _, full = setup
    assert len(full) == 3
    ori_hist = full[2]
    assert ori_hist.shape == (_NTIME, _N_GRAINS, 3)
    assert torch.isfinite(ori_hist).all()
    assert torch.allclose(ori_hist[0], ori, rtol=0.0, atol=1e-12)


def test_rotation_fixed_elastic_then_measurable_after_yield(setup):
    """Grains do not rotate in the elastic regime but do after yield."""
    _, ori, strain, full = setup
    ori_hist = full[2]
    rot = _misorientation_deg(ori_hist, ori.expand_as(ori_hist))  # (ntime, n_grains)
    elastic = strain <= _ELASTIC_STRAIN
    assert elastic.sum() >= 2
    assert rot[elastic].max() < 1e-3
    assert rot[-1].max() > 0.1


def test_simulate_forwards_return_orientation(setup):
    """TaylorModel.simulate passes return_orientation through to the forward."""
    model, ori, strain, full = setup
    experiment_data = {
        "exp_texture": [ori],
        "strain_stress": np.stack([strain.numpy(), np.zeros(_NTIME)], axis=1),
    }
    with torch.no_grad():
        stress, ori_hist = model.simulate(
            None,
            assumed_rate=_RATE,
            experiment_data=experiment_data,
            return_orientation=True,
        )
    assert torch.allclose(stress, full[0], rtol=1e-10, atol=1e-10)
    assert torch.allclose(ori_hist, full[2], rtol=1e-10, atol=1e-12)

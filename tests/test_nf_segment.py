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

"""Tests for the NF flood-fill segmentation budget handling."""

from __future__ import annotations

import pytest
import torch

from graintrace.nf import segment


def _deterministic_seed_choice(monkeypatch):
    """Always seed the flood from the first unsegmented voxel.

    ``flood`` picks its seed with ``torch.randint``; pinning it to index 0 makes
    the order in which segments are discovered deterministic.
    """
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([0]))


def _two_grain_chain():
    """A 1D chain holding one sub-threshold grain and one large grain.

    Voxels 0-2 are a 3-voxel grain at 0 deg; voxels 5-14 are a 10-voxel grain at
    90 deg. The two are far apart both in space and in orientation, so a flood
    can never bridge them.
    """
    nx = 16
    angles = torch.zeros(nx, 1, 1, 3, dtype=torch.float64)
    phase = torch.zeros(nx, 1, 1, dtype=torch.float64)
    phase[0:3] = 1.0
    phase[5:15] = 1.0
    angles[5:15, 0, 0, 0] = 90.0
    return angles, phase


def test_flood_does_not_refind_discarded_segments(monkeypatch):
    """A discarded small segment must not be re-found and re-discarded.

    With ``stop_count=2`` the 3-voxel grain is below ``grain_threshold`` and is
    thrown away. If it is reset to "unsegmented" it is picked again on the next
    iteration, the budget runs out, and the 10-voxel grain is never segmented.
    """
    _deterministic_seed_choice(monkeypatch)
    angles, phase = _two_grain_chain()

    out = segment.flood(
        angles,
        phase,
        5.0,
        connectivity=6,
        grain_threshold=5,
        stop_count=2,
        angle_convention="bunge",
        angle_type="degrees",
        symmetry="432",
    )

    # The large grain is kept with a real segment id.
    assert torch.all(out[5:15, 0, 0] >= 1)
    assert len(torch.unique(out[5:15, 0, 0])) == 1
    # The sub-threshold grain stays unsegmented, as before.
    assert torch.all(out[0:3, 0, 0] == -1)
    # Void voxels are untouched.
    assert torch.all(out[3:5, 0, 0] == 0)


@pytest.mark.parametrize("stop_count", [0, -1])
def test_flood_rejects_nonpositive_stop_count(stop_count):
    """``stop_count <= 0`` is a budget that can never be exhausted."""
    angles = torch.zeros(2, 1, 1, 3, dtype=torch.float64)
    phase = torch.zeros(2, 1, 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="stop_count"):
        segment.flood(
            angles,
            phase,
            5.0,
            connectivity=6,
            grain_threshold=5,
            stop_count=stop_count,
            angle_convention="bunge",
            angle_type="degrees",
        )

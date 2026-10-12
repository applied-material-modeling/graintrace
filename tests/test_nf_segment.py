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

from graintrace.nf import metrics, segment


def _deterministic_seed_choice(monkeypatch):
    """Always seed the flood from the first unsegmented voxel.

    ``flood`` picks its seed with ``torch.randint``; pinning it to index 0 makes
    the order in which segments are discovered deterministic.
    """
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([0]))


def _stub_misorientation(monkeypatch):
    """Replace the neml2-backed misorientation with a pure-torch stand-in.

    ``flood`` only consumes the neighbour distances as a scalar field compared
    against ``misorientation_tol``; the real metric needs neml2, which is not
    part of the base install. These tests are about the segmentation loop's
    seeding and budget accounting, not the orientation math, so the first Euler
    component's absolute difference is a faithful stand-in and keeps the test
    runnable without the compiled stack.
    """
    monkeypatch.setattr(
        metrics,
        "misorientation",
        lambda e1, e2, **kwargs: torch.abs(e1[:, 0] - e2[:, 0]),
    )


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
    _stub_misorientation(monkeypatch)
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
    """``stop_count <= 0`` is a budget that can never be exhausted.

    The grid holds only sub-threshold *material*, which is the case that
    actually hangs without the guard: every segment found is discarded, the
    counter never reaches zero, and the loop never terminates. An all-void grid
    would return immediately even without the guard and so would not exercise
    it. The guard runs before any distance is computed, so this needs no
    orientation metric.
    """
    angles = torch.zeros(4, 1, 1, 3, dtype=torch.float64)
    phase = torch.ones(4, 1, 1, dtype=torch.float64)

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


def _wrapping_neighbor_indices(monkeypatch):
    """Replace the clamping neighbour lookup with a wrapping one.

    ``flood``'s out-of-domain ``valid`` mask is inert unless it is applied, and
    with the real :func:`graintrace.nf.image.get_neighbor_indices` that is
    invisible: it clamps, so an out-of-domain offset resolves to a voxel the
    front would have reached anyway. This stand-in keeps the same ``valid``
    mask but wraps the indices instead of clamping them, which is what makes
    the mask observable -- exactly the latent trap #52 describes, where a
    change to the neighbour lookup turns a dead line into wrong labels at the
    domain boundary.
    """
    from graintrace.nf import segment as segment_module

    def _wrapping(offsets, shape):
        X, Y, Z = torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            torch.arange(shape[2]),
            indexing="ij",
        )
        Xk = X.unsqueeze(0) + offsets[:, 0][:, None, None, None]
        Yk = Y.unsqueeze(0) + offsets[:, 1][:, None, None, None]
        Zk = Z.unsqueeze(0) + offsets[:, 2][:, None, None, None]
        valid = (
            (Xk >= 0)
            & (Xk < shape[0])
            & (Yk >= 0)
            & (Yk < shape[1])
            & (Zk >= 0)
            & (Zk < shape[2])
        )
        return Xk % shape[0], Yk % shape[1], Zk % shape[2], valid

    monkeypatch.setattr(segment_module, "get_neighbor_indices", _wrapping)


def _sandwich_grid():
    """Anisotropic grid with two same-orientation slabs split by a third.

    ``nx, ny, nz = 6, 3, 2`` are all different, so an axis mix-up cannot pass
    unnoticed. Along x: voxels 0-1 and 4-5 are at 0 deg, voxels 2-3 at 90 deg.
    The two 0 deg slabs are separate grains -- unless the domain wraps, which
    would make voxel 0 and voxel 5 neighbours and merge them.
    """
    angles = torch.zeros(6, 3, 2, 3, dtype=torch.float64)
    angles[2:4, :, :, 0] = 90.0
    phase = torch.ones(6, 3, 2, dtype=torch.float64)
    return angles, phase


def _flood_sandwich(connectivity):
    angles, phase = _sandwich_grid()
    return segment.flood(
        angles,
        phase,
        5.0,
        connectivity=connectivity,
        grain_threshold=4,
        stop_count=10,
        angle_convention="bunge",
        angle_type="degrees",
        symmetry="432",
    )


class TestOutOfDomainGuardIsApplied:
    """The ``valid`` mask must actually reach ``distances`` (#52).

    The guard used the out-of-place ``masked_fill``, whose return value was
    discarded, so the mask was inert. That was label-identical on every real
    input only because the neighbour indices are clamped. These tests pin both
    halves of that claim: the guard is live, and turning it on changed nothing
    for clamped input.
    """

    @pytest.mark.parametrize("connectivity", [6, 26])
    def test_the_front_cannot_cross_the_domain_boundary(
        self, monkeypatch, connectivity
    ):
        """With a wrapping neighbour lookup, a dead mask merges the two slabs."""
        _deterministic_seed_choice(monkeypatch)
        _stub_misorientation(monkeypatch)
        _wrapping_neighbor_indices(monkeypatch)

        out = _flood_sandwich(connectivity)

        assert out[0, 0, 0] >= 1 and out[5, 0, 0] >= 1
        assert out[0, 0, 0] != out[5, 0, 0], (
            "the two 0 deg slabs were merged across the domain boundary, so "
            "the out-of-domain mask is not being applied"
        )

    @pytest.mark.parametrize("connectivity", [6, 26])
    def test_clamped_input_is_label_identical(self, monkeypatch, connectivity):
        """The real clamping lookup gives the same labels either way.

        This is the claim that made #52 LOW rather than HIGH, so it is pinned
        rather than argued: on the shipped neighbour lookup the live mask is a
        no-op, and the two slabs are still resolved as separate grains.
        """
        _deterministic_seed_choice(monkeypatch)
        _stub_misorientation(monkeypatch)

        out = _flood_sandwich(connectivity)

        assert out[0, 0, 0] != out[5, 0, 0]
        assert torch.all(out[0:2] == out[0, 0, 0])
        assert torch.all(out[4:6] == out[5, 0, 0])
        assert torch.all(out[2:4] >= 1)
        assert len(torch.unique(out)) == 3

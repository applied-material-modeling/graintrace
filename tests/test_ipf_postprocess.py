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

"""Tests for graintrace.ipf_postprocess block-colouring defaults."""

import glob

import numpy as np
import pytest


def _shipped_mesh():
    candidates = glob.glob("mwe_data/**/mesh.e", recursive=True)
    if not candidates:
        pytest.skip("no shipped Exodus hex mesh available")
    return candidates[0]


def _n_blocks(mesh_file):
    import scipy.io as sio

    with sio.netcdf_file(mesh_file, "r", mmap=False) as f:
        return int(f.dimensions["num_el_blk"])


def _read_rgb(path):
    import scipy.io as sio

    with sio.netcdf_file(str(path), "r", mmap=False) as f:
        names = [
            b"".join(row[row != b""]).decode().strip()
            for row in np.array(f.variables["name_elem_var"][:])
        ]
        out = {}
        for axis in ("rgb_x", "rgb_y", "rgb_z"):
            idx = names.index(axis) + 1
            blocks = [
                np.array(f.variables[f"vals_elem_var{idx}eb{eb + 1}"][:])
                for eb in range(int(f.dimensions["num_el_blk"]))
            ]
            out[axis] = np.concatenate([b.ravel() for b in blocks])
    return out


def _colour(tmp_path, mesh_file, ori_csv, tag, **kwargs):
    from graintrace.ipf_postprocess import IPFProcessor

    ipf = IPFProcessor(
        crystal_symmetry="432", sample_symmetry="1", save_dir=str(tmp_path)
    )
    out = ipf.add_block_rgb_to_exodus(
        mesh_file,
        str(ori_csv),
        direction=[0.0, 0.0, 1.0],
        output_file=f"rgb_{tag}.e",
        **kwargs,
    )
    return _read_rgb(out)


class TestBlockRgbDefaultConvention:
    """``add_block_rgb_to_exodus`` must default to the convention it reads (#50).

    An FF ``orientations.dat`` is Bunge in degrees. The old defaults were
    ``kocks`` + ``radians``, which silently produced a different, wrong
    colouring -- a picture, not an error.
    """

    @staticmethod
    def _orientations(tmp_path, mesh_file):
        rng = np.random.default_rng(42)
        ori = rng.uniform(
            [0.0, 0.0, 0.0], [360.0, 180.0, 360.0], size=(_n_blocks(mesh_file), 3)
        )
        path = tmp_path / "orientations.dat"
        np.savetxt(path, ori, fmt="%.12g", delimiter=",")
        return path

    def test_defaults_match_explicit_bunge_degrees(self, tmp_path):
        pytest.importorskip("neml2")
        mesh_file = _shipped_mesh()
        ori = self._orientations(tmp_path, mesh_file)

        default = _colour(tmp_path, mesh_file, ori, "default")
        explicit = _colour(
            tmp_path,
            mesh_file,
            ori,
            "explicit",
            angle_convention="bunge",
            angle_type="degrees",
        )
        for axis in ("rgb_x", "rgb_y", "rgb_z"):
            assert np.allclose(default[axis], explicit[axis]), (
                f"{axis} differs: the default convention is not bunge/degrees, "
                "so an FF orientations.dat is coloured wrong by default"
            )

    def test_the_old_defaults_really_did_colour_differently(self, tmp_path):
        """Positive control: kocks/radians is a genuinely different picture."""
        pytest.importorskip("neml2")
        mesh_file = _shipped_mesh()
        ori = self._orientations(tmp_path, mesh_file)

        new = _colour(tmp_path, mesh_file, ori, "new")
        old = _colour(
            tmp_path,
            mesh_file,
            ori,
            "old",
            angle_convention="kocks",
            angle_type="radians",
        )
        assert not all(
            np.allclose(new[axis], old[axis]) for axis in ("rgb_x", "rgb_y", "rgb_z")
        ), "kocks/radians and bunge/degrees agree, so this test proves nothing"

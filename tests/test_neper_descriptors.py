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

"""NEPER orientation descriptors: explicit suffixes in, every variant read as ``R``.

``tests/data/neper_descriptors/`` holds tessellations written by NEPER 7.0.0
(``v7_*``) and 4.10.2 (``v4_*``) from the same three Euler triplets
(``euler_input.txt``, given as ``euler-bunge:passive``), once per output
descriptor ``{rotmat, quaternion, rodrigues, axis-angle, euler-bunge} x
{active, passive}``. Regenerate with::

    neper -T -n 3 -id 1 -ori "file(euler_input.txt,des=euler-bunge:passive)" \\
        -oridescriptor <base>:<sense> -o v7_<base>_<sense>

Every file must decode to graintrace's active ``R = euler_to_matrix(input)``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from graintrace import neper_env

FIXTURES = Path(__file__).parent / "data" / "neper_descriptors"
TESS = sorted(FIXTURES.glob("v*_*.tess"))


class TestDescriptorHelpers:
    """Explicit-descriptor helpers (pure Python)."""

    def test_split_defaults_to_passive(self):
        assert neper_env.split_descriptor("rotmat") == ("rotmat", "passive")
        assert neper_env.split_descriptor("Rotmat:Active") == ("rotmat", "active")
        with pytest.raises(ValueError, match="active"):
            neper_env.split_descriptor("rotmat:sideways")

    def test_euler_descriptor_is_explicitly_passive(self):
        assert neper_env.euler_descriptor() == "euler-bunge:passive"
        assert neper_env.euler_descriptor("euler-kocks") == "euler-kocks:passive"
        assert (
            neper_env.euler_descriptor("euler-bunge:passive") == "euler-bunge:passive"
        )
        with pytest.raises(ValueError, match="passive"):
            neper_env.euler_descriptor("euler-bunge:active")
        with pytest.raises(ValueError, match="Euler"):
            neper_env.euler_descriptor("rotmat")

    def test_rotmat_descriptor_is_active(self):
        assert neper_env.ROTMAT_DESCRIPTOR == "rotmat:active"

    def test_voronoi_builder_normalises_the_descriptor(self, tmp_path):
        from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder

        kw = dict(
            input_csv=str(tmp_path / "g.csv"),
            output_dir=str(tmp_path / "o"),
            bounding_box=[0, 1, 0, 1, 0, 1],
            env={},
        )
        assert VoronoiMeshBuilder(**kw).ori_descriptor == "euler-bunge:passive"
        with pytest.raises(ValueError):
            VoronoiMeshBuilder(orientation_descriptor="euler-bunge:active", **kw)


@pytest.fixture(scope="module")
def expected_r():
    pytest.importorskip("neml2")
    import torch

    from graintrace.orientation_helper import euler_to_matrix

    eul = np.loadtxt(FIXTURES / "euler_input.txt")
    return euler_to_matrix(torch.as_tensor(eul), "bunge", "degrees")


@pytest.mark.parametrize("tess", TESS, ids=[p.stem for p in TESS])
def test_every_neper_descriptor_decodes_to_r(tess, expected_r):
    pytest.importorskip("torch_geometric")
    import torch

    from graintrace.orientation_helper import mrp_to_matrix
    from graintrace.tess_to_gnn import NeperTessToGraphNN

    parser = NeperTessToGraphNN(tess_path=str(tess), dtype=torch.float64)
    base, sense = tess.stem.split("_", 1)[1].rsplit("_", 1)
    assert parser.ori_type == base.replace("_", "-")
    assert parser.ori_sense == sense
    got = mrp_to_matrix(parser.orientations)
    assert torch.allclose(got, expected_r, atol=1e-9, rtol=0)


def test_fixture_set_is_complete():
    names = {p.stem for p in TESS}
    for ver in ("v7", "v4"):
        for base in ("rotmat", "quaternion", "rodrigues", "axis_angle", "euler_bunge"):
            for sense in ("active", "passive"):
                assert f"{ver}_{base}_{sense}" in names


def _neper():
    found = neper_env.find_neper()
    return found if found and shutil.which(found) else None


@pytest.mark.requires_neper
@pytest.mark.skipif(_neper() is None, reason="NEPER not installed")
@pytest.mark.parametrize(
    "descriptor", ["rotmat:active", "rotmat", "euler-bunge:passive"]
)
def test_installed_neper_round_trip(tmp_path, descriptor, expected_r):
    """The installed NEPER (any version) accepts the suffixes and decodes to R."""
    pytest.importorskip("torch_geometric")
    import torch

    from graintrace.orientation_helper import mrp_to_matrix
    from graintrace.tess_to_gnn import NeperTessToGraphNN

    neper = _neper()
    shutil.copy(FIXTURES / "euler_input.txt", tmp_path / "e.txt")
    subprocess.run(
        [
            neper,
            "-T",
            "-n",
            "3",
            "-id",
            "1",
            "-ori",
            "file(e.txt,des=euler-bunge:passive)",
            "-oridescriptor",
            descriptor,
            "-o",
            "t",
        ],
        check=True,
        cwd=tmp_path,
        env=neper_env.build_env(neper),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    parser = NeperTessToGraphNN(tess_path=str(tmp_path / "t.tess"), dtype=torch.float64)
    assert torch.allclose(
        mrp_to_matrix(parser.orientations), expected_r, atol=1e-9, rtol=0
    )
    assert os.path.exists(tmp_path / "t.tess")


@pytest.mark.requires_neper
@pytest.mark.skipif(_neper() is None, reason="NEPER not installed")
def test_voronoi_builder_end_to_end_writes_active_r(tmp_path):
    """FF build: O11..O33, the .tess, reconstruction.ori and orientations.dat agree on R' = Q R."""
    pytest.importorskip("neml2")
    import pandas as pd
    import torch

    from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
    from graintrace.orientation_helper import euler_to_matrix

    rng = np.random.default_rng(0)
    n = 8
    eul = np.stack(
        [rng.uniform(0, 360, n), rng.uniform(5, 175, n), rng.uniform(0, 360, n)], 1
    )
    df = pd.DataFrame(
        {
            "X": rng.uniform(-40, 40, n),
            "Y": rng.uniform(-40, 40, n),
            "Z": rng.uniform(-40, 40, n),
            "GrainRadius": np.full(n, 10.0),
            "Eul0": eul[:, 0],
            "Eul1": eul[:, 1],
            "Eul2": eul[:, 2],
        }
    )
    df.to_csv(tmp_path / "g.csv", index=False)
    builder = VoronoiMeshBuilder(
        input_csv=str(tmp_path / "g.csv"),
        output_dir=str(tmp_path / "ff"),
        bounding_box=[-60, 60, -60, 60, -60, 60],
        rotate_angles=[0, 0, 30.0],
        angle_identifier=["Eul0", "Eul1", "Eul2"],
        unit="deg",
        neper_path=_neper(),
    )
    builder.build_voronoi(
        option="voronoi", tesr_size=[8, 8, 8], CVT_iter=0, morphoalgo="subplex"
    )

    q = builder.frame_transform.Q
    r_in = euler_to_matrix(torch.as_tensor(eul), "bunge", "degrees").numpy()
    want = q @ r_in
    o_cols = [f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    got = builder.data[o_cols].to_numpy(dtype=float).reshape(-1, 3, 3)
    np.testing.assert_allclose(got, want, atol=1e-6)

    lines = (tmp_path / "ff" / "reconstruction.tess").read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.strip() == "*ori")
    assert lines[k + 1].strip() == "rotmat:active"
    tess_r = np.array(
        [list(map(float, ln.split())) for ln in lines[k + 2 : k + 2 + n]]
    ).reshape(-1, 3, 3)
    np.testing.assert_allclose(tess_r, want, atol=1e-6)

    ori = np.loadtxt(tmp_path / "ff" / "reconstruction.ori").reshape(-1, 3, 3)
    np.testing.assert_allclose(ori, want, atol=1e-6)

    dat = np.loadtxt(tmp_path / "ff" / "orientations.dat")
    r_dat = euler_to_matrix(torch.as_tensor(dat), "bunge", "degrees").numpy()
    np.testing.assert_allclose(r_dat, want, atol=1e-6)

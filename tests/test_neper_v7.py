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

"""NEPER 7 compatibility: morphoalgo default, clear errors, auto-install recipe, tet fallback."""

from __future__ import annotations

import inspect
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from graintrace import neper_env
from graintrace.construct_voronoi_mesh import (
    VoronoiMeshBuilder,
    _raise_on_neper_failure,
)


def _neper():
    found = neper_env.find_neper()
    return found if found and shutil.which(found) else None


class TestMorphoalgo:
    def test_default_is_subplex_everywhere(self):
        for fn in (VoronoiMeshBuilder.build_voronoi, VoronoiMeshBuilder.build_graph):
            assert inspect.signature(fn).parameters["morphoalgo"].default == "subplex"

    def test_rejected_algorithm_is_a_clear_value_error(self):
        proc = SimpleNamespace(
            returncode=-6,
            stdout="Error  :     > Unknown optimization algorithm `praxis'.",
        )
        with pytest.raises(ValueError, match="subplex"):
            _raise_on_neper_failure(proc, ["neper"], "praxis", "log.txt")

    def test_other_failures_still_raise_called_process_error(self):
        proc = SimpleNamespace(returncode=1, stdout="Error: something else")
        with pytest.raises(subprocess.CalledProcessError):
            _raise_on_neper_failure(proc, ["neper"], "subplex", "log.txt")
        _raise_on_neper_failure(SimpleNamespace(returncode=0, stdout=""), [], "", "")


class TestAutoInstallRecipe:
    """The v7 from-source recipe, checked as data (never executed here)."""

    def test_plan_mirrors_the_known_good_build(self, tmp_path):
        plan = neper_env._neper7_build_plan(str(tmp_path))
        assert neper_env.NEPER_AUTO_VERSION == "7.0.0"
        assert neper_env.GMSH_SDK_VERSION == "4.15.2"
        gmsh = str(tmp_path / ".local" / "opt" / "gmsh-4.15.2-Linux64-sdk")
        assert plan["gmsh_dir"] == gmsh
        assert plan["gmsh_url"].endswith("gmsh-4.15.2-Linux64-sdk.tgz")
        assert any("neper-7.0.0.tar.gz" in u for u in plan["neper_urls"])
        cmake = plan["cmake"]
        assert "-DFORCE_BUILTIN_GSL=ON" in cmake
        assert "-DCMAKE_C_COMPILER=/usr/bin/gcc" in cmake
        assert "-DCMAKE_CXX_COMPILER=/usr/bin/g++" in cmake
        assert "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" in cmake
        assert f"-DCMAKE_INSTALL_RPATH={tmp_path}/.local/lib;{gmsh}/lib" in cmake
        assert f"-DCMAKE_PREFIX_PATH={tmp_path}/.local;{gmsh}" in cmake
        assert cmake[-1] == ".."
        # conda compiler variables are cleared for the build
        for var in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CPPFLAGS"):
            assert plan["build_env"][var] is None
        assert plan["build_env"]["PATH"].endswith("/usr/bin:/bin")

    def test_existing_neper_short_circuits(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(neper_env.shutil, "which", lambda *a, **k: "/x/neper")

        def boom(*_a, **_k):
            raise AssertionError("nothing may be built when neper is present")

        monkeypatch.setattr(neper_env.subprocess, "run", boom)
        assert neper_env._auto_install_neper() == "/x/neper"

    def test_neper_version_of_a_missing_binary_is_none(self, tmp_path):
        assert neper_env.neper_version(str(tmp_path / "nope")) is None


def _tiny_ff(tmp_path, n=6):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "X": rng.uniform(-40, 40, n),
            "Y": rng.uniform(-40, 40, n),
            "Z": rng.uniform(-40, 40, n),
            "GrainRadius": np.full(n, 10.0),
            "Eul0": rng.uniform(0, 360, n),
            "Eul1": rng.uniform(5, 175, n),
            "Eul2": rng.uniform(0, 360, n),
        }
    )
    df.to_csv(tmp_path / "g.csv", index=False)
    return VoronoiMeshBuilder(
        input_csv=str(tmp_path / "g.csv"),
        output_dir=str(tmp_path / "ff"),
        bounding_box=[-60, 60, -60, 60, -60, 60],
        angle_identifier=["Eul0", "Eul1", "Eul2"],
        unit="deg",
        neper_path=_neper(),
    )


@pytest.mark.requires_neper
@pytest.mark.skipif(_neper() is None, reason="NEPER not installed")
class TestInstalledNeper:
    def test_praxis_on_neper7_raises_value_error(self, tmp_path):
        ver = neper_env.neper_version(_neper()) or ""
        if not ver.startswith("7"):
            pytest.skip(f"NEPER {ver} still accepts praxis")
        with pytest.raises(ValueError, match="subplex"):
            _tiny_ff(tmp_path).build_voronoi(
                option="centroid", CVT_iter=2, morphoalgo="praxis", tesr_size=[6, 6, 6]
            )

    def test_centroid_default_and_tet_mesh_fallback(self, tmp_path):
        """Default morphoalgo runs the CVT path; generate_mesh=True writes a .msh."""
        pytest.importorskip("neml2")
        builder = _tiny_ff(tmp_path)
        builder.build_voronoi(
            option="centroid",
            CVT_iter=5,
            tesr_size=[6, 6, 6],
            generate_mesh=True,
            relative_el_size=2.0,
            mesh_quality_min=0.5,
        )
        msh = tmp_path / "ff" / "reconstruction.msh"
        assert msh.exists() and msh.stat().st_size > 0
        assert "$Elements" in msh.read_text(errors="ignore")


class TestZSlabCut:
    """NEPER 7 reversed ``hspace`` inside ``cut``; the slab crop must follow it."""

    def test_v7_cut_expression(self):
        from graintrace.generate_random_crystal import CrystalGenerator

        assert (
            CrystalGenerator.zslab_cut_arg(0.25, 0.75)
            == "cut(hspace(-0.75,0,0,-1),hspace(0.25,0,0,1))"
        )

    @pytest.mark.requires_neper
    @pytest.mark.skipif(_neper() is None, reason="NEPER not installed")
    def test_crop_keeps_the_slab(self, tmp_path):
        """Installed NEPER: cropped cells have volume and lie inside [zlo, zhi]."""
        import numpy as np

        from graintrace.generate_random_crystal import CrystalGenerator

        bbox = [0, 1, 0, 1, 0, 1]
        cg = CrystalGenerator(
            output_dir=str(tmp_path), bounding_box=bbox, seed=3, neper_path=_neper()
        )
        cg.generate_tessellation(
            morpho_args={"type": "raw", "morpho_str": "gg", "n": 60}, iterations=2
        )
        CrystalGenerator.crop_tessellation(
            tess_name=str(tmp_path / "voronoi.tess"),
            bounding_box=bbox,
            zlo=0.25,
            zhi=0.75,
            output_dir=str(tmp_path),
            out_name="slab",
            env=cg.env,
            neper_bin=cg.neper_bin,
            verbose=False,
        )
        st = np.loadtxt(tmp_path / "slab.stcell")  # id,vol,w,x,y,z,radeq
        kept = st[st[:, 1] > 0]
        assert len(kept) > 0
        assert np.isclose(kept[:, 1].sum(), 0.5, rtol=1e-6)
        assert kept[:, 5].min() > 0.25 and kept[:, 5].max() < 0.75


def test_write_to_csv_drops_empty_cells(tmp_path):
    """Cells a NEPER 7 cut emptied (vol 0, centroid 0) must not become grains."""
    import pandas as pd

    from graintrace.generate_random_crystal import CrystalGenerator

    (tmp_path / "s.stcell").write_text(
        "1 0 1 0 0 0 0\n2 8.0 1 1.0 2.0 3.0 1.2\n3 0 1 0 0 0 0\n4 1.0 1 4 5 6 0.6\n"
    )
    (tmp_path / "s.ori").write_text("1 2 3\n4 5 6\n7 8 9\n10 11 12\n")
    out = tmp_path / "s.csv"
    CrystalGenerator.write_to_csv(
        str(tmp_path / "s.stcell"), str(tmp_path / "s.ori"), str(out), verbose=False
    )
    df = pd.read_csv(out)
    assert df["X"].tolist() == [1.0, 4.0]
    assert df["Eul0"].tolist() == [4.0, 10.0]  # orientations stay row-aligned


def test_scan_tessellation_empty_cell_is_a_clear_error(tmp_path, monkeypatch):
    """A tessellation with an empty cell (NEPER 7 keeps them) -> clear RuntimeError."""
    import numpy as np
    import pandas as pd

    from graintrace.hedm_stitching_techniques import scan_tessellation as st

    verts = np.zeros((4, 3))
    monkeypatch.setattr(st.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        st, "_parse_tess_cells", lambda path: (verts, [[0, 1, 2]], [[0], []])
    )
    df = pd.DataFrame({"X": [1.0, 0.0], "Y": [1.0, 0.0], "Z": [1.0, 0.0]})
    with pytest.raises(RuntimeError, match="empty cell"):
        st.compute_cell_geometry(
            df, [-5, 5, -5, 5, -5, 5], weighted=False, work_dir=str(tmp_path), env={}
        )


class TestNeperPin:
    """graintrace is written for NEPER 7.x and refuses other majors."""

    @pytest.mark.parametrize("ver", ["4.10.2", None, "garbage"])
    def test_non_v7_raises(self, monkeypatch, ver):
        monkeypatch.setattr(neper_env, "neper_version", lambda *a, **k: ver)
        with pytest.raises(RuntimeError, match="requires NEPER 7"):
            neper_env.require_neper_version("/fake/neper")

    def test_v7_passes(self, monkeypatch):
        monkeypatch.setattr(neper_env, "neper_version", lambda *a, **k: "7.0.0")
        assert neper_env.require_neper_version("/fake/neper") == "7.0.0"

    def test_resolve_checks_version(self, monkeypatch):
        monkeypatch.setattr(neper_env, "find_neper", lambda *a, **k: "/fake/neper")
        monkeypatch.setattr(neper_env, "neper_version", lambda *a, **k: "4.10.2")
        with pytest.raises(RuntimeError, match="requires NEPER 7"):
            neper_env.resolve_neper_env()


class TestLaguerreWeights:
    def test_power_weights_are_radius_squared(self):
        import numpy as np

        w = neper_env.laguerre_weights([20.0, 0.0, float("nan"), -3.0, 5.0])
        assert np.allclose(w, [400.0, 0.0, 0.0, 0.0, 25.0])

    @pytest.mark.requires_neper
    @pytest.mark.skipif(_neper() is None, reason="NEPER not installed")
    def test_weighted_cell_boundary_moves(self):
        """Two seeds 50 apart, radii 20 and 0: power-diagram boundary at 50 + 400/100."""
        import pandas as pd

        from graintrace.hedm_stitching_techniques.scan_tessellation import (
            compute_cell_geometry,
        )

        df = pd.DataFrame(
            {
                "X": [50.0, 50.0],
                "Y": [50.0, 50.0],
                "Z": [25.0, 75.0],
                "GrainRadius": [20.0, 0.0],
            }
        )
        box = [0, 100, 0, 100, 0, 100]
        weighted = compute_cell_geometry(df, box, weighted=True)
        plain = compute_cell_geometry(df, box, weighted=False)
        assert weighted["Zmax"].iloc[0] == pytest.approx(54.0, abs=1e-6)
        assert plain["Zmax"].iloc[0] == pytest.approx(50.0, abs=1e-6)

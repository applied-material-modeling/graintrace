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

"""Tests for scan_stitching_comparison and hedm_stitching_techniques."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _neper_available() -> bool:
    """True if a Neper tessellation actually runs in this environment."""
    try:
        from graintrace.hedm_stitching_techniques.scan_tessellation import (
            compute_cell_geometry,
        )

        df = pd.DataFrame(
            {
                "X": [10, 30, 70, 50],
                "Y": [10, 30, 20, 80],
                "Z": [10, 30, 50, 20],
                "GrainRadius": [8, 12, 10, 9],
            }
        )
        g = compute_cell_geometry(df, [0, 100, 0, 100, 0, 100], weighted=False)
        return len(g) == len(df)
    except Exception:
        return False


NEPER_AVAILABLE = _neper_available()
_needs_neper = pytest.mark.skipif(
    not NEPER_AVAILABLE, reason="Neper not available in this environment"
)


def _make_grain_csv(path, n=15, seed=0, x_offset=0.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "X": rng.uniform(0, 200, n) + x_offset,
            "Y": rng.uniform(0, 200, n),
            "Z": rng.uniform(0, 100, n),
            "GrainRadius": rng.uniform(10, 50, n),
            "Eul0": rng.uniform(0, 360, n),
            "Eul1": rng.uniform(0, 180, n),
            "Eul2": rng.uniform(0, 360, n),
        }
    )
    df.to_csv(path, index=False)
    return str(path)


class TestScanStitchingComparison:
    def test_loads_and_compares(self, tmp_path):
        from graintrace.scan_stitching_comparison import ScanStitchingComparison

        true_csv = _make_grain_csv(tmp_path / "true.csv", n=10)
        stitch_csv = _make_grain_csv(tmp_path / "stitched.csv", n=10, seed=1)

        comp = ScanStitchingComparison(
            output_dir=str(tmp_path / "comparison"),
            true_csv=true_csv,
            stitch_csv=stitch_csv,
            position_tolerance=50,
            orientation_tolerance=5,
            radius_tolerance=50,
        )
        comp.run_comparison()

    def test_output_dir_created(self, tmp_path):
        from graintrace.scan_stitching_comparison import ScanStitchingComparison

        true_csv = _make_grain_csv(tmp_path / "true2.csv", n=5)
        stitch_csv = _make_grain_csv(tmp_path / "stitch2.csv", n=5, seed=2)
        out_dir = tmp_path / "out"

        comp = ScanStitchingComparison(
            output_dir=str(out_dir),
            true_csv=true_csv,
            stitch_csv=stitch_csv,
        )
        comp.run_comparison()
        assert out_dir.exists()


class TestNaiveStitching:
    def test_combines_csvs(self, tmp_path):
        from graintrace.hedm_stitching_techniques.naive_stitching import NaiveStitching

        csv1 = tmp_path / "scan1.csv"
        csv2 = tmp_path / "scan2.csv"
        df1 = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "Z": [0, 1]})
        df2 = pd.DataFrame({"X": [2, 3], "Y": [2, 3], "Z": [2, 3]})
        df1.to_csv(csv1, index=False)
        df2.to_csv(csv2, index=False)

        out_csv = str(tmp_path / "stitched.csv")
        stitch = NaiveStitching(
            scan_files=[str(csv1), str(csv2)],
            output_csv=out_csv,
        )
        stitch.run()

        assert Path(out_csv).exists()
        df = pd.read_csv(out_csv)
        assert len(df) == 4

    def test_empty_list_raises(self, tmp_path):
        from graintrace.hedm_stitching_techniques.naive_stitching import NaiveStitching

        out_csv = str(tmp_path / "empty.csv")
        stitch = NaiveStitching(scan_files=[], output_csv=out_csv)
        with pytest.raises((ValueError, Exception)):
            stitch.run()


def _make_overlap_scan(path, zlo, zhi, seed, n=30):
    rng = np.random.default_rng(seed)
    pd.DataFrame(
        {
            "X": rng.uniform(0, 200, n),
            "Y": rng.uniform(0, 200, n),
            "Z": rng.uniform(zlo, zhi, n),
            "GrainRadius": rng.uniform(10, 25, n),
            "Eul0": rng.uniform(0, 360, n),
            "Eul1": rng.uniform(0, 180, n),
            "Eul2": rng.uniform(0, 360, n),
        }
    ).to_csv(path, index=False)
    return str(path)


@_needs_neper
class TestScanTessellation:
    def test_compute_cell_geometry(self):
        from graintrace.hedm_stitching_techniques.scan_tessellation import (
            compute_cell_geometry,
        )

        df = pd.DataFrame(
            {
                "X": [10, 30, 70, 50, 20],
                "Y": [10, 30, 20, 80, 60],
                "Z": [10, 30, 50, 20, 80],
                "GrainRadius": [8, 12, 10, 9, 11],
            }
        )
        bbox = [0, 100, 0, 100, 0, 100]
        for weighted in (False, True):
            g = compute_cell_geometry(df, bbox, weighted=weighted)
            assert list(g.columns) == ["Zmin", "Zmax", "Xc", "Yc", "Zc", "Vol"]
            assert len(g) == len(df)
            assert list(g.index) == list(df.index)  # cell i <-> row i
            assert (g["Zmax"] >= g["Zmin"]).all()
            assert (g["Vol"] > 0).all()


@_needs_neper
class TestRegionBaseStitchingTessellation:
    def _run(self, tmp_path, **extra):
        from graintrace.hedm_stitching_techniques.region_base_stitching import (
            RegionBaseStitching,
        )

        scans = [
            _make_overlap_scan(tmp_path / "s0.csv", 0, 110, 1),
            _make_overlap_scan(tmp_path / "s1.csv", 90, 200, 2),
        ]
        out = str(tmp_path / "stitched.csv")
        st = RegionBaseStitching(
            scan_files=scans,
            output_csv=out,
            position_tolerance=40,
            orientation_tolerance=10,
            radius_tolerance=-1,
            weights={"pos": 0.1, "ori": 1.0, "rad": 0},
            min_neighbors=5,
            **extra,
        )
        res = st.run(zlo=0, zhi=200, overlap_fraction=0.1)
        return res, pd.read_csv(out)

    def test_refine_extents_runs_and_output_clean(self, tmp_path):
        res, df = self._run(tmp_path, refine_extents=True, tess_weighted=True)
        assert len(res.df) > 0
        # tessellation extent columns must never leak into the output CSV
        assert "Zmin" not in df.columns and "Zmax" not in df.columns
        assert {"X", "Y", "Z", "GrainRadius"}.issubset(df.columns)

    def test_update_centroid_changes_positions(self, tmp_path):
        _, df_ff = self._run(tmp_path, refine_extents=True, update_centroid=False)
        _, df_cc = self._run(tmp_path, refine_extents=True, update_centroid=True)
        # cell centroids differ from the raw FF centroids for at least some grains
        assert not np.isclose(df_ff["X"].mean(), df_cc["X"].mean())

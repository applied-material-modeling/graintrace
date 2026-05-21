"""Tests for scan_stitching_comparison and hedm_stitching_techniques."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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

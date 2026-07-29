#!/usr/bin/env python
# Copyright 2026, UChicago Argonne, LLC -- MIT (see repo LICENSE)
"""Generate the synthetic "experiment output" for the live graintrace demo.

Produces, under ``demo/``:
  experiment/hedm_scan/scan_0..3.csv  -- 4 FF-HEDM z-scans at 25% overlap, each with
                                         X,Y,Z,GrainRadius,Eul0-2 AND residual elastic
                                         strain eKen11..33 (microstrain, per-grain
                                         consistent across overlapping scans)
  experiment/strain-stress.csv        -- hypothetical macro stress-strain curve
                                         (copied from mwe_data/ff_calibration; the
                                         material-calibration TARGET)
  experiment/sample.json              -- ALL non-inferrable metadata (sample size,
                                         loading conditions, scan geometry, units)
  _truth/voronoi.{tess,csv,...}       -- ground-truth crystal, INTERNAL only

The ground-truth crystal is NOT placed in experiment/ -- experiment/ holds only what a
real FF-HEDM experiment would hand you. Run once, then `python demo/run_demo.py`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from graintrace.generate_random_crystal import CrystalGenerator

# ---------------------------------------------------------------- INPUT
SEED = 42
# Sample bounding box (micrometers). ~2.3e8 um^3; with diameq lognormal(130) that is
# ~200 grains. Enlarge the box or lower the mean to add grains; shrink to reduce.
BOUNDING_BOX = [-300, 300, -300, 300, -320, 320]
MORPHO = {"type": "diameq", "distribution": "lognormal", "params": (130.0, 5.0)}
TESS_ITERATIONS = 1000                       # CVT relaxation (recommendations: equiaxed)
NSCAN = 4
OVERLAP_PCT = 25                             # 25% overlap between adjacent z-scans
RESIDUAL_STDEV_MICROSTRAIN = 300.0           # stdev of the synthetic residual eKen field
# Loading condition recorded in sample.json (uniaxial tension along z). 0.1% keeps
# every CPFE increment inside the rate-dependent slip model's convergence radius;
# raise it for more plasticity, but lower CPFE dt to match (see run_demo.py).
TOTAL_STRAIN = 0.001
LOADED_AXIS = "z"

HERE = Path(__file__).resolve().parent
TRUTH = HERE / "_truth"
EXP = HERE / "experiment"
SCANS_OUT = EXP / "hedm_scan"
EKEN_COLS = [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
COORD = ["X", "Y", "Z"]
# strain-stress curve source (shipped, realistic Ti macro curve)
SS_SRC = HERE.parent / "mwe_data" / "ff_calibration" / "strain-stress.csv"


def _build_residual_field(df_true: pd.DataFrame, seed: int) -> np.ndarray:
    """Per-grain symmetric residual elastic strain (n_grains, 9) in microstrain."""
    rng = np.random.default_rng(seed)
    n = len(df_true)
    E = rng.normal(0.0, RESIDUAL_STDEV_MICROSTRAIN, size=(n, 3, 3))
    E = 0.5 * (E + E.transpose(0, 2, 1))        # symmetric tensor
    return E.reshape(n, 9)                       # row-major 11,12,13,21,22,23,31,32,33


def _write_sample_json(zlo: float, zhi: float, n_grains: int) -> None:
    meta = {
        "sample": {
            "name": "demo_synthetic_200grain",
            "description": "Synthetic FF-HEDM demo microstructure (graintrace).",
            "bounding_box_um": BOUNDING_BOX,
            "n_grains_true": int(n_grains),
            "units": {
                "length": "micrometer",
                "orientation": "degrees",
                "orientation_convention": "bunge",
                "strain": "microstrain",
                "symmetry": "432",
            },
        },
        "scan_geometry": {
            "n_scans": NSCAN,
            "overlap_fraction": OVERLAP_PCT / 100.0,
            "zlo": zlo,
            "zhi": zhi,
        },
        "loading": {
            "mode": "uniaxial_tension",
            "loaded_axis": LOADED_AXIS,
            "total_strain": TOTAL_STRAIN,
            "temperature_K": 298,
            "bc": {
                "x": {"negative": "stress_free", "positive": "stress_free"},
                "y": {"negative": "stress_free", "positive": "stress_free"},
                "z": {"negative": 0, "positive": "total_strain * (zhi - zlo)"},
            },
        },
        "elastic_strain_columns": {"prefix": "eKen", "unit": "microstrain"},
        "provenance": {
            "generator": "demo/generate_experiment.py",
            "seed": SEED,
            "morpho": MORPHO,
        },
        "notes": (
            "scan_*.csv are FF-HEDM z-scans (X,Y,Z,GrainRadius,Eul0-2,eKen11..33). "
            "Residual eKen is in microstrain. This file supplies everything a raw CSV "
            "cannot: sample dimensions, loading conditions, scan geometry, and units."
        ),
    }
    (EXP / "sample.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    for d in (TRUTH, EXP, SCANS_OUT):
        d.mkdir(parents=True, exist_ok=True)

    # 1) ground-truth crystal (internal)
    print(f"==> Generating crystal in {TRUTH} (box {BOUNDING_BOX}) ...")
    cg = CrystalGenerator(output_dir=str(TRUTH), bounding_box=BOUNDING_BOX, seed=SEED)
    cg.generate_tessellation(
        morpho_args=MORPHO, iterations=TESS_ITERATIONS, extra_neper_args=["-reg", "1"]
    )
    df_true = pd.read_csv(TRUTH / "voronoi.csv")
    n_true = len(df_true)
    print(f"    true grains: {n_true}")

    # 2) simulate 4 overlapping z-scans
    print(f"==> Simulating {NSCAN} z-scans at {OVERLAP_PCT}% overlap ...")
    cg.hedm_zscan(
        tess_file=str(TRUTH / "voronoi.tess"),
        nstep=NSCAN,
        overlap_percentage=OVERLAP_PCT,
        output_hedm="hedm_scan",
        verbose=False,
    )
    raw_scan_dir = TRUTH / "hedm_scan"

    # 3) residual elastic-strain field, keyed to the nearest true-grain centroid so it
    #    is consistent for a grain seen in two overlapping scans.
    residual = _build_residual_field(df_true, SEED)
    tree = cKDTree(df_true[COORD].to_numpy())

    scan_files = sorted(raw_scan_dir.glob("scan_*.csv"))
    if not scan_files:
        raise RuntimeError(f"no scan_*.csv produced in {raw_scan_dir}")
    for sf in scan_files:
        df = pd.read_csv(sf)
        _, idx = tree.query(df[COORD].to_numpy())
        df[EKEN_COLS] = residual[idx]
        out = SCANS_OUT / sf.name
        df.to_csv(out, index=False)
        print(f"    {out.name}: {len(df)} grains (+ residual eKen)")

    # 4) hypothetical macro stress-strain curve (calibration target)
    shutil.copy(SS_SRC, EXP / "strain-stress.csv")
    print(f"    copied {SS_SRC.name} -> experiment/strain-stress.csv")

    # 5) sample.json (the non-inferrable metadata)
    _write_sample_json(zlo=BOUNDING_BOX[4], zhi=BOUNDING_BOX[5], n_grains=n_true)
    print(f"    wrote experiment/sample.json")

    print("\nDONE. experiment/ contains: hedm_scan/scan_0..%d.csv, strain-stress.csv, "
          "sample.json" % (len(scan_files) - 1))
    if not (150 <= n_true <= 260):
        print(f"NOTE: {n_true} grains is off the ~200 target; adjust BOUNDING_BOX or "
              "MORPHO params in this script and re-run.")


if __name__ == "__main__":
    main()

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

"""Timing benchmark for graintrace material calibration (neml2 v3 + pyzag Taylor).

Measures LBFGS calibration wall time as a function of `device` (cpu vs cuda),
`n_grains` (per-step state size), and `npoints` (pyzag time steps). Uses a small
FIXED LBFGS budget - we measure per-solve cost, not convergence.

In-process only (neml2 v3 + pyzag + torch); no external binaries. Must run in an
env that has a working neml2 v3 + pyzag. Use `--probe` to test whether the current
env can run a calibration at all (exit 0 = ok, 1 = broken) before a full sweep.

Example:
    python benchmark/bench_calibration.py --device cuda --n-grains 50,100,250 \
        --npoints 15,30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# pylint: disable=import-error,no-name-in-module  # local sibling module
from _harness import (  # type: ignore
    capture_sysinfo,
    parse_int_list,
    print_header,
    results_dir,
    skip,
    timer,
    write_results,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "mwe_data" / "ff_calibration"


def _cuda_available() -> bool:
    try:
        # pylint: disable=import-outside-toplevel
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pylint: disable=broad-except
        return False


def build_calib(device, n_grains, npoints, nchunk, save_dir):
    """Construct a MaterialCalibration on the checked-in ff_calibration data."""
    # pylint: disable=import-outside-toplevel  # heavy neml2/pyzag stack kept local
    import graintrace as _gt
    from graintrace.material_calibration import MaterialCalibration
    from graintrace.taylor import TaylorModel

    cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")
    return MaterialCalibration(
        model_class=TaylorModel,
        model_args={
            "neml2_path": cpfe_base + "/neml2_cpfe_calibration.i",
            "npoints": npoints,
            "nchunk": nchunk,
            "device": device,
            "compile": False,
        },
        data_args={
            "data_dir": str(DATA_DIR),
            "strain_stress_file": str(DATA_DIR / "strain-stress.csv"),
            "npoints": npoints,
            "full_field_strain_units": "microstrain",
            "straintype": "eKen",
            "max_strain": 0.006,
            "n_grains": n_grains,
            "seed": 42,
        },
        save_dir=str(save_dir),
        apply_elastic_correction=False,
        strain_window=(0.0, 0.0015),
    )


def run_one(device, n_grains, npoints, nchunk, maxiter, inner, save_dir):
    """Build + calibrate once with a fixed LBFGS budget; return timing dict."""
    with timer() as t_setup:
        calib = build_calib(device, n_grains, npoints, nchunk, save_dir)
    with timer() as t_cal:
        # plateau_window huge => never early-stops, so the work is deterministic.
        calib.calibrate(
            maxiter=maxiter,
            lr=0.3,
            max_iter_per_step=inner,
            line_search_fn="strong_wolfe",
            plateau_window=10_000,
            autosave=False,
        )
    return {
        "device": device,
        "n_grains": n_grains,
        "npoints": npoints,
        "nchunk": nchunk,
        "maxiter": maxiter,
        "inner": inner,
        "setup_s": round(t_setup[0], 3),
        "calibrate_s": round(t_cal[0], 3),
    }


def probe() -> None:
    """Minimal calibration to check the current env; exit 0 ok, 1 broken."""
    try:
        out = results_dir("calibration_probe")
        run_one(
            "cpu", n_grains=20, npoints=8, nchunk=2, maxiter=1, inner=2, save_dir=out
        )
        print("PROBE OK")
        sys.exit(0)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"PROBE FAIL: {type(exc).__name__}: {exc}")
        sys.exit(1)


def bench(args) -> None:
    """Run the device x n_grains x npoints sweep."""
    if not DATA_DIR.exists():
        skip(f"calibration data not found at {DATA_DIR}")
    try:
        import neml2  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        import pyzag  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    except ImportError as exc:
        skip(f"neml2/pyzag not importable in this env ({exc})")

    sysinfo = capture_sysinfo()
    print_header("graintrace benchmark: material calibration (timing)", sysinfo)

    if args.device == "auto":
        devices = ["cuda"] if _cuda_available() else ["cpu"]
    elif args.device == "both":
        devices = ["cpu", "cuda"] if _cuda_available() else ["cpu"]
    else:
        devices = [args.device]
    if "cuda" in devices and not _cuda_available():
        print("  (cuda requested but unavailable -> dropping cuda)")
        devices = [d for d in devices if d != "cuda"] or ["cpu"]

    out_dir = results_dir("calibration", args.out)
    rows = []
    for device in devices:
        for npoints in parse_int_list(args.npoints):
            for n_grains in parse_int_list(args.n_grains):
                print(f"\n[run] device={device} n_grains={n_grains} npoints={npoints}")
                row = run_one(
                    device,
                    n_grains,
                    npoints,
                    args.nchunk,
                    args.maxiter,
                    args.inner,
                    out_dir,
                )
                rows.append(row)
                print(f"  setup={row['setup_s']}s  calibrate={row['calibrate_s']}s")
    write_results("calibration", rows, out_dir, sysinfo)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--device",
        default="auto",
        help="cpu | cuda | auto (cuda if present) | both",
    )
    p.add_argument("--n-grains", default="50,100,250,500", dest="n_grains")
    p.add_argument("--npoints", default="30")
    p.add_argument("--nchunk", type=int, default=2)
    p.add_argument("--maxiter", type=int, default=3, help="LBFGS outer steps (fixed)")
    p.add_argument("--inner", type=int, default=4, help="LBFGS inner iters per step")
    p.add_argument("--probe", action="store_true", help="env sanity check, exit 0/1")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.probe:
        probe()
    else:
        bench(args)


if __name__ == "__main__":
    main()

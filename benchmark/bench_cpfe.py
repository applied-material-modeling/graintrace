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

"""Timing benchmark for a graintrace CPFE run (MOOSE/PUMA + neml2 v3 AOTI).

Generates a cube microstructure in-memory, dumps it straight to an Exodus hex mesh
via the voxel mesher (one cube hex per voxel - NO SCULPT/CUBIT/NEPER), then runs
CPFE on the GPU and times it. Two axes:
  * resolution -> element count (nx*ny*nz HEX8 elements)
  * device_batch (per-device NEML2 chunk / quad-points-per-call)

Per resolution the AOTI model is neml2-compiled ONCE (recompile=True for the first
device_batch), then reused (recompile=False) so the device_batch sweep isolates the
NEML2 solve throughput from the one-time compile. Each row reports setup_s (bake +
compile + launch, synchronous) separately from solve_s (async MOOSE solve).

Requires the full PUMA stack: puma-opt (--puma-bin), neml2-compile + a C/C++
toolchain, mpiexec, and a CUDA GPU. Skips cleanly if any is missing.

Example:
    python benchmark/bench_cpfe.py \
        --puma-bin /home/tranh/projects/moose_neml2_v3/puma/puma-opt \
        --resolution 16,24,32 --device-batch 5000,20000,50000
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

# pylint: disable=import-error,no-name-in-module  # local sibling module
from _harness import (  # type: ignore
    capture_sysinfo,
    parse_int_list,
    print_header,
    results_dir,
    skip,
    write_results,
)

# Gentle crystal-plasticity params from the neml2 crystal_approximated.i example
# (rate exponent n=6). A timing benchmark cares about convergence robustness, not a
# specific alloy: the stiff n=25 power law overflowed the AOTI return map (non-finite
# residual) on large strain increments at scale, so the sweep never completed.
MATERIAL = {
    "slip_constant_strength": 180.0,
    "voce_hardening_initial_slope": 2000.0,
    "voce_hardening_saturation": 500.0,
    "power_slip_n": 6,
    "power_slip_g0": 1e-4,
    "elastic_E": 209016.0,
    "elastic_nu": 0.307,
    "elastic_G": 60355.0,
    "burger_scale": 2.54,
}


def _cuda_available() -> bool:
    try:
        # pylint: disable=import-outside-toplevel
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pylint: disable=broad-except
        return False


def make_grid_csv(nx, ny, nz, n_grains, spacing, seed, path):
    """Write a dense cube grid CSV (x,y,z,Eul0-2,CellID) and return (n_vox, bbox)."""
    rng = np.random.default_rng(seed)
    ax = np.arange(nx) * spacing
    ay = np.arange(ny) * spacing
    az = np.arange(nz) * spacing
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Voronoi-by-nearest-seed grain assignment (pure numpy/scipy, no NEPER).
    # pylint: disable=import-outside-toplevel
    from scipy.spatial import cKDTree

    n_grains = min(n_grains, coords.shape[0])
    seeds = coords[rng.choice(coords.shape[0], size=n_grains, replace=False)]
    _, cell = cKDTree(seeds).query(coords)

    eul_per_grain = rng.uniform(
        [0.0, 0.0, 0.0], [360.0, 180.0, 360.0], size=(n_grains, 3)
    )
    eul = eul_per_grain[cell]
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "Eul0": eul[:, 0],
            "Eul1": eul[:, 1],
            "Eul2": eul[:, 2],
            "CellID": cell + 1,  # 1-based so nothing is treated as background (id 0)
        }
    )
    df.to_csv(path, index=False)
    h = spacing / 2.0
    bbox = [
        ax.min() - h,
        ax.max() + h,
        ay.min() - h,
        ay.max() + h,
        az.min() - h,
        az.max() + h,
    ]
    return coords.shape[0], bbox


def build_voxel_mesh(grid_csv, save_dir):
    """CSV -> Exodus hex mesh + per-block MRP orientations via the voxel mesher."""
    # pylint: disable=import-outside-toplevel  # heavy graintrace stack kept local
    from graintrace.construct_voxel_mesh import VoxelMeshBuilder

    builder = VoxelMeshBuilder(
        file_path=str(grid_csv),
        save_dir=str(save_dir),
        euler_cols=("Eul0", "Eul1", "Eul2"),
        cell_id_col="CellID",  # skips segmentation
        angle_convention="bunge",
        angle_type="degrees",
        write_vtk=False,
    )
    merged = builder.reconstruct(apply_smoothing=False)  # dense grid + provided ids
    mesh_e = builder.mesh(mesher="voxel", merged_grid=merged)
    ori_csv = Path(str(builder.mapped_orientations_path) + ".csv")
    return Path(mesh_e), ori_csv


def _puma_running() -> bool:
    # Match our own run (puma-opt -i run_cpfe.i) rather than any puma-opt process:
    # other CPFE jobs on the machine use different input decks (e.g. common.i).
    # pylint: disable=import-outside-toplevel
    import subprocess

    try:
        r = subprocess.run(
            ["pgrep", "-f", "puma-opt.*run_cpfe.i"], capture_output=True, check=False
        )
        return r.returncode == 0
    except Exception:  # pylint: disable=broad-except
        return True  # can't tell -> keep waiting


def _last_grid_csv(grid_dir):
    files = sorted(Path(grid_dir).glob("out_element_centroid_*.csv"))
    for f in reversed(files):
        try:
            with open(f, encoding="utf-8") as fh:
                fh.readline()
                if fh.readline().strip():
                    return f
        except Exception:  # pylint: disable=broad-except
            pass
    return None


def wait_for_cpfe(save_folder, total_time, timeout):
    """Poll until the MOOSE run finishes; return the last grid CSV, raise on crash.

    Adapted from examples/demo/run_demo.py: completion = block CSV reached
    total_time (or puma-opt exited) with a non-empty grid snapshot present.
    """
    sim_out = Path(save_folder) / "simulation_out"
    grid_dir = sim_out / "grid_out"
    block_csv = sim_out / "out.csv"
    log = Path(save_folder) / "cpfe_run.log"
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(5)
        logtxt = log.read_text(errors="replace") if log.exists() else ""
        if re.search(
            r"terminate called|Segmentation fault|MPI_ABORT|MPI_Abort|Fatal error in|"
            r"Command not found|CUDA error|out of memory",
            logtxt,
            re.IGNORECASE,
        ):
            raise RuntimeError(
                "CPFE crashed. Last log lines:\n" + "\n".join(logtxt.splitlines()[-30:])
            )
        reached = False
        if block_csv.exists():
            try:
                b = pd.read_csv(block_csv)
                reached = "time" in b and float(b["time"].max()) >= total_time - 1e-9
            except Exception:  # pylint: disable=broad-except
                pass
        last = _last_grid_csv(grid_dir) if grid_dir.exists() else None
        proc_done = not _puma_running()
        if last is not None and (reached or proc_done):
            return last
        if proc_done and last is None and not reached:
            raise RuntimeError(
                "puma-opt exited before completion. Last log lines:\n"
                + "\n".join(logtxt.splitlines()[-30:])
            )
    raise TimeoutError(f"CPFE did not finish within {timeout}s")


def run_cpfe(sim, ncore, save_folder, total_time, timeout):
    """Clear stale outputs, launch, and wait; return (setup_s, solve_s)."""
    stale = Path(save_folder) / "simulation_out"
    if stale.exists():
        shutil.rmtree(stale)
    log = Path(save_folder) / "cpfe_run.log"
    if log.exists():
        log.unlink()

    t0 = time.time()
    sim.run(ncore=ncore)  # synchronous: bake + neml2-compile + launch MOOSE detached
    setup_s = time.time() - t0  # compile (first run) + launch overhead

    t1 = time.time()
    wait_for_cpfe(save_folder, total_time, timeout)
    solve_s = time.time() - t1
    return setup_s, solve_s


def bench(args) -> None:
    """Run the resolution x device_batch CPFE sweep."""
    puma = Path(args.puma_bin).expanduser()
    if not puma.exists():
        skip(f"puma-opt not found at {puma} (pass --puma-bin)")
    launcher_bin = shlex.split(args.launcher)[0]
    if not shutil.which(launcher_bin):
        skip(f"{launcher_bin} (launcher) not on PATH")
    if not _cuda_available():
        skip("no CUDA GPU available (CPFE benchmark targets the GPU)")

    # MOOSE_DIR lets run_cpfe_simulation auto-locate R2IncrementToRate.py when the
    # puma binary's sibling `moose/` checkout does not contain it.
    import os  # pylint: disable=import-outside-toplevel

    if args.moose_dir:
        os.environ["MOOSE_DIR"] = str(Path(args.moose_dir).expanduser())

    sysinfo = capture_sysinfo()
    print_header("graintrace benchmark: CPFE (timing)", sysinfo)

    # pylint: disable=import-outside-toplevel
    from graintrace.run_cpfe_simulation import CPFESimulation

    out_dir = results_dir("cpfe", args.out)
    device_batches = parse_int_list(args.device_batch)
    rows = []

    for res in parse_int_list(args.resolution):
        res_dir = out_dir / f"res{res}"
        res_dir.mkdir(parents=True, exist_ok=True)
        grid_csv = res_dir / "grid.csv"
        n_vox, bbox = make_grid_csv(
            res, res, res, args.n_grains, args.spacing, args.seed, grid_csv
        )
        mesh_e, ori_csv = build_voxel_mesh(grid_csv, res_dir / "mesh")
        print(f"\n[resolution {res}] {n_vox} hex elements -> {mesh_e.name}")

        displace = args.total_strain * (bbox[5] - bbox[4])
        grid_bb = list(bbox)
        for i in range(0, 6, 2):
            grid_bb[i] += 1e-4
        for i in range(1, 6, 2):
            grid_bb[i] -= 1e-4
        ngrid = min(30, res)

        sim = CPFESimulation(
            mesh_file=str(mesh_e),
            save_simulation_folder=str(res_dir),
            moose_run_file=str(puma),
            element_order="FIRST",  # voxel HEX8
            eeres_file=None,
            ori_file=str(ori_csv),
            dim=3,
            use_ff_initial_field=False,
        )
        sim.set_parameters("material", **MATERIAL)
        sim.set_parameters(
            "boundary",
            bounding_box=bbox,
            bc={
                "x": {"negative": "stress_free", "positive": "stress_free"},
                "y": {"negative": "stress_free", "positive": "stress_free"},
                "z": {"negative": 0, "positive": displace},
            },
        )
        sim.set_parameters(
            "grid_properties",
            number_of_elements=[ngrid, ngrid, ngrid],
            bounding_box=grid_bb,
        )
        sim.set_parameters(
            "simulation_parameters",
            launcher=args.launcher,
            grid_transfer=args.grid_transfer,
            exodus_output=args.exodus_output,
            mesh_csv=args.mesh_csv,
            distributed_mesh=args.distributed_mesh,
        )

        if args.neml2_load_file:
            sim.set_parameters(
                "simulation_parameters",
                neml2_load_files=[str(Path(args.neml2_load_file).expanduser())],
            )

        for i, db in enumerate(device_batches):
            sim.set_parameters(
                "simulation_parameters",
                dt=args.dt,
                total_time=args.total_time,
                initialize_time=args.initialize_time,
                device=args.device,
                device_batch=db,
                sync_times=f"{args.total_time:.8g}",
                recompile=(i == 0),  # compile once per resolution, then reuse AOTI
            )
            print(f"  device_batch={db:>7} recompile={i == 0} ...", flush=True)
            setup_s, solve_s = run_cpfe(
                sim, args.ncore, res_dir, args.total_time, args.timeout
            )
            rows.append(
                {
                    "resolution": res,
                    "n_elements": n_vox,
                    "n_grains": min(args.n_grains, n_vox),
                    "ncore": args.ncore,
                    "device": args.device,
                    "device_batch": db,
                    "grid_transfer": args.grid_transfer,
                    "exodus_output": args.exodus_output,
                    "mesh_csv": args.mesh_csv,
                    "distributed_mesh": args.distributed_mesh,
                    "recompile": i == 0,
                    "setup_s": round(setup_s, 2),
                    "solve_s": round(solve_s, 2),
                }
            )
            print(f"    setup(compile+launch)={setup_s:.1f}s  solve={solve_s:.1f}s")

    write_results("cpfe", rows, out_dir, sysinfo)


def summarize_sweep(root) -> None:
    """Aggregate every ``cpfe.csv`` under ``root`` into ``root/cpfe_summary.csv``.

    Recovers ``ncore`` from the ``ncore<N>_db<...>`` dir name for older runs whose
    row lacked it. Prints a compact resolution/ncore/device_batch vs timing table.
    """
    # pylint: disable=import-outside-toplevel
    import glob
    import re

    import pandas as pd

    root = Path(root).expanduser()
    files = sorted(glob.glob(str(root / "**" / "cpfe.csv"), recursive=True))
    if not files:
        print(f"no cpfe.csv found under {root}")
        return
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if "ncore" not in d.columns:  # older runs: recover ncore from the dir name
            m = re.search(r"ncore(\d+)", f)
            d["ncore"] = int(m.group(1)) if m else -1
        frames.append(d)
    summary = pd.concat(frames, ignore_index=True)
    sort_cols = [
        c for c in ("resolution", "ncore", "device_batch") if c in summary.columns
    ]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)
    out = root / "cpfe_summary.csv"
    summary.to_csv(out, index=False)
    cols = [
        c
        for c in ("resolution", "ncore", "device_batch", "setup_s", "solve_s")
        if c in summary.columns
    ]
    print(
        summary[cols].to_string(index=False) if cols else summary.to_string(index=False)
    )
    print(f"\nwrote {out} ({len(summary)} rows)")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--puma-bin",
        default=None,
        dest="puma_bin",
        help="path to puma-opt (required to run; omit with --summarize)",
    )
    p.add_argument(
        "--neml2-load-file",
        default=None,
        dest="neml2_load_file",
        help="explicit R2IncrementToRate.py for neml2-compile --load "
        "(when the puma binary's sibling moose/ lacks it)",
    )
    p.add_argument(
        "--moose-dir",
        default=None,
        dest="moose_dir",
        help="MOOSE_DIR used to auto-locate R2IncrementToRate.py",
    )
    p.add_argument(
        "--resolution", default="16,24", help="cube edge lengths (elems=edge^3)"
    )
    p.add_argument("--device-batch", default="5000,20000,50000", dest="device_batch")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ncore", type=int, default=1, help="MPI ranks (launcher -n)")
    p.add_argument(
        "--launcher",
        default="mpiexec",
        help='MPI launcher for puma-opt: "mpiexec" (default) or "srun" (Cray/Slurm)',
    )
    p.add_argument(
        "--grid-transfer",
        default="final",
        choices=("final", "per_step", "off"),
        dest="grid_transfer",
        help="regular-grid MultiApp transfer frequency (default: final)",
    )
    p.add_argument(
        "--exodus-output",
        default="sync",
        choices=("sync", "per_step"),
        dest="exodus_output",
        help="native-mesh Exodus write frequency (default: sync)",
    )
    p.add_argument(
        "--mesh-csv",
        default="sync",
        choices=("sync", "per_step", "off"),
        dest="mesh_csv",
        help="CPFE-mesh element-centroid CSV frequency (default: sync)",
    )
    p.add_argument(
        "--distributed-mesh",
        action="store_true",
        dest="distributed_mesh",
        help="pre-split the mesh (--split-mesh) and run --use-split (distributed mesh); needs ncore>=2",
    )
    p.add_argument("--n-grains", type=int, default=50, dest="n_grains")
    p.add_argument("--spacing", type=float, default=5.0, help="voxel size (um)")
    p.add_argument("--total-strain", type=float, default=0.005, dest="total_strain")
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--total-time", type=float, default=2.0, dest="total_time")
    p.add_argument("--initialize-time", type=float, default=1.0, dest="initialize_time")
    p.add_argument("--timeout", type=float, default=14400, help="per-run wait cap (s)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--summarize",
        default=None,
        help="aggregate every cpfe.csv under this dir into <dir>/cpfe_summary.csv "
        "(no run); pairs with a sweep's --out root",
    )
    args = p.parse_args()
    if args.summarize:
        summarize_sweep(args.summarize)
        return
    if not args.puma_bin:
        p.error("--puma-bin is required to run (or pass --summarize to aggregate)")
    bench(args)


if __name__ == "__main__":
    main()

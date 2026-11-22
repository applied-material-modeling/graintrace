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

"""Regenerate the shipped CPFE datasets under ``mwe_data/``.

Two targets, each a full CPFE run followed by a subset/export into ``mwe_data``:

``fragmentation``
    ``mwe_data/cpfe_ff_fragmentation/``: the 10-grain ``mwe_data/cpfe_ff`` tet mesh pulled
    to 10% axial strain, per-element ``mesh_out`` (id, x, y, z, block_id, ori_rodrigues_*)
    at every 3rd step + the final one, + the per-grain ``ori_rodrigues`` block CSV.
``postprocess``
    ``mwe_data/out.csv`` + ``mwe_data/grid_out/`` (45-grain seeded NEPER microstructure,
    voxel hex mesh, 4 regular-grid snapshots) and ``mwe_data/synthetic_load_exp/`` (per-load-step
    FF-style grain tables exported from the block CSV, with TRUE Bunge Euler angles).

Run from the repo root in the ``moose-src`` env::

    python examples/data_generation/make_cpfe_mwe.py fragmentation --work out/mwe_gen
    python examples/data_generation/make_cpfe_mwe.py postprocess --work out/mwe_gen --device cuda:0

``--puma`` defaults to the ``puma_opt`` key of ``$GRAINTRACE_TOOLS_JSON`` (or ``deploy/tools.json``).
``--export-only`` skips the CPFE run and only re-exports from an existing ``--work`` run.
"""

import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MWE = REPO / "mwe_data"

MATERIAL = dict(
    slip_constant_strength=130.0,
    voce_hardening_initial_slope=1556.09,
    voce_hardening_saturation=100.0,
    power_slip_n=20,
    power_slip_g0=0.0001,
    elastic_E=209016.0,
    elastic_nu=0.307,
    elastic_G=60355.0,
    burger_scale=2.22,
)

FRAG_STRIDE = 3  # keep every 3rd mesh_out step from step 1, plus the final step
POST_GRID_STEPS = 3  # grid snapshots besides the initial one
EXP_TIMES = np.arange(100, 161, 2)  # expsyn_<t>time.csv, t = 100 * MOOSE time


def _puma(arg):
    if arg:
        return arg
    cfg = os.environ.get("GRAINTRACE_TOOLS_JSON", str(REPO / "deploy" / "tools.json"))
    with open(cfg, encoding="utf-8") as fh:
        return json.load(fh)["puma_opt"]


def _inset(bbox, eps=1e-4):
    bb = list(bbox)
    for i in range(0, 6, 2):
        bb[i] += eps
    for i in range(1, 6, 2):
        bb[i] -= eps
    return bb


def _wait(folder, poll=20.0):
    """Block until the detached puma-opt run in ``folder`` exits; raise if it failed."""
    import psutil  # pylint: disable=import-outside-toplevel

    folder = str(Path(folder).resolve())

    def running():
        for p in psutil.process_iter(["name", "cwd"]):
            try:
                if "puma" in (p.info["name"] or "") and p.info["cwd"] == folder:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    time.sleep(5.0)
    t0 = time.time()
    while running():
        time.sleep(poll)
    log = Path(folder) / "cpfe_run.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    if "Finished Executing" not in text:
        raise RuntimeError(f"CPFE run failed; see {log}")
    print(f"CPFE finished after {time.time() - t0:.0f} s")


def _run_cpfe(
    mesh,
    ori,
    folder,
    puma,
    bbox,
    order,
    device,
    ncore,
    strain,
    dt,
    total,
    init,
    sync,
    grid,
    **kw,
):
    from graintrace.run_cpfe_simulation import (  # pylint: disable=import-outside-toplevel
        CPFESimulation,
    )

    sim = CPFESimulation(
        mesh_file=str(mesh),
        save_simulation_folder=str(folder),
        moose_run_file=puma,
        element_order=order,
        eeres_file=None,
        ori_file=str(ori),
        dim=3,
        use_ff_initial_field=kw.pop("use_ff_initial_field", False),
    )
    sim.set_parameters("material", **MATERIAL)
    sim.set_parameters(
        "simulation_parameters",
        device=device,
        device_batch=kw.pop("device_batch", 0),
        dt=dt,
        total_time=total,
        initialize_time=init,
        sync_times=sync,
        **kw,
    )
    sim.set_parameters(
        "boundary",
        bounding_box=list(bbox),
        bc={
            "x": {"negative": "stress_free", "positive": "stress_free"},
            "y": {"negative": "stress_free", "positive": "stress_free"},
            "z": {"negative": 0, "positive": strain * (bbox[5] - bbox[4])},
        },
    )
    sim.set_parameters(
        "grid_properties", number_of_elements=list(grid), bounding_box=_inset(bbox)
    )
    sim.run(ncore=ncore)
    _wait(folder)


# --------------------------------------------------------------------- fragmentation
def fragmentation(work, puma, device, ncore, export_only):
    """Rerun the 10-grain fragmentation CPFE and subset it into mwe_data/cpfe_ff_fragmentation."""
    import meshio  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from graintrace import (  # pylint: disable=import-outside-toplevel
        orientation_helper as oh,
    )

    folder = Path(work) / "fragmentation"
    if not export_only:
        folder.mkdir(parents=True, exist_ok=True)
        euler = np.loadtxt(MWE / "cpfe_ff" / "orientations.dat")  # Bunge, degrees
        mrp = oh.euler_to_mrp(
            torch.tensor(euler, dtype=torch.float64), "bunge", "degrees"
        )
        ori = folder / "orientations_MRP.dat"
        np.savetxt(ori, mrp.numpy(), fmt="%.12g")
        mesh = MWE / "cpfe_ff" / "reconstruction.msh"
        pts = meshio.read(mesh).points
        bbox = [float(f(pts[:, k])) for k in range(3) for f in (np.min, np.max)]
        _run_cpfe(
            mesh,
            ori,
            folder,
            puma,
            bbox,
            "SECOND",
            device,
            ncore,
            strain=0.10,
            dt=0.1,
            total=1.0,
            init=0.1,
            sync="1.0",
            grid=(10, 10, 10),
            mesh_csv="per_step",
            grid_transfer="off",
            exodus_output="sync",
        )

    src = folder / "simulation_out"
    dst = MWE / "cpfe_ff_fragmentation"
    shutil.rmtree(dst / "mesh_out", ignore_errors=True)
    (dst / "mesh_out").mkdir(parents=True)
    keep = ["id", "x", "y", "z", "block_id"] + [f"ori_rodrigues_{c}" for c in "xyz"]
    last = max(int(f.stem.rsplit("_", 1)[1]) for f in (src / "mesh_out").glob("*.csv"))
    steps = list(range(1, last - FRAG_STRIDE + 1, FRAG_STRIDE)) + [last]
    for s in steps:
        name = f"out_element_centroid_{s:04d}.csv"
        pd.read_csv(src / "mesh_out" / name)[keep].to_csv(
            dst / "mesh_out" / name, index=False
        )
    block = pd.read_csv(src / "out.csv")
    cols = ["time"] + sorted(c for c in block.columns if c.startswith("ori_rodrigues_"))
    block[cols].to_csv(dst / "out.csv", index=False)
    print(f"wrote {dst} ({steps} mesh_out steps, {len(block)} block rows)")


# ----------------------------------------------------------------------- postprocess
POST_BOX = [-200.0, 200.0, -173.205, 173.205, 0.0, 600.0]


def _microstructure(folder, seed):
    """Seeded 45-grain NEPER tessellation, voxelized and dumped to a voxel hex Exodus."""
    from graintrace.construct_voronoi_mesh import (  # pylint: disable=import-outside-toplevel
        VoronoiMeshBuilder,
    )
    from graintrace.construct_voxel_mesh import (  # pylint: disable=import-outside-toplevel
        VoxelMeshBuilder,
    )
    from graintrace.generate_random_crystal import (  # pylint: disable=import-outside-toplevel
        CrystalGenerator,
    )

    folder.mkdir(parents=True, exist_ok=True)
    cg = CrystalGenerator(
        output_dir=str(folder), bounding_box=POST_BOX, dim=3, seed=seed, env=None
    )
    cg.generate_tessellation(
        morpho_args={"type": "raw", "morpho_str": "graingrowth", "n": 45},
        iterations=2000,
        extra_neper_args=[
            "-format",
            "tess,ori,tesr",
            "-tesrsize",
            "20,18,30",
            "-tesrformat",
            "ascii",
        ],
    )
    VoronoiMeshBuilder(
        input_csv=None,
        output_dir=str(folder),
        bounding_box=POST_BOX,
        dim=3,
        env=cg.env,
    ).reformat_tesr_file(
        tesr_file=str(folder / "voronoi.tesr"),
        orientation_file=str(folder / "voronoi.ori"),
    )
    builder = VoxelMeshBuilder(
        file_path=str(folder / "voronoi_reformatted.csv"),
        save_dir=str(folder / "mesh"),
        euler_cols=("Eul0", "Eul1", "Eul2"),
        cell_id_col="CellID",
        angle_convention="bunge",
        angle_type="degrees",
        symmetry="432",
    )
    grid = builder.reconstruct(apply_smoothing=False)
    mesh = builder.mesh(
        mesher="voxel",
        merged_grid=grid,
        mesh_path=str(folder / "mesh" / "mesh.e"),
        mapped_orientations_path=str(folder / "mesh" / "orientations"),
    )
    return Path(mesh), folder / "mesh" / "orientations.csv"


def _export_load_steps(block, dst):
    """Per-load-step FF-style grain tables from the CPFE block CSV (true Bunge Euler, radians)."""
    import torch  # pylint: disable=import-outside-toplevel
    from graintrace import (  # pylint: disable=import-outside-toplevel
        orientation_helper as oh,
    )

    gids = sorted(int(c.rsplit("_", 1)[1]) for c in block.columns if c.startswith("volume_"))
    t = block["time"].to_numpy()

    def at(time_value, field):
        cols = [f"{field}_{g}" for g in gids]
        return np.array([np.interp(time_value, t, block[c].to_numpy()) for c in cols])

    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    comp = {"11": "xx", "12": "xy", "13": "xz", "22": "yy", "23": "yz", "33": "zz"}
    for tt in EXP_TIMES:
        tv = tt / 100.0
        df = pd.DataFrame({"grain_id": gids})
        for ax, c in zip("XYZ", "xyz"):
            df[ax] = at(tv, f"centroid_{c}")
        df["GrainRadius"] = np.cbrt(3.0 * at(tv, "volume") / (4.0 * np.pi))
        for i in "123":
            for j in "123":
                key = i + j if i + j in comp else j + i
                df[f"eKen{i}{j}"] = at(tv, f"ee_{comp[key]}")
        mrp = np.stack([at(tv, f"ori_rodrigues_{c}") for c in "xyz"], axis=1)
        eul = oh.mrp_to_euler(torch.tensor(mrp, dtype=torch.float64), "bunge", "radians")
        for k in range(3):
            df[f"Eul{k}"] = eul[:, k].numpy()
        df.to_csv(dst / f"expsyn_{tt}time.csv", index=False)
    print(f"wrote {len(EXP_TIMES)} load-step tables to {dst}")


def postprocess(work, puma, device, ncore, export_only, seed=42):
    """45-grain CPFE -> mwe_data/out.csv + grid_out/ + synthetic_load_exp/."""
    folder = Path(work) / "postprocess"
    sim_dir = folder / "simulation"
    if not export_only:
        mesh, ori = _microstructure(folder / "microstructure", seed)
        _run_cpfe(
            mesh,
            ori,
            sim_dir,
            puma,
            POST_BOX,
            "FIRST",
            device,
            ncore,
            strain=0.12,
            dt=0.05,
            total=2.0,
            init=1.0,
            sync="1.2 1.6 2.0",
            grid=(20, 20, 30),
            device_batch=20000,
            grid_transfer="per_step",
            mesh_csv="off",
            exodus_output="sync",
        )

    src = sim_dir / "simulation_out"
    block = pd.read_csv(src / "out.csv")
    shutil.copy(src / "out.csv", MWE / "out.csv")
    files = sorted(glob.glob(str(src / "grid_out" / "out_element_centroid_*.csv")))
    pick = [files[0]] + [
        files[int(round(i))]
        for i in np.linspace(0, len(files) - 1, POST_GRID_STEPS + 1)[1:]
    ]
    shutil.rmtree(MWE / "grid_out", ignore_errors=True)
    (MWE / "grid_out").mkdir()
    for f in dict.fromkeys(pick):
        shutil.copy(f, MWE / "grid_out" / Path(f).name)
    print(f"wrote mwe_data/out.csv ({block.shape}) + {len(set(pick))} grid_out files")
    _export_load_steps(block, MWE / "synthetic_load_exp")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", choices=["fragmentation", "postprocess"])
    ap.add_argument("--work", default="out/mwe_gen")
    ap.add_argument("--puma", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ncore", type=int, default=4)
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()
    puma = None if a.export_only else _puma(a.puma)
    if a.target == "fragmentation":
        fragmentation(a.work, puma, a.device, a.ncore, a.export_only)
    else:
        postprocess(a.work, puma, a.device, a.ncore, a.export_only)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# Copyright 2026, UChicago Argonne, LLC -- MIT (see repo LICENSE)
"""Full graintrace pipeline on the synthetic demo experiment: ONE real run.

  stitch -> material calibration -> FF reconstruct (+GMSH mesh) -> CPFE (cuda:0)
         -> rare-event identification -> plots + "which grains/locations to look at"

This is NOT a smoke test: CPFE runs to completion (the script blocks until the
MOOSE/PUMA run finishes) before REI. Run `python demo/generate_experiment.py` first.

Stages are individually skippable via the RESTART_* flags (outputs are reused), so
you can re-run a later stage after tweaking without redoing the whole pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- make neml2 AOTI / neml2-compile / puma-opt find the env's newer libstdc++ ---
from graintrace.mcp import deps as _deps
from graintrace.mcp import tool_paths as _tp

_deps.ensure_runtime_ld_library_path()

HERE = Path(__file__).resolve().parent
EXP = HERE / "experiment"
OUT = HERE / "out"
TRUTH = HERE / "_truth"
EKEN_COLS = [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
COORD = ["X", "Y", "Z"]

# ---------------------------------------------------------------- INPUT
# External tool paths come from tools.json (~/.config/graintrace/tools.json or
# $GRAINTRACE_TOOLS_JSON); the literals are fallbacks for this box.
MOOSE_RUN_FILE = _tp.puma_opt() or "external/puma/puma-opt"  # EDIT: your built puma-opt
CVT_ITER = 300  # reconstruction CVT relaxation (fast)
REL_EL_SIZE = 3.5  # (unused by the SCULPT path; kept for reference)
TESR_SIZE = [24, 24, 26]  # voxelization grid -> hex mesh resolution (~14k hex)

# CUBIT/SCULPT hex meshing (production path; NOT GMSH tets). Recommended options
# from the 'meshing' recipe: adapt4 (-A 4 -df 1) preserves >=98% of grains.
_CUBIT_BIN = (
    "/path/to/cubit/bin"  # EDIT: your Coreform CUBIT/SCULPT install (paths only)
)
SCULPT_CONFIG = _tp.sculpt_config() or {
    "launcher": f"{_CUBIT_BIN}/mpi/bin/mpiexec",
    "psculpt": f"{_CUBIT_BIN}/psculpt",
    "epu": f"{_CUBIT_BIN}/epu",
    "nprocs": 8,
    "environment": {
        "OPAL_LIBDIR": f"{_CUBIT_BIN}/mpi/lib",
        "OPAL_PREFIX": f"{_CUBIT_BIN}/mpi",
    },
}
SCULPT_OPTIONS = ("-A", "4", "-df", "1", "-S", "2", "-CS", "4", "--void_mat", "0")
# CPFE (fast, cuda:0). MOOSE-side assembly/user-objects dominate (the NEML2 GPU
# eval is ~1s), so keep the mesh coarse, the probe grid small, and few load steps.
# CPFE and calibration MUST use the SAME strain rate (rate-dependent slip). We set
# ONE rate for both here. It also sets the physical loading time = total_strain /
# ASSUMED_RATE: at 1e-2/s, 20% -> 20 s physical, ~40 steps at dt<=0.5 s. (The
# NEML2 material integration can't take large *physical* dt, so a slow quasi-static
# rate like 1e-4/s would need ~2000 s -> thousands of tiny steps -> intractable;
# 1e-2/s is a reasonable lab rate that keeps it fast AND self-consistent.)
ASSUMED_RATE = 1.0e-2  # /s, used by BOTH TaylorModel calibration and CPFE
CPFE = dict(
    device="cuda:0",
    ncore=16,
    device_batch=20000,
    dt=0.05,
    initialize_time=0.5,  # dtmax=10*dt=0.5 s; total_time derived
    grid_elems=[16, 16, 16],
    element_order="FIRST",
)  # SCULPT -> hex8
CPFE_TIMEOUT_S = 14400  # 4 h; a matched-rate 20% pull on ~40k hex takes ~2 h
# Physically-bounded calibration ranges (default nu range is (-0.5,0.5) which lets
# the fit drift to near-incompressible -> FE locking / poor convergence).
PARAM_RANGES = {
    "elastic_tensor_E": (80_000.0, 160_000.0),
    "elastic_tensor_G": (30_000.0, 60_000.0),
    "elastic_tensor_nu": (0.25, 0.35),
    "slip_strength_constant_strength": (20.0, 400.0),
    "voce_hardening_initial_slope": (200.0, 5_000.0),
    "voce_hardening_saturated_hardening": (20.0, 400.0),
}
# REI
REI = dict(k=5, gamma=10.0, manhattan_radius=4, threshold=5e-4, n_jobs=12, seed=42)
# Stage reuse: set env DEMO_REUSE="stitch,calibrate,reconstruct,cpfe" to reuse
# existing outputs for those stages (default: run everything fresh).
_REUSE = {s.strip() for s in os.environ.get("DEMO_REUSE", "").split(",") if s.strip()}
RESTART = {k: (k in _REUSE) for k in ("stitch", "calibrate", "reconstruct", "cpfe")}

# base material (calibrated values overwrite the elastic + slip/voce terms)
BASE_MATERIAL = dict(
    slip_constant_strength=100.0,
    voce_hardening_initial_slope=1650.0,
    voce_hardening_saturation=220.0,
    power_slip_n=25,
    power_slip_g0=1e-4,
    elastic_E=109000.0,
    elastic_nu=0.307,
    elastic_G=41700.0,
    burger_scale=2.54,
)
# calibrated opt-var name -> CPFE material name
CALIB_MAP = {
    "elastic_tensor_E": "elastic_E",
    "elastic_tensor_G": "elastic_G",
    "elastic_tensor_nu": "elastic_nu",
    "slip_strength_constant_strength": "slip_constant_strength",
    "voce_hardening_initial_slope": "voce_hardening_initial_slope",
    "voce_hardening_saturated_hardening": "voce_hardening_saturation",
}


def _step(msg):
    print(f"\n{'='*70}\n>>> {msg}\n{'='*70}", flush=True)


def load_sample():
    meta = json.loads((EXP / "sample.json").read_text())
    return meta


# ---------------------------------------------------------------- 1. stitch
def stitch(meta):
    from graintrace.hedm_stitching_techniques.region_base_stitching import (
        RegionBaseStitching,
    )

    out_csv = OUT / "stitched.csv"
    if RESTART["stitch"] and out_csv.exists():
        print(f"[skip] reuse {out_csv}")
        return out_csv
    scans = sorted(str(p) for p in (EXP / "hedm_scan").glob("scan_*.csv"))
    sg = meta["scan_geometry"]
    cols = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2", "ScanID"] + EKEN_COLS
    stitcher = RegionBaseStitching(
        scan_files=scans,
        output_csv=str(out_csv),
        position_tolerance=50,
        orientation_tolerance=5.0,
        radius_tolerance=-1,
        weights={"pos": 0.1, "ori": 1.0, "rad": 0},
        min_neighbors=5,
        orientation_convention=meta["sample"]["units"]["orientation_convention"],
        orientation_units="degrees",
        symmetry=meta["sample"]["units"]["symmetry"],
        output_column=cols,
    )
    stitcher.run(zlo=sg["zlo"], zhi=sg["zhi"], overlap_fraction=sg["overlap_fraction"])

    # Ensure residual eKen survived; if not, re-attach by nearest scan grain.
    df = pd.read_csv(out_csv)
    if not set(EKEN_COLS).issubset(df.columns) or df[EKEN_COLS].isna().all().all():
        from scipy.spatial import cKDTree

        allscan = pd.concat([pd.read_csv(s) for s in scans], ignore_index=True)
        tree = cKDTree(allscan[COORD].to_numpy())
        _, idx = tree.query(df[COORD].to_numpy())
        df[EKEN_COLS] = allscan[EKEN_COLS].to_numpy()[idx]
        df.to_csv(out_csv, index=False)
    print(f"    stitched grains: {len(df)} -> {out_csv}")
    return out_csv


# ---------------------------------------------------------------- 2. calibrate
def calibrate(meta, stitched_csv):
    import graintrace as _gt
    from graintrace import orientation_helper as oh
    from graintrace.material_calibration import MaterialCalibration
    from graintrace.taylor import TaylorModel

    save_dir = OUT / "material_calibration"
    calib_json = save_dir / "calibrated_material.json"
    calib_data = OUT / "calib_data"
    if RESTART["calibrate"] and calib_json.exists():
        print(f"[skip] reuse {calib_json}")
        return _material_from_json(calib_json)

    # Build one texture CSV from the sample's OWN grains (only exp_texture[0] is
    # used by the calibration; the loss targets the macro strain-stress curve).
    calib_data.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(stitched_csv)
    M = (
        oh.euler_to_matrix(df[["Eul0", "Eul1", "Eul2"]].to_numpy(), "bunge", "degrees")
        .numpy()
        .reshape(len(df), 9)
    )
    tex = pd.DataFrame(M, columns=[f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)])
    tex[COORD] = df[COORD].to_numpy()
    tex["GrainRadius"] = df["GrainRadius"].to_numpy()
    tex[EKEN_COLS] = 0.0  # not used by the loss
    tex.to_csv(calib_data / "1.csv", index=False)  # numeric-named -> a "stress level"

    cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")
    calib = MaterialCalibration(
        model_class=TaylorModel,
        model_args=dict(
            neml2_path=cpfe_base + "/neml2_cpfe_calibration.i",
            npoints=30,
            nchunk=2,
            device="cuda",
            compile=False,
            assumed_rate=ASSUMED_RATE,
        ),  # == CPFE rate (consistency)
        data_args=dict(
            data_dir=str(calib_data),
            strain_stress_file=str(EXP / "strain-stress.csv"),
            npoints=30,
            full_field_strain_units="microstrain",
            straintype="eKen",
            max_strain=0.006,
            n_grains=min(100, len(df)),
            seed=42,
        ),
        save_dir=str(save_dir),
        apply_elastic_correction=False,
    )
    calib.calibrate(
        maxiter=15,
        lr=0.3,
        max_iter_per_step=6,
        line_search_fn="strong_wolfe",
        plateau_rtol=1e-3,
        plateau_window=2,
        param_ranges=PARAM_RANGES,
    )
    try:
        calib.plot_stress_strain(include_model=True)
    except Exception as e:
        print(f"    (calibration stress-strain plot skipped: {e})")
    return _material_from_json(calib_json)


def _material_from_json(path):
    raw = json.loads(Path(path).read_text())

    # accept {name: value} or {name: {"value": ...}}
    def val(v):
        return float(v["value"]) if isinstance(v, dict) and "value" in v else float(v)

    flat = {k: val(v) for k, v in raw.items() if _is_num(v)}
    mat = dict(BASE_MATERIAL)
    for opt_name, cpfe_name in CALIB_MAP.items():
        if opt_name in flat:
            mat[cpfe_name] = flat[opt_name]
    print(
        "    calibrated material:",
        {
            k: round(mat[k], 4)
            for k in (
                "elastic_E",
                "elastic_nu",
                "elastic_G",
                "slip_constant_strength",
                "voce_hardening_initial_slope",
                "voce_hardening_saturation",
            )
        },
    )
    return mat


def _is_num(v):
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, dict) and "value" in v:
        return True
    return False


# ---------------------------------------------------------------- 3. reconstruct
def reconstruct(meta, stitched_csv):
    """FF Voronoi reconstruction -> SCULPT hex mesh (production path, not GMSH).

    build_voronoi(generate_mesh=False) writes the voxelized grain field
    (reconstruction_reformatted.csv, CellID + Eul) + per-grain ee; VoxelMeshBuilder
    then meshes those voxels into a conformal hex Exodus mesh via CUBIT/SCULPT.
    """
    from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
    from graintrace.construct_voxel_mesh import VoxelMeshBuilder

    out_ff = OUT / "FF"
    mesh_dir = out_ff / "mesh"
    mesh_e = mesh_dir / "mesh.e"
    if RESTART["reconstruct"] and mesh_e.exists():
        print(f"[skip] reuse {mesh_e}")
        return out_ff
    bbox = meta["sample"]["bounding_box_um"]
    unit = "deg" if meta["sample"]["units"]["orientation"].startswith("deg") else "rad"
    builder = VoronoiMeshBuilder(
        input_csv=str(stitched_csv),
        output_dir=str(out_ff),
        bounding_box=bbox,
        dim=3,
        auto_fix_bbox=True,
        bbox_fix_mode="remove_points",
        bbox_tolerance=2.5,
        angle_identifier=["Eul0", "Eul1", "Eul2"],
        orientation_descriptor="euler-bunge",
        orientation_active_convention=True,
        unit=unit,
        elastic_strain_identifier=EKEN_COLS,
        strain_unit="microstrain",
    )
    builder.build_voronoi(
        generate_mesh=False,
        option="centroid",
        CVT_iter=CVT_ITER,
        morphoalgo="subplex",
        mesh_quality_min=0.7,
        relative_el_size=REL_EL_SIZE,
        tesr_size=TESR_SIZE,
    )
    vox = VoxelMeshBuilder(
        file_path=str(out_ff / "reconstruction_reformatted.csv"),
        save_dir=str(mesh_dir),
        euler_cols=["Eul0", "Eul1", "Eul2"],
        cell_id_col="CellID",
        angle_convention="bunge",
        angle_type="degrees",
        symmetry="432",
    )
    merged = vox.reconstruct(apply_smoothing=False)
    mesh_path = vox.mesh(
        sculpt_config=SCULPT_CONFIG,
        sculpt_options=list(SCULPT_OPTIONS),
        merged_grid=merged,
    )
    print(f"    SCULPT hex mesh -> {mesh_path}  (orientations.csv is MRP)")
    return out_ff


def euler_dat_to_mrp(out_ff):
    import torch
    from graintrace import orientation_helper as oh

    euler = np.loadtxt(out_ff / "orientations.dat")
    mrp = oh.euler_to_mrp(torch.tensor(euler, dtype=torch.float64), "bunge", "degrees")
    p = out_ff / "orientations_MRP.dat"
    np.savetxt(p, mrp.numpy(), fmt="%.12g")
    return p


# ---------------------------------------------------------------- 4. CPFE
def run_cpfe(meta, out_ff, material):
    from graintrace.run_cpfe_simulation import CPFESimulation

    out_sim = OUT / "simulation"
    grid_dir = out_sim / "simulation_out" / "grid_out"
    if RESTART["cpfe"] and grid_dir.exists() and any(grid_dir.glob("*.csv")):
        print(f"[skip] reuse {grid_dir}")
        return _last_grid_csv(grid_dir)

    if not Path(MOOSE_RUN_FILE).exists():
        raise FileNotFoundError(
            f"puma-opt not found at {MOOSE_RUN_FILE}; set MOOSE_RUN_FILE."
        )

    # SCULPT hex mesh + its MRP orientations (VoxelMeshBuilder already writes MRP).
    mesh_e = out_ff / "mesh" / "mesh.e"
    ori_csv = out_ff / "mesh" / "orientations.csv"
    bbox = list(meta["sample"]["bounding_box_um"])
    total_strain = meta["loading"]["total_strain"]
    displace = total_strain * (bbox[5] - bbox[4])
    grid_bb = bbox.copy()
    for i in (0, 2, 4):
        grid_bb[i] += 1e-4
    for i in (1, 3, 5):
        grid_bb[i] -= 1e-4

    # Physical time so the CPFE loads at the CALIBRATED strain rate (rate-dependent
    # slip -> the CPFE rate must equal the calibration's assumed_rate).
    init_t = CPFE["initialize_time"]
    loading_time = total_strain / ASSUMED_RATE  # seconds
    total_time = init_t + loading_time
    sync_times = f"{total_time:.8g}"
    strain_rate = total_strain / loading_time
    print(
        f"    rate {strain_rate:.2e}/s (== calibration {ASSUMED_RATE:.0e}) -> "
        f"loading {loading_time:.6g}s, total_time {total_time:.6g}s"
    )

    sim = CPFESimulation(
        mesh_file=str(mesh_e),
        save_simulation_folder=str(out_sim),
        moose_run_file=MOOSE_RUN_FILE,
        element_order=CPFE["element_order"],
        eeres_file=str(out_ff / "reconstruction_cpfe_ee.csv"),
        ori_file=str(ori_csv),
        dim=3,
        use_ff_initial_field=True,
    )
    sim.set_parameters("material", **material)
    sim.set_parameters(
        "simulation_parameters",
        dt=CPFE["dt"],
        total_time=total_time,
        initialize_time=init_t,
        device=CPFE["device"],
        device_batch=CPFE["device_batch"],
        sync_times=sync_times,
    )
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
        "grid_properties", number_of_elements=CPFE["grid_elems"], bounding_box=grid_bb
    )
    print(
        f"    launching CPFE (device={CPFE['device']}, ncore={CPFE['ncore']}, "
        f"displace={displace:.4g} um) ..."
    )
    sim.run(ncore=CPFE["ncore"])
    return _wait_for_cpfe(out_sim, total_time)


def _csv_has_rows(path) -> bool:
    try:
        with open(path) as fh:
            fh.readline()  # header
            return bool(fh.readline().strip())
    except Exception:
        return False


def _last_grid_csv(grid_dir):
    """Last grid CSV that actually has data rows (skip header-only t=0 output)."""
    files = sorted(Path(grid_dir).glob("out_element_centroid_*.csv"))
    nonempty = [f for f in files if _csv_has_rows(f)]
    return nonempty[-1] if nonempty else None


def _wait_for_cpfe(out_sim, total_time, timeout=CPFE_TIMEOUT_S):
    sim_out = Path(out_sim) / "simulation_out"
    grid_dir = sim_out / "grid_out"
    block_csv = sim_out / "out.csv"
    log = Path(out_sim) / "cpfe_run.log"
    t0 = time.time()
    print("    waiting for CPFE to finish (polling) ...", flush=True)
    while time.time() - t0 < timeout:
        time.sleep(10)
        logtxt = log.read_text(errors="replace") if log.exists() else ""
        # HARD-crash signatures only. Do NOT treat 'Solve Did NOT Converge' /
        # 'cutting timestep' as fatal; MOOSE recovers by reducing dt. Completion
        # is judged by reaching total_time (below) or the process exiting.
        if re.search(
            r"terminate called|Segmentation fault|MPI_ABORT|"
            r"Fatal error in|Command not found",
            logtxt,
        ):
            raise RuntimeError(
                "CPFE run crashed. Last log lines:\n"
                + "\n".join(logtxt.splitlines()[-30:])
            )
        # completion = per-grain block CSV reached total_time AND a non-empty
        # grid snapshot exists.
        reached = False
        if block_csv.exists():
            try:
                b = pd.read_csv(block_csv)
                reached = "time" in b and float(b["time"].max()) >= total_time - 1e-9
            except Exception:
                pass
        last = _last_grid_csv(grid_dir) if grid_dir.exists() else None
        proc_done = not _puma_running()
        # complete: block reached total_time, or the process finished and left a
        # non-empty grid snapshot.
        if last is not None and (reached or proc_done):
            print(f"    CPFE complete in {time.time()-t0:.0f}s -> {last}")
            return last
        # process died without producing a usable snapshot?
        if proc_done and last is None and not reached:
            raise RuntimeError(
                "puma-opt exited before completion (no non-empty grid output). "
                "Last log lines:\n" + "\n".join(logtxt.splitlines()[-30:])
            )
        tmax = ""
        if block_csv.exists():
            try:
                tmax = f", sim time={pd.read_csv(block_csv)['time'].max():.3g}/{total_time}"
            except Exception:
                pass
        print(f"      ... {int(time.time()-t0)}s elapsed{tmax}", flush=True)
    raise TimeoutError(f"CPFE did not finish within {timeout}s")


def _puma_running():
    try:
        r = subprocess.run(["pgrep", "-f", "puma-opt"], capture_output=True)
        return r.returncode == 0
    except Exception:
        return True  # can't tell -> keep waiting


# ---------------------------------------------------------------- 5. REI
def rei(last_grid_csv):
    from graintrace.rare_cluster_indicator import IdentifyRareClusters
    from graintrace.similarity_metric_library import SimilarityMetricLibrary
    from graintrace.user_data_class import SimilarityMetric, WeightConfig, RareCriteria
    from graintrace import rare_criteria_selection_library as rcs

    out_rei = OUT / "rei"
    out_rei.mkdir(parents=True, exist_ok=True)
    if last_grid_csv is None:
        fb = HERE.parent / "mwe_data" / "grid_out"
        cands = sorted(fb.glob("out_element_centroid_*.csv"))
        if not cands:
            raise RuntimeError("no CPFE grid_out and no mwe_data/grid_out fallback")
        last_grid_csv = cands[-1]
        print(f"    [fallback] REI on {last_grid_csv}")
    base = str(out_rei / "rei")

    lib = SimilarityMetricLibrary()
    nye_cols = [f"nye_tensor_{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    spec = lib.nye_tensor_norm(cols=nye_cols)
    spec_reduced = SimilarityMetric(
        name=spec.name + "_mean",
        feature_cols=[f"{c}_mean" for c in spec.feature_cols],
        func=spec.func,
    )
    scalar_col = spec_reduced.name + "_mean"
    rare_criteria = RareCriteria(
        selector=lambda df: rcs.select_highest_scalar(
            df, k=REI["k"], required_cols=scalar_col, min_size=1
        )
    )
    weight_cfg = WeightConfig(
        mode="rbf",
        power=2.0,
        sigma=None,
        sigma_auto={
            "sample_size": 500_000,
            "random_state": REI["seed"],
            "quantile": 0.5,
        },
    )
    irc = IdentifyRareClusters(
        input_csv_path=str(last_grid_csv), id_col="id", coord_cols=("x", "y", "z")
    )
    gsc, indicator = irc.make_stage_objects(graph_cluster_out=base + "_reduced.csv")
    bundle = irc.run_clustering(
        gsc=gsc,
        indicator=indicator,
        reduced_csv_path=base + "_reduced.csv",
        gsc_run_kwargs=dict(
            spec=spec,
            graph_mode="grid",
            manhattan_radius=REI["manhattan_radius"],
            grid_tol=1e-6,
            n_jobs=REI["n_jobs"],
            weight_chunk_size=500_000,
            segmenter="leiden",
            seed=REI["seed"],
            weight_cfg=weight_cfg,
            reduce_edges_topweights_k=20,
            networkit_kwargs={"gamma": REI["gamma"]},
            checkpoint_base_path=base + "_gsc_ckpt",
            resume_from_checkpoint=False,
        ),
        indicator_run_kwargs=dict(
            method_type="scipy_hierarchical",
            spec=spec_reduced,
            threshold=REI["threshold"],
            method="average",
            criterion="distance",
            dendrogram_path=base + "_dendrogram.png",
        ),
    )
    stats_csv = base + "_rare_cluster_stats.csv"
    out = irc.run_get_rare_cluster(
        bundle=bundle,
        criteria=rare_criteria,
        output_vtk_path=base + "_rare_clusters.vtk",
        export_control="auto",
        background_block_id=1,
        first_rare_block_id=2,
        also_write_final_label=True,
        rare_reduced_stats_csv_path=stats_csv,
        use_sample_std=False,
    )
    return out, stats_csv, str(last_grid_csv)


# ---------------------------------------------------------------- 6. plots
def make_plots(out_ff, last_grid_csv, rei_out):
    from graintrace.mcp import render

    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # reconstruction grains (IPF-colored)
    try:
        from graintrace.ipf_postprocess import IPFProcessor

        recon_vtk = out_ff / "reconstruction_reformatted.vtk"
        if recon_vtk.exists():
            ipf = IPFProcessor(
                crystal_symmetry="432", sample_symmetry="432", save_dir=str(plots)
            )
            # output_file is resolved relative to save_dir -> pass a bare filename.
            rgb_vtk = ipf.add_block_rgb_to_vtk(
                vtk_file=str(recon_vtk),
                output_file="reconstruction_ipf.vtk",
                direction=[0.0, 0.0, 1.0],
                angle_convention="bunge",
                angle_type="degrees",
                orientation_fields=("Eul0", "Eul1", "Eul2"),
            )
            render.render_vtk(
                str(rgb_vtk),
                str(plots / "grains_reconstruction.png"),
                title="Reconstruction (IPF-z)",
            )
    except Exception as e:
        print(f"    (grain render skipped: {e})")

    # REI hotspots
    try:
        rare_vtk = (
            rei_out[0].get("output_vtk_path") if isinstance(rei_out[0], dict) else None
        )
        rare_vtk = rare_vtk or (str(OUT / "rei" / "rei_rare_clusters.vtk"))
        if Path(rare_vtk).exists():
            render.render_vtk(
                rare_vtk,
                str(plots / "rei_hotspots.png"),
                threshold_rare=True,
                title="Rare clusters (nye-tensor)",
            )
    except Exception as e:
        print(f"    (REI render skipped: {e})")

    # CPFE field on the probe grid (nye norm)
    try:
        if last_grid_csv and Path(last_grid_csv).exists():
            _plot_grid_field(last_grid_csv, plots / "cpfe_nye_field.png")
    except Exception as e:
        print(f"    (CPFE field plot skipped: {e})")

    # CPFE macroscopic stress-strain (the flow curve; shows how far we pulled)
    try:
        from graintrace.simulation_postprocessing import (
            SimulationResults,
            FieldFileNaming,
        )
        from graintrace import plot_postprocessing as pp

        block = OUT / "simulation" / "simulation_out" / "out.csv"
        fdir = OUT / "simulation" / "simulation_out" / "grid_out"
        if block.exists():
            res = SimulationResults(
                block_csv=str(block),
                field_dir=str(fdir),
                field_naming=FieldFileNaming(
                    prefix="out_element_centroid", index_width=4, sep="_", suffix=".csv"
                ),
            )
            pp.plot_macroscopic_stress_strain(
                res,
                stress_tensor_prefix="cauchy_stress",
                strain_tensor_prefix="strain",
                volume_prefix="volume",
                output_folder=str(plots),
            )
    except Exception as e:
        print(f"    (CPFE stress-strain plot skipped: {e})")
    print(f"    plots -> {plots}")


def _plot_grid_field(csv, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv)
    nye = [c for c in df.columns if c.startswith("nye_tensor_")]
    df["nye_norm"] = np.linalg.norm(df[nye].to_numpy(), axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["nye_norm"], bins=60)
    ax[0].set_title("Nye-tensor norm distribution")
    ax[0].set_xlabel("|nye|")
    sc = ax[1].scatter(df["x"], df["y"], c=df["nye_norm"], s=4, cmap="viridis")
    ax[1].set_title("Nye norm (x-y)")
    ax[1].set_aspect("equal")
    fig.colorbar(sc, ax=ax[1])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 7. report
def report(stats_csv, out_ff):
    _step("LOOK HERE: rare grains / locations")
    if not Path(stats_csv).exists():
        print("no rare-cluster stats produced.")
        return
    stats = pd.read_csv(stats_csv)
    sev_col = next(
        (c for c in stats.columns if c.endswith("nye_tensor_norm_mean_mean")), None
    )
    order = stats.sort_values(sev_col, ascending=False) if sev_col else stats
    # nearest grain centroid lookup
    grains = None
    ori = out_ff / "reconstruction_reformatted.csv"
    try:
        if ori.exists():
            g = pd.read_csv(ori)
            gcoord = [
                c
                for c in (("x", "y", "z"), ("X", "Y", "Z"))
                if set(c) <= set(g.columns)
            ]
            if gcoord:
                from scipy.spatial import cKDTree

                cc = list(gcoord[0])
                idcol = next(
                    (
                        c
                        for c in ("cell_id", "id", "grain_id", "GrainID")
                        if c in g.columns
                    ),
                    None,
                )
                grains = (g, cKDTree(g[cc].to_numpy()), cc, idcol)
    except Exception:
        grains = None
    for _, r in order.iterrows():
        cx, cy, cz = r.get("x_mean"), r.get("y_mean"), r.get("z_mean")
        sev = r.get(sev_col) if sev_col else float("nan")
        line = (
            f"  cluster {int(r.get('cluster_label', -1))} "
            f"(block {int(r.get('rare_cluster_id', -1))}, n={int(r.get('n', 0))}): "
            f"centroid=({cx:.1f}, {cy:.1f}, {cz:.1f}) um, "
            f"severity(|nye|)={sev:.4g}"
        )
        if grains is not None and not any(v is None for v in (cx, cy, cz)):
            g, tree, cc, idcol = grains
            _, gi = tree.query([cx, cy, cz])
            gid = g.iloc[gi][idcol] if idcol else gi
            line += f", nearest grain={gid}"
        print(line)
    print(f"\nfull stats: {stats_csv}")
    print(
        f"plots: {OUT/'plots'} | rare VTK: {OUT/'rei'} | "
        f"mesh (open in ParaView): {out_ff/'mesh'/'mesh.e'}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (EXP / "sample.json").exists():
        sys.exit("Run `python demo/generate_experiment.py` first.")
    meta = load_sample()

    _step("1/6 stitch scans")
    stitched = stitch(meta)
    _step("2/6 material calibration")
    material = calibrate(meta, stitched)
    _step("3/6 FF reconstruct + mesh")
    out_ff = reconstruct(meta, stitched)
    _step("4/6 CPFE (run to completion)")
    last_grid = run_cpfe(meta, out_ff, material)
    _step("5/6 rare-event identification")
    rei_out = rei(last_grid)
    _step("6/6 plots")
    make_plots(out_ff, last_grid, rei_out)
    report(rei_out[1], out_ff)


if __name__ == "__main__":
    main()

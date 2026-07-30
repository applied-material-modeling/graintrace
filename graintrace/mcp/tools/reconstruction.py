# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tools: 3D microstructure reconstruction from FF / NF / EBSD data.

  * ff_reconstruct  -> VoronoiMeshBuilder (NEPER Voronoi/CVT)
  * nf_reconstruct  -> NearFieldMeshBuilder (voxel segmentation + optional SCULPT mesh)
  * voxel_mesh      -> VoxelMeshBuilder (EBSD / gridded NF -> segmentation + SCULPT mesh)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graintrace.mcp.app import mcp, workdir
from graintrace.mcp.confirm import gate

_EKEN_9 = [
    "eKen11",
    "eKen12",
    "eKen13",
    "eKen21",
    "eKen22",
    "eKen23",
    "eKen31",
    "eKen32",
    "eKen33",
]

_FF_INIT_DEFAULTS: Dict[str, Any] = {
    "dim": 3,
    "weighted": False,
    "auto_fix_bbox": True,
    "bbox_fix_mode": "remove_points",
    "bbox_tolerance": 2.5,
    "auto_rotate": False,
    "rotate_angles": [0, 0, 0],
    "rotate_convention": "xyz",
    "angle_identifier": ["Eul0", "Eul1", "Eul2"],
    "orientation_descriptor": "euler-bunge",
    "orientation_active_convention": True,
    "unit": "deg",
    "elastic_strain_identifier": _EKEN_9,
    "strain_unit": "microstrain",
}
_FF_BUILD_DEFAULTS: Dict[str, Any] = {
    "option": "centroid",
    "generate_mesh": False,
    "relative_el_size": 2.0,
    "morphoalgo": "subplex",
    "mesh_quality_min": 0.7,
    "tesr_size": [100, 100, 100],
    "CVT_iter": 1000,
}


@mcp.tool()
def ff_reconstruct(
    input_csv: str,
    bounding_box: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
    sample_json: Optional[str] = None,
    generate_mesh: bool = False,
    init_params: Optional[Dict[str, Any]] = None,
    build_params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Reconstruct a 3D Voronoi microstructure from an FF-HEDM grain CSV
    (wraps `VoronoiMeshBuilder.build_voronoi`). Read
    `get_recommended_parameters('ff_reconstruction')` first, and call
    `inspect_experiment(input_csv)` if you don't already have the metadata.

    Parameters
    ----------
    input_csv : FF grain CSV (X,Y,Z, Eul0/1/2, and eKen*/eFab* for the ee file).
    bounding_box : [xlo,xhi,ylo,yhi,zlo,zhi] micrometers. REQUIRED (from the user
        or a sample.json) -- a CSV does not contain the sample dimensions.
    sample_json : path to an experiment sample.json (supplies bounding_box + units).
    output_dir : output folder (defaults under the MCP workdir).
    generate_mesh : **default False**. When False, this only produces the Voronoi
        tessellation + reconstruction_reformatted.csv + ee file (no mesh). GMSH tet
        meshing (True) is an FF-only LAST RESORT -- the recommended CPFE mesh is
        SCULPT hex: keep this False, then call `voxel_mesh` on
        reconstruction_reformatted.csv. Only set True if CUBIT/SCULPT is missing.
    init_params : overrides for VoronoiMeshBuilder(...) -- e.g. unit ('deg'|'rad',
        must match the CSV), auto_rotate, rotate_angles, weighted,
        elastic_strain_identifier, strain_unit, bbox_fix_mode.
    build_params : overrides for build_voronoi(...) -- option, CVT_iter,
        morphoalgo, relative_el_size, tesr_size.

    If bounding_box / orientation unit are not provided (directly or via
    sample_json), returns status 'needs_input' with suggestions -- ask the user.
    Needs NEPER (and GMSH only when generate_mesh=true). Runs as a background job.
    """
    # Lazy: heavy graintrace submodule + mcp helper (pandas), imported on run.
    # pylint: disable=import-outside-toplevel
    from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
    from graintrace.mcp import sample_meta

    if output_dir is None:
        output_dir = str(workdir() / "FF")
    smeta = sample_meta.resolve_sample(sample_json)
    if bounding_box is None:
        bounding_box = smeta.get("bounding_box")
    init = {**_FF_INIT_DEFAULTS, **(init_params or {})}
    unit_provided = (init_params or {}).get("unit") is not None or "unit" in smeta
    if "unit" in smeta:
        init["unit"] = smeta["unit"]

    # Non-inferrable inputs: bounding box (sample dims) + orientation unit.
    missing, suggestions = [], {}
    if bounding_box is None:
        missing.append("bounding_box (sample dimensions [xlo,xhi,ylo,yhi,zlo,zhi] um)")
    if not unit_provided:
        missing.append(
            "orientation unit ('deg' or 'rad') -- confirm; not auto-detected"
        )
    # Pre-flight: fail SYNCHRONOUSLY (not inside the background job) if the elastic
    # strain columns aren't present -- e.g. a stitched CSV that dropped eKen.
    esid = init.get("elastic_strain_identifier")
    if esid:
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel

            cols = set(pd.read_csv(input_csv, nrows=1).columns)
            if not set(esid).issubset(cols):
                missing.append(
                    f"elastic-strain columns {esid[0]}..{esid[-1]} are not in "
                    f"input_csv ({input_csv}). If this is a stitched CSV, re-run "
                    "stitch_scans (it now re-attaches residual eKen from the scans), "
                    "or set init_params.elastic_strain_identifier=null for zero "
                    "initial strain."
                )
        # Best-effort pre-flight: skip the column check if the CSV can't be read.
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    if missing:
        try:
            suggestions = sample_meta.inspect_csv(input_csv)
        # Best-effort: surface inspection failure as a suggestion, not a crash.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            suggestions = {"inspection_error": str(exc)}

    build = {**_FF_BUILD_DEFAULTS, **(build_params or {})}
    build["generate_mesh"] = bool(generate_mesh)  # explicit switch wins; default False
    resolved = {
        "input_csv": input_csv,
        "output_dir": output_dir,
        "bounding_box": bounding_box,
        "init_params": init,
        "build_params": build,
    }
    needs = ["neper"] + (["gmsh"] if build.get("generate_mesh") else [])

    def _run():
        builder = VoronoiMeshBuilder(
            input_csv=input_csv,
            output_dir=output_dir,
            bounding_box=bounding_box,
            **init,
        )
        builder.build_voronoi(**build)
        return {
            "output_dir": output_dir,
            "expected": [
                f"{output_dir}/reconstruction_reformatted.csv",
                f"{output_dir}/reconstruction_cpfe_ee.csv",
                f"{output_dir}/orientations.dat",
            ]
            + (
                [f"{output_dir}/reconstruction.msh"]
                if build.get("generate_mesh")
                else []
            ),
        }

    return gate(
        tool="ff_reconstruct",
        confirm=confirm,
        resolved_params=resolved,
        needs=needs,
        will_write=[output_dir],
        run=_run,
        background=True,
        notes="CVT relaxation can take a while; runs in the background.",
        missing_required=missing,
        suggestions=suggestions,
    )


@mcp.tool()
def nf_reconstruct(
    input_folder: str,
    dz: float,
    nx: int,
    ny: int,
    save_dir: Optional[str] = None,
    init_params: Optional[Dict[str, Any]] = None,
    segmentation: Optional[Dict[str, Any]] = None,
    sculpt_config: Optional[Dict[str, Any]] = None,
    sculpt_options: Optional[List[str]] = None,
    confirm: bool = False,
) -> dict:
    """Reconstruct a high-resolution NF-HEDM voxel microstructure from a folder
    of .mic layers (wraps `NearFieldMeshBuilder`). Segments the voxel grid, and
    -- if `sculpt_config` is given -- also builds a hex Exodus mesh via SCULPT.

    Parameters
    ----------
    input_folder : folder of .mic layer files.
    dz, nx, ny : layer thickness (um) and in-plane grid resolution.
    save_dir : output folder (defaults under the MCP workdir).
    init_params : overrides for NearFieldMeshBuilder(...) -- exp_file_token,
        angle_convention, angle_type ('radians'|'degrees'), symmetry, prefix.
    segmentation : legacy flat segmentation dict (misorientation_tol in radians,
        connectivity, grain_threshold, etc.).
    sculpt_config : CUBIT/SCULPT config (psculpt, epu, nprocs, ...). If omitted,
        only segmentation runs (no mesh).
    sculpt_options : SCULPT CLI flags.

    Segmentation needs no external binary; meshing needs CUBIT/SCULPT. Runs in
    the background. Outputs: merged_segmented_fixed_grid.npy, mesh.e (if meshed),
    orientations.csv.
    """
    # Lazy: heavy graintrace submodule + mcp helper, imported on run.
    # pylint: disable=import-outside-toplevel
    from graintrace.construct_nf_mesh import NearFieldMeshBuilder
    from graintrace.mcp import tool_paths

    if sculpt_config is None:
        sculpt_config = tool_paths.sculpt_config()  # from tools.json if configured
    if save_dir is None:
        save_dir = str(workdir() / "NF")
    init = {
        "exp_file_token": "layer",
        "angle_convention": "bunge",
        "angle_type": "radians",
        "symmetry": "432",
        "prefix": "reconstructed",
        **(init_params or {}),
    }
    do_mesh = sculpt_config is not None
    resolved = {
        "input_folder": input_folder,
        "save_dir": save_dir,
        "dz": dz,
        "nx": nx,
        "ny": ny,
        "init_params": init,
        "segmentation": segmentation,
        "will_mesh": do_mesh,
    }
    needs = ["cubit"] if do_mesh else []

    def _run():
        builder = NearFieldMeshBuilder(
            input_folder=input_folder,
            save_dir=save_dir,
            **init,
        )
        merged = builder.reconstruct(dz=dz, nx=nx, ny=ny, segmentation=segmentation)
        out = {"save_dir": save_dir, "merged_grid": str(merged)}
        if do_mesh:
            mesh = builder.mesh(
                sculpt_config=sculpt_config,
                sculpt_options=sculpt_options,
                merged_grid=merged,
            )
            out["mesh"] = str(mesh)
            out["orientations"] = str(builder.mapped_orientations_path) + ".csv"
        return out

    return gate(
        tool="nf_reconstruct",
        confirm=confirm,
        resolved_params=resolved,
        needs=needs,
        will_write=[save_dir],
        run=_run,
        background=True,
        notes="Segmentation is Python; meshing needs CUBIT/SCULPT.",
    )


@mcp.tool()
def voxel_mesh(
    file_path: str,
    euler_cols: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    init_params: Optional[Dict[str, Any]] = None,
    reconstruct_params: Optional[Dict[str, Any]] = None,
    sculpt_config: Optional[Dict[str, Any]] = None,
    sculpt_options: Optional[List[str]] = None,
    confirm: bool = False,
) -> dict:
    """Segment a gridded orientation field (EBSD, or FF reconstruction_reformatted
    .csv) into grains and build a conformal hex mesh (wraps `VoxelMeshBuilder`).

    Parameters
    ----------
    file_path : CSV with x,y,z + Euler columns (and optionally a grain-id col).
    euler_cols : the 3 Euler column names (default ['Eul0','Eul1','Eul2']).
    save_dir : output folder (defaults under the MCP workdir).
    init_params : overrides for VoxelMeshBuilder(...) -- angle_convention,
        angle_type ('radians'|'degrees'; use 'degrees' for FF output),
        symmetry, cell_id_col, prefix.
    reconstruct_params : passed to reconstruct(...) -- apply_smoothing and the
        segmentation dict (method 'graph'|'flood', params, graph_params).
    sculpt_config / sculpt_options : CUBIT/SCULPT config. If omitted, only
        segmentation runs (no mesh).

    Meshing needs CUBIT/SCULPT. Runs in the background.
    """
    # Lazy: heavy graintrace submodule + mcp helper, imported on run.
    # pylint: disable=import-outside-toplevel
    from graintrace.construct_voxel_mesh import VoxelMeshBuilder
    from graintrace.mcp import tool_paths

    if sculpt_config is None:
        sculpt_config = tool_paths.sculpt_config()  # from tools.json if configured
    if save_dir is None:
        save_dir = str(workdir() / "voxel")
    if euler_cols is None:
        euler_cols = ["Eul0", "Eul1", "Eul2"]
    init = {
        "angle_convention": "bunge",
        "angle_type": "radians",
        "symmetry": "432",
        **(init_params or {}),
    }
    recon = reconstruct_params or {}
    do_mesh = sculpt_config is not None
    resolved = {
        "file_path": file_path,
        "euler_cols": euler_cols,
        "save_dir": save_dir,
        "init_params": init,
        "reconstruct_params": recon,
        "will_mesh": do_mesh,
    }
    needs = ["cubit"] if do_mesh else []

    def _run():
        builder = VoxelMeshBuilder(
            file_path=file_path,
            save_dir=save_dir,
            euler_cols=euler_cols,
            **init,
        )
        merged = builder.reconstruct(**recon)
        out = {"save_dir": save_dir, "merged_grid": str(merged)}
        if do_mesh:
            mesh = builder.mesh(
                sculpt_config=sculpt_config,
                sculpt_options=sculpt_options,
                merged_grid=merged,
            )
            out["mesh"] = str(mesh)
        return out

    return gate(
        tool="voxel_mesh",
        confirm=confirm,
        resolved_params=resolved,
        needs=needs,
        will_write=[save_dir],
        run=_run,
        background=True,
        notes="Segmentation is Python; meshing needs CUBIT/SCULPT.",
    )

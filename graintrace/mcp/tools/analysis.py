# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tools: stitching comparison, CPFE post-processing, and rare-event ID."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

from graintrace.mcp import deps
from graintrace.mcp.app import mcp, workdir
from graintrace.mcp.confirm import gate


# ---- stitching comparison ----------------------------------------------------


@mcp.tool()
def compare_stitching(
    true_csv: str,
    stitch_csv: str,
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Compare a stitched grain set against a known/true grain set (recall,
    precision, orientation error); wraps `ScanStitchingComparison`. Both CSVs
    need X,Y,Z,GrainRadius,Eul0,Eul1,Eul2. Pure Python.
    """
    # Lazy: heavy graintrace submodule, imported only when the tool runs.
    # pylint: disable=import-outside-toplevel
    from graintrace.scan_stitching_comparison import ScanStitchingComparison

    if output_dir is None:
        output_dir = str(workdir() / "stitching_comparison")
    p = {
        "position_tolerance": 1.0,
        "orientation_tolerance": 1.0,
        "radius_tolerance": 1.0,
        "orientation_units": "degrees",
        "orientation_convention": "bunge",
        "symmetry": "432",
        "weights": {"pos": 1.0, "ori": 0.0, "rad": 0.0},
        "min_neighbors": 5,
        **(params or {}),
    }
    resolved = {
        "true_csv": true_csv,
        "stitch_csv": stitch_csv,
        "output_dir": output_dir,
        **p,
    }

    def _run():
        cmp = ScanStitchingComparison(
            output_dir=output_dir, true_csv=true_csv, stitch_csv=stitch_csv, **p
        )
        return {"output_dir": output_dir, "comparison": cmp.run_comparison()}

    return gate(
        tool="compare_stitching",
        confirm=confirm,
        resolved_params=resolved,
        needs=[],
        will_write=[output_dir],
        run=_run,
        background=False,
    )


# ---- REI comparison ----------------------------------------------------------


@mcp.tool()
def compare_rei(
    rei_csv_1: str,
    rei_csv_2: str,
    spacing_1: Optional[float] = None,
    spacing_2: Optional[float] = None,
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Compare two rare-event (REI) point clouds: overlap metrics (IoU/Dice/
    containment), a 1-to-1 cluster correspondence, and a classified point cloud
    (only-1 / only-2 / both) exported to VTK. Wraps `REIComparison`. Pure Python.

    Each CSV is a voxelized REI region on a regular grid (columns x,y,z plus an
    optional integer rare_cluster_id). Grids may have different spacings but are
    assumed to share an origin. spacing_1/spacing_2 default to None (auto-detect
    from the CSV; pass the true grid spacing when the cloud is sparse). Such CSVs
    come from `identify_rare_events` when run with a rare-points CSV output.
    """
    # Lazy: heavy graintrace submodule, imported only when the tool runs.
    # pylint: disable=import-outside-toplevel
    from graintrace.rei_comparison import REIComparison

    if output_dir is None:
        output_dir = str(workdir() / "rei_comparison")
    p = {
        "coord_cols": ("x", "y", "z"),
        "cluster_col": "rare_cluster_id",
        "supersample": 1,
        **(params or {}),
    }
    resolved = {
        "rei_csv_1": rei_csv_1,
        "rei_csv_2": rei_csv_2,
        "spacing_1": spacing_1,
        "spacing_2": spacing_2,
        "output_dir": output_dir,
        **p,
    }

    def _run():
        cmp = REIComparison(
            rei_csv_1=rei_csv_1,
            rei_csv_2=rei_csv_2,
            output_dir=output_dir,
            spacing_1=spacing_1,
            spacing_2=spacing_2,
            **p,
        )
        return {"output_dir": output_dir, "comparison": cmp.run_comparison()}

    return gate(
        tool="compare_rei",
        confirm=confirm,
        resolved_params=resolved,
        needs=[],
        will_write=[output_dir],
        run=_run,
        background=False,
    )


# ---- CPFE post-processing ----------------------------------------------------


@mcp.tool()
def postprocess(
    block_csv: str,
    field_dir: str,
    plots: Optional[List[str]] = None,
    time: Optional[float] = None,
    output_folder: Optional[str] = None,
    field_naming: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
):
    """Load CPFE results and produce standard plots (wraps `SimulationResults` +
    plot_postprocessing). On confirm=true the generated PNGs are returned INLINE
    (shown in chat / visible to the model), not just as file paths.

    Parameters
    ----------
    block_csv : per-grain block CSV (out.csv).
    field_dir : directory of per-element grid_out CSVs.
    plots : any of ['stress_strain', 'ee_distribution', 'nye_distribution',
        'pole_figure', 'ipf_tracking']. Default ['stress_strain'].
    time : sync time for distribution/pole-figure plots.
    output_folder : where PNGs go (defaults under the MCP workdir).
    field_naming : overrides for FieldFileNaming (prefix, index_width, sep, suffix).
    params : extra plot options (e.g. direction, crystal_symmetry for pole figure;
        grain_ids, steps for ipf_tracking).

    'pole_figure' and 'ipf_tracking' need NEML2 v3 bindings; the others do not.
    'ipf_tracking' plots per-grain reorientation trajectories in the IPF triangle.
    """
    # Lazy: heavy graintrace submodules, imported only when the tool runs.
    # pylint: disable=import-outside-toplevel
    from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming
    from graintrace import plot_postprocessing as pp

    if output_folder is None:
        output_folder = str(workdir() / "postprocess")
    if plots is None:
        plots = ["stress_strain"]
    fn = {
        "prefix": "out_element_centroid",
        "index_width": 4,
        "sep": "_",
        "suffix": ".csv",
        **(field_naming or {}),
    }
    extra = params or {}
    resolved = {
        "block_csv": block_csv,
        "field_dir": field_dir,
        "plots": plots,
        "time": time,
        "output_folder": output_folder,
        "field_naming": fn,
        "params": extra,
    }
    needs = ["neml2"] if ({"pole_figure", "ipf_tracking"} & set(plots)) else []

    def _run():
        res = SimulationResults(
            block_csv=block_csv,
            field_dir=field_dir,
            field_naming=FieldFileNaming(**fn),
        )
        made = []
        if "stress_strain" in plots:
            pp.plot_macroscopic_stress_strain(
                res,
                stress_tensor_prefix="cauchy_stress",
                strain_tensor_prefix="strain",
                volume_prefix="volume",
                output_folder=output_folder,
            )
            made.append("stress_strain")
        if "ee_distribution" in plots:
            pp.plot_block_properties_distribution(
                res, time=time, tensor_prefix="ee", order=2, output_folder=output_folder
            )
            made.append("ee_distribution")
        if "nye_distribution" in plots:
            pp.plot_block_properties_distribution(
                res,
                time=time,
                tensor_prefix="nye_tensor",
                order=2,
                output_folder=output_folder,
            )
            made.append("nye_distribution")
        if "pole_figure" in plots:
            pp.plot_pole_figure(
                res,
                tensor_prefix="ori_rodrigues",
                time=time,
                direction=extra.get("direction", [0, 0, 1]),
                crystal_symmetry=extra.get("crystal_symmetry", "432"),
                device=extra.get("device", deps.default_device()),
                output_folder=output_folder,
                construct_odf=extra.get("construct_odf", False),
            )
            made.append("pole_figure")
        if "ipf_tracking" in plots:
            # pylint: disable=import-outside-toplevel
            from graintrace.ipf_orientation_tracking import OrientationTracker

            tracks = OrientationTracker().simulation_grain_tracks(
                res,
                grain_ids=extra.get("grain_ids"),
                steps=extra.get("steps"),
            )
            pp.plot_ipf_orientation_tracking(
                tracks,
                angle_convention="mrp",
                direction=extra.get("direction", [0, 0, 1]),
                crystal_symmetry=extra.get("crystal_symmetry", "432"),
                output_folder=output_folder,
                savefig_name="ipf_orientation_tracking.png",
            )
            made.append("ipf_tracking")
        return {"output_folder": output_folder, "plots_made": made}

    # Preview (confirm=false) goes through the standard gate.
    if not confirm:
        return gate(
            tool="postprocess",
            confirm=False,
            resolved_params=resolved,
            needs=needs,
            will_write=[output_folder],
            run=_run,
            background=False,
        )
    # confirm=true: check deps, run, and return the PNGs INLINE.
    msg = deps.require(*needs) if needs else None
    if msg:
        return {"status": "blocked", "tool": "postprocess", "message": msg}
    before = (
        set(glob.glob(os.path.join(output_folder, "*.png")))
        if os.path.isdir(output_folder)
        else set()
    )
    result = _run()
    after = set(glob.glob(os.path.join(output_folder, "*.png")))
    pngs = sorted(after - before) or sorted(after)
    info = {"status": "done", **result, "png_files": pngs}
    try:
        # Optional client feature; imported lazily so absence degrades gracefully.
        from mcp.server.fastmcp import Image  # pylint: disable=import-outside-toplevel

        return [Image(path=p) for p in pngs] + [json.dumps(info)]
    # Best-effort: fall back to plain metadata if inline images are unsupported.
    except Exception:  # pylint: disable=broad-exception-caught
        return info


# ---- rare-event identification ----------------------------------------------


@mcp.tool()
def identify_rare_events(
    input_csv: str,
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Find spatially coherent rare regions (e.g. high Nye-tensor / GND density)
    in a CPFE grid-output field via graph spatial clustering + hierarchical merge
    + rare-cluster selection, exporting a VTK (wraps `IdentifyRareClusters`).

    Runs the canonical Nye-tensor-norm pipeline. Pure Python (networkit); no
    external binaries. Runs as a background job.

    Parameters (all in `params`, optional)
    --------------------------------------
    id_col : id column (default 'id'); coord_cols (default ['x','y','z']).
    nye_cols : the 9 Nye-tensor component columns (default nye_tensor_11..33).
    k : number of rare clusters to keep (default 5).
    gamma : Leiden resolution (default 10.0; higher -> more clusters).
    manhattan_radius : grid neighborhood radius (default 4).
    threshold : hierarchical merge distance threshold (default 5e-4).
    n_jobs : parallel workers (default 12).
    """
    # Lazy: heavy graintrace submodules, imported only when the tool runs.
    # pylint: disable=import-outside-toplevel
    from graintrace.rare_cluster_indicator import IdentifyRareClusters
    from graintrace.similarity_metric_library import SimilarityMetricLibrary
    from graintrace.user_data_class import SimilarityMetric, WeightConfig, RareCriteria
    from graintrace import rare_criteria_selection_library as rcs

    if output_dir is None:
        output_dir = str(workdir() / "rei")
    p = {
        "id_col": "id",
        "coord_cols": ["x", "y", "z"],
        "nye_cols": [f"nye_tensor_{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)],
        "k": 5,
        "gamma": 10.0,
        "manhattan_radius": 4,
        "threshold": 5e-4,
        "n_jobs": 12,
        "seed": 42,
        **(params or {}),
    }
    resolved = {"input_csv": input_csv, "output_dir": output_dir, **p}

    def _run():
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, "rei")

        metric_lib = SimilarityMetricLibrary()
        spec = metric_lib.nye_tensor_norm(cols=p["nye_cols"])
        spec_reduced = SimilarityMetric(
            name=spec.name + "_mean",
            feature_cols=[f"{c}_mean" for c in spec.feature_cols],
            func=spec.func,
        )
        scalar_col = spec_reduced.name + "_mean"
        rare_criteria = RareCriteria(
            selector=lambda df: rcs.select_highest_scalar(
                df, k=p["k"], required_cols=scalar_col, min_size=1
            )
        )
        weight_cfg = WeightConfig(
            mode="rbf",
            power=2.0,
            sigma=None,
            sigma_auto={
                "sample_size": 500_000,
                "random_state": p["seed"],
                "quantile": 0.5,
            },
        )
        irc = IdentifyRareClusters(
            input_csv_path=input_csv,
            id_col=p["id_col"],
            coord_cols=tuple(p["coord_cols"]),
        )
        gsc, indicator = irc.make_stage_objects(graph_cluster_out=base + "_reduced.csv")
        bundle = irc.run_clustering(
            gsc=gsc,
            indicator=indicator,
            reduced_csv_path=base + "_reduced.csv",
            gsc_run_kwargs={
                "spec": spec,
                "graph_mode": "grid",
                "manhattan_radius": p["manhattan_radius"],
                "grid_tol": 1e-6,
                "n_jobs": p["n_jobs"],
                "weight_chunk_size": 500_000,
                "segmenter": "leiden",
                "seed": p["seed"],
                "weight_cfg": weight_cfg,
                "reduce_edges_topweights_k": 20,
                "networkit_kwargs": {"gamma": p["gamma"]},
                "checkpoint_base_path": base + "_gsc_ckpt",
                "resume_from_checkpoint": False,
            },
            indicator_run_kwargs={
                "method_type": "scipy_hierarchical",
                "spec": spec_reduced,
                "threshold": p["threshold"],
                "method": "average",
                "criterion": "distance",
                "dendrogram_path": base + "_dendrogram.png",
            },
        )
        out = irc.run_get_rare_cluster(
            bundle=bundle,
            criteria=rare_criteria,
            output_vtk_path=base + "_rare_clusters.vtk",
            export_control="auto",
            background_block_id=1,
            first_rare_block_id=2,
            also_write_final_label=True,
            rare_reduced_stats_csv_path=base + "_rare_cluster_stats.csv",
            use_sample_std=False,
        )
        return {
            "output_dir": output_dir,
            "rare_vtk": base + "_rare_clusters.vtk",
            "stats_csv": base + "_rare_cluster_stats.csv",
            "n_rare": len(out) if hasattr(out, "__len__") else None,
        }

    return gate(
        tool="identify_rare_events",
        confirm=confirm,
        resolved_params=resolved,
        needs=[],
        will_write=[output_dir],
        run=_run,
        background=True,
        notes="Graph clustering can be slow on large grids; runs in background.",
    )


# ---- reorientation as an REI criterion ---------------------------------------


@mcp.tool()
def reorientation_rei(
    block_csv: str,
    field_dir: str,
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Flag regions that have REORIENTED the most under load as rare events.

    Writes a per-element misorientation-from-initial field (``reorientation_deg``)
    between a reference and a loaded step, then runs the standard REI pipeline with
    the ``abs_scalar_diff`` metric (wraps ``fragmentation.write_reorientation_field``
    + ``IdentifyRareClusters``). Standalone by default; pass ``extra_cols`` to carry
    nye/stress through for a combined criterion. Pure Python (neml2 + networkit).

    Parameters (all in ``params``, optional)
    ----------------------------------------
    step : field-step index to measure (default: last field step).
    ref_step : reference step (default: first field step).
    symmetry : crystal symmetry (default '432').
    k : number of rare clusters to keep (default 5).
    gamma : Leiden resolution (default 1.0). graph_mode 'knn' (mesh_out) by default.
    extra_cols : additional feature columns to combine (default none = standalone).
    """
    # pylint: disable=import-outside-toplevel
    from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming
    from graintrace.fragmentation import FragmentationAnalyzer
    from graintrace.rare_cluster_indicator import IdentifyRareClusters
    from graintrace.similarity_metric_library import SimilarityMetricLibrary
    from graintrace.user_data_class import WeightConfig, RareCriteria
    from graintrace import rare_criteria_selection_library as rcs

    if output_dir is None:
        output_dir = str(workdir() / "reorientation_rei")
    p = {
        "step": None,
        "ref_step": None,
        "symmetry": "432",
        "k": 5,
        "gamma": 1.0,
        "graph_mode": "knn",
        "extra_cols": [],
        "n_jobs": 12,
        "seed": 42,
        **(params or {}),
    }
    resolved = {
        "block_csv": block_csv,
        "field_dir": field_dir,
        "output_dir": output_dir,
        **p,
    }

    def _run():
        os.makedirs(output_dir, exist_ok=True)
        naming = FieldFileNaming(
            prefix="out_element_centroid", index_width=4, sep="_", suffix=".csv"
        )
        res = SimulationResults(block_csv, field_dir, field_naming=naming)
        steps = sorted(res.field_files.keys())
        step = p["step"] if p["step"] is not None else steps[-1]
        reo_csv = os.path.join(output_dir, "reorientation_field.csv")
        FragmentationAnalyzer(symmetry=p["symmetry"]).write_reorientation_field(
            res,
            step,
            reo_csv,
            ref_step=p["ref_step"],
            extra_cols=p["extra_cols"],
        )
        lib = SimilarityMetricLibrary()
        spec = lib.abs_scalar_diff(["reorientation_deg"])
        spec_reduced = lib.abs_scalar_diff(["reorientation_deg_mean"])
        scalar_col = "reorientation_deg_mean_mean"
        base = os.path.join(output_dir, "reo")
        irc = IdentifyRareClusters(
            input_csv_path=reo_csv, id_col="id", coord_cols=("x", "y", "z")
        )
        gsc, indicator = irc.make_stage_objects(graph_cluster_out=base + "_reduced.csv")
        weight_cfg = WeightConfig(
            mode="rbf",
            power=2.0,
            sigma=None,
            sigma_auto={
                "sample_size": 200_000,
                "random_state": p["seed"],
                "quantile": 0.5,
            },
        )
        bundle = irc.run_clustering(
            gsc=gsc,
            indicator=indicator,
            reduced_csv_path=base + "_reduced.csv",
            gsc_run_kwargs={
                "spec": spec,
                "graph_mode": p["graph_mode"],
                "k": 16,
                "n_jobs": p["n_jobs"],
                "segmenter": "leiden",
                "seed": p["seed"],
                "weight_cfg": weight_cfg,
                "reduce_edges_topweights_k": 16,
                "networkit_kwargs": {"gamma": p["gamma"]},
            },
            indicator_run_kwargs={
                "method_type": "scipy_hierarchical",
                "spec": spec_reduced,
                "threshold": 0.5,
                "method": "average",
                "criterion": "distance",
                "dendrogram_path": base + "_dendrogram.png",
            },
        )
        rare_criteria = RareCriteria(
            selector=lambda df: rcs.select_highest_scalar(
                df, k=p["k"], required_cols=scalar_col, min_size=1
            )
        )
        irc.run_get_rare_cluster(
            bundle=bundle,
            criteria=rare_criteria,
            output_vtk_path=base + "_rare.vtk",
            export_control="auto",
            background_block_id=1,
            first_rare_block_id=2,
            also_write_final_label=True,
        )
        return {
            "output_dir": output_dir,
            "reorientation_field": reo_csv,
            "rare_vtk": base + "_rare.vtk",
            "combined": bool(p["extra_cols"]),
        }

    return gate(
        tool="reorientation_rei",
        confirm=confirm,
        resolved_params=resolved,
        needs=[],
        will_write=[output_dir],
        run=_run,
        background=True,
        notes="Writes the reorientation field then runs REI; runs in background.",
    )


# ---- simulation intragranular fragmentation ----------------------------------


@mcp.tool()
def sim_fragmentation(
    block_csv: str,
    field_dir: str,
    mesh_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Analyse simulation intragranular fragmentation at a loaded step.

    Re-segments each ORIGINAL grain's elements by orientation into sub-grains
    (keeping the parent id), reports per-grain fragment count + sizes, and, if a
    SCULPT-hex ``mesh_file`` (.e) is given, annotates copies with a per-element
    ``fragment_label`` scalar (``sim_output_fragments.e``) and a per-element RGB in a
    dark categorical palette (``sim_output_fragments_rgb.e``; ParaView: map ``rgb_*`` to
    color) so a grain's sub-grains contrast. For meaningful colors the ``mesh_file`` must
    be the ANALYZED mesh (the loaded run's own ``sim_output.e``/``mesh.e``). Wraps
    ``FragmentationAnalyzer.grain_fragments`` + ``IPFProcessor.add_element_field_to_exodus``
    / ``add_element_rgb_to_exodus``. neml2 + networkit.

    Parameters (all in ``params``, optional)
    ----------------------------------------
    step : field-step index (default: last field step).
    tol_deg : sub-grain misorientation tolerance (default 5.0).
    gamma : Leiden resolution for sub-grain segmentation (default 0.25). Lower
        under-splits (0.1 leaves grains whole); raise it to resolve finer sub-grains.
    min_subgrain_miso_deg : minimum misorientation (deg) between sub-grains to keep
        them distinct (default 1.0). After Leiden, fragments closer than this are
        merged, so a smooth intragranular gradient is not chopped into fake fragments;
        0 disables the merge.
    """
    # pylint: disable=import-outside-toplevel
    from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming
    from graintrace.fragmentation import FragmentationAnalyzer

    if output_dir is None:
        output_dir = str(workdir() / "sim_fragmentation")
    p = {
        "step": None,
        "tol_deg": 5.0,
        "gamma": 0.25,
        "min_subgrain_miso_deg": 1.0,
        **(params or {}),
    }
    resolved = {
        "block_csv": block_csv,
        "field_dir": field_dir,
        "mesh_file": mesh_file,
        "output_dir": output_dir,
        **p,
    }

    def _run():
        os.makedirs(output_dir, exist_ok=True)
        naming = FieldFileNaming(
            prefix="out_element_centroid", index_width=4, sep="_", suffix=".csv"
        )
        res = SimulationResults(block_csv, field_dir, field_naming=naming)
        step = (
            p["step"] if p["step"] is not None else sorted(res.field_files.keys())[-1]
        )
        frags_df, per_elem = FragmentationAnalyzer().grain_fragments(
            res,
            step,
            tol_deg=p["tol_deg"],
            min_subgrain_miso_deg=p["min_subgrain_miso_deg"],
            seg_kwargs={"gamma": p["gamma"]},
        )
        counts_csv = os.path.join(output_dir, "fragment_counts.csv")
        frags_df.to_csv(counts_csv, index=False)
        result = {
            "output_dir": output_dir,
            "fragment_counts": counts_csv,
            "n_grains": int(len(frags_df)),
            "n_fragmented": int((frags_df["n_fragments"] > 1).sum()),
        }
        if mesh_file is not None:
            import numpy as np  # pylint: disable=import-outside-toplevel
            from matplotlib.colors import (
                to_rgb,
            )  # pylint: disable=import-outside-toplevel

            from graintrace.ipf_postprocess import IPFProcessor

            df = res.load_field_data(step)
            ids = df["id"].to_numpy()
            codes = np.array(
                [
                    (
                        float(per_elem[int(i)].split(".")[0])
                        + float(per_elem[int(i)].split(".")[1]) / 100.0
                        if int(i) in per_elem
                        else 0.0
                    )
                    for i in ids
                ]
            )
            centroids = df[["x", "y", "z"]].to_numpy(dtype=float)
            ipf = IPFProcessor(
                crystal_symmetry="432", sample_symmetry="1", save_dir=output_dir
            )
            out_e = ipf.add_element_field_to_exodus(
                mesh_file,
                centroids,
                codes,
                field_name="fragment_label",
                output_file="sim_output_fragments.e",
            )
            result["annotated_exodus"] = str(out_e)
            # RGB copy: fragments in a dark categorical palette (a grain's sub-grains
            # contrast), matching the branching-IPF figure. ParaView: map rgb_* to color.
            blk = np.rint(df["block_id"].to_numpy()).astype(int)
            palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"]
            frag_color = {}
            for g in np.unique(blk):
                labs = sorted(
                    lab for lab in per_elem.values() if lab.startswith(f"{int(g)}.")
                )
                frag_color[int(g)] = {
                    lab: palette[i % len(palette)] for i, lab in enumerate(labs)
                }
            rgb = np.array(
                [
                    to_rgb(frag_color[int(b)].get(per_elem.get(int(i)), "0.35"))
                    for i, b in zip(ids, blk)
                ]
            )
            out_rgb = ipf.add_element_rgb_to_exodus(
                mesh_file, centroids, rgb, output_file="sim_output_fragments_rgb.e"
            )
            result["annotated_exodus_rgb"] = str(out_rgb)
        return result

    return gate(
        tool="sim_fragmentation",
        confirm=confirm,
        resolved_params=resolved,
        needs=[],
        will_write=[output_dir],
        run=_run,
        background=True,
        notes="Per-grain sub-grain segmentation; runs in background.",
    )

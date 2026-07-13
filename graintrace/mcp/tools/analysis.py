# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tools: stitching comparison, CPFE post-processing, and rare-event ID."""

from __future__ import annotations

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
    precision, orientation error) -- wraps `ScanStitchingComparison`. Both CSVs
    need X,Y,Z,GrainRadius,Eul0,Eul1,Eul2. Pure Python.
    """
    from graintrace.scan_stitching_comparison import ScanStitchingComparison

    if output_dir is None:
        output_dir = str(workdir() / "stitching_comparison")
    p = {
        "position_tolerance": 1.0, "orientation_tolerance": 1.0,
        "radius_tolerance": 1.0, "orientation_units": "degrees",
        "orientation_convention": "bunge", "symmetry": "432",
        "weights": {"pos": 1.0, "ori": 0.0, "rad": 0.0}, "min_neighbors": 5,
        **(params or {}),
    }
    resolved = {"true_csv": true_csv, "stitch_csv": stitch_csv, "output_dir": output_dir, **p}

    def _run():
        cmp = ScanStitchingComparison(
            output_dir=output_dir, true_csv=true_csv, stitch_csv=stitch_csv, **p
        )
        return {"output_dir": output_dir, "comparison": cmp.run_comparison()}

    return gate(
        tool="compare_stitching", confirm=confirm, resolved_params=resolved,
        needs=[], will_write=[output_dir], run=_run, background=False,
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
) -> dict:
    """Load CPFE results and produce standard plots (wraps `SimulationResults` +
    plot_postprocessing).

    Parameters
    ----------
    block_csv : per-grain block CSV (out.csv).
    field_dir : directory of per-element grid_out CSVs.
    plots : any of ['stress_strain', 'ee_distribution', 'nye_distribution',
        'pole_figure']. Default ['stress_strain'].
    time : sync time for distribution/pole-figure plots.
    output_folder : where PNGs go (defaults under the MCP workdir).
    field_naming : overrides for FieldFileNaming (prefix, index_width, sep, suffix).
    params : extra plot options (e.g. direction, crystal_symmetry for pole figure).

    'pole_figure' needs NEML2 v3 bindings; the others do not.
    """
    from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming
    from graintrace import plot_postprocessing as pp

    if output_folder is None:
        output_folder = str(workdir() / "postprocess")
    if plots is None:
        plots = ["stress_strain"]
    fn = {"prefix": "out_element_centroid", "index_width": 4, "sep": "_", "suffix": ".csv", **(field_naming or {})}
    extra = params or {}
    resolved = {
        "block_csv": block_csv, "field_dir": field_dir, "plots": plots,
        "time": time, "output_folder": output_folder, "field_naming": fn, "params": extra,
    }
    needs = ["neml2"] if "pole_figure" in plots else []

    def _run():
        res = SimulationResults(
            block_csv=block_csv, field_dir=field_dir,
            field_naming=FieldFileNaming(**fn),
        )
        made = []
        if "stress_strain" in plots:
            pp.plot_macroscopic_stress_strain(
                res, stress_tensor_prefix="cauchy_stress",
                strain_tensor_prefix="strain", volume_prefix="volume",
                output_folder=output_folder,
            )
            made.append("stress_strain")
        if "ee_distribution" in plots:
            pp.plot_block_properties_distribution(
                res, time=time, tensor_prefix="ee", order=2, output_folder=output_folder)
            made.append("ee_distribution")
        if "nye_distribution" in plots:
            pp.plot_block_properties_distribution(
                res, time=time, tensor_prefix="nye_tensor", order=2, output_folder=output_folder)
            made.append("nye_distribution")
        if "pole_figure" in plots:
            pp.plot_pole_figure(
                res, tensor_prefix="ori_rodrigues", time=time,
                direction=extra.get("direction", [0, 0, 1]),
                crystal_symmetry=extra.get("crystal_symmetry", "432"),
                device=extra.get("device", deps.default_device()), output_folder=output_folder,
                construct_odf=extra.get("construct_odf", False),
            )
            made.append("pole_figure")
        return {"output_folder": output_folder, "plots_made": made}

    return gate(
        tool="postprocess", confirm=confirm, resolved_params=resolved,
        needs=needs, will_write=[output_folder], run=_run, background=False,
    )


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
    from graintrace.rare_cluster_indicator import IdentifyRareClusters
    from graintrace.similarity_metric_library import SimilarityMetricLibrary
    from graintrace.user_data_class import SimilarityMetric, WeightConfig, RareCriteria
    from graintrace import rare_criteria_selection_library as rcs

    if output_dir is None:
        output_dir = str(workdir() / "rei")
    p = {
        "id_col": "id", "coord_cols": ["x", "y", "z"],
        "nye_cols": [f"nye_tensor_{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)],
        "k": 5, "gamma": 10.0, "manhattan_radius": 4, "threshold": 5e-4,
        "n_jobs": 12, "seed": 42,
        **(params or {}),
    }
    resolved = {"input_csv": input_csv, "output_dir": output_dir, **p}

    def _run():
        import os
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
                df, k=p["k"], required_cols=scalar_col, min_size=1)
        )
        weight_cfg = WeightConfig(
            mode="rbf", power=2.0, sigma=None,
            sigma_auto={"sample_size": 500_000, "random_state": p["seed"], "quantile": 0.5},
        )
        irc = IdentifyRareClusters(
            input_csv_path=input_csv, id_col=p["id_col"],
            coord_cols=tuple(p["coord_cols"]),
        )
        gsc, indicator = irc.make_stage_objects(graph_cluster_out=base + "_reduced.csv")
        bundle = irc.run_clustering(
            gsc=gsc, indicator=indicator, reduced_csv_path=base + "_reduced.csv",
            gsc_run_kwargs=dict(
                spec=spec, graph_mode="grid", manhattan_radius=p["manhattan_radius"],
                grid_tol=1e-6, n_jobs=p["n_jobs"], weight_chunk_size=500_000,
                segmenter="leiden", seed=p["seed"], weight_cfg=weight_cfg,
                reduce_edges_topweights_k=20,
                networkit_kwargs={"gamma": p["gamma"]},
                checkpoint_base_path=base + "_gsc_ckpt",
                resume_from_checkpoint=False,
            ),
            indicator_run_kwargs=dict(
                method_type="scipy_hierarchical", spec=spec_reduced,
                threshold=p["threshold"], method="average", criterion="distance",
                dendrogram_path=base + "_dendrogram.png",
            ),
        )
        out = irc.run_get_rare_cluster(
            bundle=bundle, criteria=rare_criteria,
            output_vtk_path=base + "_rare_clusters.vtk",
            export_control="auto", background_block_id=1, first_rare_block_id=2,
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
        tool="identify_rare_events", confirm=confirm, resolved_params=resolved,
        needs=[], will_write=[output_dir], run=_run, background=True,
        notes="Graph clustering can be slow on large grids; runs in background.",
    )

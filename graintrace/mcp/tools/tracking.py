# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tool: track/match grains across two load steps (VoronoiMeshBuilder.build_graph
+ GraphGrainMatcher)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graintrace.mcp.app import mcp, workdir
from graintrace.mcp.confirm import gate


@mcp.tool()
def track_grains(
    csv_a: str,
    csv_b: str,
    bounding_box: List[float],
    output_dir: Optional[str] = None,
    init_params: Optional[Dict[str, Any]] = None,
    build_params: Optional[Dict[str, Any]] = None,
    match_params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Match grains between two FF reconstructions (e.g. two load steps) by
    building a grain graph from each and matching via message passing (wraps
    `VoronoiMeshBuilder.build_graph` + `GraphGrainMatcher`).

    Parameters
    ----------
    csv_a, csv_b : the two FF grain CSVs (different loads/times).
    bounding_box : [xlo,xhi,ylo,yhi,zlo,zhi] micrometers (used for both).
    output_dir : output folder (defaults under the MCP workdir).
    init_params : overrides for VoronoiMeshBuilder(...) (e.g. unit, dim,
        angle_identifier). Applied to both reconstructions.
    build_params : overrides for build_graph(...): option, CVT_iter,
        morphoalgo, device.
    match_params : overrides for match_grains(...): message_passing_iter,
        neighbor_selection_param, and angle_convention / angle_type / symmetry
        (describe the graph Euler features; default Bunge/degrees/432, matching
        build_graph output).

    Needs NEPER (graph build) and torch-geometric. Runs as a background job;
    writes the matched correspondence under output_dir.
    """
    # Lazy: heavy graintrace submodules, imported only when the tool runs.
    # pylint: disable=import-outside-toplevel
    from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
    from graintrace.grain_graph_matching import GraphGrainMatcher

    if output_dir is None:
        output_dir = str(workdir() / "grain_tracking")
    init = {**(init_params or {})}
    build = {
        "option": "centroid",
        "CVT_iter": 100,
        "device": "cpu",
        **(build_params or {}),
    }
    match = {**(match_params or {})}
    resolved = {
        "csv_a": csv_a,
        "csv_b": csv_b,
        "bounding_box": bounding_box,
        "output_dir": output_dir,
        "init_params": init,
        "build_params": build,
        "match_params": match,
    }

    def _run():
        ga = VoronoiMeshBuilder(
            input_csv=csv_a,
            output_dir=f"{output_dir}/A",
            bounding_box=bounding_box,
            **init,
        ).build_graph(**build)
        gb = VoronoiMeshBuilder(
            input_csv=csv_b,
            output_dir=f"{output_dir}/B",
            bounding_box=bounding_box,
            **init,
        ).build_graph(**build)
        matcher = GraphGrainMatcher(graph_a=ga, graph_b=gb, output_dir=output_dir)
        result = matcher.match_grains(**match)
        return {"output_dir": output_dir, "matched": bool(result is not None)}

    return gate(
        tool="track_grains",
        confirm=confirm,
        resolved_params=resolved,
        needs=["neper", "torch_geometric"],
        will_write=[output_dir],
        run=_run,
        background=True,
    )


@mcp.tool()
def detect_fragmentation(
    nodes_a: str,
    nodes_b: str,
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    confirm: bool = False,
) -> dict:
    """Detect grain fragmentation (one grain splitting into several) between two
    load steps, beyond the strictly 1-to-1 `track_grains`. Cross-step graph +
    connected components (centroid distance + misorientation); classifies each
    lineage as match / split / merge / birth / death (wraps
    `FragmentationAnalyzer.detect_splits`). Pure NumPy+SciPy (+neml2 for the
    orientation math).

    Inputs are per-grain node tables (CSV) for the two steps:
    `grain_id, X, Y, Z, GrainRadius, Eul0/1/2`. FF supplies these directly; NF/EBSD
    reach the same detector by segmenting each step (graph/Leiden) then reducing to
    a grain table with `FragmentationAnalyzer.seg_to_grain_table`.

    Parameters (all in `params`, optional)
    --------------------------------------
    d_tol : max centroid distance for a cross-step lineage edge (default 20.0).
    theta_tol_deg : max misorientation for a lineage edge (default 15.0; generous).
    symmetry : crystal symmetry (default '432').
    angle_convention / angle_type : 'bunge'/'degrees' (or 'mrp').
    top_k : nearest B centroids probed per A grain (default 8).
    """
    # pylint: disable=import-outside-toplevel
    from graintrace.fragmentation import FragmentationAnalyzer

    if output_dir is None:
        output_dir = str(workdir() / "grain_fragmentation")
    p = {
        "d_tol": 20.0,
        "theta_tol_deg": 15.0,
        "symmetry": "432",
        "angle_convention": "bunge",
        "angle_type": "degrees",
        "top_k": 8,
        **(params or {}),
    }
    resolved = {
        "nodes_a": nodes_a,
        "nodes_b": nodes_b,
        "output_dir": output_dir,
        **p,
    }

    def _run():
        import os  # pylint: disable=import-outside-toplevel

        os.makedirs(output_dir, exist_ok=True)
        out = FragmentationAnalyzer(symmetry=p["symmetry"]).detect_splits(
            nodes_a,
            nodes_b,
            d_tol=p["d_tol"],
            theta_tol_deg=p["theta_tol_deg"],
            angle_convention=p["angle_convention"],
            angle_type=p["angle_type"],
            top_k=p["top_k"],
            out_csv=os.path.join(output_dir, "split_correspondences.csv"),
        )
        return {
            "output_dir": output_dir,
            "split_correspondences": os.path.join(
                output_dir, "split_correspondences.csv"
            ),
            "n_splits": out["n_splits"],
            "n_merges": out["n_merges"],
            "n_matches": out["n_matches"],
            "n_births": out["n_births"],
            "n_deaths": out["n_deaths"],
        }

    return gate(
        tool="detect_fragmentation",
        confirm=confirm,
        resolved_params=resolved,
        needs=[],
        will_write=[output_dir],
        run=_run,
        background=True,
    )

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

"""Thread-scaling benchmark for graintrace graph spatial clustering.

Two independent parallel kernels drive the graph-segmentation path (reused by REI,
NF, and voxel meshing):
  * the numba top-k prune (`prune_topk_per_node_parallel`, swept via `n_jobs`)
  * the NetworKit Leiden partition (`segment_graph_networkit`, swept via `n_threads`)

Pure Python (numba/networkit/numpy) - no NEPER/SCULPT/MOOSE/GPU, runs anywhere.

Example:
    python benchmark/bench_graph_segmentation.py --sizes 30 --jobs 1,2,4,8 \
        --threads 1,2,4,8 --repeat 3
"""

from __future__ import annotations

import argparse

import numpy as np

# pylint: disable=import-error,no-name-in-module  # local sibling module
from _harness import (  # type: ignore
    capture_sysinfo,
    parse_int_list,
    print_header,
    results_dir,
    timer,
    write_results,
)

FEATURE_COLS = ["sxx", "syy", "szz", "sxy", "syz", "sxz"]


def make_grid(edge: int, seed: int = 42):
    """Dense cube grid of `edge`^3 points + a random von-Mises stress field (N,6)."""
    rng = np.random.default_rng(seed)
    axis = np.arange(edge, dtype=np.float64)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    n = coords.shape[0]
    means = np.array([100.0, 50.0, 30.0, 0.0, 0.0, 0.0])
    stds = np.array([20.0, 10.0, 5.0, 5.0, 5.0, 5.0])
    x = rng.normal(means, stds, size=(n, 6))
    return coords, x


def build_graph(coords, x, manhattan_radius):
    """Build grid edges + pruned weighted graph; return (n, edges, weights)."""
    # pylint: disable=import-outside-toplevel  # heavy optional deps kept local
    from graintrace.graph_spatial_cluster import GraphSpatialCluster
    from graintrace.similarity_metric_library import SimilarityMetricLibrary
    from graintrace.user_data_class import WeightConfig

    gsc = GraphSpatialCluster.__new__(GraphSpatialCluster)
    spec = SimilarityMetricLibrary().von_mises_stress(cols=FEATURE_COLS)
    # pylint: disable=protected-access  # low-level stage API, as used in tests/
    edges = gsc._build_grid_edges(coords, manhattan_radius=manhattan_radius, tol=1e-6)
    dist = gsc.compute_edge_distances(edges=edges, X=x, spec=spec)
    weights = gsc.distances_to_weights(dist, WeightConfig(mode="inverse"))
    return gsc, coords.shape[0], edges, weights


def bench(args) -> None:
    """Run the prune (n_jobs) and Leiden (n_threads) thread sweeps."""
    sysinfo = capture_sysinfo()
    print_header("graintrace benchmark: graph segmentation (thread scaling)", sysinfo)

    jobs = parse_int_list(args.jobs)
    threads = parse_int_list(args.threads)
    rows = []

    for edge in parse_int_list(args.sizes):
        coords, x = make_grid(edge)
        gsc, n, edges, weights = build_graph(coords, x, args.manhattan_radius)
        print(
            f"\n[size] edge={edge}  nodes={n}  edges={edges.shape[0]}  "
            f"radius={args.manhattan_radius}  k={args.k}"
        )

        # Warm up the numba JIT once (excluded from timings).
        gsc.prune_topk_per_node_parallel(
            n_nodes=n, edges=edges, weights=weights, k=args.k, n_jobs=1
        )

        # --- Stage 1: numba top-k prune, sweep n_jobs ---
        base = None
        for nj in jobs:
            best = min(
                _time_prune(gsc, n, edges, weights, args.k, nj)
                for _ in range(args.repeat)
            )
            base = best if base is None else base
            rows.append(
                {
                    "stage": "prune_numba",
                    "threads": nj,
                    "nodes": n,
                    "edges": int(edges.shape[0]),
                    "time_s": round(best, 5),
                    "speedup_vs_1": round(base / best, 3),
                }
            )
            print(f"  prune   n_jobs={nj:>3}  {best:8.4f}s  speedup={base/best:5.2f}x")

        # --- Stage 2: NetworKit Leiden, sweep n_threads ---
        try:
            import networkit  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        except ImportError:
            print("  (networkit missing -> skipping Leiden n_threads sweep)")
            continue

        base = None
        for nt in threads:
            best = min(
                _time_leiden(gsc, n, edges, weights, nt) for _ in range(args.repeat)
            )
            base = best if base is None else base
            rows.append(
                {
                    "stage": "leiden_networkit",
                    "threads": nt,
                    "nodes": n,
                    "edges": int(edges.shape[0]),
                    "time_s": round(best, 5),
                    "speedup_vs_1": round(base / best, 3),
                }
            )
            print(f"  leiden  threads={nt:>3}  {best:8.4f}s  speedup={base/best:5.2f}x")

    out_dir = results_dir("graph_segmentation", args.out)
    write_results("graph_segmentation", rows, out_dir, sysinfo)
    if args.plot:
        _plot(rows, out_dir)


def _time_prune(gsc, n, edges, weights, k, n_jobs) -> float:
    with timer() as t:
        gsc.prune_topk_per_node_parallel(
            n_nodes=n, edges=edges, weights=weights, k=k, n_jobs=n_jobs
        )
    return t[0]


def _time_leiden(gsc, n, edges, weights, n_threads) -> float:
    # Running the numba prune kernel pins the shared OpenMP pool; explicitly lift
    # the cap here so Leiden's internal min(n_threads, max) is not stuck at 1.
    # pylint: disable=import-outside-toplevel
    import networkit as nk

    nk.setNumberOfThreads(int(n_threads))
    with timer() as t:
        gsc.segment_graph_networkit(
            n_nodes=n,
            edges=edges,
            weights=weights,
            method="leiden",
            seed=42,
            n_threads=n_threads,
        )
    return t[0]


def _plot(rows, out_dir) -> None:
    """Speedup-vs-threads plot per stage (optional; needs matplotlib)."""
    try:
        # pylint: disable=import-outside-toplevel  # matplotlib is optional
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib missing -> skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for stage in sorted({r["stage"] for r in rows}):
        sr = [r for r in rows if r["stage"] == stage]
        ax.plot(
            [r["threads"] for r in sr],
            [r["speedup_vs_1"] for r in sr],
            "o-",
            label=stage,
        )
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title("graph segmentation thread scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / "graph_segmentation_speedup.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Wrote {path}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes", default="30", help="comma-sep cube edge lengths (N=edge^3)"
    )
    p.add_argument(
        "--jobs", default="1,2,4,8", help="n_jobs values for the numba prune"
    )
    p.add_argument("--threads", default="1,2,4,8", help="n_threads values for Leiden")
    p.add_argument("--manhattan-radius", type=int, default=2, dest="manhattan_radius")
    p.add_argument(
        "--k", type=int, default=8, help="top-k edges per node for the prune"
    )
    p.add_argument(
        "--repeat", type=int, default=3, help="repeats per point (min taken)"
    )
    p.add_argument("--plot", action="store_true", help="write a speedup plot")
    p.add_argument("--out", default=None, help="explicit output dir (default: stamped)")
    bench(p.parse_args())


if __name__ == "__main__":
    main()

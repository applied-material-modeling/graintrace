#!/usr/bin/env python3
# Copyright 2026, UChicago Argonne, LLC -- MIT (see package LICENSE)
"""Per-stage wall-time and peak-RAM profiler for the GraphSpatialCluster pipeline
(build vs Leiden solve). Runs on synthetic grids or a real reduced CSV; see the
argparse help for usage."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graintrace.graph_spatial_cluster import GraphSpatialCluster
from graintrace.similarity_metric_library import SimilarityMetricLibrary
from graintrace.user_data_class import WeightConfig

_PAGESIZE = resource.getpagesize()


def _rss_bytes() -> int:
    """Current resident set size in bytes (Linux /proc; fast, no deps)."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * _PAGESIZE
    except OSError:
        # Fallback: ru_maxrss is a high-water mark in KiB on Linux.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class _PeakSampler:
    """Background thread that records the max RSS seen since the last reset."""

    def __init__(self, interval_s: float = 0.01) -> None:
        self._interval = interval_s
        self._peak = _rss_bytes()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            r = _rss_bytes()
            with self._lock:
                if r > self._peak:
                    self._peak = r
            time.sleep(self._interval)

    def start(self) -> "_PeakSampler":
        self._thread.start()
        return self

    def reset(self) -> int:
        with self._lock:
            self._peak = _rss_bytes()
            return self._peak

    def peak(self) -> int:
        with self._lock:
            return self._peak

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class StageRecorder:
    def __init__(self, sampler: _PeakSampler) -> None:
        self.sampler = sampler
        self.rows: List[Dict] = []

    @contextmanager
    def stage(self, name: str):
        rss0 = self.sampler.reset()
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        peak = self.sampler.peak()
        self.rows.append(
            {
                "stage": name,
                "seconds": dt,
                "rss_start_b": rss0,
                "rss_peak_b": peak,
                "rss_delta_b": max(0, peak - rss0),
            }
        )


def _gb(b: float) -> float:
    return b / (1024.0**3)


def make_synthetic_csv(path: str, n_side: int, metric: str, seed: int = 0) -> int:
    """Write a cubic n_side**3 integer-grid voxel CSV with random features; returns node count."""
    rng = np.random.default_rng(seed)
    g = np.arange(n_side, dtype=np.float64)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n = coords.shape[0]

    cols = {
        "id": np.arange(n, dtype=np.int64),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    }

    if metric == "misorientation":
        # small modified-Rodrigues vectors for angle_convention="mrp"
        mrp = rng.normal(0.0, 0.15, size=(n, 3))
        cols["ori_rodrigues_x"] = mrp[:, 0]
        cols["ori_rodrigues_y"] = mrp[:, 1]
        cols["ori_rodrigues_z"] = mrp[:, 2]
    elif metric == "nye":
        for c in (
            "nye_tensor_11", "nye_tensor_12", "nye_tensor_13",
            "nye_tensor_21", "nye_tensor_22", "nye_tensor_23",
            "nye_tensor_31", "nye_tensor_32", "nye_tensor_33",
        ):
            cols[c] = rng.normal(0.0, 1.0, size=n)
    else:
        raise ValueError(f"synthetic generator has no columns for metric={metric!r}")

    pd.DataFrame(cols).to_csv(path, index=False)
    return n


def build_spec(metric: str, device: str = "cpu"):
    lib = SimilarityMetricLibrary()
    if metric == "misorientation":
        return lib.misorientation(device=device)  # ori_rodrigues_{x,y,z}, mrp
    if metric == "nye":
        return lib.nye_tensor_norm()
    if metric == "vonmises":
        return lib.von_mises_stress()
    raise ValueError(f"unknown metric: {metric}")


def profile_one(
    csv_path: str,
    metric: str,
    graph_mode: str,
    radius: int,
    knn_k: int,
    topk: Optional[int],
    n_jobs: int,
    weight_chunk: int,
    nodes_chunk: int,
    sampler: _PeakSampler,
    device: str = "cpu",
    skip_loop: bool = False,
    n_threads: int = 1,
) -> Dict:
    import networkit as nk

    nk_set_threads = min(int(n_threads), nk.getMaxNumberOfThreads())
    nk.setNumberOfThreads(nk_set_threads)

    rec = StageRecorder(sampler)
    gsc = GraphSpatialCluster(csv_path=csv_path)
    gsc.load_data()
    spec = build_spec(metric, device=device)
    gsc.check_feature_matrix(spec)

    df = gsc.data
    coords = df[list(gsc.coord_cols)].to_numpy(dtype=np.float64)
    Xfeat = df[spec.feature_cols].to_numpy(dtype=np.float64)
    n_nodes = coords.shape[0]

    mode = graph_mode
    if mode == "auto":
        mode = "grid" if gsc._detect_grid(coords, tol=1e-6) else "knn"

    # 1. build edges
    with rec.stage("build_edges"):
        if mode == "grid":
            edges = gsc._build_grid_edges(coords, manhattan_radius=radius, tol=1e-6)
        else:
            edges = gsc._build_mutual_knn_edges(coords, k=knn_k)
    n_edges = int(edges.shape[0])

    # 2. edge distances (the metric kernel)
    with rec.stage("edge_distances"):
        distances = gsc.compute_edge_distances(
            edges=edges, X=Xfeat, spec=spec, n_jobs=n_jobs, chunk_size=weight_chunk
        )

    # 3. distances -> weights
    weight_cfg = WeightConfig(
        mode="rbf", power=2.0, sigma=None,
        sigma_auto={"sample_size": 200_000, "random_state": 42, "quantile": 0.5},
    )
    with rec.stage("weights"):
        sigma = gsc.estimate_sigma_from_distances(
            distances=distances, quantile=weight_cfg.sigma_auto["quantile"]
        )
        weight_cfg = WeightConfig(**{**weight_cfg.__dict__, "sigma": sigma})
        weights = gsc.distances_to_weights(distances, weight_cfg)

    # 4. prune top-k per node (optional)
    if topk is not None:
        with rec.stage("prune_topk"):
            edges, weights = gsc.prune_topk_per_node_parallel(
                n_nodes=n_nodes, edges=edges, weights=weights,
                k=topk, n_jobs=n_jobs, nodes_chunk=nodes_chunk,
            )
        n_edges = int(edges.shape[0])

    # 5/6/7. segmentation: benchmark both graph-build paths (addEdge loop vs GraphFromCoo)
    if not skip_loop:
        with rec.stage("nk_build_loop"):
            G_loop = nk.Graph(n_nodes, weighted=True, directed=False)
            for (u, v), w in zip(edges, weights):
                G_loop.addEdge(int(u), int(v), float(w))
        del G_loop

    with rec.stage("nk_build_coo"):
        row = np.ascontiguousarray(edges[:, 0], dtype=np.uint64)
        col = np.ascontiguousarray(edges[:, 1], dtype=np.uint64)
        w = np.ascontiguousarray(weights, dtype=np.float64)
        G = nk.GraphFromCoo((w, (row, col)), n=n_nodes, weighted=True, directed=False)

    nk.setSeed(42, True)
    if not hasattr(nk.community, "ParallelLeiden"):
        raise RuntimeError("NetworKit ParallelLeiden unavailable in this install.")
    algo = nk.community.ParallelLeiden(G)
    with rec.stage("leiden_solve"):
        algo.run()
    part = algo.getPartition()

    with rec.stage("label_extract"):
        labels = np.asarray(part.getVector(), dtype=np.int64)

    edge_bytes = edges.nbytes + weights.nbytes
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "edges_per_node": n_edges / max(1, n_nodes),
        "n_clusters": int(labels.max()) + 1 if labels.size else 0,
        "graph_mode": mode,
        "metric": metric,
        "edge_array_gb": _gb(edge_bytes),
        "nk_threads": nk_set_threads,
        "stages": rec.rows,
    }


def print_report(result: Dict) -> None:
    rows = result["stages"]
    total = sum(r["seconds"] for r in rows)
    print(f"  NetworKit Leiden threads set to: {result.get('nk_threads', '?')}")
    print(
        f"\n  nodes={result['n_nodes']:,}  edges={result['n_edges']:,}  "
        f"({result['edges_per_node']:.1f}/node)  mode={result['graph_mode']}  "
        f"metric={result['metric']}  clusters={result['n_clusters']:,}"
    )
    print(f"  edge+weight arrays: {result['edge_array_gb']:.3f} GB")
    print(f"  {'stage':<16}{'time (s)':>12}{'%time':>8}{'peak RSS':>12}{'Δ RSS':>12}")
    print(f"  {'-'*16}{'-'*12}{'-'*8}{'-'*12}{'-'*12}")
    for r in rows:
        pct = 100.0 * r["seconds"] / total if total else 0.0
        bar = "#" * int(round(pct / 4.0))
        print(
            f"  {r['stage']:<16}{r['seconds']:>12.3f}{pct:>7.1f}%"
            f"{_gb(r['rss_peak_b']):>10.2f}G{_gb(r['rss_delta_b']):>10.2f}G  {bar}"
        )
    print(f"  {'-'*16}{'-'*12}{'-'*8}{'-'*12}{'-'*12}")
    print(f"  {'TOTAL':<16}{total:>12.3f}{100.0:>7.1f}%")

    by = {r["stage"]: r["seconds"] for r in rows}
    coo = by.get("nk_build_coo", 0)
    loop = by.get("nk_build_loop", 0)
    build = by.get("build_edges", 0) + coo
    solve = by.get("leiden_solve", 0)
    kernel = by.get("edge_distances", 0)
    print(
        f"\n  >> SPLIT (fixed build): build(edges+coo)={build:.2f}s "
        f"| leiden_solve={solve:.2f}s | metric_kernel={kernel:.2f}s"
    )
    if loop and coo:
        print(
            f"     nk_build:  addEdge loop {loop:.2f}s  ->  GraphFromCoo {coo:.2f}s "
            f"= {loop / max(coo, 1e-9):.0f}x faster build"
        )
    if loop and result["n_edges"]:
        print(
            f"     loop cost {1e9*loop/result['n_edges']:.0f} ns/edge"
            f"  vs  coo {1e9*coo/result['n_edges']:.0f} ns/edge"
        )


def project(results: List[Dict], targets: List[int]) -> None:
    """Crude linear extrapolation from the largest measured run."""
    if not results:
        return
    base = max(results, key=lambda r: r["n_nodes"])
    by = {r["stage"]: r["seconds"] for r in base["stages"]}
    epn = base["edges_per_node"]
    n0 = base["n_nodes"]
    e0 = base["n_edges"]
    coo_per_edge = by.get("nk_build_coo", 0) / max(1, e0)
    leiden_per_edge = by.get("leiden_solve", 0) / max(1, e0)
    kernel_per_edge = by.get("edge_distances", 0) / max(1, e0)
    bytes_per_edge = (base["edge_array_gb"] * 1024**3) / max(1, e0)

    print("\n=== Linear projection from largest run "
          f"(n={n0:,}, {epn:.1f} edges/node) ===")
    print(f"  {'nodes':>14}{'edges':>16}{'coo_build':>11}{'leiden':>10}"
          f"{'kernel':>10}{'edge GB':>10}")
    print(f"  {'-'*14}{'-'*16}{'-'*11}{'-'*10}{'-'*10}{'-'*10}")
    for nt in targets:
        et = nt * epn
        print(
            f"  {nt:>14,}{int(et):>16,}{coo_per_edge*et:>10.1f}s{leiden_per_edge*et:>9.1f}s"
            f"{kernel_per_edge*et:>9.1f}s{bytes_per_edge*et/1024**3:>9.2f}G"
        )
    print("  NOTE: NetworKit/Leiden internal memory is several x the raw edge")
    print("        array; treat 'edge GB' as a lower bound on host RAM.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sizes", type=str,
                     help="comma list of cube side lengths, e.g. 25,35,45 (nodes=side^3)")
    src.add_argument("--csv", type=str, help="path to a real reduced CSV")
    ap.add_argument("--metric", default="nye",
                    choices=["nye", "misorientation", "vonmises"],
                    help="distance metric (misorientation/vonmises need matching cols / neml2)")
    ap.add_argument("--graph-mode", default="auto", choices=["auto", "grid", "knn"])
    ap.add_argument("--radius", type=int, default=2, help="manhattan radius (grid mode)")
    ap.add_argument("--knn-k", type=int, default=16, help="k (knn mode)")
    ap.add_argument("--topk", type=int, default=None,
                    help="prune to top-k edges per node before segmentation")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default="cpu",
                    help="device for vectorized metrics, e.g. cpu | cuda | cuda:0 "
                         "(misorientation kernel runs on GPU when set)")
    ap.add_argument("--skip-loop", action="store_true",
                    help="skip the slow addEdge-loop baseline (for very large runs)")
    ap.add_argument("--threads", type=int, default=1,
                    help="NetworKit/Leiden thread count (default 1; ~physical cores is optimal)")
    ap.add_argument("--weight-chunk", type=int, default=1_000_000)
    ap.add_argument("--nodes-chunk", type=int, default=250_000)
    ap.add_argument("--project", type=str, default="1000000,10000000,100000000",
                    help="comma list of node counts to extrapolate to")
    ap.add_argument("--tmpdir", type=str, default=".",
                    help="where to write synthetic CSVs")
    ap.add_argument("--json", type=str, default=None, help="dump raw results to JSON")
    args = ap.parse_args()

    sampler = _PeakSampler().start()
    results: List[Dict] = []

    try:
        if args.csv:
            res = profile_one(
                csv_path=args.csv, metric=args.metric, graph_mode=args.graph_mode,
                radius=args.radius, knn_k=args.knn_k, topk=args.topk,
                n_jobs=args.n_jobs, weight_chunk=args.weight_chunk,
                nodes_chunk=args.nodes_chunk, sampler=sampler,
                device=args.device, skip_loop=args.skip_loop,
                n_threads=args.threads,
            )
            print_report(res)
            results.append(res)
        else:
            for side in [int(s) for s in args.sizes.split(",")]:
                csv_path = os.path.join(args.tmpdir, f"_profile_grid_{side}.csv")
                print(f"\n### side={side}  -> generating {side**3:,}-node grid "
                      f"({csv_path}) ...")
                make_synthetic_csv(csv_path, side, args.metric)
                res = profile_one(
                    csv_path=csv_path, metric=args.metric, graph_mode=args.graph_mode,
                    radius=args.radius, knn_k=args.knn_k, topk=args.topk,
                    n_jobs=args.n_jobs, weight_chunk=args.weight_chunk,
                    nodes_chunk=args.nodes_chunk, sampler=sampler,
                    device=args.device, skip_loop=args.skip_loop,
                    n_threads=args.threads,
                )
                res["side"] = side
                print_report(res)
                results.append(res)
                try:
                    os.remove(csv_path)
                except OSError:
                    pass

        targets = [int(t) for t in args.project.split(",")] if args.project else []
        if targets:
            project(results, targets)

        if args.json:
            with open(args.json, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"\nWrote {args.json}")
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()

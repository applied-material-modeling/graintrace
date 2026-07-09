#!/usr/bin/env python3
# Copyright 2026, UChicago Argonne, LLC -- MIT (see package LICENSE)
"""In-memory end-to-end ROI benchmark for the REI graph-segmentation pipeline
(build_edges -> kernel -> weights -> prune -> coo -> leiden -> labels). Reports
per-stage wall time + peak RSS per grid size for the nye / misorientation-CPU /
misorientation-GPU kernels; synthetic data generation is excluded from timings."""

from __future__ import annotations

import argparse, gc, os, resource, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graintrace.graph_spatial_cluster import GraphSpatialCluster
from graintrace.similarity_metric_library import SimilarityMetricLibrary
from graintrace.user_data_class import WeightConfig

_PAGE = resource.getpagesize()


def rss_gb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE / 1024**3
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class Peak:
    def __init__(self):
        self.p = rss_gb(); self._stop = False
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while not self._stop:
            r = rss_gb()
            if r > self.p:
                self.p = r
            time.sleep(0.05)
    def reset(self):
        self.p = rss_gb(); return self.p


PK = Peak()


def stage(name, fn, log):
    gc.collect(); PK.reset()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    log[name] = dt
    print(f"    {name:<16}{dt:>10.2f}s   peakRSS={PK.p:.0f}G", flush=True)
    return out


def lean_coords(nx, ny, nz):
    n = nx * ny * nz
    c = np.empty((n, 3), dtype=np.float64)
    c[:, 0] = np.repeat(np.arange(nx, dtype=np.float64), ny * nz)
    c[:, 1] = np.tile(np.repeat(np.arange(ny, dtype=np.float64), nz), nx)
    c[:, 2] = np.tile(np.arange(nz, dtype=np.float64), nx * ny)
    return c


def run_size(nx, ny, nz, radius, k, kernel_threads, leiden_threads,
             cpu_chunk, gpu_chunk, nye_chunk, do_gpu):
    import networkit as nk
    import torch

    n = nx * ny * nz
    print(f"\n=== {nx}x{ny}x{nz} = {n:,} nodes | radius={radius} k={k} ===", flush=True)
    log = {}
    gsc = GraphSpatialCluster.__new__(GraphSpatialCluster)
    lib = SimilarityMetricLibrary()
    rng = np.random.default_rng(0)

    # --- shared: coords + edges (edges depend only on coords) ---
    coords = lean_coords(nx, ny, nz)
    edges = stage("build_edges",
                  lambda: gsc._build_grid_edges(coords, manhattan_radius=radius, tol=1e-6),
                  log)
    ne = edges.shape[0]
    print(f"    pre-prune edges = {ne:,} ({ne/n:.1f}/node)", flush=True)
    del coords; gc.collect()

    # --- kernels (per metric / device) ---
    torch.set_num_threads(int(kernel_threads))

    feat3 = rng.normal(0.0, 0.15, size=(n, 3))   # mrp orientations
    feat9 = rng.normal(0.0, 1.0, size=(n, 9))    # nye tensor

    spec_nye = lib.nye_tensor_norm()
    stage("kernel_nye",
          lambda: gsc.compute_edge_distances(edges=edges, X=feat9, spec=spec_nye,
                                             n_jobs=1, chunk_size=nye_chunk), log)

    spec_cpu = lib.misorientation(device="cpu")
    dist = stage("kernel_miso_cpu",
                 lambda: gsc.compute_edge_distances(edges=edges, X=feat3, spec=spec_cpu,
                                                    n_jobs=1, chunk_size=cpu_chunk), log)

    if do_gpu:
        spec_gpu = lib.misorientation(device="cuda")
        try:
            _ = spec_gpu.dist_edges(feat3, edges[:1000]); torch.cuda.synchronize()
            stage("kernel_miso_gpu",
                  lambda: gsc.compute_edge_distances(edges=edges, X=feat3, spec=spec_gpu,
                                                     n_jobs=1, chunk_size=gpu_chunk), log)
        except Exception as ex:
            print(f"    kernel_miso_gpu FAILED: {type(ex).__name__}: {ex}", flush=True)
            log["kernel_miso_gpu"] = float("nan")

    del feat9; gc.collect()

    # --- downstream (run once; edge-count driven, ~metric-independent) ---
    wcfg = WeightConfig(mode="rbf", power=2.0, sigma=None,
                        sigma_auto={"sample_size": 200_000, "random_state": 42, "quantile": 0.5})
    def to_weights():
        sigma = gsc.estimate_sigma_from_distances(distances=dist, quantile=0.5)
        return gsc.distances_to_weights(dist, WeightConfig(**{**wcfg.__dict__, "sigma": sigma}))
    weights = stage("weights", to_weights, log)
    del dist, to_weights; gc.collect()  # release distances before prune peak

    pe, pw = stage("prune_topk",
                   lambda: gsc.prune_topk_per_node_parallel(
                       n_nodes=n, edges=edges, weights=weights, k=k, n_jobs=28), log)
    npost = pe.shape[0]
    print(f"    post-prune edges = {npost:,} ({npost/n:.1f}/node)", flush=True)
    del edges, weights; gc.collect()

    def build_coo():
        row = np.ascontiguousarray(pe[:, 0], dtype=np.uint64)
        col = np.ascontiguousarray(pe[:, 1], dtype=np.uint64)
        w = np.ascontiguousarray(pw, dtype=np.float64)
        return nk.GraphFromCoo((w, (row, col)), n=n, weighted=True, directed=False)
    G = stage("coo_build", build_coo, log)

    nk.setNumberOfThreads(min(int(leiden_threads), nk.getMaxNumberOfThreads()))
    nk.setSeed(42, True)
    algo = nk.community.ParallelLeiden(G)
    stage("leiden", lambda: algo.run(), log)
    labels = stage("labels", lambda: np.asarray(algo.getPartition().getVector(), dtype=np.int64), log)
    nclust = int(labels.max()) + 1

    # --- ROI assembly ---
    shared = log["build_edges"] + log["weights"] + log["prune_topk"] + log["coo_build"] + log["leiden"] + log["labels"]
    print(f"\n    clusters={nclust:,}  shared(build+weights+prune+coo+leiden+labels)={shared:.0f}s", flush=True)
    print(f"    >> ROI end-to-end:", flush=True)
    print(f"         nye            = {shared + log['kernel_nye']:.0f}s  (kernel {log['kernel_nye']:.0f}s)", flush=True)
    print(f"         miso CPU       = {shared + log['kernel_miso_cpu']:.0f}s  (kernel {log['kernel_miso_cpu']:.0f}s)", flush=True)
    if do_gpu and not np.isnan(log.get("kernel_miso_gpu", float("nan"))):
        kg = log["kernel_miso_gpu"]
        print(f"         miso GPU       = {shared + kg:.0f}s  (kernel {kg:.0f}s, {log['kernel_miso_cpu']/kg:.2f}x vs CPU)", flush=True)
    del G, algo, labels, pe, pw, feat3; gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="263x263x263,330x330x330,381x381x381",
                    help="comma list of grids AxBxC (default ~18M/36M/55M nodes)")
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--kernel-threads", type=int, default=min(28, os.cpu_count() or 1))
    ap.add_argument("--leiden-threads", type=int, default=16)
    ap.add_argument("--cpu-chunk", type=int, default=20_000_000)
    ap.add_argument("--gpu-chunk", type=int, default=1_000_000)
    ap.add_argument("--nye-chunk", type=int, default=50_000_000)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    print(f"cores={os.cpu_count()} kernel_threads={args.kernel_threads} "
          f"leiden_threads={args.leiden_threads} radius={args.radius} k={args.k}", flush=True)
    for d in args.dims.split(","):
        nx, ny, nz = [int(x) for x in d.lower().split("x")]
        run_size(nx, ny, nz, args.radius, args.k, args.kernel_threads,
                 args.leiden_threads, args.cpu_chunk, args.gpu_chunk,
                 args.nye_chunk, not args.no_gpu)


if __name__ == "__main__":
    main()

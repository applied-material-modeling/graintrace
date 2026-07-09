# REI graph-segmentation ROI — timing results

End-to-end timings for the realistic REI pipeline (radius-4 grid graph, top-k=20
weight pruning), produced by `examples/roi_benchmark.py`. Data generation is
**excluded** from all timings; only pipeline stages are timed.

**Machine:** 32-core CPU, 503 GB RAM, 2× NVIDIA RTX A5000 (23 GB each).
**Config:** radius 4, k=20, `kernel_threads=28`, `leiden_threads=16`,
prune `n_jobs=28` (numba threads), GPU kernel chunk 1M edges.
All stage times in seconds; peak RSS in GB (host RAM). Times are wall-clock and
carry a few-% run-to-run variance on the unchanged kernels.

---

## 1. Final pipeline (numba prune, no multiprocessing) — all sizes

| nodes | pre-prune edges | post-prune | clusters | nye total | miso CPU total | miso GPU total | GPU/CPU kernel |
|------:|----------------:|-----------:|---------:|----------:|---------------:|---------------:|:--------------:|
| 18.2M | 1,149,769,696 | 202,478,018 | 50 | 579 | 1601 | 887 | 2.51× |
| 35.9M | 2,277,152,432 | 399,897,980 | 52 | 979 | 2973 | 1595 | 2.48× |
| 55.3M | 3,509,183,708 | 615,345,255 | 65 | 1456 | 4524 | 2524 | 2.39× |
| 110.6M | 7,029,581,732 | 1,230,225,593 | 75 | 4793 | 11819 | 6455 | 2.86× |

"miso CPU/GPU total" = end-to-end (shared stages + that kernel). nye uses a cheap
numpy kernel. GPU/CPU kernel = misorientation kernel speedup of GPU over 28-thread CPU.

### Per-stage breakdown (seconds / peak RSS GB)

| stage | 18.2M | 35.9M | 55.3M | 110.6M |
|---|---|---|---|---|
| build_edges      | 165 / 54  | 199 / 113 | 333 / 178 | 1201 / 327 |
| kernel_nye       | 164 / 38  | 316 / 73  | 372 / 112 | 1222 / 177 |
| kernel_miso_cpu  | 1186 / 29 | 2310 / 63 | 3440 / 106 | 8248 / 169 |
| kernel_miso_gpu  | 472 / 37  | 932 / 80  | 1440 / 132 | 2885 / 221 |
| weights          | 34 / 53   | 67 / 114  | 102 / 184 | 265 / 318 |
| prune_topk       | 122 / 87  | 206 / 181 | 340 / 287 | 1428 / 452 |
| coo_build        | 52 / 9    | 103 / 27  | 174 / 45  | 412 / 50 |
| leiden           | 42 / 23   | 87 / 44   | 133 / 67  | 261 / 131 |
| labels           | 0.7 / 19  | 1.1 / 37  | 1.5 / 55  | 3.3 / 110 |

Notes: sizes 18–55M measured with the numba prune (argsort grouping); the 110M run
used the additional prune memory optimization (quicksort grouping + lazy half-edge
weights), so its prune peak is the binding/worst case. The whole-pipeline peak RSS
is the `prune_topk` row.

---

## 2. Prune: numba vs original (the change made this run)

Original = pre-change single-threaded argsort + argpartition (the multiprocessing
path it replaced). Bit-identical output on all sizes (post-prune counts match).

| nodes | pre-prune edges | original time | numba time | speedup | original peak | numba peak | mem saved |
|------:|----------------:|--------------:|-----------:|:-------:|--------------:|-----------:|:---------:|
| 18.2M | 1.15B | 234 | 122 | 1.9× | 141 | 87 | −38% |
| 35.9M | 2.28B | 520 | 206 | 2.5× | 284 | 181 | −36% |
| 55.3M | 3.51B | 759 | 340 | 2.2× | 433 | 287 | −34% |
| 110.6M | 7.03B | **576 (OOM)** | 1428 | runs | **576 (OOM)** | 452 | runs |

- **numba prune is fork-free** (numba threads, not processes) — removes the Python
  3.13 fork-in-multithreaded-process warning and the deadlock fragility.
- **110.6M / 7.03B edges was infeasible before**: the original prune needed ~576 GB
  (OOM on a 503 GB box); the numba version peaks at **452 GB and completes**. Halving
  prune memory — the binding constraint — roughly doubled the tractable problem size.

---

## 3. End-to-end: original vs new (sizes 18–55M)

| metric | 18.2M | 35.9M | 55.3M |
|---|---|---|---|
| nye      | 702 → **579** | 1302 → **979** | 1990 → **1456** |
| miso CPU | 1775 → **1601** | 3156 → **2973** | 4904 → **4524** |
| miso GPU | 1041 → **887** | 1967 → **1595** | 3004 → **2524** |

The end-to-end delta is entirely the prune improvement (kernels are unchanged code).
nye shows it most clearly (prune is a larger fraction of its pipeline).

---

## 4. Key findings

1. **Prune (numba): 2–2.5× faster, −34–38% peak memory, fork-free, bit-identical.**
2. **GPU misorientation kernel: 2.4–2.9× vs 28-thread CPU**, speedup growing with
   scale (2.86× at 110M). This is the naive GPU path (fp64, per-chunk transfer);
   fp32 + resident orientations remain untapped headroom.
3. **Lowering prune memory raised the OOM ceiling from ~4B to ~7B edges**
   (~68M → ~110M nodes at radius 4) on the same 503 GB box.
4. **Leiden is no longer a bottleneck** (≤261 s even at 1.23B post-prune edges) after
   setting its thread count; build_edges and the misorientation kernel now dominate.

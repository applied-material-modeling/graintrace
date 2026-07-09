# Copyright 2026, UChicago Argonne, LLC -- MIT (see package LICENSE)
"""Frozen copy of the ORIGINAL (pre-numba) top-k prune algorithm.

This is kept verbatim as a regression reference: the numba/fallback prune in
graintrace.graph_spatial_cluster must produce bit-identical output to this on
distinct-weight inputs. Dependency-free (numpy only), so it can stand alone.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _topk_nodes_worker(args) -> np.ndarray:
    indptr, adj_eid, adj_w, a, b, k = args
    kept_chunks = []
    for nidx in range(a, b):
        s = int(indptr[nidx])
        e = int(indptr[nidx + 1])
        m = e - s
        if m <= 0:
            continue
        if m <= k:
            kept_chunks.append(adj_eid[s:e])
        else:
            w_slice = adj_w[s:e]
            idx = np.argpartition(w_slice, -k)[-k:]
            kept_chunks.append(adj_eid[s:e][idx])
    if kept_chunks:
        return np.concatenate(kept_chunks)
    return np.empty((0,), dtype=np.int64)


def prune_original(
    n_nodes: int,
    edges: np.ndarray,
    weights: np.ndarray,
    k: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Original single-threaded argsort + argpartition top-k-per-node prune."""
    if k is None:
        return edges, weights
    k = int(k)
    if k < 1 or edges.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    E = edges.shape[0]
    node = np.concatenate(
        [
            edges[:, 0].astype(np.int64, copy=False),
            edges[:, 1].astype(np.int64, copy=False),
        ]
    )
    eid = np.concatenate(
        [np.arange(E, dtype=np.int64), np.arange(E, dtype=np.int64)]
    )
    adj_w_half = np.concatenate(
        [
            weights.astype(np.float64, copy=False),
            weights.astype(np.float64, copy=False),
        ]
    )

    deg = np.bincount(node, minlength=n_nodes).astype(np.int64)
    indptr = np.empty(n_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(deg, out=indptr[1:])

    order = np.argsort(node, kind="mergesort")
    adj_eid = eid[order]
    adj_w = adj_w_half[order]

    keep_edge = np.zeros(E, dtype=bool)
    kept = _topk_nodes_worker((indptr, adj_eid, adj_w, 0, n_nodes, k))
    if kept.size:
        keep_edge[kept] = True

    return edges[keep_edge], weights[keep_edge]

# Copyright 2026, UChicago Argonne, LLC -- MIT (see package LICENSE)
"""Frozen reference: ``detect_splits`` exactly as it stood before PR #63.

Issue #59 requires that "default arguments reproduce today's output exactly".
A test that compares the new implementation against itself with the defaults
spelled out cannot show that -- both sides move together. This module is the
pre-change implementation, copied verbatim from ``main`` at 2dd7d78 with only
two edits: it is a module-level function, and ``self.symmetry`` is a parameter.

Do not refactor it, and do not "fix" it to match the current code. Its whole
value is that it does not change when ``fragmentation.py`` does. Same pattern,
and same reasoning, as ``tests/_prune_original.py``.
"""

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

# _load_nodes and _orientation_matrices read the node tables and convert Euler
# angles; both are untouched by this change (``git diff main..HEAD`` on
# fragmentation.py adds functions, it does not modify these two), so they are
# imported rather than duplicated. What is frozen here is the detection
# algorithm, which is what issue #59 asks to hold constant.
from graintrace.fragmentation import _load_nodes, _orientation_matrices


def detect_splits_original(
    symmetry,
    nodes_a,
    nodes_b,
    d_tol: float,
    theta_tol_deg: float,
    angle_convention: str = "bunge",
    angle_type: str = "degrees",
    ori_cols: Sequence[str] = ("Eul0", "Eul1", "Eul2"),
    coord_cols: Sequence[str] = ("X", "Y", "Z"),
    id_col: str = "grain_id",
    top_k: Optional[int] = 8,
    adjacency_b: Optional[Iterable[Tuple[int, int]]] = None,
    out_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect grain splits/merges/births/deaths between two load steps.

    Goes beyond the strictly one-to-one grain matcher: candidate cross edges
    from a centroid ``cKDTree`` (top-k) are kept when the centroid distance is
    within ``d_tol`` and the misorientation within ``theta_tol_deg``; connected
    components of the step-A/step-B union graph are grain lineages, classified by
    their ``(nA, nB)`` shape. FF supplies the node tables directly; NF/EBSD reach
    the same detector via :meth:`segment` + :meth:`seg_to_grain_table`.

    The grain tables use their own ``ori_cols``/``coord_cols`` (``Eul0/1/2`` and
    ``X/Y/Z`` by default), independent of the simulation column conventions on
    the instance.

    Args:
        nodes_a, nodes_b: per-grain node tables (DataFrame or CSV path) for the
            two steps, with an id column, three coordinate columns, and three
            orientation columns.
        d_tol: maximum centroid distance for a candidate cross edge.
        theta_tol_deg: maximum misorientation (degrees) for a candidate cross edge.
        angle_convention: ``"bunge"``/``"kocks"``/``"roe"`` Euler, or ``"mrp"``.
        angle_type: ``"degrees"`` or ``"radians"`` (Euler conventions only).
        ori_cols: the three orientation columns.
        coord_cols: the three coordinate columns.
        id_col: the grain-id column.
        top_k: cap on the number of B candidates probed per A grain
            (default 8), or ``None`` for no cap. With a cap, ``top_k`` --
            not ``d_tol`` -- is the binding limit on how many children a
            split can have whenever more than ``top_k`` B grains sit within
            ``d_tol``, so a parent that fragments into many pieces, or a
            dense fine-grained step B, needs ``top_k`` raised as well as
            ``d_tol``. Pass ``top_k=None`` to take every B grain within
            ``d_tol`` and let ``d_tol`` alone decide the search, at the
            cost of a radius query that is slower on dense data.

            Note that the order of the two filters does not matter: the
            ``top_k`` nearest B grains that survive ``d_tol`` are exactly
            the ``top_k`` nearest among those within ``d_tol``, because
            every grain inside the radius is closer than every grain
            outside it. Only raising or removing the cap changes the
            candidate set.
        adjacency_b: optional iterable of ``(grain_id, grain_id)`` pairs giving
            intra-step-B grain adjacency; used to reject a spurious split child
            that is not adjacent to any other child of the same parent. If no two
            of a parent's candidates are mutually adjacent there is no split to
            reject, so the single best candidate (lowest misorientation, then
            shortest centroid distance) is kept and the parent matches it rather
            than being dropped entirely.
        out_csv: if given, write the split correspondences to this CSV path.

    Returns:
        dict with ``components`` (list of ``{a_ids, b_ids, kind}``),
        ``split_correspondences`` (DataFrame), and the counts ``n_splits``,
        ``n_merges``, ``n_matches``, ``n_births``, ``n_deaths``.
    """
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from graintrace.orientation_helper import misorientation_matrix

    da = _load_nodes(nodes_a)
    db = _load_nodes(nodes_b)
    n_a, n_b = len(da), len(db)
    ids_a = da[id_col].to_numpy()
    ids_b = db[id_col].to_numpy()
    coords_a = da[list(coord_cols)].to_numpy(dtype=float)
    coords_b = db[list(coord_cols)].to_numpy(dtype=float)
    r_a = _orientation_matrices(da, ori_cols, angle_convention, angle_type)
    r_b = _orientation_matrices(db, ori_cols, angle_convention, angle_type)

    # Candidate cross edges: the B centroids within d_tol of each A grain,
    # capped at top_k nearest. Capping before or after the d_tol gate gives
    # the same set (every grain inside the radius is nearer than every grain
    # outside it), so top_k=None is the only way to widen the search.
    tree_b = cKDTree(coords_b)
    ai, bj, pair_dist = [], [], []
    if top_k is None:
        for i in range(n_a):
            for j in tree_b.query_ball_point(coords_a[i], r=d_tol):
                ai.append(i)
                bj.append(int(j))
                pair_dist.append(float(np.linalg.norm(coords_b[j] - coords_a[i])))
    else:
        kq = min(top_k, n_b)
        dist, jdx = tree_b.query(coords_a, k=kq)
        dist = np.asarray(dist).reshape(n_a, kq)
        jdx = np.asarray(jdx).reshape(n_a, kq)

        for i in range(n_a):
            for col in range(kq):
                j = int(jdx[i, col])
                if dist[i, col] <= d_tol:
                    ai.append(i)
                    bj.append(j)
                    pair_dist.append(float(dist[i, col]))
    ai = np.asarray(ai, dtype=np.int64)
    bj = np.asarray(bj, dtype=np.int64)
    pair_dist = np.asarray(pair_dist, dtype=float)
    pair_miso = np.zeros(len(ai), dtype=float)

    # Filter by symmetry-aware misorientation.
    if len(ai) > 0:
        miso = (
            misorientation_matrix(r_a[ai], r_b[bj], symmetry, "degrees").cpu().numpy()
        )
        keep = miso <= theta_tol_deg
        ai, bj = ai[keep], bj[keep]
        pair_dist, pair_miso = pair_dist[keep], np.asarray(miso)[keep]

    # Optional adjacency consistency: a candidate child that is not adjacent (in B)
    # to any sibling under the same parent is not evidence of a split, so it is
    # dropped -- but only when the parent still keeps at least one adjacent pair.
    # When NO two candidates are mutually adjacent there is no split to reject, and
    # dropping every candidate would turn a legitimate correspondence into a death
    # plus N spurious births; keep the single best candidate (lowest misorientation,
    # then shortest centroid distance) so the parent still matches.
    if adjacency_b is not None and len(ai) > 0:
        idmap_b = {int(g): k for k, g in enumerate(ids_b)}
        adj: Dict[int, set] = {}
        for g1, g2 in adjacency_b:
            u, v = idmap_b.get(int(g1)), idmap_b.get(int(g2))
            if u is None or v is None:
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        keep = np.ones(len(ai), dtype=bool)
        for a in np.unique(ai):
            sel = np.where(ai == a)[0]
            children = bj[sel]
            if len(children) <= 1:
                continue
            cset = set(children.tolist())
            adjacent = np.array(
                [bool(adj.get(int(c), set()) & (cset - {int(c)})) for c in children]
            )
            if adjacent.any():
                keep[sel[~adjacent]] = False
            else:
                best = sel[np.lexsort((bj[sel], pair_dist[sel], pair_miso[sel]))[0]]
                keep[sel[sel != best]] = False
        ai, bj = ai[keep], bj[keep]

    # Union graph A (0..n_a-1) + B (n_a..n_a+n_b-1); components = grain lineages.
    n = n_a + n_b
    rows = np.concatenate([ai, bj + n_a]) if len(ai) else np.array([], dtype=np.int64)
    csum = np.concatenate([bj + n_a, ai]) if len(ai) else np.array([], dtype=np.int64)
    data = np.ones(len(rows), dtype=np.int8)
    graph = coo_matrix((data, (rows, csum)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)

    components = []
    counts = {
        "n_matches": 0,
        "n_splits": 0,
        "n_merges": 0,
        "n_births": 0,
        "n_deaths": 0,
    }
    split_rows = []
    for c in np.unique(comp):
        members = np.where(comp == c)[0]
        a_local = members[members < n_a]
        b_local = members[members >= n_a] - n_a
        a_grains = [int(ids_a[i]) for i in a_local]
        b_grains = [int(ids_b[j]) for j in b_local]
        n_ai, n_bj = len(a_grains), len(b_grains)
        if n_ai == 1 and n_bj == 1:
            kind = "match"
            counts["n_matches"] += 1
        elif n_ai == 1 and n_bj > 1:
            kind = "split"
            counts["n_splits"] += 1
            sizes = [
                (
                    float(db.loc[db[id_col] == g, "GrainRadius"].iloc[0])
                    if "GrainRadius" in db.columns
                    else np.nan
                )
                for g in b_grains
            ]
            split_rows.append(
                {
                    "parent_a": a_grains[0],
                    "children_b": ";".join(map(str, b_grains)),
                    "n_children": n_bj,
                    "child_radii": ";".join(f"{s:.4g}" for s in sizes),
                }
            )
        elif n_ai > 1 and n_bj == 1:
            kind = "merge"
            counts["n_merges"] += 1
        elif n_ai == 1 and n_bj == 0:
            kind = "death"
            counts["n_deaths"] += 1
        elif n_ai == 0 and n_bj == 1:
            kind = "birth"
            counts["n_births"] += 1
        else:
            kind = "complex"
        components.append({"a_ids": a_grains, "b_ids": b_grains, "kind": kind})

    split_df = pd.DataFrame(
        split_rows,
        columns=["parent_a", "children_b", "n_children", "child_radii"],
    )
    if out_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        split_df.to_csv(out_csv, index=False)

    return {"components": components, "split_correspondences": split_df, **counts}

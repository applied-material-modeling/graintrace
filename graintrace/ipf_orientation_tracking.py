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

"""Build per-entity orientation tracks across load steps for IPF trajectory plots.

:class:`OrientationTracker` returns a list of ``(k, 3)`` orientation arrays (MRP for
simulation, Euler for experiment), one per tracked entity, ready for
:func:`graintrace.plot_postprocessing.plot_ipf_orientation_tracking`. Simulation
entities keep a stable id across time (no matching); experiment grains are tracked
either by a stable ``grain_id`` column or, when ids are not stable across loads, by
graph matching (:class:`~graintrace.grain_graph_matching.GraphGrainMatcher`).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def _resolve_steps(n_steps: int, steps: Optional[Sequence[int]]) -> List[int]:
    """Normalize a requested step subset to a sorted list of valid indices."""
    if steps is None:
        return list(range(n_steps))
    out = [int(s) for s in steps]
    for s in out:
        if s < 0 or s >= n_steps:
            raise IndexError(f"step index {s} out of range [0, {n_steps - 1}]")
    return out


def _graph_euler(graph, eul_cols: Sequence[str]) -> np.ndarray:
    """Extract the ``(N, 3)`` Euler node features from a grain graph, in node order."""
    x = graph.x.detach().cpu().numpy()
    cols = []
    for name in eul_cols:
        start, end = graph.feature_slices[name]
        cols.append(x[:, start:end])
    return np.concatenate(cols, axis=1)


class OrientationTracker:
    """Build per-entity orientation tracks across load steps for IPF plots.

    Holds the column/tensor conventions shared by the track builders; each method
    returns a list of ``(k, 3)`` orientation arrays (one per tracked entity) to feed
    :func:`graintrace.plot_postprocessing.plot_ipf_orientation_tracking`.

    Args:
        ori_tensor: CPFE block tensor name for per-grain simulation orientations
            (neml2 MRP; default ``"ori_rodrigues"``).
        block_id_col: per-element field-CSV column giving each element's grain/block
            id (default ``"block_id"``).
        eul_cols: experiment Euler-angle column/feature names (default Bunge triple).
    """

    def __init__(
        self,
        ori_tensor: str = "ori_rodrigues",
        block_id_col: str = "block_id",
        eul_cols: Sequence[str] = ("Eul0", "Eul1", "Eul2"),
    ) -> None:
        self.ori_tensor = ori_tensor
        self.block_id_col = block_id_col
        self.eul_cols = tuple(eul_cols)

    @staticmethod
    def compose_tracks(
        a_to_b_list: Sequence[Sequence[int]],
        seed_ids: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        """Chain pairwise ``a_to_b`` maps into multi-step row tracks.

        ``a_to_b_list[k]`` maps a row in step ``k`` to its match in step ``k+1``
        (``-1`` unmatched). Each track follows one seed row and truncates at the first
        ``-1``. ``seed_ids`` defaults to every row of the first map.
        """
        maps = [np.asarray(m).astype(np.int64).reshape(-1) for m in a_to_b_list]
        if seed_ids is None:
            seed_ids = list(range(len(maps[0]) if maps else 1))

        tracks: List[List[int]] = []
        for s in seed_ids:
            rows = [int(s)]
            cur = int(s)
            for m in maps:
                if cur < 0 or cur >= len(m) or int(m[cur]) < 0:
                    break
                cur = int(m[cur])
                rows.append(cur)
            tracks.append(rows)
        return tracks

    def simulation_grain_tracks(
        self,
        results,
        grain_ids: Optional[Sequence[int]] = None,
        steps: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        """Per-grain orientation (MRP) tracks from a CPFE block output.

        Grain identity is stable across time, so no matching is needed.

        Args:
            results: a :class:`~graintrace.simulation_postprocessing.SimulationResults`.
            grain_ids: grains to track; defaults to all ``results.grain_ids``.
            steps: time-step indices to keep; defaults to all steps.

        Returns:
            list of ``(k, 3)`` MRP arrays, one per grain.
        """
        if grain_ids is None:
            grain_ids = list(results.grain_ids)
        sel = np.asarray(_resolve_steps(len(results.time), steps), dtype=int)
        return [
            np.asarray(
                results.get_tensor_block(
                    self.ori_tensor, order=1, sample="time", grain_id=gid
                )
            )[sel]
            for gid in grain_ids
        ]

    def simulation_element_tracks(
        self,
        results,
        block_id: Optional[int] = None,
        element_ids: Optional[Sequence[int]] = None,
        steps: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        """Per-element orientation (MRP) tracks within one grain (fragmentation).

        Element membership comes from the ``block_id`` column of the per-element field
        CSVs (block == grain subdomain), so no mesh file is needed; alternatively pass
        ``element_ids`` explicitly.

        Args:
            results: a :class:`~graintrace.simulation_postprocessing.SimulationResults`
                whose ``field_dir`` holds per-element CSVs.
            block_id: grain/block id whose elements to track (ignored if
                ``element_ids`` is given).
            element_ids: explicit element ids to track (one track each).
            steps: field-step indices to keep; defaults to all available field steps.

        Returns:
            list of ``(k, 3)`` MRP arrays, one per element.
        """
        field_steps = sorted(results.field_files.keys())
        sel = [s for s in _resolve_steps(results.n_steps, steps) if s in field_steps]
        if not sel:
            raise ValueError("No requested steps have a per-element field CSV.")

        if element_ids is None:
            if block_id is None:
                raise ValueError("Provide either block_id or element_ids.")
            df0 = results.load_field_data(sel[0])
            if self.block_id_col not in df0.columns:
                raise KeyError(
                    f"Field CSV has no '{self.block_id_col}' column; re-run CPFE with "
                    f"the block-id output, or pass element_ids explicitly."
                )
            mask = np.rint(df0[self.block_id_col].to_numpy()).astype(np.int64) == int(
                block_id
            )
            element_ids = df0.loc[mask, "id"].astype(np.int64).tolist()
            if not element_ids:
                raise ValueError(f"No elements found for block_id={block_id}.")

        cols = [f"{self.ori_tensor}_x", f"{self.ori_tensor}_y", f"{self.ori_tensor}_z"]
        per_step = {s: results.load_field_data(s).set_index("id") for s in sel}
        return [
            np.asarray([per_step[s].loc[eid, cols].to_numpy(dtype=float) for s in sel])
            for eid in element_ids
        ]

    def experiment_grain_tracks_by_id(
        self,
        step_csvs: Sequence[str],
        seed_grain_ids: Optional[Sequence[int]] = None,
        grain_id_col: str = "grain_id",
    ) -> List[np.ndarray]:
        """Per-grain Euler tracks across experiment steps by a stable grain-id column.

        Use when the per-step CSVs carry a consistent ``grain_id`` (no matching).

        Args:
            step_csvs: per-step FF CSV paths, ordered by load step.
            seed_grain_ids: grain ids to track; defaults to all ids in the first step.
            grain_id_col: name of the stable grain-id column.

        Returns:
            list of ``(k, 3)`` Euler arrays, one per seed grain; a grain missing from a
            step truncates its track there.
        """
        # pylint: disable=import-outside-toplevel  # heavy/optional dep
        import pandas as pd

        dfs = []
        for path in step_csvs:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            dfs.append(df.set_index(grain_id_col))

        if seed_grain_ids is None:
            seed_grain_ids = list(dfs[0].index)
        eul_cols = list(self.eul_cols)

        tracks: List[np.ndarray] = []
        for gid in seed_grain_ids:
            seq = []
            for df in dfs:
                if gid not in df.index:
                    break
                seq.append(df.loc[gid, eul_cols].to_numpy(dtype=float))
            if seq:
                tracks.append(np.asarray(seq))
        return tracks

    def experiment_grain_tracks(
        self,
        graphs: Sequence,
        seed_grain_ids: Optional[Sequence[int]] = None,
        message_passing_iter: int = 3,
        neighbor_selection_param: Optional[dict] = None,
        output_dir: str = "grain_tracking_output",
    ) -> List[np.ndarray]:
        """Per-grain Euler tracks across experiment load steps via graph matching.

        Matches grains on consecutive step pairs with
        :class:`~graintrace.grain_graph_matching.GraphGrainMatcher`, composes the
        pairwise ``a_to_b`` maps (:meth:`compose_tracks`), and reads each tracked
        grain's Euler angles from the graph node features -- the exact nodes the
        matcher used, so indices stay aligned even when the tessellation dropped
        out-of-box points.

        Args:
            graphs: one grain graph per step (e.g. from
                ``VoronoiMeshBuilder.build_graph``), ordered by load step.
            seed_grain_ids: first-step node indices to track; defaults to all nodes.
            message_passing_iter: matcher message-passing iterations.
            neighbor_selection_param: matcher neighbor-selection params (matcher
                default if ``None``).
            output_dir: base directory for the matcher's per-pair result files.

        Returns:
            list of ``(k, 3)`` Euler arrays, one per seed grain; a grain that stops
            matching is truncated there.
        """
        # pylint: disable=import-outside-toplevel  # heavy/optional deps
        import os

        from .grain_graph_matching import GraphGrainMatcher

        if not graphs:
            raise ValueError("graphs must be non-empty.")

        euler = [_graph_euler(g, self.eul_cols) for g in graphs]

        a_to_b_list = []
        for k in range(len(graphs) - 1):
            result = GraphGrainMatcher(
                graphs[k],
                graphs[k + 1],
                output_dir=os.path.join(output_dir, f"pair_{k}"),
            ).match_grains(
                message_passing_iter=message_passing_iter,
                neighbor_selection_param=neighbor_selection_param,
            )
            a_to_b_list.append(np.asarray(result["match"]["a_to_b"]).reshape(-1))

        return [
            np.asarray([euler[j][row] for j, row in enumerate(rows)])
            for rows in self.compose_tracks(a_to_b_list, seed_ids=seed_grain_ids)
        ]

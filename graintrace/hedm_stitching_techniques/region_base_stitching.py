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

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import os
from typing import List, Tuple, Dict, Optional
from .pair_stitching_utils import PairwiseStitcher, merge_properties
from .dataclass_utils import ScanMetadata, GrainSet
from graintrace.orientation_helper import misorientation
from scipy.optimize import linear_sum_assignment


class RegionBaseStitching:
    def __init__(
        self,
        scan_files: List[str],
        output_csv: str,
        position_tolerance: float,
        orientation_tolerance: float,
        radius_tolerance: float,
        weights: Dict[str, float],
        min_neighbors: int = 5,
        orientation_convention: str = "bunge",
        orientation_units: str = "degrees",
        symmetry: str = "432",
        output_column: List[str] = [
            "X",
            "Y",
            "Z",
            "GrainRadius",
            "Eul0",
            "Eul1",
            "Eul2",
            "ScanID",
        ],
    ):
        self.scan_files = scan_files
        self.output_csv = output_csv
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.radius_tolerance = radius_tolerance
        self.weights = weights
        self.min_neighbors = min_neighbors
        self.output_column = output_column

        self.angle_convention = orientation_convention
        self.angle_type = orientation_units
        self.symmetry = symmetry

        self.scans: List[GrainSet] = []
        self.stitched: Optional[GrainSet] = None

    def run(self, zlo: float, zhi: float, overlap_fraction: float) -> GrainSet:
        """
        Entry point: load and sort scans, then iteratively stitch S0 → S01 → S012 → ...
        and write the final output.
        """
        if zhi <= zlo:
            raise ValueError(f"Invalid z-window: zhi ({zhi}) must be > zlo ({zlo}).")

        self.global_zlo = float(zlo)
        self.global_zhi = float(zhi)

        self._load_and_sort_scans()

        all_zmin = min(gs.meta.zmin for gs in self.scans)
        all_zmax = max(gs.meta.zmax for gs in self.scans)

        # no overlap
        if overlap_fraction == 0.0:
            current = self.scans[0]
            for k in range(len(self.scans) - 1):
                current = self._nonoverlap_stitch_pair(
                    current, self.scans[k + 1], pair_id=k
                )

            self.stitched = current
            self._write_output(self.stitched, self.output_column)

            return self.stitched

        # with overlap
        nscan = len(self.scans)
        if nscan == 0:
            raise RuntimeError("No scans available.")
        if nscan == 1:
            self.stitched = self.scans[0]
            self._write_output(self.stitched, self.output_column)
            return self.stitched

        H = zhi - zlo
        denom = nscan - (nscan - 1) * overlap_fraction
        if denom <= 0:
            raise ValueError(
                f"Inconsistent configuration. Check overlap_fraction value."
            )

        z_scan_height = H / denom
        z_step = z_scan_height * (1.0 - overlap_fraction)

        current = self.scans[0]

        for k in range(nscan - 1):
            A = current
            B = self.scans[k + 1]

            # overlap region between theoretical scan k and k+1:
            # [zlo + (k+1)*z_step, zlo + k*z_step + z_scan_height]
            zol = zlo + (k + 1) * z_step
            zoh = zlo + k * z_step + z_scan_height

            current = self._overlap_stitch_pair(A, B, zol, zoh, pair_id=k)

        self.stitched = current
        self._write_output(self.stitched, self.output_column)
        return self.stitched

    def _load_and_sort_scans(self) -> None:
        """Load all scan CSVs into GrainSet objects and sort by zmin (bottom -> top)."""
        scans: List[GrainSet] = []

        required_cols = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]

        for idx, path in enumerate(self.scan_files):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Scan file not found: {path}")

            df = pd.read_csv(path)

            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Scan file '{path}' is missing required columns: {missing}. "
                    f"Each scan CSV must contain at least: {required_cols}."
                )

            df["ScanID"] = idx

            base = os.path.basename(path)
            name, _ = os.path.splitext(base)

            zmin = float(df["Z"].min())
            zmax = float(df["Z"].max())

            meta = ScanMetadata(name=name, scan_id=idx, zmin=zmin, zmax=zmax)

            scans.append(GrainSet(df=df, meta=meta))

        scans.sort(key=lambda gs: gs.meta.zmin)  # bottom (smallest zmin) to top

        self.scans = scans

    def _overlap_stitch_pair(
        self, A: GrainSet, B: GrainSet, zol: float, zoh: float, pair_id: int
    ) -> GrainSet:
        """
        Apply the pairwise stiching algorithm to GrainSets A and B
        over the overlap region [zol, zoh].
        """

        out_dir = os.path.dirname(self.output_csv)
        stem = os.path.splitext(os.path.basename(self.output_csv))[0]

        debug_folder = os.path.join(out_dir, "debug_logs")
        os.makedirs(debug_folder, exist_ok=True)

        debug_log_csv = os.path.join(
            debug_folder,
            f"{stem}_pair{pair_id}_decision_log.csv",
        )

        print(
            "Debug log for this pairwise stitching step will be saved to:",
            debug_log_csv,
            "\n",
        )

        stitcher = PairwiseStitcher(
            A=A,
            B=B,
            zol=zol,
            zoh=zoh,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            radius_tolerance=self.radius_tolerance,
            weights=self.weights,
            min_neighbors=self.min_neighbors,
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
            symmetry=self.symmetry,
            debug_log_csv=debug_log_csv,
        )

        return stitcher.run()

    def _nonoverlap_stitch_pair(
        self, A: GrainSet, B: GrainSet, pair_id: int
    ) -> GrainSet:
        """
        Non-overlap stitching: for each A grain in the top slab, find k nearest B
        candidates, keep the smallest-misorientation match within tolerance and slab
        distance, resolved via Hungarian assignment.
        """
        from scipy.spatial import cKDTree
        import numpy as np
        import pandas as pd

        dfA, dfB = A.df, B.df
        rA_all = dfA["GrainRadius"].to_numpy(float)
        rB_all = dfB["GrainRadius"].to_numpy(float)

        # slab thickness scale
        t = 2.0 * float(np.median(np.concatenate([rA_all, rB_all])))

        zA_max = float(dfA["Z"].max())
        zB_min = float(dfB["Z"].min())

        A_slab = dfA[dfA["Z"] >= (zA_max - t)].copy()
        B_slab = dfB[dfB["Z"] <= (zB_min + t)].copy()

        if len(A_slab) == 0 or len(B_slab) == 0:
            out = pd.concat([dfA, dfB], ignore_index=True)
            meta = ScanMetadata(
                name=f"{A.meta.name}_{B.meta.name}",
                scan_id=-1,
                zmin=float(out["Z"].min()),
                zmax=float(out["Z"].max()),
            )
            return GrainSet(df=out, meta=meta)

        coords_A = A_slab[["X", "Y", "Z"]].to_numpy(float)
        coords_B = B_slab[["X", "Y", "Z"]].to_numpy(float)

        tree_B = cKDTree(coords_B)

        k_query = min(int(self.min_neighbors), len(coords_B))
        dist, idxB = tree_B.query(coords_A, k=k_query, workers=-1)

        if k_query == 1:
            dist = dist[:, None]
            idxB = idxB[:, None]

        nA = len(A_slab)
        nB = len(B_slab)
        if nA == 0 or nB == 0:
            out = pd.concat([dfA, dfB], ignore_index=True)
            meta = ScanMetadata(
                name=f"{A.meta.name}_{B.meta.name}",
                scan_id=-1,
                zmin=float(out["Z"].min()),
                zmax=float(out["Z"].max()),
            )
            return GrainSet(df=out, meta=meta)

        a_idx, _ = np.indices(dist.shape)
        a_idx = a_idx.ravel()
        b_idx = idxB.ravel()
        dpos = dist.ravel().astype(float)

        oriA = A_slab[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
        oriB = B_slab[["Eul0", "Eul1", "Eul2"]].to_numpy(float)

        eA = oriA[a_idx]
        eB = oriB[b_idx]

        # misorientation for each candidate edge
        dori_t = misorientation(
            eA,
            eB,
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
            symmetry=self.symmetry,
        )
        dori = dori_t.detach().cpu().numpy().astype(float)

        ok = dori <= self.orientation_tolerance
        a_idx = a_idx[ok]
        b_idx = b_idx[ok]
        dpos = dpos[ok]
        dori = dori[ok]

        # Hungarian assignment with unmatched allowed (slab-local indices)
        nA = len(A_slab)
        nB = len(B_slab)

        # distance gate on top of orientation filter (avoids merging far grains for large k)
        ok2 = dpos <= t
        a_idx2 = a_idx[ok2]
        b_idx2 = b_idx[ok2]
        dpos2 = dpos[ok2]
        dori2 = dori[ok2]

        if a_idx2.size == 0:  # nothing feasible
            out = pd.concat([dfA, dfB], ignore_index=True)
            meta = ScanMetadata(
                name=f"{A.meta.name}_{B.meta.name}",
                scan_id=-1,
                zmin=float(out["Z"].min()),
                zmax=float(out["Z"].max()),
            )
            return GrainSet(df=out, meta=meta)

        # Cost: orientation dominates, small position term breaks ties (w_pos=0 => pure orientation)
        w_ori = 1.0
        w_pos = 0.1

        # scale terms so both are ~O(1)
        otol = self.orientation_tolerance if self.orientation_tolerance > 0 else 1.0
        ptol = t if t > 0 else 1.0

        edge_cost = w_ori * (dori2 / otol) + w_pos * (dpos2 / ptol)

        # unmatch cost just above max feasible cost
        max_real = w_ori * 1.0 + w_pos * 1.0
        UNMATCH_COST = max_real + 1e-6
        BIG = 1e9

        # best edge per (a, b)
        pair = {}
        for a, b, dp, do, cc in zip(a_idx2, b_idx2, dpos2, dori2, edge_cost):
            key = (int(a), int(b))
            if key not in pair or cc < pair[key][4]:
                pair[key] = (float(dp), float(do), float(cc), int(a), int(b))

        # Augmented square matrix: rows A + dummy rows for B, cols B + dummy cols for A
        N = nA + nB
        C = np.full((N, N), BIG, dtype=float)

        for (a, b), (dp, do, cc, _, _) in pair.items():
            C[a, b] = cc

        C[0:nA, nB : nB + nA] = UNMATCH_COST      # Real A -> dummy cols (unmatch A)
        C[nA : nA + nB, 0:nB] = UNMATCH_COST      # Dummy rows -> real B (unmatch B)
        C[nA : nA + nB, nB : nB + nA] = 0.0       # Dummy -> dummy

        row_ind, col_ind = linear_sum_assignment(C)

        keep_pairs_local = []
        for r, c in zip(row_ind, col_ind):
            if r >= nA:
                continue  # dummy row
            if c < nB and C[r, c] < UNMATCH_COST and C[r, c] < BIG * 0.5:
                keep_pairs_local.append((int(r), int(c)))

        A_slab_idx = A_slab.index.to_numpy()
        B_slab_idx = B_slab.index.to_numpy()

        matched_A_global = set()
        matched_B_global = set()
        merged_rows = []

        for a_loc, b_loc in keep_pairs_local:
            idxA = int(A_slab_idx[a_loc])  # global index in dfA
            idxB = int(B_slab_idx[b_loc])  # global index in dfB

            matched_A_global.add(idxA)
            matched_B_global.add(idxB)

            rowA = dfA.loc[idxA]
            rowB = dfB.loc[idxB]

            new_row = rowA.copy()
            new_row = merge_properties(new_row, rowA, rowB)
            merged_rows.append(new_row)

        keepA = dfA.drop(index=list(matched_A_global), errors="ignore")
        keepB = dfB.drop(index=list(matched_B_global), errors="ignore")

        out_frames = [keepA, keepB]
        if merged_rows:
            out_frames.insert(0, pd.DataFrame(merged_rows))

        out = pd.concat(out_frames, ignore_index=True)
        meta = ScanMetadata(
            name=f"{A.meta.name}_{B.meta.name}",
            scan_id=-1,
            zmin=float(out["Z"].min()),
            zmax=float(out["Z"].max()),
        )
        return GrainSet(df=out, meta=meta)

    def _write_output(self, stitched: GrainSet, required: List[str]) -> None:
        """Write the final stitched dataframe (with required columns) to self.output_csv."""

        df = stitched.df.copy()

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Stitched dataframe missing required columns: {missing}")

        debug_cols = ["Matched_when_merged", "Cell", "Unmatched_location"]
        for c in debug_cols:
            if c in df.columns:
                required.append(c)

        df = df[required]
        df.to_csv(self.output_csv, index=False)

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

"""Pairwise HEDM scan stitching: region classification, matching, merging."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from graintrace.orientation_helper import misorientation, average_orientation

from .dataclass_utils import ScanMetadata, GrainSet


class RegionClassifier:
    """Classify a grain's z-region relative to the scan overlap window."""

    @staticmethod
    def classify(zl: float, zh: float, zol: float, zoh: float) -> int:
        """
        Assign a grain's region from its z-extent [zl, zh] relative to overlap [zol, zoh].

            1 CORE, 2 HIGH, 3 LOW, 4 BND-HIGH, 5 BND-LOW, 6 CROSS-BOTH
        """

        if zh <= zol:
            return 3  # LOW

        if zl >= zoh:
            return 2  # HIGH

        if zl < zol and zh > zoh:
            return 6  # CROSS-BOTH

        if zl < zol < zh:
            return 5  # BND-LOW

        if zl < zoh < zh:
            return 4  # BND-HIGH

        return 1  # CORE


class MatchRuleTable:
    """
    Returns action code for matched pair given region(A) and region(B).
    Actions:
        'MC'       : merge (core-core)
        'KA'       : keep A, drop B
        'KB'       : keep B, drop A
        'RJ'       : reject match (send both to unmatched logic)
    """

    def __init__(self):
        self.table = self._build_table()

    def _build_table(self, optiona=True) -> Dict[Tuple[int, int], str]:

        t: Dict[Tuple[int, int], str] = {}

        RJ = "RJ"
        MC = "MC"
        KA = "KA"
        KB = "KB"

        # Regions: 1 CORE, 2 HIGH, 3 LOW, 4 BND-H, 5 BND-L, 6 CROSS
        if optiona:
            # Row A=1 (CORE)
            t[(1, 1)] = MC
            t[(1, 2)] = KB
            t[(1, 3)] = RJ
            t[(1, 4)] = KA
            t[(1, 5)] = KA
            t[(1, 6)] = KA

            # Row A=2 (HIGH)
            t[(2, 1)] = RJ
            t[(2, 2)] = RJ
            t[(2, 3)] = RJ
            t[(2, 4)] = RJ
            t[(2, 5)] = RJ
            t[(2, 6)] = RJ

            # Row A=3 (LOW)
            t[(3, 1)] = KA
            t[(3, 2)] = MC
            t[(3, 3)] = RJ
            t[(3, 4)] = MC
            t[(3, 5)] = KA
            t[(3, 6)] = KA

            # Row A=4 (BND-H)
            t[(4, 1)] = KB
            t[(4, 2)] = KB
            t[(4, 3)] = RJ
            t[(4, 4)] = KB
            t[(4, 5)] = MC
            t[(4, 6)] = KB

            # Row A=5 (BND-L)
            t[(5, 1)] = KB
            t[(5, 2)] = KB
            t[(5, 3)] = RJ
            t[(5, 4)] = MC
            t[(5, 5)] = KA
            t[(5, 6)] = KA

            # Row A=6 (CROSS)
            t[(6, 1)] = KB
            t[(6, 2)] = KB
            t[(6, 3)] = RJ
            t[(6, 4)] = KB
            t[(6, 5)] = MC
            t[(6, 6)] = MC
        else:
            # Alternate rule set: merge CORE/BND/CROSS A with any overlap-intersecting B
            t[(1, 1)] = MC
            t[(1, 2)] = RJ
            t[(1, 3)] = RJ
            t[(1, 4)] = MC
            t[(1, 5)] = MC
            t[(1, 6)] = MC

            t[(2, 1)] = RJ
            t[(2, 2)] = RJ
            t[(2, 3)] = RJ
            t[(2, 4)] = RJ
            t[(2, 5)] = RJ
            t[(2, 6)] = RJ

            t[(3, 1)] = RJ
            t[(3, 2)] = RJ
            t[(3, 3)] = RJ
            t[(3, 4)] = RJ
            t[(3, 5)] = RJ
            t[(3, 6)] = RJ

            t[(4, 1)] = MC
            t[(4, 2)] = RJ
            t[(4, 3)] = RJ
            t[(4, 4)] = MC
            t[(4, 5)] = MC
            t[(4, 6)] = MC

            t[(5, 1)] = MC
            t[(5, 2)] = RJ
            t[(5, 3)] = RJ
            t[(5, 4)] = MC
            t[(5, 5)] = MC
            t[(5, 6)] = MC

            t[(6, 1)] = MC
            t[(6, 2)] = RJ
            t[(6, 3)] = RJ
            t[(6, 4)] = MC
            t[(6, 5)] = MC
            t[(6, 6)] = MC

        return t

    def action(self, rA: int, rB: int) -> str:
        """Return the merge action code for matched regions (rA, rB)."""
        return self.table[(rA, rB)]


class UnmatchedRules:
    """
    Region to decision for unmatched grains.
    Returns: 'KEEP' or 'REMOVE' or 'ERROR'.
    """

    def __init__(self):
        self.rules_A = self._rules_A()
        self.rules_B = self._rules_B()

    def _rules_A(self) -> Dict[int, str]:
        """Region → 'KEEP' or 'REMOVE' for unmatched A grains."""
        return {
            1: "REMOVE",  # CORE
            2: "ERROR",  # HIGH
            3: "KEEP",  # LOW
            4: "REMOVE",  # BND-HIGH
            5: "KEEP",  # BND-LOW
            6: "REMOVE",  # CROSS-BOTH
        }

    def _rules_B(self) -> Dict[int, str]:
        """Region → 'KEEP' or 'REMOVE' for unmatched B grains."""
        return {
            1: "KEEP",  # CORE
            2: "KEEP",  # HIGH
            3: "ERROR",  # LOW
            4: "KEEP",  # BND-HIGH
            5: "REMOVE",  # BND-LOW
            6: "KEEP",  # CROSS-BOTH
        }

    def decide(self, which: str, region: int) -> str:
        """
        which = 'A' or 'B'
        region = 1 to 6
        """
        if which == "A":
            return self.rules_A[region]
        if which == "B":
            return self.rules_B[region]
        raise ValueError(f"Unknown scan label '{which}', expected 'A' or 'B'.")


class PairwiseStitcher:
    """Stitch a single pair of overlapping scans (A, B) into one grain set."""

    def __init__(
        self,
        A: GrainSet,
        B: GrainSet,
        zol: float,
        zoh: float,
        position_tolerance: float,
        orientation_tolerance: float,
        radius_tolerance: float,
        weights: Dict[str, float],
        min_neighbors: int,
        debug_log_csv: str | None = None,
        angle_convention: str = "bunge",
        angle_type: str = "degrees",
        symmetry: str = "432",
    ):
        # Copy the frames so region/debug columns are never written onto the
        # caller's GrainSets (A is the growing accumulator in RegionBaseStitching).
        self.A = GrainSet(df=A.df.copy(), meta=A.meta)
        self.B = GrainSet(df=B.df.copy(), meta=B.meta)
        self.zol = zol
        self.zoh = zoh

        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.radius_tolerance = radius_tolerance
        self.weights = weights
        self.min_neighbors = min_neighbors

        self.region_A = None
        self.region_B = None

        self.matches = None
        self.unmatched_A = None
        self.unmatched_B = None

        # Rejected-match indices, populated in _apply_matched_rules().
        self.rj_A = None
        self.rj_B = None

        self.matched_rules = MatchRuleTable()
        self.unmatched_rules = UnmatchedRules()

        self.debug_log_csv = debug_log_csv
        self._decision_log = []

        self.angle_convention = angle_convention
        self.angle_type = angle_type
        self.symmetry = symmetry

    def run(self) -> GrainSet:
        """Execute the full A→B stitching step and return a new GrainSet."""
        self._classify_regions()
        self._match()
        merged, keepA_matched, keepB_matched = self._apply_matched_rules()
        keepA_unmatched, keepB_unmatched = self._apply_unmatched_rules()

        result = self._build_output(
            merged, keepA_matched, keepB_matched, keepA_unmatched, keepB_unmatched
        )

        if self.debug_log_csv is not None:
            pd.DataFrame(self._decision_log).to_csv(self.debug_log_csv, index=False)

        return result

    def _classify_regions(self, delta_p=0.001) -> None:
        """Assign a region to each grain, stored in self.region_A / self.region_B."""
        if self.zoh <= self.zol:
            raise ValueError(
                f"Invalid overlap window in PairwiseStitcher: zoh ({self.zoh}) <= zol ({self.zol})."
            )

        def _z_extent(df):
            """Grain z-extent [zl, zh]: true tessellation extent if Zmin/Zmax columns
            are present (see scan_tessellation.compute_cell_geometry), else the
            equivalent-sphere approximation z +/- GrainRadius (inflated by delta_p)."""
            if "Zmin" in df.columns and "Zmax" in df.columns:
                return df["Zmin"].to_numpy(dtype=float), df["Zmax"].to_numpy(
                    dtype=float
                )
            z = df["Z"].to_numpy(dtype=float)
            r = df["GrainRadius"].to_numpy(dtype=float)
            return z - r - delta_p * r, z + r + delta_p * r

        zl_A, zh_A = _z_extent(self.A.df)
        zl_B, zh_B = _z_extent(self.B.df)

        regions_A = np.empty(len(zl_A), dtype=int)
        for i, zl_val in enumerate(zl_A):
            regions_A[i] = RegionClassifier.classify(
                zl=float(zl_val),
                zh=float(zh_A[i]),
                zol=self.zol,
                zoh=self.zoh,
            )

        regions_B = np.empty(len(zl_B), dtype=int)
        for i, zl_val in enumerate(zl_B):
            regions_B[i] = RegionClassifier.classify(
                zl=float(zl_val),
                zh=float(zh_B[i]),
                zol=self.zol,
                zoh=self.zoh,
            )

        self.region_A = regions_A
        self.region_B = regions_B

        self.A.df["Region"] = self.region_A
        self.B.df["Region"] = self.region_B

        for df in (self.A.df, self.B.df):
            df["Matched_when_merged"] = 0
            df["Cell"] = ""
            df["Unmatched_location"] = 1

    def _match(self) -> None:
        dfA = self.A.df
        dfB = self.B.df
        nA = len(dfA)
        nB = len(dfB)

        if nA == 0 or nB == 0:
            self.matches = pd.DataFrame(
                columns=[
                    "idx_A",
                    "idx_B",
                    "diff_pos_norm2",
                    "diff_rad_percentage",
                    "diff_ori_norm2",
                ]
            )
            self.unmatched_A = np.arange(nA, dtype=int)
            self.unmatched_B = np.arange(nB, dtype=int)
            return

        coords_A = dfA[["X", "Y", "Z"]].to_numpy(float)
        coords_B = dfB[["X", "Y", "Z"]].to_numpy(float)

        tree_A = cKDTree(coords_A)
        k = min(int(self.min_neighbors), nA)
        dist, idx = tree_A.query(coords_B, k=k, workers=-1)

        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        # Flatten candidate edges (B_s, A_t)
        s_idx, _ = np.indices(dist.shape)
        s_idx = s_idx.ravel()
        t_idx = idx.ravel()
        diff_pos = dist.ravel().astype(float)

        ori_A = dfA[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
        ori_B = dfB[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
        rad_A = dfA["GrainRadius"].to_numpy(float)
        rad_B = dfB["GrainRadius"].to_numpy(float)

        diff_ori_t = misorientation(
            ori_B[s_idx],
            ori_A[t_idx],
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
            symmetry=self.symmetry,
        )
        diff_ori = diff_ori_t.detach().cpu().numpy().astype(float)

        diff_rad = np.abs(rad_B[s_idx] - rad_A[t_idx]) / np.maximum(rad_A[t_idx], 1e-14)

        w_pos = float(self.weights.get("pos", 1.0))
        w_ori = float(self.weights.get("ori", 1.0))
        w_rad = float(self.weights.get("rad", 0.0))

        # Gating is independent of cost weighting: a tolerance of -1 disables the
        # gate for that dimension; a weight of 0 only removes it from the cost.
        gate_pos = self.position_tolerance != -1.0
        gate_ori = self.orientation_tolerance != -1.0
        gate_rad = self.radius_tolerance != -1.0

        ok = np.ones_like(diff_pos, dtype=bool)
        if gate_pos:
            ok &= diff_pos <= self.position_tolerance
        if gate_ori:
            ok &= diff_ori <= self.orientation_tolerance
        if gate_rad:
            ok &= diff_rad <= self.radius_tolerance

        s_idx = s_idx[ok]
        t_idx = t_idx[ok]
        diff_pos = diff_pos[ok]
        diff_ori = diff_ori[ok]
        diff_rad = diff_rad[ok]

        if s_idx.size == 0:
            self.matches = pd.DataFrame(
                columns=[
                    "idx_A",
                    "idx_B",
                    "diff_pos_norm2",
                    "diff_rad_percentage",
                    "diff_ori_norm2",
                ]
            )
            self.unmatched_A = np.arange(nA, dtype=int)
            self.unmatched_B = np.arange(nB, dtype=int)
            return

        ptol = self.position_tolerance if self.position_tolerance > 0 else 1.0
        otol = self.orientation_tolerance if self.orientation_tolerance > 0 else 1.0
        rtol = self.radius_tolerance if self.radius_tolerance > 0 else 1.0

        cost = (
            w_pos * (diff_pos / ptol)
            + w_ori * (diff_ori / otol)
            + w_rad * (diff_rad / rtol)
        )

        # Hungarian assignment with unmatched allowed.
        # Unmatch cost sits just above any valid match but far below an infeasible (BIG) one.
        # Each gated term is normalized to <= 1, so the max real cost is the sum of weights.
        max_real = w_pos + w_ori + w_rad

        UNMATCH_COST = max_real + 1e-6
        BIG = 1e9  # must be >> UNMATCH_COST

        # Best-cost lookup per feasible (b, a) pair
        pair_dp = {}
        pair_dr = {}
        pair_do = {}
        pair_c = {}

        for b, a, dp, dr, do, cc in zip(
            s_idx, t_idx, diff_pos, diff_rad, diff_ori, cost
        ):
            key = (int(b), int(a))
            if key not in pair_c or cc < pair_c[key]:
                pair_c[key] = float(cc)
                pair_dp[key] = float(dp)
                pair_dr[key] = float(dr)
                pair_do[key] = float(do)

        # Square augmented cost matrix:
        # rows = B real (nB) + A-dummy rows (nA); cols = A real (nA) + B-dummy cols (nB)
        N = nA + nB
        C = np.full((N, N), BIG, dtype=float)

        # Real B -> Real A (feasible pairs; rest stay BIG)
        for (b, a), cc in pair_c.items():
            C[b, a] = cc

        C[0:nB, nA : nA + nB] = UNMATCH_COST  # Real B -> dummy cols (unmatch B)
        C[nB : nB + nA, 0:nA] = UNMATCH_COST  # Dummy rows -> Real A cols (unmatch A)
        C[nB : nB + nA, nA : nA + nB] = 0.0  # Dummy -> dummy (no-op)

        row_ind, col_ind = linear_sum_assignment(C)

        rows = []
        matched_A = set()
        matched_B = set()

        for r, c in zip(row_ind, col_ind):
            if r >= nB:
                continue  # only real B rows

            b = int(r)
            if c < nA:  # matched to real A
                a = int(c)
                cc = C[r, c]
                if cc >= UNMATCH_COST or cc >= BIG * 0.5:
                    continue  # treat as unmatched

                key = (b, a)
                dp = pair_dp.get(key, np.nan)
                dr = pair_dr.get(key, np.nan)
                do = pair_do.get(key, np.nan)

                rows.append((a, b, float(dp), float(dr), float(do)))
                matched_A.add(a)
                matched_B.add(b)

        self.matches = pd.DataFrame(
            rows,
            columns=[
                "idx_A",
                "idx_B",
                "diff_pos_norm2",
                "diff_rad_percentage",
                "diff_ori_norm2",
            ],
        )

        all_A = set(range(nA))
        all_B = set(range(nB))
        self.unmatched_A = np.array(sorted(all_A - matched_A), dtype=int)
        self.unmatched_B = np.array(sorted(all_B - matched_B), dtype=int)

    def _apply_matched_rules(self):
        """
        Apply MatchRuleTable actions (MC merge / KA / KB / RJ reject) to each matched pair.

        Returns (merged_df, keepA_matched, keepB_matched); sets self.rj_A / self.rj_B
        with indices deferred to the unmatched stage.
        """

        if self.matches is None:
            raise RuntimeError(
                "self.matches is not set. Run _match() before _apply_matched_rules()."
            )

        merged_rows = []
        keepA_idx = []
        keepB_idx = []
        rj_A_idx = set()
        rj_B_idx = set()

        dfA = self.A.df
        dfB = self.B.df

        for row in self.matches.itertuples(index=False):
            idx_A = int(row.idx_A)
            idx_B = int(row.idx_B)

            rA = int(self.region_A[idx_A])
            rB = int(self.region_B[idx_B])

            action = self.matched_rules.action(rA, rB)

            cell = f"({rA},{rB})"

            outA = "UNKNOWN"
            outB = "UNKNOWN"

            if action == "MC":
                outA, outB = "MERGED", "MERGED"
            elif action == "KA":
                outA, outB = "KEPT", "REMOVED"
            elif action == "KB":
                outA, outB = "REMOVED", "KEPT"
            elif action == "RJ":
                outA, outB = "DEFER_UNMATCHED", "DEFER_UNMATCHED"

            self._decision_log.append(
                {
                    "stage": "matched",
                    "idx_A": idx_A,
                    "idx_B": idx_B,
                    "rA": rA,
                    "rB": rB,
                    "cell": cell,
                    "action": action,
                    "outcome_A": outA,
                    "outcome_B": outB,
                }
            )

            dfA.at[dfA.index[idx_A], "Cell"] = cell
            dfB.at[dfB.index[idx_B], "Cell"] = cell
            dfA.at[dfA.index[idx_A], "Unmatched_location"] = 0
            dfB.at[dfB.index[idx_B], "Unmatched_location"] = 0

            if action == "MC":

                dfA.at[dfA.index[idx_A], "Matched_when_merged"] = 1
                dfB.at[dfB.index[idx_B], "Matched_when_merged"] = 1

                # Merge: start from A's row and overwrite averaged fields
                rowA = dfA.iloc[idx_A]
                rowB = dfB.iloc[idx_B]

                new_row = rowA.copy()

                new_row["Matched_when_merged"] = 1
                new_row["Cell"] = cell
                new_row["Unmatched_location"] = 0

                new_row = merge_properties(
                    new_row,
                    rowA,
                    rowB,
                    angle_convention=self.angle_convention,
                    angle_type=self.angle_type,
                    symmetry=self.symmetry,
                )
                merged_rows.append(new_row)

            elif action == "KA":
                keepA_idx.append(idx_A)

            elif action == "KB":
                keepB_idx.append(idx_B)

            elif action == "RJ":
                rj_A_idx.add(idx_A)
                rj_B_idx.add(idx_B)

        if merged_rows:
            merged_df = pd.DataFrame(merged_rows)
        else:
            merged_df = pd.DataFrame(columns=dfA.columns)

        if keepA_idx:
            keepA_matched = dfA.iloc[keepA_idx].copy()
        else:
            keepA_matched = pd.DataFrame(columns=dfA.columns)

        if keepB_idx:
            keepB_matched = dfB.iloc[keepB_idx].copy()
        else:
            keepB_matched = pd.DataFrame(columns=dfB.columns)

        self.rj_A = (
            np.array(sorted(rj_A_idx), dtype=int)
            if rj_A_idx
            else np.array([], dtype=int)
        )
        self.rj_B = (
            np.array(sorted(rj_B_idx), dtype=int)
            if rj_B_idx
            else np.array([], dtype=int)
        )

        return merged_df, keepA_matched, keepB_matched

    def _apply_unmatched_rules(self):
        """
        Apply UnmatchedRules to never-matched grains and rejected-match grains,
        returning (keepA_unmatched, keepB_unmatched) DataFrames of kept grains.
        """

        dfA = self.A.df
        dfB = self.B.df

        # Collect candidate indices: never-matched (_match) + rejected (_apply_matched_rules)
        base_unmatched_A = getattr(self, "unmatched_A", None)
        base_unmatched_B = getattr(self, "unmatched_B", None)
        rj_A = getattr(self, "rj_A", None)
        rj_B = getattr(self, "rj_B", None)

        idx_A_set = set()
        idx_B_set = set()

        if base_unmatched_A is not None:
            idx_A_set.update(int(i) for i in base_unmatched_A)
        if rj_A is not None:
            idx_A_set.update(int(i) for i in rj_A)

        if base_unmatched_B is not None:
            idx_B_set.update(int(i) for i in base_unmatched_B)
        if rj_B is not None:
            idx_B_set.update(int(i) for i in rj_B)

        # apply rules for A
        keepA_indices = []
        for idx in sorted(idx_A_set):
            region = int(self.region_A[idx])
            dfA.at[dfA.index[idx], "Unmatched_location"] = int(self.region_A[idx])
            decision = self.unmatched_rules.decide("A", region)

            self._decision_log.append(
                {
                    "stage": "unmatched",
                    "which": "A",
                    "idx": int(idx),
                    "region": region,
                    "decision": decision,
                }
            )

            if decision == "KEEP":
                keepA_indices.append(idx)
            if decision == "ERROR":
                print(
                    f"Unmatched decision=ERROR for A idx={idx} (region={region}). Dropping this grain."
                )

        if keepA_indices:
            keepA_unmatched = dfA.iloc[keepA_indices].copy()
        else:
            keepA_unmatched = pd.DataFrame(columns=dfA.columns)

        # apply rules for B
        keepB_indices = []
        for idx in sorted(idx_B_set):
            region = int(self.region_B[idx])
            decision = self.unmatched_rules.decide("B", region)
            dfB.at[dfB.index[idx], "Unmatched_location"] = int(self.region_B[idx])

            self._decision_log.append(
                {
                    "stage": "unmatched",
                    "which": "B",
                    "idx": int(idx),
                    "region": region,
                    "decision": decision,
                }
            )

            if decision == "KEEP":
                keepB_indices.append(idx)
            if decision == "ERROR":
                print(
                    f"Unmatched decision=ERROR for B idx={idx} (region={region}). Dropping this grain."
                )

        if keepB_indices:
            keepB_unmatched = dfB.iloc[keepB_indices].copy()
        else:
            keepB_unmatched = pd.DataFrame(columns=dfB.columns)

        return keepA_unmatched, keepB_unmatched

    def _build_output(
        self,
        merged: pd.DataFrame,
        keepA_matched: pd.DataFrame,
        keepB_matched: pd.DataFrame,
        keepA_unmatched: pd.DataFrame,
        keepB_unmatched: pd.DataFrame,
    ) -> GrainSet:
        """Concatenate merged/kept frames into a new stitched GrainSet with updated metadata."""

        frames = [
            merged,
            keepA_matched,
            keepB_matched,
            keepA_unmatched,
            keepB_unmatched,
        ]

        # drop empty frames to avoid dtype issues on concat
        non_empty = [f for f in frames if f is not None and len(f) > 0]

        if non_empty:
            df_new = pd.concat(non_empty, ignore_index=True)
        else:
            df_new = pd.DataFrame(columns=self.A.df.columns)

        if len(df_new) > 0 and "Z" in df_new.columns:
            zmin_new = float(df_new["Z"].min())
            zmax_new = float(df_new["Z"].max())
        else:
            zmin_new = min(self.A.meta.zmin, self.B.meta.zmin)
            zmax_new = max(self.A.meta.zmax, self.B.meta.zmax)

        meta_new = ScanMetadata(
            name=f"{self.A.meta.name}_{self.B.meta.name}",
            scan_id=-1,  # stitched composite
            zmin=zmin_new,
            zmax=zmax_new,
        )

        if "Region" in df_new.columns:
            df_new = df_new.drop(columns=["Region"])

        return GrainSet(df=df_new, meta=meta_new)


def merge_properties(
    new_row,
    rowA,
    rowB,
    angle_convention: str = "bunge",
    angle_type: str = "degrees",
    symmetry: str = "432",
):
    """
    Volume-weighted merge of two grains.
    Requires columns: ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]

    Orientation is a symmetry-aware, volume-weighted average of the two grains:
    B is brought into the symmetry-equivalent closest to A, the two rotation
    matrices are volume-weighted and re-projected onto SO(3) (via SVD), then
    converted back to Euler angles in the given convention/units.
    """

    rA = float(rowA["GrainRadius"])
    rB = float(rowB["GrainRadius"])

    vA = (4.0 / 3.0) * np.pi * rA**3
    vB = (4.0 / 3.0) * np.pi * rB**3
    vT = vA + vB

    for col in ["X", "Y", "Z"]:
        new_row[col] = (vA * rowA[col] + vB * rowB[col]) / vT

    # merged size = average of the two grain volumes (not their sum)
    vAvg = 0.5 * (vA + vB)
    new_row["GrainRadius"] = (3.0 * vAvg / (4.0 * np.pi)) ** (1.0 / 3.0)

    # orientation: symmetry-aware, volume-weighted average of A and B
    e_avg = average_orientation(
        [
            [float(rowA["Eul0"]), float(rowA["Eul1"]), float(rowA["Eul2"])],
            [float(rowB["Eul0"]), float(rowB["Eul1"]), float(rowB["Eul2"])],
        ],
        weights=[vA, vB],
        convention=angle_convention,
        angle_type=angle_type,
        symmetry=symmetry,
    )
    new_row["Eul0"] = float(e_avg[0])
    new_row["Eul1"] = float(e_avg[1])
    new_row["Eul2"] = float(e_avg[2])

    return new_row

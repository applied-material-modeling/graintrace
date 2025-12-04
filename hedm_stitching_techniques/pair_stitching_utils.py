import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from .dataclass_utils import ScanMetadata, GrainSet
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


# Region classifier (zl, zh vs zol, zoh -> region 1 to 6)
class RegionClassifier:
    @staticmethod
    def classify(zl: float, zh: float, zol: float, zoh: float) -> int:
        """
        Assign region for a grain based on zl = z_centroid - radius,
        zh = z_centroid + radius, relative to [zol, zoh].

            1 CORE       : entirely inside overlap
            2 HIGH       : entirely above overlap
            3 LOW        : entirely below overlap
            4 BND-HIGH   : intersects upper boundary only
            5 BND-LOW    : intersects lower boundary only
            6 CROSS-BOTH : spans both boundaries
        """

        if zh <= zol:
            return 3  # LOW

        if zl >= zoh:
            return 2  # HIGH

        if zl < zol and zh > zoh:
            return 6  # CROSS-BOTH

        if zl < zol and zh > zol:
            return 5  # BND-LOW

        if zl < zoh and zh > zoh:
            return 4  # BND-HIGH

        return 1  # CORE


# Matched rule table (6x6)
class MatchRuleTable:
    """
    Returns action code for matched pair given region(A) and region(B).
    Actions:
        'MC'       : merge (core-core)
        'KA'       : keep A, drop B
        'KB'       : keep B, drop A
        'MB_merge' : merge, boundary/large case
        'RJ'       : reject match (send both to unmatched logic)
    """
    def __init__(self):
        self.table = self._build_table()

    def _build_table(self) -> Dict[Tuple[int, int], str]:

        t: Dict[Tuple[int, int], str] = {}

        # Convenience
        RJ = "RJ"
        MC = "MC"
        KA = "KA"
        KB = "KB"
        MB = "MB_merge"

        # Regions:
        # 1 CORE, 2 HIGH, 3 LOW, 4 BND-H, 5 BND-L, 6 CROSS

        # Row A=1 (CORE)
        t[(1, 1)] = MC
        t[(1, 2)] = RJ
        t[(1, 3)] = RJ
        t[(1, 4)] = KA
        t[(1, 5)] = KA
        t[(1, 6)] = KA

        # Row A=2 (HIGH) – all RJ
        for rb in range(1, 7):
            t[(2, rb)] = RJ

        # Row A=3 (LOW) – all RJ
        for rb in range(1, 7):
            t[(3, rb)] = RJ

        # Row A=4 (BND-H)
        t[(4, 1)] = KB
        t[(4, 2)] = RJ
        t[(4, 3)] = RJ
        t[(4, 4)] = KB
        t[(4, 5)] = MB
        t[(4, 6)] = KB

        # Row A=5 (BND-L)
        t[(5, 1)] = KB
        t[(5, 2)] = RJ
        t[(5, 3)] = RJ
        t[(5, 4)] = MB
        t[(5, 5)] = KA
        t[(5, 6)] = KA

        # Row A=6 (CROSS)
        t[(6, 1)] = KB
        t[(6, 2)] = RJ
        t[(6, 3)] = RJ
        t[(6, 4)] = KB
        t[(6, 5)] = KA
        t[(6, 6)] = MB

        return t

    def action(self, rA: int, rB: int) -> str:
        return self.table[(rA, rB)]


# Unmatched rule table (separate for A and B)
class UnmatchedRules:
    """
    Region to decision for unmatched grains.
    Returns: 'KEEP' or 'REMOVE'.

    For scan A (lower):
        1 CORE       : KEEP   (stitching error)
        2 HIGH       : REMOVE (geometry error for lower scan)
        3 LOW        : KEEP
        4 BND-HIGH   : REMOVE (A lets B own top-side unmatched)
        5 BND-LOW    : KEEP   (warning)
        6 CROSS-BOTH : KEEP   (warning)

    For scan B (higher):
        1 CORE       : KEEP   (stitching error)
        2 HIGH       : KEEP
        3 LOW        : REMOVE (geometry error for higher scan)
        4 BND-HIGH   : KEEP   (warning)
        5 BND-LOW    : REMOVE (A owns bottom-side unmatched)
        6 CROSS-BOTH : KEEP   (warning)
    """
    def __init__(self):
        self.rules_A = self._rules_A()
        self.rules_B = self._rules_B()

    def _rules_A(self) -> Dict[int, str]:
        """Region → 'KEEP' or 'REMOVE' for unmatched A grains."""
        return {
            1: "KEEP",   # CORE
            2: "REMOVE", # HIGH
            3: "KEEP",   # LOW
            4: "REMOVE", # BND-HIGH
            5: "KEEP",   # BND-LOW
            6: "KEEP",   # CROSS-BOTH
        }

    def _rules_B(self) -> Dict[int, str]:
        """Region → 'KEEP' or 'REMOVE' for unmatched B grains."""
        return {
            1: "KEEP",   # CORE
            2: "KEEP",   # HIGH
            3: "REMOVE", # LOW
            4: "KEEP",   # BND-HIGH
            5: "REMOVE", # BND-LOW
            6: "KEEP",   # CROSS-BOTH
        }

    def decide(self, which: str, region: int) -> str:
        """
        which = 'A' or 'B'
        region = 1 to 6
        """
        if which == "A":
            return self.rules_A[region]
        elif which == "B":
            return self.rules_B[region]
        else:
            raise ValueError(f"Unknown scan label '{which}', expected 'A' or 'B'.")

# -------------------------------------------------------------------------
# MAIN CLASS
# -------------------------------------------------------------------------
class PairwiseStitcher:
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
    ):
        self.A = A
        self.B = B
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

        self.matched_rules = MatchRuleTable()
        self.unmatched_rules = UnmatchedRules()

    def run(self) -> GrainSet:
        """
        Executes the full A→B stitching step.
        Returns a new GrainSet.
        """
        # 1. classify regions
        self._classify_regions()

        # 2. match B → A
        self._match()

        # 3. apply matched rules
        merged, keepA_matched, keepB_matched = self._apply_matched_rules()

        # 4. apply unmatched rules
        keepA_unmatched, keepB_unmatched = self._apply_unmatched_rules()

        # 5. build final grain set AB
        result = self._build_output(
            merged,
            keepA_matched, keepB_matched,
            keepA_unmatched, keepB_unmatched
        )

        return result

    # ----------------------------------------------------------
    # Internal steps
    # ----------------------------------------------------------

    def _classify_regions(self, delta_p = 0.001) -> None:
        """
        Assigned regions to self.region_A, self.region_B for each grain in A and B
        
        Results are stored in:
            self.region_A  # np.ndarray[int] of length len(A.df)
            self.region_B  # np.ndarray[int] of length len(B.df)
        """
        if self.zoh <= self.zol:
            raise ValueError(
                f"Invalid overlap window in PairwiseStitcher: zoh ({self.zoh}) <= zol ({self.zol})."
            )

        z_A = self.A.df["Z"].to_numpy(dtype=float)
        r_A = self.A.df["GrainRadius"].to_numpy(dtype=float)
        zl_A = z_A - r_A - delta_p*r_A
        zh_A = z_A + r_A + delta_p*r_A

        regions_A = np.empty(len(z_A), dtype=int)
        for i in range(len(z_A)):
            regions_A[i] = RegionClassifier.classify(
                zl=float(zl_A[i]),
                zh=float(zh_A[i]),
                zol=self.zol,
                zoh=self.zoh,
            )

        z_B = self.B.df["Z"].to_numpy(dtype=float)
        r_B = self.B.df["GrainRadius"].to_numpy(dtype=float)
        zl_B = z_B - r_B - delta_p*r_B
        zh_B = z_B + r_B + delta_p*r_B

        regions_B = np.empty(len(z_B), dtype=int)
        for i in range(len(z_B)):
            regions_B[i] = RegionClassifier.classify(
                zl=float(zl_B[i]),
                zh=float(zh_B[i]),
                zol=self.zol,
                zoh=self.zoh,
            )

        self.region_A = regions_A
        self.region_B = regions_B

        self.A.df["Region"] = self.region_A
        self.B.df["Region"] = self.region_B
        
    def _match(self) -> None:
        """
        Match grains of B to grains of A using KD-tree neighbor search
        and linear-sum assignment, following the ScanStitchingComparison
        algorithm.

        Produces:
            self.matches      : DataFrame with columns
                                ['idx_A', 'idx_B',
                                 'diff_pos_norm2',
                                 'diff_rad_percentage',
                                 'diff_ori_norm2']
            self.unmatched_A  : np.ndarray of indices in A with no valid match
            self.unmatched_B  : np.ndarray of indices in B with no valid match
        """

        dfA = self.A.df
        dfB = self.B.df

        nA = len(dfA)
        nB = len(dfB)

        if nA == 0 or nB == 0:
            # trivial case: nothing to match
            self.matches = pd.DataFrame(
                columns=[
                    "idx_A", "idx_B",
                    "diff_pos_norm2",
                    "diff_rad_percentage",
                    "diff_ori_norm2",
                ]
            )
            self.unmatched_A = np.arange(nA, dtype=int)
            self.unmatched_B = np.arange(nB, dtype=int)
            return

        # --- build KD-tree on A (target) ---
        coords_A = dfA[["X", "Y", "Z"]].to_numpy(dtype=float)
        tree_A = cKDTree(coords_A)

        # --- source = B, target = A ---
        coords_B = dfB[["X", "Y", "Z"]].to_numpy(dtype=float)

        k = min(self.min_neighbors, nA)
        dist, idx = tree_A.query(coords_B, k=k, workers=-1)

        # shape handling (k=1 returns 1D arrays from scipy)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        # indices for all candidate pairs
        # s_idx: indices in source (B)
        # t_idx: indices in target (A)
        s_idx, n_idx = np.indices(dist.shape)
        s_idx = s_idx.ravel()
        t_idx = idx.ravel()
        diff_pos = dist.ravel()

        # --- orientation and radius differences ---
        ori_A = dfA[["Eul0", "Eul1", "Eul2"]].to_numpy(dtype=float)
        ori_B = dfB[["Eul0", "Eul1", "Eul2"]].to_numpy(dtype=float)

        rad_A = dfA["GrainRadius"].to_numpy(dtype=float)
        rad_B = dfB["GrainRadius"].to_numpy(dtype=float)

        # orientation: norm of Euler difference
        diff_ori = np.linalg.norm(ori_B[s_idx] - ori_A[t_idx], axis=1)

        # radius: relative difference wrt target (A)
        diff_rad = np.abs(rad_B[s_idx] - rad_A[t_idx]) / rad_A[t_idx]

        # --- cost matrix for Hungarian assignment ---
        w_pos = self.weights.get("pos", 1.0)
        w_ori = self.weights.get("ori", 0.0)
        w_rad = self.weights.get("rad", 0.0)

        eps = 1e-14

        ptol = self.position_tolerance if self.position_tolerance != 0.0 else eps
        otol = self.orientation_tolerance if self.orientation_tolerance != 0.0 else eps
        rtol = self.radius_tolerance if self.radius_tolerance != 0.0 else eps

        if ptol < 0 or otol < 0 or rtol < 0:
            raise ValueError(
                f"Tolerances must be non-negative; got "
                f"position_tolerance={self.position_tolerance}, "
                f"orientation_tolerance={self.orientation_tolerance}, "
                f"radius_tolerance={self.radius_tolerance}."
            )

        cost = (
            w_pos * (diff_pos / ptol) +
            w_ori * (diff_ori / otol) +
            w_rad * (diff_rad / rtol)
        )

        # build full cost matrix (source = B, target = A)
        cost_matrix = np.full((nB, nA), 1e12, dtype=float)
        cost_matrix[s_idx, t_idx] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        match_rows = []
        matched_A = set()
        matched_B = set()
        for s, t in zip(row_ind, col_ind):

            # recover the original diff_pos/diff_rad/diff_ori for this pair
            mask = (s_idx == s) & (t_idx == t)
            if not np.any(mask):
                continue

            dp = diff_pos[mask][0]
            dr = diff_rad[mask][0]
            do = diff_ori[mask][0]

            if (
                (dp > self.position_tolerance) or
                (do > self.orientation_tolerance) or
                (dr > self.radius_tolerance)
            ):
                continue

            # idx_A = target index (A), idx_B = source index (B)
            match_rows.append(
                (int(t), int(s), dp, dr, do)
            )
            matched_A.add(int(t))
            matched_B.add(int(s))

        if match_rows:
            self.matches = pd.DataFrame(
                match_rows,
                columns=[
                    "idx_A", "idx_B",
                    "diff_pos_norm2",
                    "diff_rad_percentage",
                    "diff_ori_norm2",
                ],
            )
        else:
            self.matches = pd.DataFrame(
                columns=[
                    "idx_A", "idx_B",
                    "diff_pos_norm2",
                    "diff_rad_percentage",
                    "diff_ori_norm2",
                ]
            )

        all_A = set(range(nA))
        all_B = set(range(nB))

        self.unmatched_A = np.array(sorted(all_A - matched_A), dtype=int)
        self.unmatched_B = np.array(sorted(all_B - matched_B), dtype=int)

    def _apply_matched_rules(self):
        """
        For each matched pair (idx_A, idx_B):

          - Look up regions: rA, rB
          - Get action from MatchRuleTable:
              'MC'       : merge (core-core) -- this is the good one
              'MB_merge' : merge, boundary/large -- this is the warning one
              'KA'       : keep A, drop B
              'KB'       : keep B, drop A
              'RJ'       : reject match (both go to unmatched logic)

        Returns:
            merged_df       : new rows created by merging A/B
            keepA_matched   : rows kept from A (matched pairs)
            keepB_matched   : rows kept from B (matched pairs)

        Side effects:
            self.rj_A, self.rj_B : np.ndarray of indices from A/B involved in RJ.
        """

        if self.matches is None:
            raise RuntimeError("self.matches is not set. Run _match() before _apply_matched_rules().")

        # Containers
        merged_rows = []
        keepA_idx = []
        keepB_idx = []
        rj_A_idx = set()
        rj_B_idx = set()

        dfA = self.A.df
        dfB = self.B.df

        for row in self.matches.itertuples(index=False):
            # Expect columns: idx_A, idx_B
            idx_A = int(row.idx_A)
            idx_B = int(row.idx_B)

            rA = int(self.region_A[idx_A])
            rB = int(self.region_B[idx_B])

            action = self.matched_rules.action(rA, rB)

            if action in ("MC", "MB_merge"):
                # Merge: start from A's row and overwrite averaged fields
                rowA = dfA.iloc[idx_A]
                rowB = dfB.iloc[idx_B]

                new_row = rowA.copy()

                # average the relevant columns .. to be modified as needed
                for col in ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]:
                    new_row[col] = 0.5 * (rowA[col] + rowB[col])
                # ------------------------------------------------------

                merged_rows.append(new_row)

            elif action == "KA":
                # keep A, drop B
                keepA_idx.append(idx_A)

            elif action == "KB":
                # keep B, drop A
                keepB_idx.append(idx_B)

            elif action == "RJ":
                # rejected match: both will be treated as unmatched
                rj_A_idx.add(idx_A)
                rj_B_idx.add(idx_B)

        # Build DataFrames from collected items
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

        # Store rejected indices for unmatched stage
        self.rj_A = np.array(sorted(rj_A_idx), dtype=int) if rj_A_idx else np.array([], dtype=int)
        self.rj_B = np.array(sorted(rj_B_idx), dtype=int) if rj_B_idx else np.array([], dtype=int)

        return merged_df, keepA_matched, keepB_matched

    def _apply_unmatched_rules(self):
        """
        Apply unmatched rules to:

          - grains that were never matched (unmatched_A, unmatched_B)
          - grains from rejected matches (rj_A, rj_B)

        For each such grain:
          - use its region (self.region_A / self.region_B)
          - call unmatched_rules.decide('A' or 'B', region)
          - keep or drop accordingly

        Returns:
            keepA_unmatched : DataFrame of A-grains kept
            keepB_unmatched : DataFrame of B-grains kept
        """

        dfA = self.A.df
        dfB = self.B.df

        # --- collect candidate indices for unmatched processing ---

        # From _match()
        base_unmatched_A = getattr(self, "unmatched_A", None)
        base_unmatched_B = getattr(self, "unmatched_B", None)

        # From _apply_matched_rules()
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

        # --- apply rules for A ---
        keepA_indices = []
        for idx in sorted(idx_A_set):
            region = int(self.region_A[idx])
            decision = self.unmatched_rules.decide("A", region)
            if decision == "KEEP":
                keepA_indices.append(idx)

        if keepA_indices:
            keepA_unmatched = dfA.iloc[keepA_indices].copy()
        else:
            keepA_unmatched = pd.DataFrame(columns=dfA.columns)

        # --- apply rules for B ---
        keepB_indices = []
        for idx in sorted(idx_B_set):
            region = int(self.region_B[idx])
            decision = self.unmatched_rules.decide("B", region)
            if decision == "KEEP":
                keepB_indices.append(idx)

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
        """
        Construct the new stitched GrainSet from:
          - merged             (MC, MB_merge results)
          - keepA_matched      (KA)
          - keepB_matched      (KB)
          - keepA_unmatched    (unmatched A that we keep)
          - keepB_unmatched    (unmatched B that we keep)

        Returns:
            GrainSet with concatenated df and updated metadata.
        """

        frames = [
            merged,
            keepA_matched,
            keepB_matched,
            keepA_unmatched,
            keepB_unmatched,
        ]

        # filter out empty frames to avoid issues with differing dtypes
        non_empty = [f for f in frames if f is not None and len(f) > 0]

        if non_empty:
            df_new = pd.concat(non_empty, ignore_index=True)
        else:
            df_new = pd.DataFrame(columns=self.A.df.columns)

        # Compute new z-range
        if len(df_new) > 0 and "Z" in df_new.columns:
            zmin_new = float(df_new["Z"].min())
            zmax_new = float(df_new["Z"].max())
        else:
            zmin_new = min(self.A.meta.zmin, self.B.meta.zmin)
            zmax_new = max(self.A.meta.zmax, self.B.meta.zmax)

        meta_new = ScanMetadata(
            name=f"{self.A.meta.name}_{self.B.meta.name}",
            scan_id=-1,          # stitched / composite
            zmin=zmin_new,
            zmax=zmax_new,
        )

        if "Region" in df_new.columns:
            df_new = df_new.drop(columns=["Region"])

        return GrainSet(df=df_new, meta=meta_new)

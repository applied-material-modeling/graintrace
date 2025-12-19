import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from .dataclass_utils import ScanMetadata, GrainSet
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from orientation_helper import misorientation

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

        # Regions:
        # 1 CORE, 2 HIGH, 3 LOW, 4 BND-H, 5 BND-L, 6 CROSS

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

        return t

    def action(self, rA: int, rB: int) -> str:
        return self.table[(rA, rB)]


# Unmatched rule table (separate for A and B)
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
            1: "REMOVE", # CORE
            2: "ERROR", # HIGH
            3: "KEEP",   # LOW
            4: "REMOVE", # BND-HIGH
            5: "KEEP",   # BND-LOW
            6: "REMOVE",   # CROSS-BOTH
        }

    def _rules_B(self) -> Dict[int, str]:
        """Region → 'KEEP' or 'REMOVE' for unmatched B grains."""
        return {
            1: "KEEP", # CORE
            2: "KEEP",   # HIGH
            3: "ERROR", # LOW
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
        debug_log_csv: str | None = None,
        angle_convention: str = "bunge",
        angle_type: str = "degrees",
        symmetry: str = "432",
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

        self.debug_log_csv = debug_log_csv
        self._decision_log = []   # list of dict rows

        self.angle_convention = angle_convention
        self.angle_type = angle_type
        self.symmetry = symmetry

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

        if self.debug_log_csv is not None:
            pd.DataFrame(self._decision_log).to_csv(self.debug_log_csv, index=False)

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
            self.matches = pd.DataFrame(columns=[
                "idx_A", "idx_B", "diff_pos_norm2", "diff_rad_percentage", "diff_ori_norm2"
            ])
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

        # Flatten candidate edges: (B_s, A_t)
        s_idx, _ = np.indices(dist.shape)
        s_idx = s_idx.ravel()
        t_idx = idx.ravel()
        diff_pos = dist.ravel().astype(float)

        ori_A = dfA[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
        ori_B = dfB[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
        rad_A = dfA["GrainRadius"].to_numpy(float)
        rad_B = dfB["GrainRadius"].to_numpy(float)

        diff_ori_t = misorientation(
            ori_B[s_idx], ori_A[t_idx],
            angle_convention=self.angle_convention,
            angle_type=self.angle_type,
            symmetry=self.symmetry,
        )
        diff_ori = diff_ori_t.detach().cpu().numpy().astype(float)

        diff_rad = np.abs(rad_B[s_idx] - rad_A[t_idx]) / np.maximum(rad_A[t_idx], 1e-14)

        w_pos = float(self.weights.get("pos", 1.0))
        w_ori = float(self.weights.get("ori", 1.0))
        w_rad = float(self.weights.get("rad", 0.0))

        ok = np.ones_like(diff_pos, dtype=bool)

        use_pos = (w_pos > 0.0) and (self.position_tolerance != -1.0)
        use_ori = (w_ori > 0.0) and (self.orientation_tolerance != -1.0)
        use_rad = (w_rad > 0.0) and (self.radius_tolerance != -1.0)

        ok = np.ones_like(diff_pos, dtype=bool)
        if use_pos: ok &= (diff_pos <= self.position_tolerance)
        if use_ori: ok &= (diff_ori <= self.orientation_tolerance)
        if use_rad: ok &= (diff_rad <= self.radius_tolerance)

        s_idx = s_idx[ok]
        t_idx = t_idx[ok]
        diff_pos = diff_pos[ok]
        diff_ori = diff_ori[ok]
        diff_rad = diff_rad[ok]

        if s_idx.size == 0:
            self.matches = pd.DataFrame(columns=[
                "idx_A", "idx_B", "diff_pos_norm2", "diff_rad_percentage", "diff_ori_norm2"
            ])
            self.unmatched_A = np.arange(nA, dtype=int)
            self.unmatched_B = np.arange(nB, dtype=int)
            return



        ptol = self.position_tolerance if self.position_tolerance > 0 else 1.0
        otol = self.orientation_tolerance if self.orientation_tolerance > 0 else 1.0
        rtol = self.radius_tolerance if self.radius_tolerance > 0 else 1.0

        cost = (
            w_pos * (diff_pos / ptol) +
            w_ori * (diff_ori / otol) +
            w_rad * (diff_rad / rtol)
        )

        # --- for each B, pick best A by min cost ---
        order = np.lexsort((cost, s_idx))
        s_s = s_idx[order]
        t_s = t_idx[order]
        dp_s = diff_pos[order]
        dr_s = diff_rad[order]
        do_s = diff_ori[order]
        c_s  = cost[order]

        best_t = np.full(nB, -1, dtype=int)
        best_dp = np.full(nB, np.inf, dtype=float)
        best_dr = np.full(nB, np.inf, dtype=float)
        best_do = np.full(nB, np.inf, dtype=float)
        best_c  = np.full(nB, np.inf, dtype=float)

        seenB = np.zeros(nB, dtype=bool)
        for s, t, dp, dr, do, cc in zip(s_s, t_s, dp_s, dr_s, do_s, c_s):
            if not seenB[s]:
                best_t[s] = int(t)
                best_dp[s] = float(dp)
                best_dr[s] = float(dr)
                best_do[s] = float(do)
                best_c[s]  = float(cc)
                seenB[s] = True

        chosen_B = np.where(best_t >= 0)[0]
        chosen_A = best_t[chosen_B]
        chosen_c = best_c[chosen_B]

        order2 = np.argsort(chosen_c, kind="mergesort")
        used_A = set()
        keep = []
        for j in order2:
            b = int(chosen_B[j])
            a = int(chosen_A[j])
            if a in used_A:
                continue
            used_A.add(a)
            keep.append(b)

        keep = np.array(keep, dtype=int)

        rows = []
        matched_A = set()
        matched_B = set()
        for b in keep:
            a = int(best_t[b])
            rows.append((a, b, float(best_dp[b]), float(best_dr[b]), float(best_do[b])))
            matched_A.add(a)
            matched_B.add(b)

        self.matches = pd.DataFrame(
            rows,
            columns=["idx_A", "idx_B", "diff_pos_norm2", "diff_rad_percentage", "diff_ori_norm2"],
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
              'MC'       : merge
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

            self._decision_log.append({
                "stage": "matched",
                "idx_A": idx_A, "idx_B": idx_B,
                "rA": rA, "rB": rB, "cell": cell,
                "action": action,
                "outcome_A": outA,
                "outcome_B": outB,
            })

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

                new_row = merge_properties(new_row, rowA, rowB)
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
            dfA.at[dfA.index[idx], "Unmatched_location"] = int(self.region_A[idx])
            decision = self.unmatched_rules.decide("A", region)

            self._decision_log.append({
                "stage": "unmatched",
                "which": "A",
                "idx": int(idx),
                "region": region,
                "decision": decision,
            })
            
            if decision == "KEEP":
                keepA_indices.append(idx)
            if decision == "ERROR":
                raise RuntimeError(f"Points center locates outside of A's region.")

        if keepA_indices:
            keepA_unmatched = dfA.iloc[keepA_indices].copy()
        else:
            keepA_unmatched = pd.DataFrame(columns=dfA.columns)

        # --- apply rules for B ---
        keepB_indices = []
        for idx in sorted(idx_B_set):
            region = int(self.region_B[idx])
            decision = self.unmatched_rules.decide("B", region)
            dfB.at[dfB.index[idx], "Unmatched_location"] = int(self.region_B[idx])

            self._decision_log.append({
                "stage": "unmatched",
                "which": "B",
                "idx": int(idx),
                "region": region,
                "decision": decision,
            })

            if decision == "KEEP":
                keepB_indices.append(idx)
            if decision == "ERROR":
                raise RuntimeError(f"Points center locates outside of B's region.")

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
          - merged             (MC)
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

def merge_properties(new_row, rowA, rowB):
    '''
    Require columns: ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]
    '''
    
    # --- volumes ---
    rA = float(rowA["GrainRadius"])
    rB = float(rowB["GrainRadius"])

    vA = (4.0 / 3.0) * np.pi * rA**3
    vB = (4.0 / 3.0) * np.pi * rB**3
    vT = vA + vB

    for col in ["X", "Y", "Z"]:
        new_row[col] = (vA * rowA[col] + vB * rowB[col]) / vT

    new_row["GrainRadius"] = (3.0 * vT / (4.0 * np.pi)) ** (1.0 / 3.0)

    # orientation, keep one for now
    for col in ["Eul0", "Eul1", "Eul2"]:
        new_row[col] = rowA[col]

    return new_row
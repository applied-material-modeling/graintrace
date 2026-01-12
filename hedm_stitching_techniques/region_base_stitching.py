from email.mime import base
import pandas as pd
import os
from typing import List, Tuple, Dict, Optional
from .pair_stitching_utils import PairwiseStitcher, merge_properties
from .dataclass_utils import ScanMetadata, GrainSet
from orientation_helper import misorientation


class RegionBaseStitching():
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
    ):
        self.scan_files = scan_files            # raw per-scan CSVs
        self.output_csv = output_csv
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.radius_tolerance = radius_tolerance
        self.weights = weights
        self.min_neighbors = min_neighbors

        self.angle_convention = orientation_convention
        self.angle_type = orientation_units
        self.symmetry = symmetry

        self.scans: List[GrainSet] = []         # loaded & sorted
        self.stitched: Optional[GrainSet] = None

    def run(self, zlo: float, zhi: float, overlap_fraction: float) -> GrainSet:
        """
        Entry point.

        Steps:
          1. Validate global z-window (zlo < zhi).
          2. Load scans and sort them.
          3. Compute uniform overlap region (zol, zoh).
          4. Iteratively stitch:
                S0 → S01 → S012 → ... 
          5. Write final stitched output.
        """
        if zhi <= zlo:
            raise ValueError(f"Invalid z-window: zhi ({zhi}) must be > zlo ({zlo}).")

        self.global_zlo = float(zlo)
        self.global_zhi = float(zhi)

        self._load_and_sort_scans()

        all_zmin = min(gs.meta.zmin for gs in self.scans)
        all_zmax = max(gs.meta.zmax for gs in self.scans)

        # if zlo > all_zmin:
        #     raise ValueError(
        #         f"Given zlo={zlo} is ABOVE the actual lowest scan boundary {all_zmin}."
        #     )
        # if zhi < all_zmax:
        #     raise ValueError(
        #         f"Given zhi={zhi} is BELOW the actual highest scan boundary {all_zmax}."
        #     )
        
        # if there is no overlap
        if overlap_fraction == 0.0:
            current = self.scans[0]
            for k in range(len(self.scans) - 1):
                current = self._nonoverlap_stitch_pair(current, self.scans[k + 1], pair_id=k)

            self.stitched = current
            self._write_output(self.stitched)

            return self.stitched
        
        # if there is overlap
        nscan = len(self.scans)
        if nscan == 0:
            raise RuntimeError("No scans available.")
        if nscan == 1:
            # trivial case
            self.stitched = self.scans[0]
            self._write_output(self.stitched)
            return self.stitched
        
        # Pass in zol and zoh of the overlap region
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

            # overlap region between theoretical scan k and k+1
            # ... (I think the math is correct here) ...
            # scan_k:     [zlo + k*z_step, zlo + k*z_step + z_scan_height]
            # scan_k+1:   [zlo + (k+1)*z_step, zlo + (k+1)*z_step + z_scan_height]
            # overlap:    [zlo + (k+1)*z_step, zlo + k*z_step + z_scan_height]
            zol = zlo + (k + 1) * z_step
            zoh = zlo + k * z_step + z_scan_height

            current = self._overlap_stitch_pair(A, B, zol, zoh, pair_id=k)

        self.stitched = current
        self._write_output(self.stitched)
        return self.stitched

    def _load_and_sort_scans(self) -> None:
        """Load all scan CSVs into GrainSet objects and sort by zmin (bottom -> top)."""
        scans: List[GrainSet] = []

        # Columns required by the stitching logic and final output
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

        # sort from bottom (smallest zmin) to top
        scans.sort(key=lambda gs: gs.meta.zmin)

        self.scans = scans

    def _overlap_stitch_pair(self, A: GrainSet, B: GrainSet, zol: float, zoh: float, pair_id: int) -> GrainSet:
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
            f"{stem}_pair{pair_id}_{A.meta.name}_to_{B.meta.name}_decision_log.csv"
        )

        print("Debug log for this pairwise stitching step will be saved to:", debug_log_csv, "\n")

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

    def _nonoverlap_stitch_pair(self, A: GrainSet, B: GrainSet, pair_id: int) -> GrainSet:
        """
        Non-overlap stitching (orientation-first, per-A selection):

        For each A grain in the top slab:
        1) Find k nearest B candidates (KDTree built on B, query with A).
        2) Among those k, pick the one with smallest misorientation within tolerance.
        3) Absurd-distance rejection (using slab scale t).
        4) If multiple A pick same B,
            keep the pair with smallest misorientation.
        """
        from scipy.spatial import cKDTree
        import numpy as np
        import pandas as pd

        dfA, dfB = A.df, B.df
        rA_all = dfA["GrainRadius"].to_numpy(float)
        rB_all = dfB["GrainRadius"].to_numpy(float)

        # slab thickness scale (your heuristic)
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

        # Orientations
        oriA = A_slab[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
        oriB = B_slab[["Eul0", "Eul1", "Eul2"]].to_numpy(float)

        eA = oriA[a_idx]
        eB = oriB[b_idx]

        # Misorientation for each candidate edge
        dori_t = misorientation(
            eA, eB,
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

        if a_idx.size == 0:
            out = pd.concat([dfA, dfB], ignore_index=True)
            meta = ScanMetadata(
                name=f"{A.meta.name}_{B.meta.name}",
                scan_id=-1,
                zmin=float(out["Z"].min()),
                zmax=float(out["Z"].max()),
            )
            return GrainSet(df=out, meta=meta)

        best_b = np.full(nA, -1, dtype=int)
        best_dp = np.full(nA, np.inf, dtype=float)
        best_do = np.full(nA, np.inf, dtype=float)

        order = np.lexsort((dori, a_idx))
        a_s = a_idx[order]
        b_s = b_idx[order]
        dp_s = dpos[order]
        do_s = dori[order]

        seenA = np.zeros(nA, dtype=bool)
        for a, b, dp, do in zip(a_s, b_s, dp_s, do_s):
            if not seenA[a]:
                best_b[a] = int(b)
                best_dp[a] = float(dp)
                best_do[a] = float(do)
                seenA[a] = True

        validA = (best_b >= 0) & (best_dp <= t)

        A_choices = np.where(validA)[0]
        B_choices = best_b[validA]
        DO_choices = best_do[validA]

        if A_choices.size == 0:
            out = pd.concat([dfA, dfB], ignore_index=True)
            meta = ScanMetadata(
                name=f"{A.meta.name}_{B.meta.name}",
                scan_id=-1,
                zmin=float(out["Z"].min()),
                zmax=float(out["Z"].max()),
            )
            return GrainSet(df=out, meta=meta)

        order2 = np.argsort(DO_choices, kind="mergesort")
        matched_B = set()
        keep_pairs_local = []

        for j in order2:
            a = int(A_choices[j])
            b = int(B_choices[j])
            if b in matched_B:
                continue
            keep_pairs_local.append((a, b))
            matched_B.add(b)

        A_slab_idx = A_slab.index.to_numpy()
        B_slab_idx = B_slab.index.to_numpy()

        matched_A_global = set()
        matched_B_global = set()
        merged_rows = []

        for a_loc, b_loc in keep_pairs_local:
            idxA = int(A_slab_idx[a_loc])  # global row index in dfA
            idxB = int(B_slab_idx[b_loc])  # global row index in dfB

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

    def _write_output(self, stitched: GrainSet) -> None:
        """
        Write final stitched dataframe to self.output_csv.
        Required output columns:
            X, Y, Z, GrainRadius, Eul0, Eul1, Eul2, ScanID
        """

        df = stitched.df.copy()

        required = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2", "ScanID"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Stitched dataframe missing required columns: {missing}")

        debug_cols = ["Matched_when_merged", "Cell", "Unmatched_location"]
        for c in debug_cols:
            if c in df.columns:
                required.append(c)

        df = df[required]
        df.to_csv(self.output_csv, index=False)


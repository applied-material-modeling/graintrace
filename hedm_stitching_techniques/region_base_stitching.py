import pandas as pd
import os
from typing import List, Tuple, Dict, Optional
from .pair_stitching_utils import PairwiseStitcher
from .dataclass_utils import ScanMetadata, GrainSet

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
    ):
        self.scan_files = scan_files            # raw per-scan CSVs
        self.output_csv = output_csv
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.radius_tolerance = radius_tolerance
        self.weights = weights
        self.min_neighbors = min_neighbors

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

        if zlo > all_zmin:
            raise ValueError(
                f"Given zlo={zlo} is ABOVE the actual lowest scan boundary {all_zmin}."
            )
        if zhi < all_zmax:
            raise ValueError(
                f"Given zhi={zhi} is BELOW the actual highest scan boundary {all_zmax}."
            )
        
        # concatenation if there is no overlap
        if overlap_fraction == 0.0:
            print("overlap_fraction=0.0: performing simple concatenation of scans. zlo and zhi are ignored.")
            # concatenate all scans (already sorted)
            df_all = pd.concat([gs.df for gs in self.scans], ignore_index=True)
            # metadata for stitched set: global z-range
            stitched_meta = ScanMetadata(
                name="stitched",
                scan_id=-1,
                zmin=all_zmin,
                zmax=all_zmax,
            )
            self.stitched = GrainSet(df=df_all, meta=stitched_meta)
            self._write_output(self.stitched)
            return self.stitched
        
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

            current = self._stitch_pair(A, B, zol, zoh)

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

    def _stitch_pair(self, A: GrainSet, B: GrainSet, zol: float, zoh: float) -> GrainSet:
        """
        Apply the pairwise stiching algorithm to GrainSets A and B
        over the overlap region [zol, zoh].
        """
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
        )

        return stitcher.run()

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

        df = df[required]
        df.to_csv(self.output_csv, index=False)


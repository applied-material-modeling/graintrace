from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass
class ScanMetadata:
    name: str               # e.g. "scan_0"
    scan_id: int            # integer index in original stack
    zmin: float             # min Z in this scan
    zmax: float             # max Z in this scan

@dataclass
class GrainSet:
    """
    Represents a set of grains (either a raw scan or a stitched aggregate).
    """
    df: pd.DataFrame        # columns: X,Y,Z,GrainRadius,Eul0,Eul1,Eul2
    meta: ScanMetadata
from __future__ import annotations

from typing import List
import os
import pandas as pd

class NaiveStitching():
    def __init__(self, scan_files: list[str], output_csv: str):
        """
        Parameters
        ----------
        scan_files : list of str
            List of CSV files from individual scans.
        output_csv : str
            Path for the stitched combined CSV.
        """
        self.scan_files = scan_files
        self.output_csv = output_csv

    def run(self) -> None:
        """Combine all scan CSVs in order."""
        dataframes = []
        for i, f in enumerate(self.scan_files):
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing scan file: {f}")
            df = pd.read_csv(f)
            df["ScanID"] = i
            dataframes.append(df)

        df_all = pd.concat(dataframes, ignore_index=True)
        df_all.to_csv(self.output_csv, index=False)
        print(f"Naive stitching complete. \nOutput saved to {self.output_csv}\n")
        return df_all

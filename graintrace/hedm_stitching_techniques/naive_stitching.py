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

"""Naive HEDM scan stitching: concatenate scan CSVs in order."""

from __future__ import annotations

import os

import pandas as pd


class NaiveStitching:
    """Combine per-scan HEDM CSVs by simple ordered concatenation."""

    def __init__(self, scan_files: list[str], output_csv: str):
        """scan_files: per-scan CSVs. output_csv: path for combined output."""
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

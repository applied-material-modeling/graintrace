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

"""Load and query experimental (FF/NF/EBSD) field data (ExperimentResults)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class FieldFileNaming:
    """Naming convention for per-step CSVs (identifier is between sep and suffix)."""

    def __init__(
        self,
        prefix: str,
        index_width: Optional[int] = None,
        sep: str = "_",
        suffix: str = ".csv",
    ) -> None:
        self.prefix = prefix
        self.index_width = index_width
        self.sep = sep
        self.suffix = suffix


class ExperimentResults:
    """
    Far-field HEDM experiment results (one CSV per load/time step).
    """

    def __init__(
        self,
        exp_dir: Union[str, Path],
        exp_naming: FieldFileNaming,
        step_to_time: Optional[Union[Dict, Callable]] = None,
    ) -> None:

        self.exp_dir = Path(exp_dir).expanduser().resolve()
        self.exp_naming = exp_naming

        self.step_to_time = step_to_time

        self.grain_files = None
        self.step_ids = None
        self.time = None
        self.n_steps = None

        self.grain_ids = None
        self._grain_row_map = None  # dict: step_id -> dict(grain_id -> row_idx)

        self.check_input()
        self._run_grain_tracking()

        self._block_df = pd.DataFrame({"time": self.time})

    def check_input(self) -> None:
        """Discover experiment CSVs, order the step ids, and derive times."""
        if not self.exp_dir.exists() or not self.exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {self.exp_dir}")

        prefix = re.escape(self.exp_naming.prefix)
        sep = re.escape(self.exp_naming.sep)
        suffix = re.escape(self.exp_naming.suffix)

        rx = re.compile(rf"^{prefix}{sep}(.+){suffix}$")

        step_map = {}
        for p in self.exp_dir.iterdir():
            if not p.is_file():
                continue
            m = rx.match(p.name)
            if not m:
                continue
            grain_id = m.group(1)
            step_map[grain_id] = p

        if not step_map:
            raise FileNotFoundError(
                f"No experiment CSVs found in {self.exp_dir} matching "
                f"'{self.exp_naming.prefix}{self.exp_naming.sep}<id>{self.exp_naming.suffix}'."
            )

        def _sort_key(s):
            try:
                return (0, float(s))
            except (ValueError, TypeError):
                return (1, s)

        self.step_ids = sorted(step_map.keys(), key=_sort_key)
        self.grain_files = {sid: step_map[sid] for sid in self.step_ids}
        self.n_steps = len(self.step_ids)

        times = []
        if self.step_to_time is None:
            for i, sid in enumerate(self.step_ids):
                try:
                    times.append(float(sid))
                except (ValueError, TypeError):
                    times.append(float(i))
        elif callable(self.step_to_time):
            for sid in self.step_ids:
                times.append(float(self.step_to_time(sid)))
        else:
            for sid in self.step_ids:
                if sid not in self.step_to_time:
                    raise KeyError(f"step_to_time missing mapping for grain_id='{sid}'")
                times.append(float(self.step_to_time[sid]))

        self.time = pd.Series(times, name="time")

    def load_data(self, grain_id: str) -> pd.DataFrame:
        """Read and return the field CSV for the given step/grain id."""
        if grain_id not in self.grain_files:
            raise KeyError(f"Unknown grain_id='{grain_id}'. Available: {self.step_ids}")
        df = pd.read_csv(self.grain_files[grain_id])
        df.columns = [c.strip() for c in df.columns]
        return df

    def _tensor_from_df(
        self,
        df: pd.DataFrame,
        tensor_prefix: str,
        order: int,
        suffix: str = "",
        return_comp_names: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        if order not in (0, 1, 2):
            raise ValueError("order must be 0, 1, or 2")

        def require(cols, where):
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns for {where}: {missing}")

        if order == 0:
            col = f"{tensor_prefix}{suffix}"
            require([col], "order-0 scalar")
            data = df[[col]].to_numpy()
            comp_names = [tensor_prefix]
            return (data, comp_names) if return_comp_names else data

        if order == 1:
            cols = [
                f"{tensor_prefix}0{suffix}",
                f"{tensor_prefix}1{suffix}",
                f"{tensor_prefix}2{suffix}",
            ]
            require(cols, "order-1 vector")
            data = df[cols].to_numpy()
            comp_names = [f"{tensor_prefix}0", f"{tensor_prefix}1", f"{tensor_prefix}2"]
            return (data, comp_names) if return_comp_names else data

        full = ["11", "12", "13", "21", "22", "23", "31", "32", "33"]
        full_cols = [f"{tensor_prefix}{ij}{suffix}" for ij in full]
        if all(c in df.columns for c in full_cols):
            data = df[full_cols].to_numpy()
            comp_names = [f"{tensor_prefix}{ij}" for ij in full]
            return (data, comp_names) if return_comp_names else data

        sym = ["xx", "xy", "xz", "yy", "yz", "zz"]
        sym_cols = [f"{tensor_prefix}{c}{suffix}" for c in sym]
        require(sym_cols, "order-2 symmetric tensor")

        a = df[sym_cols].to_numpy()
        data = np.column_stack(
            [
                a[:, 0],
                a[:, 1],
                a[:, 2],
                a[:, 1],
                a[:, 3],
                a[:, 4],
                a[:, 2],
                a[:, 4],
                a[:, 5],
            ]
        )
        comp_names = [
            f"{tensor_prefix}xx",
            f"{tensor_prefix}xy",
            f"{tensor_prefix}xz",
            f"{tensor_prefix}yx",
            f"{tensor_prefix}yy",
            f"{tensor_prefix}yz",
            f"{tensor_prefix}zx",
            f"{tensor_prefix}zy",
            f"{tensor_prefix}zz",
        ]
        return (data, comp_names) if return_comp_names else data

    def get_tensor_block(
        self,
        tensor_prefix: str,
        order: int,
        sample: str,
        grain_id: Optional[int] = None,
        block_id: Optional[int] = None,
        return_comp_names: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        sample='id'   -> requires block_id (step index), returns (n_grains_in_step, 1|3|9)
        sample='time' -> requires grain_id, returns (n_steps, 1|3|9) using tracking map
        """
        if sample not in ("time", "id"):
            raise ValueError("sample must be 'time' or 'id'")

        if sample == "id":
            if block_id is None:
                raise ValueError("block_id must be provided when sample='id'")
            if block_id < 0 or block_id >= self.n_steps:
                raise IndexError(f"block_id out of range: {block_id}")

            step_id = self.step_ids[block_id]
            df = self.load_data(step_id)
            return self._tensor_from_df(
                df, tensor_prefix, order, suffix="", return_comp_names=return_comp_names
            )

        if grain_id is None:
            raise ValueError("grain_id must be provided when sample='time'")
        if not isinstance(grain_id, int):
            raise TypeError("grain_id must be an int")

        data_list = []
        comp_names = None

        for sid in self.step_ids:
            row_idx = self._grain_row_map[sid].get(grain_id, None)
            if row_idx is None:
                # missing grain in this step
                if order == 0:
                    d = np.array([[np.nan]])
                    cn = [tensor_prefix]
                elif order == 1:
                    d = np.full((1, 3), np.nan)
                    cn = [f"{tensor_prefix}x", f"{tensor_prefix}y", f"{tensor_prefix}z"]
                else:
                    d = np.full((1, 9), np.nan)
                    cn = None

                if return_comp_names and comp_names is None and cn is not None:
                    comp_names = cn

                data_list.append(d[0])
                continue

            df = self.load_data(sid)
            row = df.iloc[[row_idx]]

            out = self._tensor_from_df(
                row,
                tensor_prefix,
                order,
                suffix="",
                return_comp_names=return_comp_names,
            )
            if return_comp_names:
                d, cn = out
                if comp_names is None:
                    comp_names = cn
            else:
                d = out

            data_list.append(d[0])

        data = np.asarray(data_list)

        if return_comp_names:
            return data, comp_names
        return data

    def get_tensor_element(self, *args: object, **kwargs: object) -> None:
        """Not implemented: near-field experiment tensor access."""
        raise NotImplementedError("Near-field experiment support not yet implemented.")

    def _run_grain_tracking(self) -> None:
        """Assign grain IDs based on row indices in the first step."""
        first_step = self.step_ids[0]
        df0 = self.load_data(first_step)
        n0 = df0.shape[0]

        self.grain_ids = list(range(n0))

        row_map = {}
        for sid in self.step_ids:
            df = self.load_data(sid)
            n = df.shape[0]
            m = {}
            for gid in self.grain_ids:
                if gid < n:
                    m[gid] = gid
            row_map[sid] = m

        self._grain_row_map = row_map

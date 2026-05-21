from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class FieldFileNaming:
    """
    Naming convention for per-time-step field CSVs, no path for prefix.

    Example:
        prefix="xxxxx"
        index_width=4   -> xxxxx_0000.csv
        index_width=None -> xxxxx_0.csv
    """

    def __init__(self, prefix: str, index_width: Optional[int] = None, sep: str = "_", suffix: str = ".csv") -> None:
        self.prefix = prefix
        self.index_width = index_width
        self.sep = sep
        self.suffix = suffix


class SimulationResults:

    def __init__(
        self,
        block_csv: Union[str, Path],
        field_dir: Union[str, Path],
        field_naming: FieldFileNaming,
    ) -> None:

        self.block_csv = Path(block_csv).expanduser().resolve()
        self.field_dir = Path(field_dir).expanduser().resolve()
        self.field_naming = field_naming

        self._block_df = None

        self.n_steps = None
        self.time = None
        self.grain_ids = None

        self.field_files = None

        self.check_input()

    def check_input(self) -> None:

        if not self.block_csv.exists() or not self.block_csv.is_file():
            raise FileNotFoundError(
                f"Block (per grain) properties CSV not found: {self.block_csv}"
            )

        block = self.load_block_data()

        if "time" not in block.columns:
            raise ValueError("Block CSV must contain a 'time' column.")

        self.n_steps = block.shape[0]
        self.time = block["time"]

        grain_id_pattern = re.compile(r".+_(\d+)$")
        grain_ids = set()

        for col in block.columns:
            if col == "time":
                continue
            m = grain_id_pattern.match(col)
            if m:
                grain_ids.add(int(m.group(1)))

        if not grain_ids:
            raise ValueError(
                "No per-grain columns found. Expected '<field>_<grainId>' pattern."
            )

        self.grain_ids = sorted(grain_ids)

        if not self.field_dir.exists() or not self.field_dir.is_dir():
            raise FileNotFoundError(
                f"Field (per element) properties directory not found: {self.field_dir}"
            )

        prefix = re.escape(self.field_naming.prefix)
        sep = re.escape(self.field_naming.sep)
        suffix = re.escape(self.field_naming.suffix)
        rx = re.compile(rf"^{prefix}{sep}(\d+){suffix}$")

        field_map = {}
        for p in self.field_dir.iterdir():
            if not p.is_file():
                continue
            m = rx.match(p.name)
            if not m:
                continue

            idx = int(m.group(1))
            if idx < 0 or idx >= self.n_steps:
                raise ValueError(
                    f"Field file index {idx} out of range [0, {self.n_steps - 1}]: {p.name}"
                )
            field_map[idx] = p

        if not field_map:
            raise FileNotFoundError(
                f"No field CSVs found matching "
                f"'{self.field_naming.prefix}{self.field_naming.sep}<index>{self.field_naming.suffix}'."
            )

        self.field_files = dict(sorted(field_map.items()))

    def load_block_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.block_csv)
        df.columns = [c.strip() for c in df.columns]
        self._block_df = df
        return df

    def load_field_data(self, block_row_idx: int) -> pd.DataFrame:

        if not isinstance(block_row_idx, int):
            raise TypeError("block_row_idx must be an int")

        if block_row_idx not in self.field_files:
            raise KeyError(
                f"No field data for block_row_idx={block_row_idx}. "
                f"Available indices: {list(self.field_files.keys())}"
            )

        df = pd.read_csv(self.field_files[block_row_idx])
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
        """
        df: pandas DataFrame
        suffix: "" for element df; for block df use f"_{grain_id}" or f"_{gid}"
        Returns:
          order=0 -> (N, 1)
          order=1 -> (N, 3)
          order=2 -> (N, 9)  (full 11..33 preferred, else symmetric xx..zz expanded)
          comp_names : component names for plotting labeling afterwards
        """
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
                f"{tensor_prefix}_x{suffix}",
                f"{tensor_prefix}_y{suffix}",
                f"{tensor_prefix}_z{suffix}",
            ]
            require(cols, "order-1 vector")
            data = df[cols].to_numpy()
            comp_names = [
                f"{tensor_prefix}_x",
                f"{tensor_prefix}_y",
                f"{tensor_prefix}_z",
            ]
            return (data, comp_names) if return_comp_names else data

        # if order == 2
        full = ["11", "12", "13", "21", "22", "23", "31", "32", "33"]
        full_cols = [f"{tensor_prefix}_{ij}{suffix}" for ij in full]
        if all(c in df.columns for c in full_cols):
            data = df[full_cols].to_numpy()
            comp_names = [f"{tensor_prefix}_{ij}" for ij in full]
            return (data, comp_names) if return_comp_names else data

        sym = ["xx", "xy", "xz", "yy", "yz", "zz"]
        sym_cols = [f"{tensor_prefix}_{c}{suffix}" for c in sym]
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
            f"{tensor_prefix}_xx",
            f"{tensor_prefix}_xy",
            f"{tensor_prefix}_xz",
            f"{tensor_prefix}_yx",
            f"{tensor_prefix}_yy",
            f"{tensor_prefix}_yz",
            f"{tensor_prefix}_zx",
            f"{tensor_prefix}_zy",
            f"{tensor_prefix}_zz",
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
        Block (per-grain) extraction.

        sample='time' -> requires grain_id, returns (n_times, 1|3|9)
        sample='id'   -> requires block_id, returns (n_grains, 1|3|9)
        """
        if sample not in ("time", "id"):
            raise ValueError("sample must be 'time' or 'id'")

        block = self._block_df
        if block is None:
            raise RuntimeError("Block dataframe not loaded.")

        if sample == "time":
            if grain_id is None:
                raise ValueError("grain_id must be provided when sample='time'")
            if grain_id not in self.grain_ids:
                raise KeyError(f"grain_id {grain_id} not found in block data")

            suffix = f"_{grain_id}"

            return self._tensor_from_df(
                block,
                tensor_prefix,
                order,
                suffix=suffix,
                return_comp_names=return_comp_names,
            )

        # if sample == "id"
        if block_id is None:
            raise ValueError("block_id must be provided when sample='id'")
        if block_id < 0 or block_id >= self.n_steps:
            raise IndexError(f"block_id out of range: {block_id}")

        one_row = block.loc[[block_id], :]
        data_list = []
        comp_names = None

        for gid in self.grain_ids:
            out = self._tensor_from_df(
                one_row,
                tensor_prefix,
                order,
                suffix=f"_{gid}",
                return_comp_names=return_comp_names,
            )
            if return_comp_names:
                d, comp_names = out
            else:
                d = out
            data_list.append(d[0])

        data = np.asarray(data_list)

        if return_comp_names:
            return data, comp_names
        return data

    def get_tensor_element(
        self,
        tensor_prefix: str,
        order: int,
        sample: str,
        element_id: Optional[int] = None,
        block_id: Optional[int] = None,
        return_comp_names: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Element (per-point/per-element) extraction.

        sample='id'   -> requires block_id, returns (n_elements, 1|3|9) from that field CSV
        sample='time' -> requires element_id, returns (n_available_times, 1|3|9) by scanning field CSVs
        """
        if sample not in ("time", "id"):
            raise ValueError("sample must be 'time' or 'id'")

        if sample == "id":
            if block_id is None:
                raise ValueError("block_id must be provided when sample='id'")
            if not isinstance(block_id, int):
                raise TypeError("block_id must be an int")

            df = self.load_field_data(block_id)
            return self._tensor_from_df(
                df,
                tensor_prefix,
                order,
                suffix="",
                return_comp_names=return_comp_names,
            )

        # sample == "time"
        if element_id is None:
            raise ValueError("element_id must be provided when sample='time'")

        data_list = []
        comp_names = None

        for bid in sorted(self.field_files.keys()):
            df = self.load_field_data(bid)

            if "id" not in df.columns:
                raise KeyError(f"Field CSV for block_id={bid} has no 'id' column.")

            row = df.loc[df["id"] == element_id]
            if row.shape[0] == 0:
                raise KeyError(
                    f"element_id={element_id} not found in field file for block_id={bid}"
                )
            if row.shape[0] > 1:
                raise ValueError(
                    f"element_id={element_id} appears multiple times in field file for block_id={bid}"
                )

            out = self._tensor_from_df(
                row,
                tensor_prefix,
                order,
                suffix="",
                return_comp_names=return_comp_names,
            )
            if return_comp_names:
                d, comp_names = out
            else:
                d = out
            data_list.append(d[0])

        data = np.asarray(data_list)

        if return_comp_names:
            return data, comp_names
        return data

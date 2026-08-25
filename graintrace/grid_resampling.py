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

"""Offline resampling of a saved CPFE Exodus onto a regular grid via MOOSE.

When a CPFE run is done with ``grid_transfer="off"`` (or ``"final"``), the per-step
regular-grid field output is skipped to save the expensive MultiApp transfers. This
module regenerates that gridded output afterward from the native-mesh Exodus
(``sim_output.e``) using the same MOOSE shape-function-evaluation transfer: a no-solve
sub-app (``resample_source.i``) loads the CPFE fields at a chosen Exodus timestep, and
the main app (``resample_grid.i``) shape-evaluates them onto a ``GeneratedMesh`` grid and
writes ``grid_out/out_element_centroid_<idx>.csv`` — the exact schema consumed by
``SimulationResults`` / ``IdentifyRareClusters`` / ``GraphSpatialCluster``.

This is a cheap, **smoothed** approximation of the online per-step grid: the CPFE fields
are order-FIRST ``MONOMIAL`` and MOOSE stores them in the Exodus as nodal projections, so
grain-boundary extremes are compressed. For extreme-sensitive REI use the crisp online grid
(``grid_transfer="per_step"``) or run REI on the true mesh; a denser resample grid recovers
spatial detail but not the extremes lost to the nodal projection.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


def _package_lib_dirs():
    """Lib dirs of the installed neml2 + torch packages (libneml2_eager.so / libtorch).

    puma-opt links these even for a solve-free resample, so they must be resolvable.
    """
    import importlib  # pylint: disable=import-outside-toplevel

    dirs = []
    for mod in ("neml2", "torch"):
        try:
            pkg = importlib.import_module(mod)
        except ImportError:
            continue
        lib = Path(pkg.__file__).parent / "lib"
        if lib.is_dir():
            dirs.append(str(lib))
    return dirs


class GridResampler:
    """Resample a saved CPFE Exodus onto a regular grid, offline, via ``puma-opt``.

    Drives the ``resample_grid.i`` + ``resample_source.i`` templates once per requested
    Exodus timestep, emitting ``grid_out/out_element_centroid_<idx4>.csv`` under
    ``save_dir`` (indices 0-based, matching ``FieldFileNaming``). See
    ``examples/demonstrate_postprocess.py`` and the ``/post-processing`` skill.
    """

    def __init__(
        self,
        cpfe_exodus,
        save_dir,
        number_of_elements,
        bounding_box,
        moose_run_file,
        launcher="mpiexec",
        extra_ld_library_paths=None,
        ncore=1,
    ):
        self.cpfe_exodus = Path(cpfe_exodus).resolve()
        if not self.cpfe_exodus.exists():
            raise FileNotFoundError(f"CPFE Exodus not found: {self.cpfe_exodus}")

        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if len(number_of_elements) != 3:
            raise ValueError("number_of_elements must be [nx, ny, nz].")
        self.number_of_elements = [int(n) for n in number_of_elements]

        if len(bounding_box) != 6:
            raise ValueError("bounding_box must be [xlo, xhi, ylo, yhi, zlo, zhi].")
        self.bounding_box = [float(b) for b in bounding_box]

        self.moose_run_file = Path(moose_run_file).resolve()
        if not self.moose_run_file.exists():
            raise FileNotFoundError(f"MOOSE run file not found: {self.moose_run_file}")

        self.launcher = launcher
        self.extra_ld_library_paths = extra_ld_library_paths
        self.ncore = int(ncore)

    def num_timesteps(self):
        """Number of time steps stored in the CPFE Exodus (via the netCDF reader)."""
        from scipy.io import netcdf_file  # pylint: disable=import-outside-toplevel

        try:
            with netcdf_file(str(self.cpfe_exodus), "r", mmap=False) as ncf:
                tw = ncf.variables.get("time_whole")
                return int(tw.shape[0]) if tw is not None else 0
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Could not read timesteps from {self.cpfe_exodus}: {exc}"
            ) from exc

    def _runtime_env(self):
        """Env for puma-opt: neml2/torch (+ optional libtorch/PETSc) on LD_LIBRARY_PATH."""
        env = os.environ.copy()
        paths = []
        if self.extra_ld_library_paths:
            paths.extend(str(p) for p in self.extra_ld_library_paths)
        else:
            root = self.moose_run_file.parent.parent
            for cand in (
                root / "libtorch" / "lib",
                root / "moose" / "petsc" / "arch-moose" / "lib",
            ):
                if cand.is_dir():
                    paths.append(str(cand))
        paths.extend(_package_lib_dirs())
        if paths:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                paths + ([existing] if existing else [])
            )
        return env

    def resample(self, timesteps="all"):
        """Resample the requested Exodus timesteps to grid CSVs; return their paths.

        timesteps: "all" (every stored step) or a list of 1-based Exodus timestep
        indices (MOOSE ``initial_from_file_timestep`` convention). Output CSVs are
        numbered 0-based under ``save_dir/grid_out/``.
        """
        if timesteps == "all":
            n = self.num_timesteps()
            if n <= 0:
                raise RuntimeError(f"No timesteps found in {self.cpfe_exodus}")
            ts_indices = list(range(1, n + 1))
        else:
            ts_indices = [int(t) for t in timesteps]

        cpfe_base = Path(__file__).parent / "cpfe_base"
        for fname in ("resample_grid.i", "resample_source.i"):
            shutil.copy(cpfe_base / fname, self.save_dir / fname)

        grid_out = self.save_dir / "grid_out"
        grid_out.mkdir(parents=True, exist_ok=True)

        nx, ny, nz = self.number_of_elements
        xlo, xhi, ylo, yhi, zlo, zhi = self.bounding_box
        launcher = shlex.split(str(self.launcher))
        env = self._runtime_env()

        outputs = []
        for out_idx, ts in enumerate(ts_indices):
            run_base = f"resample_ts{ts:04d}"
            (self.save_dir / run_base).mkdir(parents=True, exist_ok=True)
            argv = [
                *launcher,
                "-n",
                str(self.ncore),
                str(self.moose_run_file),
                "-i",
                "resample_grid.i",
                f"cpfe_exodus={self.cpfe_exodus}",
                f"cpfe_timestep={ts}",
                f"base_folder={run_base}",
                f"grid_nx={nx:d}",
                f"grid_ny={ny:d}",
                f"grid_nz={nz:d}",
                f"grid_min_x={xlo:.12g}",
                f"grid_max_x={xhi:.12g}",
                f"grid_min_y={ylo:.12g}",
                f"grid_max_y={yhi:.12g}",
                f"grid_min_z={zlo:.12g}",
                f"grid_max_z={zhi:.12g}",
            ]
            log_path = self.save_dir / f"{run_base}.log"
            print(f"\n==> resample timestep {ts} -> {run_base}", flush=True)
            with open(log_path, "w", encoding="utf-8") as logf:
                subprocess.run(
                    argv,
                    cwd=self.save_dir,
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

            produced = sorted(
                (self.save_dir / run_base / "grid_out").glob(
                    "out_element_centroid_*.csv"
                )
            )
            if not produced:
                raise RuntimeError(
                    f"resample produced no grid CSV for timestep {ts}; see {log_path}"
                )
            dst = grid_out / f"out_element_centroid_{out_idx:04d}.csv"
            shutil.move(str(produced[-1]), str(dst))
            outputs.append(dst)
            print(f"    -> {dst}")

        return outputs

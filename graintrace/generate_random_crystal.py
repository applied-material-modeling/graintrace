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

"""Generate synthetic crystal tessellations with NEPER and simulate HEDM z-scans."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from textwrap import dedent
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class CrystalGenerator:
    """Build NEPER Voronoi tessellations and derive HEDM-like scan CSVs from them."""

    _MORPHO_SCHEMA = {
        "gg": {"required": ["mean"], "optional": []},
        "lamellar": {"required": ["n", "v"], "optional": []},
        "columnar": {"required": ["n", "v"], "optional": []},
        "bamboo": {"required": ["n", "v"], "optional": []},
        "diameq": {"required": ["distribution", "params"], "optional": []},
        "size": {"required": ["distribution", "params"], "optional": []},
        # 'raw' = literal NEPER -morpho string (e.g. bi-modal mixtures)
        "raw": {"required": ["morpho_str"], "optional": ["n"]},
    }

    _DISTRIBUTION_SCHEMA = {
        "normal": ["mean", "sigma"],
        "lognormal": ["mean", "sigma"],
        "dirac": ["mean"],
        "beta": ["x", "y"],
        "lorentzian": ["mean", "sigma"],
        "studentst": ["mean", "sigma"],
        "weibull": ["k", "sigma"],
        "breitwigner": ["mean", "sigma", "gamma"],
        "expnormal": ["mean", "sigma", "gamma"],
        "moffat": ["mean", "sigma", "gamma"],
        "pearson7": ["mean", "sigma", "gamma"],
        "pseudovoigt": ["mean", "sigma", "gamma"],
        "skewnormal": ["mean", "sigma", "gamma"],
    }

    def __init__(
        self,
        output_dir,
        bounding_box,
        dim=3,
        neper_version="4.10.1",
        seed=12345,
        env=None,
    ):

        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        expected_len = 6 if dim == 3 else 4 if dim == 2 else None

        if expected_len is None:
            raise ValueError("dim must be either 2 or 3.")

        self.bounding_box = np.array(bounding_box, dtype=float)

        if self.bounding_box.size != expected_len:
            raise ValueError(
                f"For dim={dim}, bounding_box must have {expected_len} values "
                f"(min/max pairs per axis)."
            )

        self.dim = int(dim)
        self.neper_version = neper_version
        self.seed = int(seed)

        self.env = self.check_dependencies() if env is None else env

    def _build_morpho(self, morpho_arg: dict) -> str:
        """Construct the Neper -morpho string from a validated morphology dictionary."""

        self.validate_morpho(morpho_arg)

        mtype = morpho_arg["type"]

        if mtype == "gg":
            return f"gg({morpho_arg['mean']})"

        if mtype == "lamellar":
            return f"lamellar(n={morpho_arg['n']},v={morpho_arg['v']})"

        if mtype in ["columnar", "bamboo"]:
            return f"{mtype}({morpho_arg['v']})"

        if mtype in ["size", "diameq"]:
            dist_name = morpho_arg["distribution"]
            params = morpho_arg["params"]
            self.validate_distribution(dist_name, params)
            param_str = ",".join(map(str, params))
            return f"{mtype}:{dist_name}({param_str})"

        # mtype == "raw" (validate_morpho guarantees a known type)
        return morpho_arg["morpho_str"]

    def generate_tessellation(
        self,
        morpho_args: Dict[str, Any],
        iterations: int = 10,
        extra_neper_args: Optional[List[str]] = None,
    ) -> None:
        """Run neper -T. `extra_neper_args` are appended verbatim (e.g. ['-reg','1'])."""
        morpho_str = self._build_morpho(morpho_args)

        # compute domain
        if self.dim == 2:
            xmin, xmax, ymin, ymax = self.bounding_box
            sx, sy = xmax - xmin, ymax - ymin
            tx, ty = xmin, ymin
            domain_arg = f"square({sx},{sy}):translate({tx},{ty})"
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounding_box
            sx, sy, sz = xmax - xmin, ymax - ymin, zmax - zmin
            tx, ty, tz = xmin, ymin, zmin
            domain_arg = f"cube({sx},{sy},{sz}):translate({tx},{ty},{tz})"

        mtype = morpho_args["type"]

        if mtype in ("gg", "size", "diameq", "lamellar", "raw"):
            # 'raw' with no size term needs an explicit -n via morpho_args["n"]
            if mtype == "raw" and "n" in morpho_args:
                n_arg = ["-n", str(int(morpho_args["n"]))]
            else:
                n_arg = ["-n", "from_morpho"]

        elif mtype in ("columnar", "bamboo"):
            n_arg = ["-n", str(int(morpho_args["n"]))]
        else:
            raise ValueError(f"Unsupported morpho type: {mtype}")

        tess_name = os.path.join(self.output_dir, "voronoi")
        output_cell = "id,vol,w,x,y,z,radeq"
        neper_cmd = [
            "neper",
            "-T",
            "-id",
            str(self.seed),
            "-dim",
            str(self.dim),
            "-domain",
            domain_arg,
            "-oridescriptor",
            "euler-bunge",
            "-o",
            tess_name,
            "-statcell",
            output_cell,
            "-format",
            "tess,geo,ori",
            "-morpho",
            morpho_str,
            "-morphooptistop",
            f"iter={iterations}",
        ] + n_arg

        if extra_neper_args:
            neper_cmd.extend(extra_neper_args)

        print("\n=== Running Neper Tessellation ===\n")
        print("> " + " ".join(neper_cmd))

        subprocess.run(
            neper_cmd,
            cwd=self.output_dir,
            env=self.env,
            check=True,
        )

        CrystalGenerator.write_to_csv(
            stat_file=os.path.join(self.output_dir, "voronoi.stcell"),
            ori_file=os.path.join(self.output_dir, "voronoi.ori"),
            output_csv=os.path.join(self.output_dir, "voronoi.csv"),
        )

    def hedm_zscan(
        self,
        tess_file: str,
        nstep: int,
        overlap_percentage: float,
        output_hedm: str = "hedm_scan",
        verbose: bool = False,
        apply_noise: bool = False,
        apply_noise_method: str = "gaussian",
        noise_level: float = 1e-4,
        remove_minimum_volume: bool = False,
        min_vol: float = 0.0,
    ):
        """Generate overlapping z-direction tessellation cuts to simulate HEDM scans.

        nstep slices span the domain z-extent; overlap_percentage (0-100) sets the
        overlap between adjacent scans; output_hedm is the output subfolder name."""
        if not os.path.exists(tess_file):
            raise FileNotFoundError(f"Tessellation file not found: {tess_file}")

        output_hedm_dir = os.path.join(self.output_dir, output_hedm)
        if os.path.exists(output_hedm_dir):
            shutil.rmtree(output_hedm_dir)
        os.makedirs(output_hedm_dir, exist_ok=True)

        # scanning window geometry
        _xmin, _xmax, _ymin, _ymax, zmin, zmax = self.bounding_box
        total_height = zmax - zmin
        overlap_fraction = overlap_percentage / 100.0
        z_scan_height = total_height / (nstep - (nstep - 1) * overlap_fraction)
        # starting from zmin, each next scan shifts by z_step
        z_step = z_scan_height * (1 - overlap_fraction)

        scans = []
        z_current = zmin
        eps = (zmax - zmin) * 1e-4
        for i in range(nstep):
            zlo = z_current
            if i == 0:
                zlo = zlo - eps
            zhi = z_current + z_scan_height
            if i == nstep - 1:
                zhi = zhi + eps
            scans.append((zlo, zhi))
            z_current += z_step
            if zhi >= zmax:
                break

        raw_folder = os.path.join(output_hedm_dir, "raw_files")
        os.makedirs(raw_folder, exist_ok=True)

        print(f"\n=== Generating {len(scans)} HEDM z-scans from {tess_file} ===\n")
        print(
            f"z_scan_height = {z_scan_height:.3f}, overlap = {overlap_percentage:.1f}%, z_step = {z_step:.3f}\n"
        )

        # crop each slice
        for i, (zlo, zhi) in enumerate(scans):

            out_csv_name = os.path.join(output_hedm_dir, f"scan_{i}")
            out_name = os.path.join(raw_folder, f"scan_{i}")

            print(f"--- Scan {i}: z = [{zlo:.3f}, {zhi:.3f}] ---")

            CrystalGenerator.crop_tessellation(
                tess_name=tess_file,
                bounding_box=self.bounding_box,
                zlo=zlo,
                zhi=zhi,
                dim=self.dim,
                output_dir=output_hedm_dir,
                out_name=out_name,
                env=self.env,
                seed=self.seed,
                verbose=verbose,
            )

            CrystalGenerator.write_to_csv(
                stat_file=f"{out_name}.stcell",
                ori_file=f"{out_name}.ori",
                output_csv=f"{out_csv_name}.csv",
                verbose=verbose,
                apply_noise=apply_noise,
                apply_noise_method=apply_noise_method,
                noise_level=noise_level,
                remove_minimum_volume=remove_minimum_volume,
                min_vol=min_vol,
            )

        print("\n=== HEDM z-scan generation complete ===")
        print(f"Outputs saved in: {output_hedm_dir}\n")

    def check_dependencies(self) -> None:
        """Check and install Neper (and its GSL/OpenBLAS deps) locally under ~/.local if missing."""

        home = os.path.expanduser("~")
        prefix = os.path.join(home, ".local")
        env = os.environ.copy()
        env["PATH"] = f"{prefix}/bin:" + env["PATH"]
        env["LD_LIBRARY_PATH"] = f"{prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")

        def is_installed(cmd):
            return shutil.which(cmd, path=env["PATH"]) is not None

        def run(cmd, cwd=None):
            print(">", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=cwd, env=env)

        os.makedirs(prefix, exist_ok=True)

        # install GSL locally if missing
        gsl_lib = os.path.join(prefix, "lib", "libgsl.so")
        if not os.path.exists(gsl_lib):
            print("Installing GSL locally...")
            run(
                [
                    "wget",
                    "https://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz",
                    "-O",
                    os.path.join(home, "gsl.tar.gz"),
                ]
            )
            run(["tar", "-xzf", "gsl.tar.gz"], cwd=home)
            gsl_src = next(
                (
                    os.path.join(home, d)
                    for d in os.listdir(home)
                    if d.startswith("gsl-")
                ),
                None,
            )
            if gsl_src:
                run(["./configure", f"--prefix={prefix}"], cwd=gsl_src)
                run(["make", "-j", str(os.cpu_count())], cwd=gsl_src)
                run(["make", "install"], cwd=gsl_src)
            else:
                raise RuntimeError("GSL extraction failed.")

        # install OpenBLAS locally if missing
        openblas_lib = os.path.join(prefix, "lib", "libopenblas.so.0")
        if not os.path.exists(openblas_lib):
            print("Installing OpenBLAS locally...")
            progs_dir = os.path.expanduser("~/Progs")
            os.makedirs(progs_dir, exist_ok=True)
            run(
                ["git", "clone", "https://github.com/xianyi/OpenBLAS.git"],
                cwd=progs_dir,
            )
            openblas_src = os.path.join(progs_dir, "OpenBLAS")
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            run(
                [
                    "make",
                    f"PREFIX={prefix}",
                    f"FC={conda_prefix}/bin/x86_64-conda-linux-gnu-gfortran",
                    "-j",
                    str(os.cpu_count()),
                ],
                cwd=openblas_src,
            )
            run(["make", "install", f"PREFIX={prefix}"], cwd=openblas_src)

        # install Neper if missing
        neper_installed = is_installed("neper")

        if not neper_installed:
            print("Installing Neper locally...")
            stable_version = "4.10.1"
            stable_url = f"https://neper.info/download/neper-{stable_version}.tar.gz"
            progs_dir = os.path.expanduser("~/Progs")
            os.makedirs(progs_dir, exist_ok=True)

            try:
                # try official stable release first
                print(f"Attempting official stable release v{stable_version}...")
                run(
                    [
                        "wget",
                        stable_url,
                        "-O",
                        os.path.join(progs_dir, f"neper-{stable_version}.tar.gz"),
                    ]
                )
                run(["tar", "-zxf", f"neper-{stable_version}.tar.gz"], cwd=progs_dir)
                neper_src_dir = os.path.join(
                    progs_dir, f"neper-{stable_version}", "src"
                )
            except subprocess.CalledProcessError:
                # fall back to GitHub master
                print("Stable release unavailable, cloning GitHub master instead...")
                repo_url = "https://github.com/rquey/neper.git"
                neper_src_dir = os.path.join(progs_dir, "neper", "src")
                if not os.path.exists(os.path.join(progs_dir, "neper")):
                    run(["git", "clone", repo_url, os.path.join(progs_dir, "neper")])
                else:
                    run(["git", "-C", os.path.join(progs_dir, "neper"), "pull"])

            # build & install
            build_dir = os.path.join(neper_src_dir, "build")
            os.makedirs(build_dir, exist_ok=True)
            run(
                [
                    "cmake",
                    f"-DCMAKE_INSTALL_PREFIX={prefix}",
                    "-DNEPER_INSTALL_BASH_COMPLETION=OFF",  # disable sudo path
                    "..",
                ],
                cwd=build_dir,
            )
            run(["make", "-j", str(os.cpu_count())], cwd=build_dir)
            run(["make", "install"], cwd=build_dir)

            if is_installed("neper"):
                neper_installed = True
                print("Neper installed successfully (stable or GitHub build).")
            else:
                raise RuntimeError("Neper installation failed.")

        else:
            print("Neper already available in PATH.")

        home = os.path.expanduser("~")
        prefix = os.path.join(home, ".local")

        final_env = os.environ.copy()
        final_env["PATH"] = f"{prefix}/bin:" + final_env.get("PATH", "")
        final_env["LD_LIBRARY_PATH"] = (
            f"{prefix}/lib:{prefix}/lib64:/usr/local/lib:/usr/lib:/usr/lib64:/lib/x86_64-linux-gnu"
        )

        self.env = final_env

        return final_env

    @staticmethod
    def write_to_csv(
        stat_file: str,
        ori_file: str,
        output_csv: str,
        verbose: bool = True,
        apply_noise: bool = False,
        apply_noise_method: str = "gaussian",
        noise_level: float = 1e-4,
        remove_minimum_volume: bool = False,
        min_vol: float = 0.0,
    ):
        """Combine Neper .stcell (geometry) and .ori (orientation) into one CSV
        mimicking HEDM APS output (columns X, Y, Z, GrainRadius, Eul0, Eul1, Eul2).

        apply_noise adds gaussian noise scaled by value magnitude (noise_level =
        relative stdev); remove_minimum_volume drops grains below min_vol."""

        if not os.path.exists(stat_file):
            raise FileNotFoundError(f"Missing .stcell file: {stat_file}")
        if not os.path.exists(ori_file):
            raise FileNotFoundError(f"Missing .ori file: {ori_file}")

        # load .stcell
        df_stat = pd.read_csv(
            stat_file,
            sep=r"\s+",
            comment="*",
            header=None,
            names=["id", "vol", "w", "x", "y", "z", "radeq"],
            engine="python",
        )

        # load .ori
        df_ori = pd.read_csv(
            ori_file,
            sep=r"\s+",
            comment="*",
            header=None,
            names=["Eul0", "Eul1", "Eul2"],
            engine="python",
        )

        # sanity check
        if len(df_stat) != len(df_ori):
            print(
                f"Warning: Mismatch in number of grains ({len(df_stat)} vs {len(df_ori)})"
            )

        df = pd.concat([df_stat[["vol", "x", "y", "z", "radeq"]], df_ori], axis=1)

        # optionally remove small grains
        if remove_minimum_volume and min_vol > 0:
            before = len(df)
            df = df[df["vol"] >= min_vol].reset_index(drop=True)
            print(f"        Removed {before - len(df)} grains below min_vol={min_vol}")

        df_final = df.rename(
            columns={"x": "X", "y": "Y", "z": "Z", "radeq": "GrainRadius"}
        )[["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]]

        # optionally apply noise
        if apply_noise:
            if apply_noise_method.lower() == "gaussian":
                numeric_cols = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]
                for col in numeric_cols:
                    df_final[col] += df_final[col] * np.random.normal(
                        0, noise_level, size=len(df_final)
                    )
                print(
                    f"        Applied Gaussian noise ({noise_level*100:.2f}% mean) to all properties."
                )
            else:
                raise ValueError(
                    f"Noise method '{apply_noise_method}' has not yet been implemented."
                )

        df_final.to_csv(output_csv, index=False)

        if verbose:
            print(f"Written combined CSV to: {output_csv}")

    @staticmethod
    def crop_tessellation(
        tess_name: str,
        bounding_box=None,
        seed: int = 12345,
        zlo: float = 0.0,
        zhi: float = 1.0,
        dim: int = 3,
        output_dir: str = ".",
        out_name: str = "tess_cropped",
        env: dict | None = None,
        verbose: bool = True,
    ):
        """Crop an existing .tess file along z between zmin and zmax."""

        if bounding_box is None:
            raise ValueError("bounding_box must be provided.")
        bounding_box = np.array(bounding_box, dtype=float).ravel()

        if not os.path.exists(tess_name):
            raise FileNotFoundError(f"Tessellation file not found: {tess_name}")

        os.makedirs(output_dir, exist_ok=True)

        # extract ncell
        def _get_ncell(tess_file):
            with open(tess_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.lstrip().startswith("**cell"):
                        next_line = next(f, "").strip()
                        if next_line.isdigit():
                            return int(next_line)
                        raise RuntimeError(
                            f"Unexpected format after **cell: '{next_line}'"
                        )
            raise RuntimeError("Failed to find cell count in .tess file.")

        ncell = _get_ncell(tess_name)

        # compute domain
        if dim == 2:
            xmin, xmax, ymin, ymax = bounding_box
            sx, sy = xmax - xmin, ymax - ymin
            tx, ty = xmin, ymin
            domain_arg = f"square({sx},{sy}):translate({tx},{ty})"
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
            sx, sy, sz = xmax - xmin, ymax - ymin, zmax - zmin
            tx, ty, tz = xmin, ymin, zmin
            domain_arg = f"cube({sx},{sy},{sz}):translate({tx},{ty},{tz})"

        # define cut
        cut_arg = f"cut(hspace({zhi},0,0,1),hspace({-zlo},0,0,-1))"

        neper_cmd = [
            "neper",
            "-T",
            "-n",
            str(ncell),
            "-id",
            str(seed),
            "-domain",
            domain_arg,
            "-morphooptiini",
            f"file({tess_name})",
            "-oridescriptor",
            "euler-bunge",
            "-transform",
            cut_arg,
            "-o",
            out_name,
            "-statcell",
            "id,vol,w,x,y,z,radeq",
            "-format",
            "tess,geo,ori",
        ]

        if verbose:
            print("\n=== Cropping Tessellation ===\n")
            print("> " + " ".join(neper_cmd) + " \n")

            subprocess.run(
                neper_cmd,
                cwd=output_dir,
                env=env,
                check=True,
            )
        else:
            subprocess.run(
                neper_cmd,
                cwd=output_dir,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        return os.path.join(output_dir, f"{os.path.basename(out_name)}.tess")

    @staticmethod
    def show_morpho_options(exit_after: bool = False) -> None:
        """Display available morphology configurations and supported distributions."""
        morpho_info = dedent(
            """
        Available morphology configurations:

        | Type          | Required Keys            | Description                                                                         |
        | ------------- | ------------------------ | ----------------------------------------------------------------------------------- |
        | "gg"          | mean                     | Grain growth morphology with absolute mean grain size.                              |
        | "lamellar"    | n, v                     | Layered grains: number of lamellae n, direction v ('x','y','z','random','crysdir') |
        | "columnar"    | n, v                     | Columnar n grains aligned with along direction ('x','y','z').                              |
        | "bamboo"      | n, v                     | 1D bamboo-like n grains along direction ('x','y','z').                                |
        | "diameq"      | distribution, params     | Equivalent-diameter distribution, e.g. lognormal(0.1,0.03).                         |
        | "size"        | distribution, params     | Volume-based size distribution, same format as diameq.                              |

        Example dictionary inputs:
        gg:        {"type": "gg", "mean": 1.0}
        lamellar:  {"type": "lamellar", "n": 8, "v": "z"}
        columnar:  {"type": "columnar", "n": 8, "v": "y"}
        bamboo:    {"type": "bamboo", "n": 8, "v": "z"}
        diameq:    {"type": "diameq", "distribution": "lognormal", "params": (0.1, 0.03)}
        size:      {"type": "size", "distribution": "weibull", "params": (2.5, 1.0)}
        """
        )

        dist_info = dedent(
            """
        Supported distributions (for 'size' or 'diameq'):

        | Distribution    | Required Parameters    |
        | --------------- | ---------------------- |
        | normal          | mean, sigma            |
        | lognormal       | mean, sigma            |
        | dirac           | mean                   |
        | beta            | x, y                   |
        | lorentzian      | mean, sigma            |
        | studentst       | mean, sigma            |
        | weibull         | k, sigma               |
        | breitwigner     | mean, sigma, gamma     |
        | expnormal       | mean, sigma, gamma     |
        | moffat          | mean, sigma, gamma     |
        | pearson7        | mean, sigma, gamma     |
        | pseudovoigt     | mean, sigma, gamma     |
        | skewnormal      | mean, sigma, gamma     |

        Example dictionary inputs:
        lognormal:  {"distribution": "lognormal", "params": (1.0, 0.3)}
        weibull:    {"distribution": "weibull", "params": (2.5, 1.0)}
        dirac:      {"distribution": "dirac", "params": (0.5,)}
        """
        )

        print(morpho_info)
        print()
        print(dist_info)

        if exit_after:
            sys.exit(0)

    @staticmethod
    def validate_morpho(morpho: Dict[str, Any]) -> None:
        """Validate morphology dictionary against schema."""
        if not isinstance(morpho, dict):
            raise TypeError("Morphology input must be a dictionary.")
        if "type" not in morpho:
            raise ValueError("Morphology dictionary must include a 'type' key.")

        mtype = morpho["type"]
        if mtype not in CrystalGenerator._MORPHO_SCHEMA:
            raise ValueError(f"Unsupported morphology type: '{mtype}'")

        schema = CrystalGenerator._MORPHO_SCHEMA[mtype]
        required, optional = schema["required"], schema["optional"]
        allowed = set(required + optional + ["type"])

        missing = [k for k in required if k not in morpho]
        extra = [k for k in morpho if k not in allowed]

        if missing:
            raise ValueError(f"Missing required key(s) for '{mtype}': {missing}")
        if extra:
            raise ValueError(f"Unexpected key(s) for '{mtype}': {extra}")

        # Validate distribution for size/diameq
        if mtype in ["size", "diameq"]:
            dist_name = morpho.get("distribution")
            params = morpho.get("params")
            CrystalGenerator.validate_distribution(dist_name, params)

    @staticmethod
    def validate_distribution(dist_name: str, params: Any) -> None:
        """Validate distribution name and its parameter count."""
        schema = CrystalGenerator._DISTRIBUTION_SCHEMA

        if dist_name not in schema:
            raise ValueError(f"Unsupported distribution: '{dist_name}'")

        required = schema[dist_name]
        if not isinstance(params, (list, tuple)):
            raise TypeError(f"Parameters for '{dist_name}' must be a tuple or list.")

        if len(params) != len(required):
            raise ValueError(
                f"Distribution '{dist_name}' requires {len(required)} parameters {required}, "
                f"but got {len(params)}."
            )

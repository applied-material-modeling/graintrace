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

"""Locate a user-provided NEPER binary and build a subprocess environment for it.

graintrace drives NEPER as an external tool; like CUBIT/SCULPT it is *not* a pip
package and must be installed by the user. Install NEPER once (see
https://neper.info/doc/introduction.html#installing-neper) and graintrace will
find it, in this order:

  1. an explicit ``neper_path`` / ``env=`` passed to the builder,
  2. the ``NEPER`` environment variable (absolute path to the ``neper`` binary),
  3. the ``neper`` key of a ``graintrace_tools.json`` (see ``graintrace.mcp.tool_paths``),
  4. ``neper`` on ``PATH`` (``shutil.which``).

gmsh, by contrast, *is* a pip package (declared in ``pyproject.toml`` and
installed with graintrace), so it needs no discovery here.

For the legacy convenience of a from-source build on Linux, pass
``auto_install=True`` (opt-in): it builds GSL, OpenBLAS and NEPER into
``~/.local`` exactly as older graintrace did. The default path never downloads or
compiles anything.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

NEPER_INSTALL_URL = "https://neper.info/doc/introduction.html#installing-neper"


def find_neper(neper_path: Optional[str] = None) -> Optional[str]:
    """Resolve the ``neper`` binary path, or ``None`` if it cannot be found.

    Order: explicit ``neper_path`` -> ``NEPER`` env var -> tools.json ``neper``
    key -> ``shutil.which("neper")``.
    """
    if neper_path and Path(neper_path).exists():
        return str(neper_path)

    env_path = os.environ.get("NEPER")
    if env_path and Path(env_path).exists():
        return env_path

    try:
        # Lazy import: keeps this module light and avoids an import cycle.
        from graintrace.mcp import tool_paths  # pylint: disable=import-outside-toplevel

        cfg = tool_paths.neper()
        if cfg:
            return cfg
    # Best-effort: the MCP subpackage is optional; ignore any failure.
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    return shutil.which("neper")


def build_env(neper_bin: Optional[str] = None) -> dict:
    """Copy ``os.environ`` and, given a ``neper_bin``, put its dir/libs on the path.

    Prepends the binary's directory to ``PATH`` and the sibling ``lib``/``lib64``
    of its install prefix to ``LD_LIBRARY_PATH`` so a self-built NEPER (e.g. under
    ``~/.local/bin``) finds its GSL/OpenBLAS shared libraries.
    """
    env = os.environ.copy()
    if not neper_bin:
        return env

    bindir = os.path.dirname(os.path.abspath(neper_bin))
    prefix = os.path.dirname(bindir)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    libdirs = [os.path.join(prefix, "lib"), os.path.join(prefix, "lib64")]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [d for d in libdirs if d] + ([existing] if existing else [])
    )
    return env


def resolve_neper_env(
    neper_path: Optional[str] = None,
    auto_install: bool = False,
) -> Tuple[str, dict]:
    """Return ``(neper_bin, env)`` for driving NEPER, or raise a clear error.

    Looks NEPER up via :func:`find_neper`. If not found and ``auto_install`` is
    True, builds it from source under ``~/.local`` (Linux only); otherwise raises
    ``RuntimeError`` with install guidance.
    """
    neper_bin = find_neper(neper_path)

    if neper_bin is None and auto_install:
        neper_bin = _auto_install_neper()

    if neper_bin is None:
        raise RuntimeError(
            "NEPER not found. graintrace does not ship NEPER; install it yourself "
            f"({NEPER_INSTALL_URL}), then let graintrace find it via one of:\n"
            "  - the NEPER environment variable (absolute path to the `neper` binary),\n"
            "  - a graintrace_tools.json with a `neper` key "
            "(see graintrace/mcp/tools.example.json),\n"
            "  - `neper` on your PATH,\n"
            "  - passing neper_path=/abs/path/to/neper (or a prepared env=) to the builder.\n"
            "As a last resort on Linux, pass auto_install=True to build it into ~/.local."
        )

    return neper_bin, build_env(neper_bin)


def _auto_install_neper() -> str:
    """Opt-in Linux from-source build of GSL + OpenBLAS + NEPER into ``~/.local``.

    Returns the installed ``neper`` path. This is the legacy behavior, retained
    only behind ``auto_install=True``; the default resolution never calls it.
    """
    home = os.path.expanduser("~")
    prefix = os.path.join(home, ".local")
    env = os.environ.copy()
    env["PATH"] = f"{prefix}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")

    def is_installed(cmd):
        return shutil.which(cmd, path=env["PATH"]) is not None

    def run(cmd, cwd=None):
        print(">", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=cwd, env=env)

    os.makedirs(prefix, exist_ok=True)

    # GSL
    if not os.path.exists(os.path.join(prefix, "lib", "libgsl.so")):
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
            (os.path.join(home, d) for d in os.listdir(home) if d.startswith("gsl-")),
            None,
        )
        if not gsl_src:
            raise RuntimeError("GSL extraction failed.")
        run(["./configure", f"--prefix={prefix}"], cwd=gsl_src)
        run(["make", "-j", str(os.cpu_count())], cwd=gsl_src)
        run(["make", "install"], cwd=gsl_src)

    # OpenBLAS
    if not os.path.exists(os.path.join(prefix, "lib", "libopenblas.so.0")):
        print("Installing OpenBLAS locally...")
        progs_dir = os.path.expanduser("~/Progs")
        os.makedirs(progs_dir, exist_ok=True)
        run(["git", "clone", "https://github.com/xianyi/OpenBLAS.git"], cwd=progs_dir)
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

    # NEPER
    if not is_installed("neper"):
        print("Installing Neper locally...")
        stable_version = "4.10.1"
        stable_url = f"https://neper.info/download/neper-{stable_version}.tar.gz"
        progs_dir = os.path.expanduser("~/Progs")
        os.makedirs(progs_dir, exist_ok=True)
        try:
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
            neper_src_dir = os.path.join(progs_dir, f"neper-{stable_version}", "src")
        except subprocess.CalledProcessError:
            print("Stable release unavailable, cloning GitHub master instead...")
            repo_url = "https://github.com/rquey/neper.git"
            neper_src_dir = os.path.join(progs_dir, "neper", "src")
            if not os.path.exists(os.path.join(progs_dir, "neper")):
                run(["git", "clone", repo_url, os.path.join(progs_dir, "neper")])
            else:
                run(["git", "-C", os.path.join(progs_dir, "neper"), "pull"])

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

        if not is_installed("neper"):
            raise RuntimeError("Neper installation failed.")
        print("Neper installed successfully (stable or GitHub build).")
    else:
        print("Neper already available in PATH.")

    return shutil.which("neper", path=env["PATH"]) or os.path.join(
        prefix, "bin", "neper"
    )

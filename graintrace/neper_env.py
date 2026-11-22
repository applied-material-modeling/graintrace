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

For the convenience of a from-source build on Linux, pass ``auto_install=True``
(opt-in): it builds NEPER 7.0.0 into ``~/.local`` against the Gmsh 4.15.2 SDK
(unpacked into ``~/.local/opt``, linked via RPATH) with NEPER's builtin GSL and
the system gcc (see :func:`_neper7_build_plan`). The default path never downloads
or compiles anything. graintrace supports NEPER 4.x and 7.x; with NEPER 7 the
``-morphooptialgo`` choices are ``subplex`` (graintrace's default) and ``lloyd``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

NEPER_INSTALL_URL = "https://neper.info/doc/introduction.html#installing-neper"

#: Descriptor graintrace asks NEPER to write orientation matrices in: the active
#: ``R`` (crystal -> sample, = graintrace / MIDAS / neml2). Accepted by NEPER 4.x
#: and 7.x; NEPER's own default (no suffix) is the passive ``g = R^T``.
ROTMAT_DESCRIPTOR = "rotmat:active"


def split_descriptor(descriptor: str) -> Tuple[str, str]:
    """Split a NEPER orientation descriptor into ``(base, sense)``.

    ``"rotmat:active"`` -> ``("rotmat", "active")``; no suffix means NEPER's
    default, ``"passive"``.

    Raises:
        ValueError: if the suffix is neither ``active`` nor ``passive``.
    """
    base, _, sense = descriptor.strip().lower().partition(":")
    sense = sense or "passive"
    if sense not in ("active", "passive"):
        raise ValueError(
            f"NEPER orientation descriptor {descriptor!r}: the convention suffix "
            "must be ':active' or ':passive'"
        )
    return base, sense


def euler_descriptor(descriptor: str = "euler-bunge") -> str:
    """The explicit NEPER descriptor for graintrace Euler angles: ``<euler-*>:passive``.

    graintrace Euler angles are the standard (passive) triplet, so they are always
    handed to NEPER with an explicit ``:passive`` suffix rather than relying on
    NEPER's default.

    Args:
        descriptor: ``"euler-bunge"``, ``"euler-kocks"`` or ``"euler-roe"``, with
            or without ``":passive"``.

    Returns:
        str: e.g. ``"euler-bunge:passive"``.

    Raises:
        ValueError: for a non-Euler descriptor or an ``:active`` Euler triplet
            (the Euler angles of ``R^T``, which graintrace never uses).
    """
    base, sense = split_descriptor(descriptor)
    if base not in ("euler-bunge", "euler-kocks", "euler-roe"):
        raise ValueError(
            f"orientation_descriptor {descriptor!r}: graintrace Euler angles need "
            "'euler-bunge', 'euler-kocks' or 'euler-roe'"
        )
    if sense != "passive":
        raise ValueError(
            f"orientation_descriptor {descriptor!r}: graintrace Euler angles are the "
            "standard (passive) triplet; ':active' Euler angles are not supported"
        )
    return f"{base}:passive"


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


def neper_version(neper_bin: Optional[str] = None) -> Optional[str]:
    """The ``neper --version`` string (e.g. ``"7.0.0"``), or ``None`` if it cannot run.

    Args:
        neper_bin: the binary; resolved with :func:`find_neper` when ``None``.
    """
    neper_bin = neper_bin or find_neper()
    if neper_bin is None:
        return None
    try:
        proc = subprocess.run(
            [neper_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=build_env(neper_bin),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (proc.stdout or proc.stderr or "").strip().splitlines()
    return lines[0].strip() if proc.returncode == 0 and lines else None


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


#: NEPER version built by ``auto_install=True`` and the Gmsh SDK it links against.
NEPER_AUTO_VERSION = "7.0.0"
GMSH_SDK_VERSION = "4.15.2"


def _neper7_build_plan(home: str) -> dict:
    """The NEPER 7 from-source build recipe (pure data, so it can be tested unrun).

    Mirrors a known-good manual build: the Gmsh SDK unpacked into
    ``~/.local/opt`` and found through ``CMAKE_PREFIX_PATH``, linked via RPATH
    (no ``LD_LIBRARY_PATH`` needed at run time), NEPER's builtin GSL
    (``-DFORCE_BUILTIN_GSL=ON``), and the SYSTEM gcc/g++ (a conda compiler in the
    environment breaks the link), with the compiler variables cleared.

    Returns:
        dict with ``prefix``, ``gmsh_dir``, ``gmsh_url``, ``neper_urls``,
        ``src_dir``, ``build_env`` (overrides; ``None`` = unset) and ``cmake``
        (the configure command).
    """
    prefix = os.path.join(home, ".local")
    progs = os.path.join(home, "Progs")
    gmsh_name = f"gmsh-{GMSH_SDK_VERSION}-Linux64-sdk"
    gmsh_dir = os.path.join(prefix, "opt", gmsh_name)
    rpath = f"{prefix}/lib;{gmsh_dir}/lib"
    return {
        "prefix": prefix,
        "progs": progs,
        "gmsh_dir": gmsh_dir,
        "gmsh_url": f"https://gmsh.info/bin/Linux/{gmsh_name}.tgz",
        "neper_urls": [
            f"https://neper.info/download/neper-{NEPER_AUTO_VERSION}.tar.gz",
            "https://github.com/neperfepx/neper/archive/refs/tags/"
            f"v{NEPER_AUTO_VERSION}.tar.gz",
        ],
        "src_dir": os.path.join(progs, f"neper-{NEPER_AUTO_VERSION}", "src"),
        "build_env": {
            "CC": None,
            "CXX": None,
            "CFLAGS": None,
            "CXXFLAGS": None,
            "LDFLAGS": None,
            "CPPFLAGS": None,
            "CMAKE_PREFIX_PATH": None,
            "PATH": f"{prefix}/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": f"{prefix}/lib:{gmsh_dir}/lib",
        },
        "cmake": [
            "cmake",
            "-DCMAKE_C_COMPILER=/usr/bin/gcc",
            "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            f"-DCMAKE_PREFIX_PATH={prefix};{gmsh_dir}",
            f"-DCMAKE_INSTALL_RPATH={rpath}",
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DNEPER_INSTALL_BASH_COMPLETION=OFF",  # no sudo path
            "-DFORCE_BUILTIN_GSL=ON",
            "..",
        ],
    }


def _auto_install_neper() -> str:
    """Opt-in Linux from-source build of NEPER 7 (+ Gmsh SDK, OpenBLAS) into ``~/.local``.

    Returns the installed ``neper`` path. Retained only behind
    ``auto_install=True``; the default resolution never calls it. The recipe is
    :func:`_neper7_build_plan`; it needs ``wget``, ``tar``, ``cmake``, ``make``
    and the system ``gcc``/``g++``.
    """
    home = os.path.expanduser("~")
    plan = _neper7_build_plan(home)
    prefix, progs = plan["prefix"], plan["progs"]
    env = os.environ.copy()
    env["PATH"] = f"{prefix}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")

    tool_env = env.copy()
    for key, val in plan["build_env"].items():
        if val is None:
            tool_env.pop(key, None)
        else:
            tool_env[key] = val

    def is_installed(cmd):
        return shutil.which(cmd, path=env["PATH"]) is not None

    def run(cmd, cwd=None, run_env=None):
        print(">", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=cwd, env=run_env or env)

    os.makedirs(prefix, exist_ok=True)
    os.makedirs(progs, exist_ok=True)

    if is_installed("neper"):
        print("Neper already available in PATH.")
        return shutil.which("neper", path=env["PATH"])

    # OpenBLAS (NEPER's linear algebra)
    if not os.path.exists(os.path.join(prefix, "lib", "libopenblas.so.0")):
        print("Installing OpenBLAS locally...")
        run(["git", "clone", "https://github.com/xianyi/OpenBLAS.git"], cwd=progs)
        openblas_src = os.path.join(progs, "OpenBLAS")
        run(
            ["make", f"PREFIX={prefix}", "-j", str(os.cpu_count())],
            cwd=openblas_src,
            run_env=tool_env,
        )
        run(["make", "install", f"PREFIX={prefix}"], cwd=openblas_src)

    # Gmsh SDK (NEPER 7 links libgmsh for meshing)
    if not os.path.isdir(plan["gmsh_dir"]):
        print(f"Installing the Gmsh {GMSH_SDK_VERSION} SDK...")
        os.makedirs(os.path.dirname(plan["gmsh_dir"]), exist_ok=True)
        tgz = os.path.join(progs, os.path.basename(plan["gmsh_url"]))
        run(["wget", plan["gmsh_url"], "-O", tgz])
        run(["tar", "-xzf", tgz, "-C", os.path.dirname(plan["gmsh_dir"])])

    # NEPER source
    if not os.path.isdir(plan["src_dir"]):
        tarball = os.path.join(progs, f"neper-{NEPER_AUTO_VERSION}.tar.gz")
        for url in plan["neper_urls"]:
            try:
                print(f"Downloading NEPER {NEPER_AUTO_VERSION} from {url} ...")
                run(["wget", url, "-O", tarball])
                break
            except subprocess.CalledProcessError:
                continue
        else:
            raise RuntimeError(f"Could not download NEPER {NEPER_AUTO_VERSION}.")
        run(["tar", "-zxf", tarball], cwd=progs)
        if not os.path.isdir(plan["src_dir"]):  # GitHub archive layout
            alt = os.path.join(progs, f"neper-{NEPER_AUTO_VERSION}")
            raise RuntimeError(f"NEPER sources not found under {alt}/src")

    build_dir = os.path.join(plan["src_dir"], "build")
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir)
    run(plan["cmake"], cwd=build_dir, run_env=tool_env)
    run(["make", "-j", str(os.cpu_count())], cwd=build_dir, run_env=tool_env)
    run(["make", "install"], cwd=build_dir, run_env=tool_env)

    neper_bin = os.path.join(prefix, "bin", "neper")
    if not os.path.exists(neper_bin):
        raise RuntimeError("Neper installation failed.")
    clean = {k: v for k, v in env.items() if k != "LD_LIBRARY_PATH"}
    run([neper_bin, "--version"], run_env=clean)  # RPATH: runs without LD_LIBRARY_PATH
    print(f"Neper {NEPER_AUTO_VERSION} installed successfully.")
    return neper_bin

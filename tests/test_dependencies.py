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

"""Check that optional heavy dependencies are available."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# The working neml2 is the repo-pinned build PUMA installs into the PUMA env
# (pyzag comes from graintrace's pip deps); on a plain checkout without neml2
# these skip (not error).


def test_neml2_importable():
    """neml2 must be importable (repo-pinned build supplied by PUMA)."""
    pytest.importorskip("neml2")
    import neml2  # noqa: F401


def test_neml2_has_types():
    """neml2.types (v3) must expose the tensor wrappers used for orientation math."""
    pytest.importorskip("neml2")
    from neml2 import types

    assert hasattr(types, "MRP"), "neml2.types.MRP not found"
    assert hasattr(types, "Vec"), "neml2.types.Vec not found"
    assert hasattr(types, "R2"), "neml2.types.R2 not found"


def test_neml2_has_symmetry():
    """neml2 (v3) must expose crystal symmetry operators."""
    pytest.importorskip("neml2")
    from neml2.ops import symmetry

    assert callable(symmetry), "neml2.ops.symmetry not callable"


def test_neml2_has_texture():
    """neml2.texture (v3, was neml2.postprocessing) must expose pole-figure tools."""
    pytest.importorskip("neml2")
    from neml2 import texture

    assert hasattr(texture, "pretty_plot_pole_figure_points")
    assert hasattr(texture, "symmetry_operators_as_R2")


def test_pyzag_backend_available():
    """The NEML2 v3 pyzag adapter + pyzag must be importable for calibration."""
    pytest.importorskip("neml2")
    pytest.importorskip("pyzag")
    import neml2
    from neml2.pyzag import NEML2PyzagModel  # noqa: F401
    from pyzag import nonlinear, chunktime, reparametrization  # noqa: F401

    assert hasattr(neml2, "compile")


def _resolved_sculpt_config():
    """CUBIT/SCULPT config from a tools.json.

    Tries the canonical graintrace resolver (env var / ``graintrace_tools.json`` /
    ``~/.config``) first, then this repo's dev ``deploy/tools.json`` as a fallback.
    Returns None when nothing resolves (the tests then skip).
    """
    try:
        from graintrace.mcp import tool_paths  # pylint: disable=import-outside-toplevel

        cfg = tool_paths.sculpt_config()
        if cfg:
            return cfg
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    deploy = Path(__file__).parent.parent / "deploy" / "tools.json"
    if deploy.is_file():
        try:
            cfg = json.loads(deploy.read_text(encoding="utf-8")).get("sculpt_config")
            if isinstance(cfg, dict) and cfg.get("psculpt"):
                return cfg
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    return None


def test_cubit_psculpt_exists():
    """psculpt binary (Coreform CUBIT/SCULPT) must exist and be executable.

    CUBIT is proprietary and machine-specific. The test resolves it from the
    ``PSCULPT``/``CUBIT_BIN_DIR`` env var or, failing that, from a configured
    ``tools.json`` (``sculpt_config``); it skips only when none resolve.
    """
    cfg = _resolved_sculpt_config()
    psculpt = (
        os.environ.get("PSCULPT")
        or (
            os.path.join(os.environ["CUBIT_BIN_DIR"], "psculpt")
            if os.environ.get("CUBIT_BIN_DIR")
            else None
        )
        or (cfg.get("psculpt") if cfg else None)
    )
    if not psculpt:
        pytest.skip("No CUBIT/SCULPT install (set PSCULPT/CUBIT_BIN_DIR or tools.json)")
    assert os.path.isfile(psculpt), f"psculpt not found at {psculpt}"
    assert os.access(psculpt, os.X_OK), f"psculpt not executable: {psculpt}"


def test_cubit_mpiexec_exists():
    """mpiexec for CUBIT must exist.

    Resolved from ``CUBIT_MPIEXEC``/``CUBIT_BIN_DIR`` or a configured ``tools.json``
    (``sculpt_config``'s ``launcher``); skips only when none resolve.
    """
    cfg = _resolved_sculpt_config()
    mpiexec = (
        os.environ.get("CUBIT_MPIEXEC")
        or (
            os.path.join(os.environ["CUBIT_BIN_DIR"], "mpi", "bin", "mpiexec")
            if os.environ.get("CUBIT_BIN_DIR")
            else None
        )
        or (cfg.get("launcher") if cfg else None)
    )
    if not mpiexec:
        pytest.skip("No CUBIT mpiexec (set CUBIT_MPIEXEC/CUBIT_BIN_DIR or tools.json)")
    assert os.path.isfile(mpiexec), f"mpiexec not found at {mpiexec}"
    assert os.access(mpiexec, os.X_OK), f"mpiexec not executable: {mpiexec}"


# ---- NEPER resolution (pure Python; no external tool needed) ----------------


def test_neper_env_var_takes_precedence(tmp_path, monkeypatch):
    """find_neper resolves the NEPER env var to an existing binary."""
    from graintrace import neper_env

    fake = tmp_path / "bin" / "neper"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("NEPER", str(fake))
    assert neper_env.find_neper() == str(fake)


def test_neper_explicit_path_wins(tmp_path, monkeypatch):
    """An explicit neper_path beats the NEPER env var."""
    from graintrace import neper_env

    explicit = tmp_path / "explicit_neper"
    explicit.write_text("#!/bin/sh\n")
    monkeypatch.setenv("NEPER", "/does/not/exist/neper")
    assert neper_env.find_neper(str(explicit)) == str(explicit)


def test_build_env_puts_neper_on_path(tmp_path):
    """build_env prepends the binary's dir to PATH and its libs to LD_LIBRARY_PATH."""
    from graintrace import neper_env

    neper_bin = tmp_path / "bin" / "neper"
    neper_bin.parent.mkdir(parents=True)
    env = neper_env.build_env(str(neper_bin))
    assert env["PATH"].split(os.pathsep)[0] == str(tmp_path / "bin")
    libdirs = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(tmp_path / "lib") in libdirs
    assert str(tmp_path / "lib64") in libdirs


def test_resolve_neper_env_raises_when_missing(monkeypatch):
    """With no NEPER anywhere and auto_install off, resolution raises with guidance."""
    from graintrace import neper_env

    monkeypatch.setattr(neper_env, "find_neper", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="NEPER not found"):
        neper_env.resolve_neper_env(auto_install=False)

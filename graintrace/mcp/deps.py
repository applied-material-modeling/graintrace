# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Best-effort detection of the external / compiled stack graintrace drives.

``pip install graintrace`` gives you the Python code only. The heavy stack
(NEPER, MOOSE/PUMA ``puma-opt``, CUBIT/SCULPT ``psculpt``, NEML2/pyzag) is built
separately (see the repo's ``external/`` submodules + PUBLISHING.md). These
checks let a tool tell the user up front, in its preview, whether the tools it
needs are present -- and if a run is attempted anyway, produce a friendly
"not built yet" message instead of a raw traceback.

Detection is best-effort: PATH lookup for binaries, import probes for Python
packages, plus a couple of well-known submodule build locations. A tool that
passes an explicit binary path (e.g. ``moose_run_file``) is always validated
against that path at run time too.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class DepStatus:
    name: str
    ok: bool
    detail: str          # where it was found, or why it's missing
    build_hint: str      # how to get it


# ---- individual probes -------------------------------------------------------

def _repo_external() -> Path:
    """Location of the repo's external/ submodule tree, if running from a checkout."""
    # graintrace/mcp/deps.py -> graintrace/ -> repo root
    return Path(__file__).resolve().parents[2] / "external"


def _which_or_external(binary: str, *rel_candidates: str) -> Optional[str]:
    found = shutil.which(binary)
    if found:
        return found
    ext = _repo_external()
    for rel in rel_candidates:
        cand = ext / rel
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _check_neper() -> DepStatus:
    p = _which_or_external("neper", "neper/build/neper", "neper/src/neper")
    if p:
        return DepStatus("neper", True, f"found: {p}", "")
    return DepStatus(
        "neper", False, "not found on PATH or in external/neper",
        "Build NEPER (https://neper.info) and put `neper` on PATH, "
        "e.g. ~/.local/bin/neper.",
    )


def _check_puma() -> DepStatus:
    p = _which_or_external(
        "puma-opt", "puma/puma-opt", "moose/puma/puma-opt", "puma/build/puma-opt"
    )
    if p:
        return DepStatus("puma-opt", True, f"found: {p}", "")
    return DepStatus(
        "puma-opt", False,
        "MOOSE/PUMA `puma-opt` not found on PATH or in external/puma",
        "Build MOOSE + PUMA from external/puma (see README). Then pass its "
        "path as `moose_run_file`, or put it on PATH.",
    )


def _check_cubit() -> DepStatus:
    psculpt = _which_or_external("psculpt")
    epu = _which_or_external("epu")
    if psculpt and epu:
        return DepStatus("cubit", True, f"psculpt: {psculpt}; epu: {epu}", "")
    missing = [n for n, v in (("psculpt", psculpt), ("epu", epu)) if not v]
    return DepStatus(
        "cubit", False, f"missing CUBIT/SCULPT binaries: {', '.join(missing)}",
        "Install CUBIT/Coreform and point `sculpt_config` at its bin/ "
        "(psculpt, epu, mpiexec). CUBIT is licensed; never commit its license.",
    )


def _check_gpu() -> DepStatus:
    """CUDA GPU visible to torch. Not a hard dependency -- runs fall back to CPU
    -- but the policy is: if a GPU is available, always use it (CPFE + material
    calibration are the GPU-accelerated steps)."""
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
            return DepStatus("gpu", True, f"{n} CUDA device(s): {names}", "")
        return DepStatus(
            "gpu", False, "no CUDA device visible to torch",
            "Runs fall back to CPU (much slower for CPFE/calibration). Ensure "
            "CUDA drivers + a cuda-enabled torch build if a GPU is expected.",
        )
    except Exception as exc:  # torch missing/broken
        return DepStatus("gpu", False, f"torch CUDA probe failed: {exc}", "")


def gpu_available() -> bool:
    return _check_gpu().ok


def default_device() -> str:
    """'cuda' when a GPU is available, else 'cpu'. Policy: always prefer GPU."""
    return "cuda" if gpu_available() else "cpu"


def default_cpfe_device() -> str:
    """'cuda:0' when a GPU is available, else 'cpu' (CPFE wants an indexed device)."""
    return "cuda:0" if gpu_available() else "cpu"


def _check_python_pkg(pkg: str, build_hint: str) -> DepStatus:
    spec = importlib.util.find_spec(pkg)
    if spec is None:
        return DepStatus(pkg, False, f"python package `{pkg}` not importable", build_hint)
    # Importable, but neml2's AOTI runtime binding may still be broken; flag softly.
    return DepStatus(pkg, True, f"importable: {spec.origin}", "")


def _check_neml2() -> DepStatus:
    """neml2 python bindings importable. Sufficient for calibration/pyzag paths;
    the CPFE run additionally needs the AOTI runtime -- see _check_neml2_aoti."""
    st = _check_python_pkg(
        "neml2",
        "Install NEML2 v3 python bindings from external/neml2 "
        "(`pip install . -v`) into this env.",
    )
    if not st.ok:
        return st
    try:  # pragma: no cover - depends on local build
        import neml2  # noqa: F401
    except Exception as exc:  # importable-but-broken
        return DepStatus("neml2", False, f"neml2 import raised: {exc}", st.build_hint)
    return st


def _check_neml2_aoti() -> DepStatus:
    """The compiled AOTI runtime that CPFE needs to load neml2-compile'd models.

    Distinct from `neml2` because the material-calibration (pyzag) path does NOT
    need AOTI -- only the MOOSE/PUMA CPFE run does. A common failure here is an
    ABI mismatch (e.g. libstdc++ CXXABI too old for the compiled _aoti.so).
    """
    base = _check_neml2()
    if not base.ok:
        return DepStatus("neml2-aoti", False, base.detail, base.build_hint)
    hint = (
        "The neml2 AOTI runtime failed to load (often an ABI mismatch: the "
        "system libstdc++/CXXABI is older than what _aoti.so was built against). "
        "Rebuild NEML2 v3 against this env's toolchain, or update libstdc++."
    )
    try:  # pragma: no cover - depends on local build
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import neml2.aoti._aoti  # noqa: F401
    except Exception as exc:
        return DepStatus("neml2-aoti", False, f"AOTI runtime unavailable: {exc}", hint)
    return DepStatus("neml2-aoti", True, "AOTI runtime importable", "")


# name -> probe
_PROBES: Dict[str, Callable[[], DepStatus]] = {
    "gpu": _check_gpu,
    "neper": _check_neper,
    "puma-opt": _check_puma,
    "cubit": _check_cubit,
    "neml2": _check_neml2,
    "neml2-aoti": _check_neml2_aoti,
    "pyzag": lambda: _check_python_pkg(
        "pyzag", "Install pyzag from external/pyzag (`pip install . -v`)."
    ),
    "gmsh": lambda: _check_python_pkg("gmsh", "`pip install gmsh`."),
    "torch_geometric": lambda: _check_python_pkg(
        "torch_geometric", "`pip install 'graintrace[gnn]'` (torch-geometric)."
    ),
}


def check(name: str) -> DepStatus:
    probe = _PROBES.get(name)
    if probe is None:
        return DepStatus(name, False, f"unknown dependency '{name}'", "")
    return probe()


def check_all() -> List[DepStatus]:
    return [check(n) for n in _PROBES]


def require(*names: str) -> Optional[str]:
    """Return None if all named deps are present, else a user-facing message.

    Tools call this at the top of the ``confirm=True`` branch. The returned
    string is meant to be relayed verbatim to the user -- it names what's
    missing and how to build it, with no traceback.
    """
    missing = [check(n) for n in names]
    missing = [s for s in missing if not s.ok]
    if not missing:
        return None
    lines = ["This step needs external tools that aren't ready on this machine yet:\n"]
    for s in missing:
        lines.append(f"  - {s.name}: {s.detail}\n      -> {s.build_hint}")
    lines.append(
        "\nBuild the missing piece(s) and try again. `pip install graintrace` "
        "ships only the Python code; the compiled stack is separate."
    )
    return "\n".join(lines)


def summary() -> Dict[str, dict]:
    """Machine-readable status of the whole external stack (for a status tool)."""
    return {
        s.name: {"ok": s.ok, "detail": s.detail, "build_hint": s.build_hint}
        for s in check_all()
    }

# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Machine-local paths to the compiled/licensed external tools graintrace drives.

`pip install graintrace` ships only Python; the heavy stack (MOOSE/PUMA `puma-opt`,
CUBIT/SCULPT `psculpt`/`epu`, NEPER) is built/installed per machine at arbitrary
locations. Rather than hard-code those, point graintrace at a small JSON:

    {
      "puma_opt": "/abs/path/to/puma-opt",
      "neper":    "/abs/path/to/neper",            // optional; PATH used otherwise
      "sculpt_config": {                            // CUBIT/SCULPT for hex meshing
        "launcher": "/abs/cubit/bin/mpi/bin/mpiexec",
        "psculpt":  "/abs/cubit/bin/psculpt",
        "epu":      "/abs/cubit/bin/epu",
        "nprocs":   8,
        "environment": {"OPAL_LIBDIR": "/abs/cubit/bin/mpi/lib",
                        "OPAL_PREFIX": "/abs/cubit/bin/mpi"}
      }
    }

Search order (first found wins):
  1. $GRAINTRACE_TOOLS_JSON
  2. ./graintrace_tools.json  (current working dir)
  3. ~/.config/graintrace/tools.json
  4. <repo>/deploy/tools.json

`sculpt_config` is returned ready to hand to ``VoxelMeshBuilder.mesh`` /
``NearFieldMeshBuilder.mesh``. See ``deploy/tools.example.json``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


def _candidates() -> List[Path]:
    out = []
    env = os.environ.get("GRAINTRACE_TOOLS_JSON")
    if env:
        out.append(Path(env))
    out.append(Path.cwd() / "graintrace_tools.json")
    out.append(Path.home() / ".config" / "graintrace" / "tools.json")
    out.append(Path(__file__).resolve().parents[2] / "deploy" / "tools.json")
    return out


def config_path() -> Optional[str]:
    for c in _candidates():
        if c.is_file():
            return str(c)
    return None


@lru_cache(maxsize=1)
def _load() -> Dict[str, Any]:
    p = config_path()
    if not p:
        return {}
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def reload() -> Dict[str, Any]:
    """Drop the cached config (call after editing the JSON)."""
    _load.cache_clear()
    return _load()


def puma_opt() -> Optional[str]:
    """Configured puma-opt path (if it exists), else None."""
    p = _load().get("puma_opt")
    return p if p and Path(p).exists() else None


def neper() -> Optional[str]:
    p = _load().get("neper")
    return p if p and Path(p).exists() else None


def sculpt_config() -> Optional[Dict[str, Any]]:
    """CUBIT/SCULPT config dict (ready for ``builder.mesh``) if its binaries exist."""
    cfg = _load().get("sculpt_config")
    if not isinstance(cfg, dict):
        return None
    psculpt, epu = cfg.get("psculpt"), cfg.get("epu")
    if psculpt and epu and Path(psculpt).exists() and Path(epu).exists():
        return cfg
    return None

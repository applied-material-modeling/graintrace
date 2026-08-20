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

"""Shared utilities for the graintrace performance benchmarks.

Results are written under ``benchmark/results/`` (gitignored) as CSV + JSON, each
stamped with host/GPU/git info. These are timing measurements for a given machine,
NOT pass/fail regression gates.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Sequence

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BENCH_DIR / "results"


@contextmanager
def timer():
    """Context manager yielding a 1-element list; on exit list[0] = elapsed seconds."""
    box = [0.0]
    start = time.perf_counter()
    try:
        yield box
    finally:
        box[0] = time.perf_counter() - start


def _git_commit() -> str:
    """Best-effort short git hash of the working tree (empty string on failure)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BENCH_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()
    except Exception:  # pylint: disable=broad-except
        return ""


def _gpu_info() -> List[str]:
    """Names of visible CUDA devices via torch, empty if torch/CUDA unavailable."""
    try:
        # pylint: disable=import-outside-toplevel  # torch is a heavy optional dep here
        import torch

        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # pylint: disable=broad-except
        return []


def capture_sysinfo() -> Dict[str, Any]:
    """Collect machine metadata stored alongside every benchmark result."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "gpus": _gpu_info(),
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def results_dir(name: str, out: str | None = None) -> Path:
    """Create and return a stamped output dir: results/<host>_<timestamp>/<name>/."""
    if out is not None:
        path = Path(out)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = RESULTS_ROOT / f"{socket.gethostname()}_{stamp}" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_results(
    name: str,
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    sysinfo: Dict[str, Any],
) -> None:
    """Write rows to <out_dir>/<name>.csv and a <name>.json (rows + sysinfo)."""
    rows = list(rows)
    json_path = out_dir / f"{name}.json"
    json_path.write_text(
        json.dumps({"sysinfo": sysinfo, "rows": rows}, indent=2), encoding="utf-8"
    )
    if rows:
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        csv_path = out_dir / f"{name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def skip(reason: str) -> None:
    """Print a SKIP notice and exit 0 (dependency missing / not applicable here)."""
    print(f"SKIP: {reason}")
    sys.exit(0)


def parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated integer list, e.g. '1,2,4' -> [1, 2, 4]."""
    return [int(t) for t in str(text).split(",") if t.strip() != ""]


def print_header(title: str, sysinfo: Dict[str, Any]) -> None:
    """Print a labeled banner with machine info at the top of a benchmark run."""
    print("=" * 70)
    print(title)
    print(
        f"  host={sysinfo['hostname']} cpus={sysinfo['cpu_count']} "
        f"gpus={sysinfo['gpus'] or 'none'} git={sysinfo['git_commit'] or '?'}"
    )
    print("=" * 70)

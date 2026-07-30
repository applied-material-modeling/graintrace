# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Read-only access to the graintrace repository source, so the connected LLM can
see the codebase (list / read / search), not just call the workflow tools.

All access is jailed to the repo root (``GRAINTRACE_REPO_ROOT`` or the tree that
contains the installed ``graintrace`` package). Listing/search default to the
CODE directories and skip the huge data/build trees (``external/``,
``experiment_*``, outputs, ``.git``) so it stays fast on this ~600 GB checkout;
``read_code_file`` can still read any text file under the repo by path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional

from graintrace.mcp.app import mcp

# Directories that are code (default scope for list/search).
_CODE_ROOTS = ["graintrace", "examples", "tests", "mwe_data"]
# Never walk these (huge / binary / generated).
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "external",
    "dist",
    "build",
    ".vscode",
    "graintrace_mcp_out",
    "graintrace.egg-info",
    "out",
    "_truth",  # demo outputs
}
_MAX_BYTES = 400_000


def repo_root() -> Path:
    """Return the graintrace repo root (env override, else the installed package)."""
    env = os.environ.get("GRAINTRACE_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # Lazy self-import to locate the installed package without a load-time cycle.
    import graintrace as _gt  # pylint: disable=import-outside-toplevel

    return Path(_gt.__file__).resolve().parent.parent  # <repo>/graintrace/ -> <repo>


def _resolve(rel: str) -> Path:
    """Resolve a repo-relative (or absolute-in-repo) path, jailed to the repo."""
    root = repo_root()
    p = (root / rel).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"path escapes the repo root: {rel}")
    return p


def _iter_files(base: Path, pattern: str):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.match(pattern):
                yield p


@mcp.tool()
def list_code_files(
    subdir: Optional[str] = None, pattern: str = "*.py", max_files: int = 500
) -> dict:
    """List repository source files (repo-relative paths).

    subdir: limit to a directory (repo-relative), e.g. 'graintrace/mcp'. Default
        scans the code dirs (graintrace, examples, tests, demo, mwe_data).
    pattern: filename glob, e.g. '*.py', '*.md', '*.i'. The huge data/build trees
        (external/, experiment_*, .git, outputs) are always skipped.
    """
    root = repo_root()
    bases = [_resolve(subdir)] if subdir else [root / d for d in _CODE_ROOTS]
    out: List[str] = []
    for base in bases:
        if not base.exists():
            continue
        for p in _iter_files(base, pattern):
            out.append(str(p.relative_to(root)))
            if len(out) >= max_files:
                break
    return {
        "repo_root": str(root),
        "n_files": len(out),
        "truncated": len(out) >= max_files,
        "files": sorted(out),
    }


@mcp.tool()
def read_code_file(
    path: str, start_line: Optional[int] = None, end_line: Optional[int] = None
) -> dict:
    """Read a text file from the repo (repo-relative path), optionally a line
    range. Capped at ~400 KB. Use for source, recipes, configs, docs, .i files."""
    p = _resolve(path)
    if not p.is_file():
        return {"error": f"not a file: {path}"}
    if p.stat().st_size > _MAX_BYTES and not (start_line or end_line):
        return {
            "error": f"file too large ({p.stat().st_size} bytes); pass a line range"
        }
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    # Best-effort: return a friendly error instead of a traceback (e.g. binary).
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"error": f"cannot read (binary?): {exc}"}
    lines = text.splitlines()
    n = len(lines)
    if start_line or end_line:
        s = max(1, start_line or 1)
        e = min(n, end_line or n)
        body = "\n".join(lines[s - 1 : e])
        return {"path": path, "lines": f"{s}-{e}", "total_lines": n, "content": body}
    return {"path": path, "total_lines": n, "content": text[:_MAX_BYTES]}


@mcp.tool()
def search_code(
    query: str,
    subdir: Optional[str] = None,
    pattern: str = "*.py",
    regex: bool = False,
    max_results: int = 150,
) -> dict:
    """Search the repository source for a string (or regex), like grep.

    query: text to find. regex=true to treat it as an extended regex.
    subdir: repo-relative directory to scope to (default: the code dirs).
    pattern: filename glob to include (e.g. '*.py', '*.md', '*.i').
    Returns file:line:text matches (huge data/build trees are skipped).
    """
    root = repo_root()
    bases = (
        [str(_resolve(subdir))]
        if subdir
        else [str(root / d) for d in _CODE_ROOTS if (root / d).exists()]
    )
    cmd = ["grep", "-rInH", "--include", pattern]
    for d in _SKIP_DIRS:
        cmd += ["--exclude-dir", d]
    cmd += (["-E"] if regex else ["-F"]) + ["--", query] + bases
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    # Best-effort: return a friendly error instead of a traceback.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"error": f"search failed: {exc}"}
    lines = (r.stdout or "").splitlines()
    rel = []
    for ln in lines[:max_results]:
        try:
            fp, rest = ln.split(":", 1)
            rel.append(str(Path(fp).resolve().relative_to(root)) + ":" + rest)
        # Best-effort: keep the raw grep line if it can't be made repo-relative.
        except Exception:  # pylint: disable=broad-exception-caught
            rel.append(ln)
    return {
        "query": query,
        "n_matches": len(lines),
        "truncated": len(lines) > max_results,
        "matches": rel[:max_results],
    }

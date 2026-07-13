# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Recommendation recipes: per-setup vetted parameter presets as Markdown.

Each ``recipes/<name>.md`` file documents recommended parameters for a specific
setup (e.g. FF reconstruction of an equiaxed Ti alloy). They are:
  * served as MCP resources at ``recipe://<name>`` so a client can read them,
  * looked up by the ``get_recommended_parameters`` tool, and
  * meant to be edited/extended by domain experts -- they are plain Markdown,
    not code, so improving guidance never touches the server.

The optional YAML-ish front-matter block (``--- ... ---``) may carry a
``defaults:`` mapping the tools use to prefill parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

_RECIPE_DIR = Path(__file__).resolve().parent / "recipes"


def _split_front_matter(text: str):
    """Return (front_matter_text, body). Front matter is an optional leading
    ``---`` fenced block. Parsing is intentionally tiny -- no YAML dependency."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[3:end].strip()
            body = text[end + 4 :].lstrip("\n")
            return fm, body
    return "", text


def _parse_defaults(fm: str) -> Dict[str, str]:
    """Very small parser for a ``defaults:`` block of ``key: value`` lines."""
    defaults: Dict[str, str] = {}
    in_block = False
    for line in fm.splitlines():
        if line.strip() == "defaults:":
            in_block = True
            continue
        if in_block:
            if not line.startswith((" ", "\t")):
                break
            if ":" in line:
                k, v = line.split(":", 1)
                defaults[k.strip()] = v.strip()
    return defaults


# Markdown files in the recipe dir that are docs, not recipes.
_NON_RECIPES = {"readme"}


def list_recipes() -> List[str]:
    if not _RECIPE_DIR.exists():
        return []
    return sorted(
        p.stem for p in _RECIPE_DIR.glob("*.md")
        if p.stem.lower() not in _NON_RECIPES
    )


def get_recipe(name: str) -> Optional[dict]:
    if name.lower() in _NON_RECIPES:
        return None
    path = _RECIPE_DIR / f"{name}.md"
    if not path.exists():
        return None
    text = path.read_text()
    fm, body = _split_front_matter(text)
    return {
        "name": name,
        "markdown": body,
        "defaults": _parse_defaults(fm),
        "path": str(path),
    }

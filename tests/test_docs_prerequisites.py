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

"""Wherever the docs build is documented, its system prerequisites must be too.

``nbsphinx`` shells out to the ``pandoc`` binary, which is not a Python package
and so is not installed by ``pip install -e ".[docs]"``. A reader who follows
only the documented steps on a clean checkout hits ``nbsphinx.PandocMissing``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every file that tells a reader how to build the documentation.
DOCS_BUILD_GUIDES = ("README.md", "docs/development.rst", "docs/install.rst")

# Commands whose presence means the file is documenting the docs build.
BUILD_MARKERS = ('[docs]"', "make html", "sphinx-build")


def _read(relative_path):
    """Return the text of a repo file, skipping the test if it is absent."""
    path = REPO_ROOT / relative_path
    if not path.is_file():
        pytest.skip(f"{relative_path} is not present in this checkout")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("relative_path", DOCS_BUILD_GUIDES)
def test_docs_build_guide_names_pandoc(relative_path):
    """Any file documenting the docs build must name the pandoc prerequisite."""
    text = _read(relative_path)
    assert any(
        marker in text for marker in BUILD_MARKERS
    ), f"{relative_path} no longer documents the docs build; update DOCS_BUILD_GUIDES"
    assert (
        "pandoc" in text.lower()
    ), f"{relative_path} documents the docs build without naming pandoc"


def test_development_guide_gives_a_pandoc_install_route():
    """Naming pandoc is not enough; the reader needs a way to install it."""
    text = _read("docs/development.rst").lower()
    assert "pandoc.org" in text, "docs/development.rst does not link pandoc's own docs"
    assert any(
        route in text for route in ("apt-get install", "conda install", "brew install")
    ), "docs/development.rst names pandoc but gives no concrete install command"

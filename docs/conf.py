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

# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

# Put the package on the path so autodoc can import it.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "graintrace"
copyright = "2026, Argonne National Laboratory"
author = "Applied Material Modeling, Argonne National Laboratory"

try:
    release = _pkg_version("graintrace")
except PackageNotFoundError:
    release = "0.1.2"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
    "sphinx.ext.doctest",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

imgmath_latex_preamble = "\\usepackage{amsmath}"

# -- autodoc / napoleon ------------------------------------------------------
# graintrace lazy-imports the compiled/heavy stack. Mock everything that is not
# needed to read signatures and docstrings so the docs build (and CI) needs no
# NEML2/MOOSE/NEPER/CUBIT and no GPU. Mirrors ignored-modules in .pylintrc.
# Mock ONLY the packages that are genuinely not installed by ``pip install
# ".[docs]"`` (NEML2 is not on PyPI; torch_geometric is the [gnn] extra; mcp and
# meshio are other extras). Do NOT mock installed packages such as torch — some
# modules use ``torch.pi`` at import time, and a mock breaks that arithmetic.
autodoc_mock_imports = [
    "neml2",
    "torch_geometric",
    "mcp",
    "meshio",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Notebooks ship pre-executed; do not run them at build time (no GPU / stack).
nbsphinx_execute = "never"

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"graintrace {release}"

# -- copybutton --------------------------------------------------------------
# Strip shell / REPL prompts so a pasted snippet does not carry the prompt.
copybutton_prompt_text = r"^\$ |^>>> |^\.\.\. "
copybutton_prompt_is_regexp = True


# -- linkcheck ---------------------------------------------------------------
# PyTorch's docs use client-side (JS) anchors that the link checker cannot see;
# still verify the page exists, but skip anchor validation for those hosts.
linkcheck_anchors_ignore_for_url = [
    r"https://docs\.pytorch\.org/.*",
    r"https://pytorch\.org/.*",
]
linkcheck_timeout = 15
linkcheck_retries = 2


def _write_nojekyll(app, exception):
    # GitHub Pages runs Jekyll by default, which strips `_static/`, `_sources/`,
    # etc. A `.nojekyll` marker at the site root disables it. Sphinx has no
    # native option, so emit it from the build-finished hook.
    if app.builder.name == "html" and getattr(app, "outdir", None):
        (Path(app.outdir) / ".nojekyll").touch()


def setup(app):
    app.connect("build-finished", _write_nojekyll)

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

"""The lazy ``__getattr__`` must explain a missing optional dependency."""

from __future__ import annotations

import importlib.util

import pytest

import graintrace

INSTALL_DOCS = "https://applied-material-modeling.github.io/graintrace/install.html"


def _fail_import_with(monkeypatch, missing):
    """Make the lazy import of any submodule fail as if ``missing`` were absent."""

    def fake_import_module(name, package=None):  # pylint: disable=unused-argument
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(graintrace, "import_module", fake_import_module)


def test_missing_extra_dependency_names_symbol_extra_and_docs(monkeypatch):
    """A dependency that an extras group provides must point at that group."""
    _fail_import_with(monkeypatch, "torch_geometric")

    with pytest.raises(ModuleNotFoundError) as excinfo:
        graintrace.__getattr__("GraphGrainMatcher")

    message = str(excinfo.value)
    assert "graintrace.GraphGrainMatcher" in message
    assert "grain_graph_matching" in message
    assert "torch_geometric" in message
    assert 'pip install "graintrace[gnn]"' in message
    assert INSTALL_DOCS in message
    # The machine-readable attribute still names the real missing module.
    assert excinfo.value.name == "torch_geometric"


def test_missing_compiled_tier_dependency_points_at_the_build(monkeypatch):
    """NEML2 is not on PyPI, so no extras group may be suggested for it."""
    _fail_import_with(monkeypatch, "neml2")

    with pytest.raises(ModuleNotFoundError) as excinfo:
        graintrace.__getattr__("IPFProcessor")

    message = str(excinfo.value)
    assert "graintrace.IPFProcessor" in message
    assert "neml2" in message
    assert "pip install" not in message
    assert "PUMA" in message
    assert INSTALL_DOCS in message


def test_submodule_import_error_is_not_rewritten(monkeypatch):
    """A missing graintrace submodule is a packaging bug and must surface as-is."""

    def fake_import_module(name, package=None):  # pylint: disable=unused-argument
        raise ModuleNotFoundError(
            "No module named 'graintrace.taylor'", name="graintrace.taylor"
        )

    monkeypatch.setattr(graintrace, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        graintrace.__getattr__("TaylorModel")

    assert str(excinfo.value) == "No module named 'graintrace.taylor'"


def test_unknown_symbol_still_raises_attribute_error():
    """Unrelated names keep the normal AttributeError."""
    with pytest.raises(AttributeError):
        graintrace.__getattr__("NotASymbol")


@pytest.mark.skipif(
    importlib.util.find_spec("torch_geometric") is not None,
    reason="torch_geometric is installed, so the real import succeeds",
)
def test_real_missing_dependency_message():
    """End-to-end: a genuinely absent optional dependency gives the same message."""
    with pytest.raises(ModuleNotFoundError) as excinfo:
        getattr(graintrace, "NeperTessToGraphNN")

    message = str(excinfo.value)
    assert "graintrace.NeperTessToGraphNN" in message
    assert 'pip install "graintrace[gnn]"' in message
    assert INSTALL_DOCS in message

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

"""The MCP layer runs on the ``mcp`` SDK 1.x (FastMCP) and 2.x (MCPServer)."""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_server_registers_tools_on_installed_sdk():
    pytest.importorskip("mcp")
    import asyncio

    from graintrace.mcp import _compat, server

    assert _compat.MCP_SDK_MAJOR in (1, 2)
    assert isinstance(server.mcp, _compat.MCPServer)
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"run_cpfe", "calibrate_material", "dependency_status"} <= names


@pytest.mark.parametrize("major", [1, 2])
def test_run_streamable_http_dispatch(monkeypatch, major):
    pytest.importorskip("mcp")
    from types import SimpleNamespace

    from graintrace.mcp import _compat

    calls = {}

    class _Fake:
        settings = SimpleNamespace(host=None, port=None)

        def run(self, **kw):
            calls.update(
                kw, host_setting=self.settings.host, port_setting=self.settings.port
            )

    monkeypatch.setattr(_compat, "MCP_SDK_MAJOR", major)
    _compat.run_streamable_http(_Fake(), "0.0.0.0", 9001)
    assert calls["transport"] == "streamable-http"
    if major == 2:
        assert (calls["host"], calls["port"]) == ("0.0.0.0", 9001)
    else:
        assert (calls["host_setting"], calls["port_setting"]) == ("0.0.0.0", 9001)


def test_tool_paths_import_without_the_sdk():
    """``graintrace.mcp`` is lazy: tool_paths/deps import with ``mcp`` blocked."""
    code = (
        "import sys; sys.modules['mcp'] = None;"
        "from graintrace.mcp import tool_paths, deps;"
        "import graintrace.mcp as m;"
        "\ntry:\n    m.mcp\nexcept ImportError:\n    print('lazy-ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "lazy-ok"

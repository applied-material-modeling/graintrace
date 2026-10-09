# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
"""Tests for the ``python -m graintrace.doctor`` dependency report."""

import json
import subprocess
import sys

from graintrace import doctor


def test_report_covers_every_probe_with_a_remedy():
    """Each row names a dependency and, when absent, how to get it."""
    rows = doctor.report()
    assert rows, "the probe registry should not be empty"
    names = {r["name"] for r in rows}
    # The external stack the workflows depend on must all be reported.
    assert {"neper", "puma-opt", "cubit", "neml2", "gpu"} <= names
    for row in rows:
        assert set(row) == {"name", "ok", "detail", "build_hint"}
        assert isinstance(row["ok"], bool)
        assert row["detail"], f"{row['name']} reported no detail"
        if not row["ok"]:
            assert row["build_hint"], f"{row['name']} is missing with no remedy"


def test_format_report_marks_missing_and_lists_remedies():
    """The table flags missing entries and appends their build hints."""
    rows = [
        {"name": "neper", "ok": False, "detail": "not found", "build_hint": "build it"},
        {"name": "neml2", "ok": True, "detail": "importable", "build_hint": ""},
    ]
    text = doctor.format_report(rows)
    assert "MISSING  neper" in text
    assert "ok       neml2" in text
    assert "neper: build it" in text


def test_require_flag_fails_only_on_an_absent_dependency():
    """``--require`` is the scripting hook: exit 1 iff something named is absent."""
    rows = {r["name"]: r["ok"] for r in doctor.report()}
    present = next((n for n, ok in rows.items() if ok), None)
    absent = next((n for n, ok in rows.items() if not ok), None)
    if present is not None:
        assert doctor.main(["--require", present]) == 0
    if absent is not None:
        assert doctor.main(["--require", absent]) == 1
    # An unknown name is treated as absent rather than silently ignored.
    assert doctor.main(["--require", "no-such-dependency"]) == 1


def test_json_output_is_parseable():
    """``--json`` emits the same rows in a machine-readable form."""
    out = subprocess.run(
        [sys.executable, "-m", "graintrace.doctor", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(out.stdout)
    assert {r["name"] for r in parsed} == {r["name"] for r in doctor.report()}


def test_runs_without_the_mcp_extra_imported():
    """The probes must work on a bare install, where ``mcp`` is not available.

    ``graintrace.mcp.__init__`` imports FastMCP, so a subprocess that blocks the
    ``mcp`` package stands in for an install without the optional extra.
    """
    code = (
        "import sys;"
        "sys.modules['mcp'] = None;"
        "from graintrace import doctor;"
        "print(len(doctor.report()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert int(out.stdout.strip()) > 0

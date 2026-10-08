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

"""Example scripts must write to private, untracked output paths.

The three REI examples each regenerate a synthetic dataset. If they share an
output path, running them in sequence silently changes each other's inputs; if
that path is tracked by git, running a tutorial dirties the reader's checkout.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Examples that generate their own synthetic dataset, and the module-level
# string variables naming the files they write.
REI_EXAMPLES = (
    "examples/demonstrate_rei_example_2D.py",
    "examples/demonstrate_rei_example_3D.py",
    "examples/demonstrate_rei_pipeline.py",
)
OUTPUT_VARIABLES = ("filename", "output_folder", "vtk_out")


def _string_assignments(relative_path):
    """Module-level ``name = "literal"`` assignments in an example script.

    Args:
        relative_path (str): path to the script, relative to the repo root

    Returns:
        dict: variable name -> string literal, for the names in OUTPUT_VARIABLES
    """
    path = REPO_ROOT / relative_path
    if not path.is_file():
        pytest.skip(f"{relative_path} is not present in this checkout")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in OUTPUT_VARIABLES:
                found[target.id] = node.value.value
    return found


def _git(*args):
    """Run a git command in the repo, returning (returncode, stdout)."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout


def _require_git_checkout():
    """Skip unless the tests are running inside the git working tree."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout; cannot ask git what is tracked or ignored")
    if _git("rev-parse", "--is-inside-work-tree")[0] != 0:
        pytest.skip("git is unavailable")


def test_rei_examples_write_to_distinct_datasets():
    """No two REI examples may generate over the same dataset path."""
    paths = {script: _string_assignments(script)["filename"] for script in REI_EXAMPLES}
    duplicates = [p for p in paths.values() if list(paths.values()).count(p) > 1]
    assert not duplicates, (
        "REI examples share a generated dataset path, so running one changes "
        f"another's input: {paths}"
    )


@pytest.mark.parametrize("script", REI_EXAMPLES)
def test_rei_example_outputs_are_untracked(script):
    """Everything an example writes must be git-ignored, never a tracked file."""
    _require_git_checkout()
    for variable, value in sorted(_string_assignments(script).items()):
        tracked = _git("ls-files", "--error-unmatch", "--", value)[0] == 0
        assert not tracked, (
            f"{script} writes {variable}={value!r}, which is tracked by git; "
            "running the example dirties the checkout"
        )
        # A value with no file extension is an output directory. Probe a file
        # inside it, so the answer does not depend on the directory happening
        # to exist from an earlier run.
        probe = value if Path(value).suffix else f"{value}/probe"
        ignored = _git("check-ignore", "-q", "--", probe)[0] == 0
        assert ignored, (
            f"{script} writes {variable}={value!r}, which is neither tracked "
            "nor git-ignored; add it to .gitignore"
        )

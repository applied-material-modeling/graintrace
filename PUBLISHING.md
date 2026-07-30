# Publishing graintrace

Runbook for building and publishing the `graintrace` wheel to PyPI, plus how to re-point the
external submodules. **Nothing here is automated** — run each step yourself with your own
credentials. Do **not** `git push` or `twine upload` until the release is reviewed.

## 0. Pre-flight

- Confirm the version in `pyproject.toml` (`version = "0.1.0"`) is the intended release.
- Check the name is free / yours on PyPI: <https://pypi.org/project/graintrace/>.
- Ensure the working tree is clean of build artifacts: `rm -rf build dist *.egg-info`.
- Verify no proprietary/private data is tracked:
  ```bash
  git ls-files | grep -Ei '\.lic$|licen|cubit|coreform|clarolic' | grep -v '^LICENSE$'   # -> empty
  git grep -n "/home/tranh"                                                                # -> empty (author paths; covers demo/, .claude/, examples/)
  git ls-files | grep -E '\.webui_secret_key$|^graintrace_mcp_out/'                        # -> empty (secret / run artifacts untracked)
  ```

## 1. Build

```bash
conda activate graintrace_env
pip install -U build twine        # or: pip install -e ".[dev]"
python -m build                   # writes dist/graintrace-<ver>-py3-none-any.whl + .tar.gz
```

## 2. Check

```bash
twine check dist/*
# confirm the wheel ships the runtime data + all subpackages:
python -m zipfile -l dist/graintrace-*.whl | grep -E 'cpfe_base/.*\.i|mcp/recipes/.*\.md'
```

## 3. TestPyPI dry-run (recommended)

```bash
twine upload --repository testpypi dist/*
# then, in a scratch env, confirm the listing installs + imports:
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ graintrace
python -c "import graintrace; print(graintrace.__version__)"
```

## 4. PyPI upload

```bash
twine upload dist/*
```

Use a PyPI API token (`__token__` / `pypi-...`), ideally via `~/.pypirc` or `TWINE_PASSWORD`.

## 5. Tag the release (after confirmation)

```bash
git tag -a v0.1.0 -m "graintrace 0.1.0"
git push origin v0.1.0        # only after the release is confirmed
```

---

## Re-pointing the PUMA submodule

The native stack is pinned via a **single** submodule, `external/puma` (PUMA carries `moose/` and
`neml2/` as its own submodules, so graintrace pins PUMA once and inherits the whole stack). NEML2/
pyzag are **not** graintrace dependencies — they are the repo-pinned NEML2 source that PUMA builds
and installs into `graintrace_env`. Source-of-truth is `.gitmodules` (URL + branch) + the committed
gitlink (commit).

### Procedure

```bash
git submodule set-branch --branch <branch> external/puma      # currently: development
git submodule update --init --recursive external/puma
git -C external/puma fetch origin
git -C external/puma checkout <new-tag-or-commit>             # pin to a tag/commit, not a branch
git add .gitmodules external/puma
git commit -m "external/puma: re-point to applied-material-modeling/puma@<ref>"
```

Pin to a **tag or commit** (not a moving branch) so re-points are deliberate and reproducible.
To bump the underlying MOOSE/NEML2, update PUMA's own moose/neml2 submodule pins in the puma repo
and re-point `external/puma` to that PUMA commit. After re-pointing, rebuild (README "Install" →
tier 2, via PUMA) and re-run `pytest` + `examples/demonstrate_cpfe.py` before releasing.

> Note: `external/puma` is a pinned gitlink, **not** checked out in a fresh clone until
> `git submodule update --init --recursive external/puma`. This keeps clones small by default; the
> multi-GB MOOSE tree (a submodule of PUMA) is only fetched on recursive init.

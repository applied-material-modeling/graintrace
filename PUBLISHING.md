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
  git grep -n "/home/" -- ':!examples'                                                     # -> empty
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

## Re-pointing the external submodules

The compiled stack is pinned as submodules under `external/`. All source-of-truth lives in
`.gitmodules` (URL + branch) + the committed gitlink (commit). To move from the current
development forks to the official upstream repos, update **only** `.gitmodules` + the gitlink:

### Current pins → intended official targets

| Submodule | Current (fork) | Commit | Intended official |
|---|---|---|---|
| `external/moose`  | hugary1995/moose @ `neml2-v3-migration` | `54c0d6a` | idaholab/moose |
| `external/neml2`  | hdt5kt/neml2 @ `pyzag_v3_port` | `7b3c8d0` | applied-material-modeling/neml2 |
| `external/puma`   | applied-material-modeling/puma @ `development` | `525cc29` | applied-material-modeling/puma |
| `external/pyzag`  | applied-material-modeling/pyzag @ `huy_pyzag_abstraction_neml2_v3` | `e053563` | applied-material-modeling/pyzag |

### Procedure (per submodule)

```bash
git submodule set-url    external/<name> <new-official-url>
git submodule set-branch --branch <new-branch> external/<name>
git submodule update --init external/<name>
git -C external/<name> fetch origin
git -C external/<name> checkout <new-tag-or-commit>
git add .gitmodules external/<name>
git commit -m "external/<name>: re-point to <official>@<ref>"
```

Pin to a **tag or commit** (not a moving branch) so re-points are deliberate and reproducible.
After re-pointing, rebuild (README "Building MOOSE + NEML2 + PUMA") and re-run `pytest` +
`examples/demonstrate_cpfe.py` before releasing.

> Note: the submodules are registered as pinned gitlinks and are **not** checked out in a fresh
> clone until `git submodule update --init` (or `git clone --recursive`). This keeps clones
> small by default; the multi-GB MOOSE tree is only fetched when you initialize it.

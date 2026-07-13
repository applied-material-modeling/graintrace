# graintrace

Crystal-plasticity finite-element (CPFE) pipeline for APS HEDM experiments.

`graintrace` links experimental grain-scale characterization data (far-field / near-field
HEDM, EBSD) to MOOSE/PUMA CPFE simulations with NEML2 v3 material models. It reconstructs 3D
microstructure meshes from raw experimental data, calibrates crystal-plasticity parameters,
runs CPFE simulations, and post-processes results for rare-event identification (REI).

## What you get from `pip` vs. what you must build

`graintrace` is the **Python** layer. The heavy compiled / licensed stack it *drives* is **not**
on PyPI and cannot be `pip install`ed:

| Provided by `pip install graintrace` | Must be provided separately |
|---|---|
| the `graintrace` package + its Python deps (numpy, pandas, scipy, torch, pyvista, gmsh Python API, …) | **NEML2 v3**, **pyzag**, **MOOSE + PUMA** (`puma-opt`), **libtorch** — built from source (git submodules below) |
| | **NEPER** + standalone **gmsh** binary — installed separately |
| | **Coreform CUBIT/SCULPT** — proprietary, licensed; obtain your own license (never commit it) |

So `pip install graintrace` gives you an importable package and lets you run the
Python-only parts (post-processing, REI, stitching); running reconstruction/CPFE end-to-end
requires the external tools.

## Requirements

- python **>= 3.10**, conda with pip
- NEPER, gmsh
- CUBIT/SCULPT (Coreform license required)
- MOOSE with the PUMA app, linked with NEML2 (v3) + libtorch
- NEML2 (v3) and pyzag

## Quick install (Python package only)

```bash
pip install graintrace            # once published to PyPI
# optional extras:
pip install "graintrace[gnn]"     # grain-graph / GNN utilities (torch-geometric)
pip install "graintrace[mcp]"     # MCP server (drive graintrace from Claude Desktop / Open WebUI)
pip install "graintrace[examples]"  # deps used by examples/ (meshio)
```

Or from source with the conda environment:

```bash
git clone --recursive https://github.com/applied-material-modeling/graintrace.git
cd graintrace
conda env create -f environment.yml      # creates env "graintrace_env"
conda activate graintrace_env
pip install -e .
```

`--recursive` (or `git submodule update --init`) fetches the pinned external sources under
`external/` (see below). The `environment.yml` mirrors a working `moose-src`-style env
(neml2 v3 + editable pyzag + torch/CUDA).

## External compiled stack (git submodules)

The exact working versions of MOOSE/NEML2/PUMA/pyzag are pinned as git submodules under
`external/`:

| Submodule | Repo | Branch |
|---|---|---|
| `external/moose`  | github.com/hugary1995/moose | neml2-v3-migration |
| `external/neml2`  | github.com/hdt5kt/neml2 | pyzag_v3_port |
| `external/puma`   | github.com/applied-material-modeling/puma | development |
| `external/pyzag`  | github.com/applied-material-modeling/pyzag | huy_pyzag_abstraction_neml2_v3 |

```bash
git submodule update --init external/moose external/neml2 external/puma external/pyzag
```

> These currently track development forks/branches; they are expected to move to the official
> upstream repos over time (see `PUBLISHING.md` for the one-command re-point procedure). The
> submodules are pinned to specific commits, so updates are deliberate.

## Building MOOSE + NEML2 + PUMA (for reconstruction/CPFE)

Build inside your conda env (worked on Ubuntu 20.04 with the appropriate MPI/compilers). Use
the submodule checkouts under `external/`:

```bash
export CC=mpicc CXX=mpicxx FC=mpif90 F90=mpif90 F77=mpif77
export MOOSE_DIR=${PWD}/external/moose MOOSE_JOBS=12 METHODS=opt

# 1) MOOSE deps
cd external/moose/scripts
./update_and_rebuild_petsc.sh
./update_and_rebuild_libmesh.sh
./update_and_rebuild_wasp.sh
cd ../../..

# 2) libtorch (GPU build; pick the CUDA version matching your driver)
#    https://pytorch.org/get-started/locally/ -> Stable / Linux / LibTorch
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.10.0%2Bcu126.zip
unzip libtorch-shared-with-deps-2.10.0+cu126.zip
export LIBTORCH_DIR=${PWD}/libtorch

# 3) NEML2 + MOOSE
cd external/moose
./configure --with-libtorch --with-neml2
./scripts/update_and_rebuild_neml2.sh
cd ../..

# 4) PUMA (MOOSE app that runs CPFE)
cd external/puma
make -j 12
./run_tests          # all should pass
cd ../..

# 5) NEML2 Python bindings (v3) into the active env
pip install ./external/neml2 -v
# pyzag (editable) into the active env
pip install -e ./external/pyzag
```

The PUMA binary is then `external/puma/puma-opt` — pass this as `moose_run_file` to
`CPFESimulation` (see `examples/demonstrate_cpfe.py`).

## CUBIT/SCULPT (proprietary — bring your own license)

Coreform CUBIT (National-Lab, commercial, or education license) provides CUBIT + SCULPT:
<https://coreform.com/>. Obtain and install it under your own account.

> **Never commit CUBIT license material** (`*.lic`, license servers, keys) to this or any repo.
> `sculpt_config` in the examples takes only **executable paths** — no license tokens. Set them
> to your install, e.g.:
> ```python
> sculpt_config = {
>     "launcher": "/path/to/cubit/bin/mpi/bin/mpiexec",
>     "psculpt":  "/path/to/cubit/bin/psculpt",
>     "epu":      "/path/to/cubit/bin/epu",
>     "nprocs":   int(ncore),
>     "environment": {"OPAL_LIBDIR": "/path/to/cubit/bin/mpi/lib",
>                     "OPAL_PREFIX": "/path/to/cubit/bin/mpi"},
> }
> ```

## Examples & workflow segments

Runnable examples live in `examples/demonstrate_*.py` (flat top-level `## INPUT` style). Many
use the small self-contained datasets shipped under `mwe_data/`. Each workflow segment also has
a `/skill` guide (`.claude/skills/…`); `.claude/CLAUDE.md` §12 maps segment → example → skill.

Minimum end-to-end check (needs the full external stack): `examples/demonstrate_cpfe_nfff.py`.
Before running, edit the `sculpt_config`, `moose_run_file`, `ncore`, and `device` at the top of
the script to match your machine.

Python-only checks (no MOOSE/CUBIT/NEPER needed):

- `examples/demonstrate_postprocess.py`
- `examples/demonstrate_rei_pipeline.py`, `examples/demonstrate_rei_example_2D.py`, `..._3D.py`
- `examples/demonstrate_graintracking.py` (needs NEPER)

## Running the test suite

```bash
conda activate graintrace_env
pip install -e ".[dev]"
pytest tests/
```

All **86 tests** should pass. The suite covers data classes, similarity metrics, clustering,
orientation math, simulation/experiment post-processing, and stitching. It also checks that
`neml2`/`pyzag` import; CUBIT-binary tests skip unless you set `PSCULPT` / `CUBIT_MPIEXEC`
(or `CUBIT_BIN_DIR`).

```bash
pytest tests/ -m "not slow"          # fast pure-Python subset
pytest tests/test_dependencies.py -v # environment checks
```

## License

MIT — see [LICENSE](LICENSE). Copyright 2026, UChicago Argonne, LLC / Argonne National
Laboratory.

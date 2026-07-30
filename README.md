# graintrace

Crystal plasticity finite element (CPFE) pipeline for APS HEDM experiments.

graintrace is a Python toolkit that links experimental grain scale characterization (far field and near field high energy diffraction microscopy, HEDM, plus EBSD) to CPFE simulation. It ingests raw scan data, reconstructs a 3D microstructure mesh, calibrates a crystal plasticity material model, runs the CPFE simulation, and analyzes the results to pinpoint the grains and regions worth measuring in more detail.

**Architecture.** graintrace is the Python orchestration layer over a compiled scientific stack:

- **MOOSE / PUMA**: finite element framework and solver.
- **NEML2 v3**: GPU accelerated crystal plasticity constitutive models, AOTI compiled.
- **pyzag**: analytic adjoint gradients for material calibration.
- **NEPER**: Voronoi/CVT tessellation; **Coreform CUBIT/SCULPT** (or gmsh): meshing.

The pipeline runs reconstruct, calibrate, simulate (CPFE), then analyze and identify rare events.

**Features**

- **Experiment data handling**: ingest and register real FF/NF HEDM and EBSD scans, and reconstruct a 3D microstructure. FF grain centroids become a Voronoi/CVT tessellation (NEPER); NF and EBSD voxel fields are segmented into grains (flood fill or graph/Leiden clustering) and meshed to conformal hex with CUBIT/SCULPT; multiple FF scan layers are stitched into one grain set by region matching. *See `examples/demonstrate_farfield.py`, `examples/demonstrate_cpfe_nfff.py`, `examples/demonstrate_grid_segmentation_mesh.py`.*
- **Virtual microstructure generation**: synthesize microstructures faithful to input grain size and orientation distributions via NEPER morphology control, accounting for HEDM scanning strategies. *See `examples/demonstrate_hedm_study.py`.*
- **GPU CPFE and fast calibration**: run CPFE with NEML2 AOTI compiled crystal plasticity models on GPU; calibrate the material to a macroscopic stress vs strain curve with a pyzag analytic adjoint Taylor model, in under 30 minutes for 100+ grains. *See `examples/demonstrate_cpfe.py`, `examples/demonstrate_material_calibration.py`.*
- **Grain tracking**: match grains across load steps by building a grain graph from each reconstruction and matching via message passing. *See `examples/demonstrate_graintracking.py`.*
- **Rare event identification (REI)**: locate spatially coherent rare regions in CPFE fields via graph spatial clustering (Leiden) with hierarchical merging, flagging grains and locations for targeted measurement; scales to tens of millions of query points in under an hour. *See `examples/demonstrate_rei_pipeline.py`.*

Analysis (field distributions, macroscopic stress vs strain, pole figures, IPF coloring) rounds out the pipeline. *See `examples/demonstrate_postprocess.py`.*

## What you get from `pip` vs. what you must build

`graintrace` is the **Python** layer. The heavy compiled / licensed stack it *drives* is **not**
on PyPI and cannot be `pip install`ed:

| Provided by `pip install graintrace` | Must be provided separately |
|---|---|
| the `graintrace` package + its Python deps (numpy, pandas, scipy, torch, pyvista, gmsh Python API, …) | **NEML2 v3 + pyzag**: provided by the **PUMA** build (see below), *not* pinned by graintrace |
| | **MOOSE + PUMA** (`puma-opt`), **libtorch**: built from source via PUMA (git submodule below) |
| | **NEPER** + standalone **gmsh** binary: installed separately |
| | **Coreform CUBIT/SCULPT**: proprietary, licensed; obtain your own license (never commit it) |

`graintrace/__init__.py` lazy-imports the compiled stack, so `pip install graintrace` gives you
an importable package that runs the **Python-only** parts (post-processing, REI, stitching,
similarity metrics) with no NEML2 present. Features that use NEML2 (material calibration, pole
figures, CPFE) require the **PUMA-built NEML2** in the shared `graintrace_env`; graintrace does
**not** install neml2/pyzag itself, because the working, ABI-matched build is the repo-pinned
NEML2 source that PUMA builds (so `puma-opt`'s C++ library and the Python package stay in lockstep;
a public PyPI neml2 wheel is a different build and is deliberately not relied upon).

## Requirements

- python **>= 3.10**, conda with pip
- NEPER, gmsh
- CUBIT/SCULPT (Coreform license required)
- MOOSE with the PUMA app, linked with NEML2 (v3) + libtorch
- NEML2 (v3) and pyzag

## Install

There are two tiers, depending on whether you need the NEML2/CPFE features.

**1. Python-only** (post-processing, REI, stitching, rei_comparison, similarity metrics). No
conda, NEML2, or native build needed:

```bash
pip install graintrace            # once published to PyPI
# from a source checkout instead:  pip install -e .
# optional extras:
pip install "graintrace[gnn]"       # grain-graph / GNN utilities (torch-geometric)
pip install "graintrace[mcp]"       # MCP server (drive graintrace from Claude Desktop/Code or any MCP client)
pip install "graintrace[examples]"  # deps used by examples/ (meshio)
pip install "graintrace[dev]"       # test/lint/build tooling (pytest, black, isort, build, twine)
```

`import graintrace` works with no NEML2 present (the compiled stack is lazy-imported).

**2. NEML2 / CPFE features** (material calibration, pole figures, reconstruction, CPFE). These
need the native stack, which is provided entirely by **PUMA**. Build PUMA first; it creates the
shared conda env `graintrace_env`, builds MOOSE + libtorch + the repo-pinned NEML2 v3, and installs
the ABI-matched NEML2/pyzag Python packages into that env. Then install graintrace on top:

```bash
git clone https://github.com/applied-material-modeling/graintrace.git
cd graintrace
git submodule update --init --recursive external/puma   # pulls puma -> moose + neml2

# Build the native stack via PUMA (see external/puma/README.md for prerequisites):
cd external/puma
conda env create -f environment.yml        # creates "graintrace_env" (toolchain + NEML2 v3 stack)
conda activate graintrace_env
scripts/get_dependencies.sh --build        # inits moose/neml2, builds NEML2 into MOOSE
make -j                                     # builds puma-opt
cd ../..

# Install graintrace into the same env (does NOT reinstall neml2/pyzag):
pip install -e .        # or:  pip install -e ".[dev]"  to run the tests
```

graintrace depends on PUMA (one-way): the only thing it needs at runtime is the `puma-opt` binary
at `external/puma/puma-opt`; pass it as `moose_run_file` to `CPFESimulation` (see
`examples/demonstrate_cpfe.py`). graintrace intentionally does **not** pin or install neml2/pyzag;
they come from PUMA's build so `puma-opt`'s C++ library and the Python package stay in lockstep.

## External compiled stack (single PUMA submodule)

The native stack is pinned via **one** submodule; PUMA carries MOOSE and NEML2 as its own
submodules, so graintrace pins PUMA once and gets the whole stack:

| Submodule | Repo | Branch | Carries |
|---|---|---|---|
| `external/puma` | github.com/applied-material-modeling/puma | development | `moose/` + `neml2/` (its own submodules) |

```bash
git submodule update --init --recursive external/puma
```

> Pinned to a specific PUMA commit (recursive init also pins PUMA's moose/neml2), so updates are
> deliberate. See `PUBLISHING.md` for the re-point procedure.

## CUBIT/SCULPT (proprietary; bring your own license)

Coreform CUBIT (National-Lab, commercial, or education license) provides CUBIT + SCULPT:
<https://coreform.com/>. Obtain and install it under your own account.

> **Never commit CUBIT license material** (`*.lic`, license servers, keys) to this or any repo.
> `sculpt_config` in the examples takes only **executable paths**, no license tokens. Set them
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
pip install -e ".[dev]"
pytest tests/
```

**101 tests.** On a plain checkout without the PUMA-built NEML2, the NEML2/pyzag-dependent tests
(orientation math, dependency checks) **skip** via `pytest.importorskip` rather than error, so the
pure-Python subset (data classes, similarity metrics, clustering, post-processing, stitching, REI
comparison) runs green. Run inside the PUMA-built `graintrace_env` to exercise the full suite
(then **99 pass, 2 skip**; the 2 skips are the CUBIT-binary checks, which skip unless you set
`PSCULPT` / `CUBIT_MPIEXEC` or `CUBIT_BIN_DIR`).

```bash
pytest tests/ -m "not slow"          # fast pure-Python subset
pytest tests/test_dependencies.py -v # environment checks
```

## License

MIT. See [LICENSE](LICENSE). Copyright 2026, UChicago Argonne, LLC / Argonne National
Laboratory.

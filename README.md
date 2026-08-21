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

## Capabilities and what each one needs

`graintrace` is the **Python** layer. Some features are pure-Python and run straight from
`pip install graintrace`; others drive a compiled or licensed stack you install separately.
`graintrace/__init__.py` lazy-imports the compiled stack, so `import graintrace` always works and
a feature raises a clear error only when a tool it needs is missing.

| Capability | `pip install graintrace` | NEML2 (Python) | PUMA `puma-opt` | NEPER | CUBIT/SCULPT |
|---|:---:|:---:|:---:|:---:|:---:|
| Post-processing (distributions, stress–strain) | ✅ | | | | |
| Rare-event identification (REI) + REI comparison | ✅ | | | | |
| HEDM stitching, similarity metrics | ✅ | | | | |
| FF Voronoi reconstruction, grain tracking | ✅ | | | ✅ | |
| NF / EBSD segmentation + hex meshing | ✅ | | | | ✅ |
| Material calibration | ✅ | ✅ | | | |
| Pole figures / orientation math | ✅ | ✅ | | | |
| CPFE simulation | ✅ | ✅ | ✅ | ✅¹ | ✅¹ |

¹ CPFE needs a mesh: FF meshes come from NEPER, NF/EBSD meshes from CUBIT/SCULPT.

`pyzag` is installed automatically by `pip install graintrace` (it is pure-Python and on PyPI). The
**NEML2 (Python)** column is the `neml2` package built by PUMA (see [Install](#install)); **PUMA
`puma-opt`** is the compiled solver binary, built with MOOSE + libtorch. **CUBIT/SCULPT** is
proprietary (Coreform license); **NEPER** is a separate tool graintrace drives. graintrace does not
install NEML2 from PyPI: PUMA builds NEML2 from its pinned submodule so `puma-opt`'s C++ library and
the Python `neml2` stay ABI-matched.

## Requirements

- python **>= 3.10**, conda with pip
- gmsh and pyzag 2.0.0 (both installed automatically with graintrace via pip)
- NEPER (bring your own; install from <https://neper.info> and point graintrace at it)
- CUBIT/SCULPT (Coreform license required)
- NEML2 v3 (Python) — built by PUMA
- MOOSE + PUMA (`puma-opt`), linked with NEML2 v3 + libtorch — for CPFE

## Install

Three tiers by capability (see the table above). Each higher tier adds the PUMA-built native
stack on top of the pip install.

### 1. Python-only

Post-processing, REI, REI comparison, stitching, similarity metrics. No conda or native build
needed:

```bash
pip install graintrace            # once published to PyPI
# from a source checkout instead:  pip install -e .
# optional extras:
pip install "graintrace[gnn]"       # grain-graph / GNN utilities (torch-geometric)
pip install "graintrace[mcp]"       # MCP server (drive graintrace from Claude Desktop/Code or any MCP client)
pip install "graintrace[examples]"  # deps used by examples/ (meshio)
pip install "graintrace[dev]"       # test/lint/build tooling (pytest, black, isort, build, twine)
```

`import graintrace` works with no NEML2 present (the compiled stack is lazy-imported). This tier
also installs `pyzag`.

### 2. NEML2 features without CPFE

Material calibration and pole figures need the NEML2 **Python** package but **not** the full
MOOSE/`puma-opt` solver. Build only NEML2 from PUMA's submodule — the lightweight "NEML2-only" PUMA
path (no PETSc/libMesh/MOOSE build):

```bash
git clone https://github.com/applied-material-modeling/graintrace.git
cd graintrace
git submodule update --init external/puma        # pulls puma -> moose + neml2

# Create the PUMA Python environment (see external/puma/README.md "Python environment"):
conda create -n puma python=3.13 mpich gcc_linux-64 gxx_linux-64 gfortran_linux-64 \
  cmake make ninja hdf5 netcdf4 zlib libaec bison flex m4 pkg-config
conda activate puma
pip install torch nmhit scikit-build-core ninja

# Build ONLY the NEML2 Python package from the submodule (skip the MOOSE/puma-opt build):
cd external/puma
git submodule update --init neml2
pip install ./neml2 --no-deps                      # NEML2 v3 Python + `neml2-compile`
cd ../..

pip install -e .                                   # graintrace + pyzag into the same env
```

`--no-deps` on NEML2 keeps the `pyzag==2.0.0` that graintrace installs; the published NEML2 PyPI
wheel pins an older, incompatible pyzag, which is why NEML2 comes from source here.

### 3. Full CPFE

Reconstruct → simulate needs the whole native stack: MOOSE + libtorch + `puma-opt` + NEML2. Build
it via PUMA, then install graintrace into that same env:

```bash
git clone https://github.com/applied-material-modeling/graintrace.git
cd graintrace
git submodule update --init external/puma

# Build the native stack: follow external/puma/README.md end to end (conda env, submodules,
# PETSc/libMesh/WASP, NEML2, `make -j`, and `neml2-compile` for the material models).
cd external/puma
# ... PUMA build steps ...
cd ../..

pip install -e .        # or:  pip install -e ".[dev]"  to run the tests
```

graintrace depends on PUMA one-way: at runtime it needs the `puma-opt` binary at
`external/puma/puma-opt`; pass it as `moose_run_file` to `CPFESimulation` (see
`examples/demonstrate_cpfe.py`). NEML2 comes from PUMA's build so `puma-opt`'s C++ library and the
Python `neml2` stay in lockstep; `pyzag` is provided by graintrace's pip install.

## External compiled stack (single PUMA submodule)

The native stack is pinned via **one** submodule; PUMA carries MOOSE and NEML2 as its own
submodules, so graintrace pins PUMA once and gets the whole stack:

| Submodule | Repo | Branch | Carries |
|---|---|---|---|
| `external/puma` | github.com/applied-material-modeling/puma | development | `moose/` + `neml2/` (its own submodules) |

```bash
git submodule update --init external/puma          # graintrace -> puma
cd external/puma && git submodule update --init    # puma -> moose/ + neml2/
```

The init is intentionally **not** recursive: it stops at PUMA's `moose/` and `neml2/`. MOOSE's own
submodules (PETSc, libMesh, WASP) are initialized by MOOSE's build scripts during the PUMA build,
not here. Each gitlink pins a specific commit, so updates are deliberate: to re-point, update
`.gitmodules` and the `external/puma` gitlink, then re-init.

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

## NEPER and gmsh

**gmsh** is a pip package (declared in `pyproject.toml`), so it is installed automatically with
`pip install graintrace` (or `pip install -e .`). Nothing extra to do.

**NEPER** is an external tool graintrace drives (like CUBIT/SCULPT); it is **not** a pip package, so
you install it yourself. Install NEPER once, then point graintrace at it:

1. Install NEPER: see <https://neper.info/doc/introduction.html#installing-neper> (a distro package,
   the official tarball, or a from-source build all work).
2. Let graintrace find it, in this precedence order:
   - the `NEPER` environment variable set to the absolute path of the `neper` binary
     (`export NEPER=/abs/path/to/neper`),
   - a `graintrace_tools.json` with a `"neper"` key (see `graintrace/mcp/tools.example.json`; the
     search order is `$GRAINTRACE_TOOLS_JSON` → `./graintrace_tools.json` → `~/.config/graintrace/tools.json`),
   - `neper` on your `PATH`,
   - or pass `neper_path=/abs/path/to/neper` (or a prepared `env=`) to `VoronoiMeshBuilder` /
     `CrystalGenerator`.

If NEPER cannot be found, the builders raise a clear error with these instructions. On Linux you can
also pass `auto_install=True` to build GSL + OpenBLAS + NEPER into `~/.local`.

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
comparison) runs green. Run inside the PUMA-built env to exercise the full suite
(then **99 pass, 2 skip**; the 2 skips are the CUBIT-binary checks, which skip unless you set
`PSCULPT` / `CUBIT_MPIEXEC` or `CUBIT_BIN_DIR`).

```bash
pytest tests/ -m "not slow"          # fast pure-Python subset
pytest tests/test_dependencies.py -v # environment checks
```

## License

MIT. See [LICENSE](LICENSE). Copyright 2026, UChicago Argonne, LLC / Argonne National
Laboratory.

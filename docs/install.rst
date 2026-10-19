Installation
============

Requirements
------------

- Python **>= 3.10**, with pip (conda recommended for the native stack).
- gmsh and pyzag 2.0.0: installed automatically with graintrace via pip.
- NEPER: bring your own. Install from https://neper.info and point graintrace
  at it (see :ref:`neper-gmsh`).
- CUBIT/SCULPT: for hex meshing; Coreform license required.
- NEML2 v3 (Python): for calibration and pole figures; built by PUMA.
- MOOSE + PUMA (``puma-opt``): for CPFE; linked with NEML2 v3 + libtorch.
- pandoc: only to *build* the documentation. A system binary, not a pip
  package, so the ``docs`` extra cannot install it
  (see :ref:`pandoc-prerequisite`).

Installation is organized in three tiers by capability. Each higher tier adds
the PUMA-built native stack on top of the pip install.

Tier 1: Python only
-------------------

Post-processing, rare-event identification, REI comparison, stitching, and
similarity metrics. No conda or native build needed:

.. code-block:: bash

   pip install graintrace
   # from a source checkout instead:
   pip install -e .

   # optional extras:
   pip install "graintrace[gnn]"       # grain-graph / GNN utilities (torch-geometric)
   pip install "graintrace[mcp]"       # MCP server
   pip install "graintrace[examples]"  # deps used by examples/ (meshio)
   pip install "graintrace[dev]"       # test/lint/build tooling
   pip install "graintrace[docs]"      # documentation build (Sphinx)

``import graintrace`` works with no NEML2 present. The compiled stack is
lazy-imported. This tier also installs ``pyzag``.

Accessing a symbol whose dependency is missing raises a ``ModuleNotFoundError``
that names the symbol, the dependency, and the remedy -- the extras group for a
pip-installable one (for example ``pip install "graintrace[gnn]"`` for
``GraphGrainMatcher``), or the tier 2/3 PUMA build for NEML2 -- and links back
to this page.

To run the examples or benchmarks, clone the repo and install from the checkout.
The PyPI package ships only the ``graintrace`` library, not ``examples/``,
``benchmark/``, or the ``mwe_data/`` sample datasets:

.. code-block:: bash

   git clone https://github.com/applied-material-modeling/graintrace.git
   cd graintrace
   pip install -e ".[examples]"

Tier 2: NEML2 features without CPFE
-----------------------------------

Material calibration and pole figures need the NEML2 **Python** package but not
the full MOOSE/``puma-opt`` solver. Build only NEML2 from PUMA's submodule:

.. code-block:: bash

   git clone https://github.com/applied-material-modeling/graintrace.git
   cd graintrace
   git submodule update --init external/puma        # pulls puma -> moose + neml2

   # Create the PUMA Python environment (see external/puma/README.md):
   conda create -n puma python=3.13 mpich gcc_linux-64 gxx_linux-64 gfortran_linux-64 \
     cmake make ninja hdf5 netcdf4 zlib libaec bison flex m4 pkg-config
   conda activate puma
   pip install torch nmhit scikit-build-core ninja

   # Build ONLY the NEML2 Python package (skip the MOOSE/puma-opt build):
   cd external/puma
   git submodule update --init neml2
   pip install ./neml2 --no-deps                     # NEML2 v3 Python + neml2-compile
   cd ../..

   pip install -e .                                  # graintrace + pyzag into the same env

``--no-deps`` on NEML2 keeps the ``pyzag==2.0.0`` that graintrace installs; the
published NEML2 PyPI wheel pins an older, incompatible pyzag, which is why NEML2
is built from source here.

Tier 3: Full CPFE
-----------------

Reconstruct then simulate needs the whole native stack: MOOSE + libtorch +
``puma-opt`` + NEML2. Build it via PUMA, then install graintrace into that env:

.. code-block:: bash

   git clone https://github.com/applied-material-modeling/graintrace.git
   cd graintrace
   git submodule update --init external/puma

   # Build the native stack: follow external/puma/README.md end to end
   # (conda env, submodules, PETSc/libMesh/WASP, NEML2, `make -j`, and
   # `neml2-compile` for the material models).
   cd external/puma
   # ... PUMA build steps ...
   cd ../..

   pip install -e .        # or: pip install -e ".[dev]" to run the tests

At runtime graintrace needs the ``puma-opt`` binary at ``external/puma/puma-opt``;
pass it as ``moose_run_file`` to :class:`~graintrace.CPFESimulation` (see
``examples/demonstrate_cpfe.py``). NEML2 comes from PUMA's build so ``puma-opt``'s
C++ library and the Python ``neml2`` stay in lockstep; ``pyzag`` is provided by
graintrace's pip install.

.. _checking-the-install:

Checking what you actually have
-------------------------------

Which tier you ended up on is a property of the machine, not of the command you
typed, so check it rather than assume it:

.. code-block:: bash

   python -m graintrace.doctor      # or: graintrace-doctor

It probes every external dependency graintrace drives and prints where each one
was found, or why it was not and how to get it. Sample output from a tier 1
machine:

.. code-block:: text

   graintrace dependency report

     MISSING  gpu              no CUDA device visible to torch
     MISSING  neper            not found via NEPER env var, tools.json, PATH, or external/neper
     MISSING  puma-opt         MOOSE/PUMA `puma-opt` not found (tools.json, PATH, or external/puma)
     MISSING  cubit            missing CUBIT/SCULPT binaries: psculpt, epu
     ok       neml2            importable: .../site-packages/neml2/__init__.py
     ok       neml2-aoti       AOTI runtime OK (requires LD_LIBRARY_PATH to include .../lib)
     ok       pyzag            importable: .../site-packages/pyzag/__init__.py
     ok       gmsh             importable: .../site-packages/gmsh.py
     MISSING  torch_geometric  python package `torch_geometric` not importable

   How to get the missing pieces:
     neper: Install NEPER (https://neper.info/doc/introduction.html#installing-neper) and
     point graintrace at it via the NEPER env var, a graintrace_tools.json `neper` key, or
     `neper` on PATH.
     ...

A ``MISSING`` line is not an error on its own -- most of the stack is optional
for some subset of the workflows, and the command exits 0 regardless. Two flags
make it usable from a script:

``--json``
   emit the same rows as JSON, one object per dependency with ``name``, ``ok``,
   ``detail`` and ``build_hint``.

``--require NAME[,NAME...]``
   exit 1 unless every named dependency is present, so a job script can refuse
   to start rather than fail an hour in:

   .. code-block:: bash

      python -m graintrace.doctor --require puma-opt,neml2-aoti || exit 1

The probe names are the ones in the report above: ``gpu``, ``neper``,
``puma-opt``, ``cubit``, ``neml2``, ``neml2-aoti``, ``pyzag``, ``gmsh`` and
``torch_geometric``. The same probes back the MCP server's
``dependency_status`` tool, so the two always agree.

External compiled stack (single PUMA submodule)
-----------------------------------------------

The native stack is pinned via one submodule. PUMA carries MOOSE and NEML2 as
its own submodules, so graintrace pins PUMA once and gets the whole stack:

.. code-block:: bash

   git submodule update --init external/puma          # graintrace -> puma
   cd external/puma && git submodule update --init    # puma -> moose/ + neml2/

The init is intentionally not recursive: it stops at PUMA's ``moose/`` and
``neml2/``. MOOSE's own submodules (PETSc, libMesh, WASP) are initialized by
MOOSE's build scripts during the PUMA build.

CUBIT/SCULPT (proprietary; bring your own license)
--------------------------------------------------

Coreform CUBIT provides CUBIT + SCULPT: https://coreform.com/. Obtain and
install it under your own account.

Never commit CUBIT license material (``*.lic``, license servers, keys) to any
repo. ``sculpt_config`` in the examples takes only executable paths, no license
tokens. See :ref:`config-sculpt` for the ``sculpt_config`` layout.

.. _neper-gmsh:

NEPER and gmsh
--------------

**gmsh** is a pip dependency, installed automatically with graintrace.

**NEPER** is an external tool graintrace drives; it is not a pip package. Install
it yourself, then let graintrace find it, in this precedence order:

1. the ``NEPER`` environment variable set to the absolute path of the ``neper``
   binary (``export NEPER=/abs/path/to/neper``);
2. a ``graintrace_tools.json`` with a ``"neper"`` key (search order
   ``$GRAINTRACE_TOOLS_JSON`` → ``./graintrace_tools.json`` →
   ``~/.config/graintrace/tools.json``);
3. ``neper`` on your ``PATH``;
4. or pass ``neper_path=/abs/path/to/neper`` (or a prepared ``env=``) to
   :class:`~graintrace.VoronoiMeshBuilder` / :class:`~graintrace.CrystalGenerator`.

If NEPER cannot be found, the builders raise a clear error with these
instructions. On Linux you can pass ``auto_install=True`` to build GSL +
OpenBLAS + NEPER into ``~/.local``.

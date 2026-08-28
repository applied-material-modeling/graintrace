Configuration reference
=======================

This page collects the configuration objects used across graintrace: meshing,
boundary conditions, segmentation, clustering weights, the CPFE parameter
groups, and external-tool paths.

.. _config-sculpt:

sculpt_config
-------------

Required for any builder step that calls ``builder.mesh(sculpt_config=...)``.
It takes executable paths only, no license tokens.

.. code-block:: python

   sculpt_config = {
       "launcher": "/path/to/cubit/bin/mpi/bin/mpiexec",
       "psculpt":  "/path/to/cubit/bin/psculpt",
       "epu":      "/path/to/cubit/bin/epu",
       "nprocs":   10,
       "environment": {
           "OPAL_LIBDIR": "/path/to/cubit/bin/mpi/lib",
           "OPAL_PREFIX": "/path/to/cubit/bin/mpi",
       },
   }

Required keys: ``psculpt``, ``epu``, ``nprocs``. ``launcher`` and
``environment`` are needed for MPI-based execution.

sculpt_options
--------------

A tuple of CLI flag strings passed to SCULPT:

.. code-block:: python

   sculpt_options = (
       "--adapt", "-A", "7",    # mesh adaptation level
       "-df", "1",              # dilation factor
       "-S", "2",               # smoothing passes
       "-CS", "4",              # curve smoothing
       "--void_mat", "0",       # void material ID
   )

For FF Voronoi meshes without adaptation, use just ``("--void_mat", "0")``.

Boundary conditions (bc)
------------------------

.. code-block:: python

   bc = {
       "x": {"negative": "stress_free", "positive": "stress_free"},
       "y": {"negative": "stress_free", "positive": "stress_free"},
       "z": {"negative": 0, "positive": displace_amount},
   }

``"stress_free"`` means traction-free (no constraint). An integer/float is a
prescribed displacement (``0`` = fixed).

Segmentation
------------

Two methods are available for voxel segmentation (EBSD / NF-as-voxel).

Flood fill (simpler and faster):

.. code-block:: python

   segmentation = {
       "method": "flood",
       "params": {
           "misorientation_tol": 5.0 / 180 * np.pi,  # radians for VoxelMeshBuilder
           "connectivity": 26,                        # 6 or 26
           "grain_threshold_final": 1000,
           "batch_norm": 200_000,
           "grain_threshold": 1000,
           "stop_count": 500,
       },
   }

Graph-based (better for complex textures):

.. code-block:: python

   segmentation = {
       "method": "graph",
       "params": {
           "misorientation_tol": 5.0,   # degrees if angle_type="degrees", else radians
           "connectivity": 26,
           "grain_threshold_final": 100,
       },
       "graph_params": {
           "segmenter": "leiden",
           "graph_mode": "grid",
           "manhattan_radius": 2,
           "grid_tol": 1e-6,
           "n_jobs": 10,
           "weight_chunk_size": 1_000_000,
           "reduce_edges_topweights_k": 8,
           "nodes_chunk": 500_000,
           "seed": 42,
           "networkit_kwargs": {"gamma": 0.001},   # lower = fewer clusters
           "weight_cfg": {
               "mode": "rbf",
               "sigma": None,
               "sigma_auto": {"sample_size": 20_000, "random_state": 42, "quantile": 0.5},
               "power": 2.0,
           },
           "plot": True,
       },
   }

For :meth:`~graintrace.NearFieldMeshBuilder.reconstruct`, the ``segmentation``
argument is a flat dict (no ``method``/``params`` nesting), with
``misorientation_tol`` in radians.

WeightConfig
------------

Controls edge weighting for graph clustering.

.. code-block:: python

   from graintrace.user_data_class import WeightConfig

   weight_cfg = WeightConfig(
       mode="rbf",           # "rbf" | "inverse"
       power=2.0,
       sigma=None,           # if None, use sigma_auto
       sigma_auto={"sample_size": 500_000, "random_state": 42, "quantile": 0.5},
   )

CPFE parameters
---------------

:meth:`~graintrace.CPFESimulation.set_parameters` takes named groups. The main
groups and common keys:

**material**: ``slip_constant_strength``, ``voce_hardening_initial_slope``,
``voce_hardening_saturation``, ``power_slip_n``, ``power_slip_g0``, ``elastic_E``,
``elastic_nu``, ``elastic_G``, ``burger_scale``.

**simulation_parameters**: ``dt``, ``total_time``, ``initialize_time`` (load
ramps from ``initialize_time`` to ``total_time``), ``device`` (``"cpu"``,
``"cuda:N"``, or a space-separated list), ``device_batch`` (per-device NEML2
chunk), ``sync_times`` (space-separated grid-output times). Output-frequency
knobs:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Knob
     - Values
   * - ``grid_transfer``
     - ``"final"`` (default) | ``"per_step"`` | ``"off"`` (regular-grid MultiApp transfer)
   * - ``exodus_output``
     - ``"sync"`` (default, only at ``sync_times``) | ``"per_step"``
   * - ``mesh_csv``
     - ``"sync"`` (default) | ``"per_step"`` | ``"off"`` (per-element CSV on the true mesh)
   * - ``distributed_mesh``
     - ``False`` (default, replicated) | ``True`` — pre-split the mesh to ``ncore`` and run
       ``--use-split`` (distributed mesh; low per-rank memory for large meshes; requires ``ncore >= 2``)

The defaults are the cheap settings; the per-step grid transfer dominates wall
time. Three sources of REI field data are described in
:doc:`tutorials/rare-event-identification`.

Set ``distributed_mesh=True`` for large meshes that exhaust memory as a replicated
mesh: it pre-splits the mesh once (``--split-mesh ncore``) and runs ``--use-split``
so each rank reads only its partition. It is pre-split only (no in-situ option),
requires ``ncore >= 2``, and needs a ``puma-opt`` build with the
``EqualValueBoundaryConstraint`` distributed-mesh fix. Outputs are unchanged (a
single ``sim_output.e`` via gather, plus complete ``mesh_out/`` and ``grid_out/`` CSVs).

**boundary**: ``bounding_box`` and the ``bc`` dict (above).

**grid_properties**: ``number_of_elements`` and ``bounding_box``. The grid box
should be inset by a small amount on each face to avoid mesh-boundary issues:

.. code-block:: python

   grid_bb = bounding_box.copy()
   for i in range(0, 6, 2):   # xlo, ylo, zlo
       grid_bb[i] += 0.0001
   for i in range(1, 6, 2):   # xhi, yhi, zhi
       grid_bb[i] -= 0.0001

.. _config-tools-json:

External-tool paths (tools.json)
--------------------------------

External tool locations can be supplied via a JSON file instead of environment
variables. The template is ``graintrace/mcp/tools.example.json`` with keys
``puma_opt``, ``neper``, and a ``sculpt_config`` block. Search order:
``$GRAINTRACE_TOOLS_JSON`` → ``./graintrace_tools.json`` →
``~/.config/graintrace/tools.json``.

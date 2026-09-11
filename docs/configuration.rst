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

**Graph/Leiden is the default and better pathway** for NF/EBSD; flood over-merges
into percolating grains. The vetted recipe (validated on Fe-9Cr NF
reconstruction): hard 5° misorientation cutoff; ``manhattan_radius=2``;
a **fixed broad RBF sigma ≈ half the cutoff** (``sigma_auto`` collapses to ~0.1° and
shatters grains); ``reduce_edges_topweights_k=12``; optional ``downsample=(2,2,1)``;
Leiden ``gamma`` chosen by **percolation** (smallest gamma whose biggest grain is
below ~3% of solid) over a sweep ``[0.5, 1, 2, 4, 8]``; then absorb sub-30-voxel
fragments and an adjacency-only re-merge (~3°). This recipe is promoted into the
library as :meth:`graintrace.fragmentation.FragmentationAnalyzer.segment` and its
staticmethods (``gamma_sweep_leiden``, ``pick_gamma_by_percolation``,
``absorb_fragments``, ``adjacency_remerge``). The percolation gamma-pick is for large, many-grain data;
for few-grain or intragranular sub-grain segmentation pass a fixed low ``gamma``
(~0.1).

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
   * - ``solver_route``
     - ``"timestep_optimized"`` (default) | ``"hpc_memory"`` — swaps the linear solver / Executioner deck

The defaults are the cheap settings; the per-step grid transfer dominates wall
time. Three sources of REI field data are described in
:doc:`tutorials/rare-event-identification`.

The per-element ``mesh_out`` CSVs also carry a ``block_id`` column (the element
subdomain id == grain id). It lets post-processing select all elements of a
grain without the mesh file &mdash; e.g. intragranular reorientation /
fragmentation via
:meth:`graintrace.ipf_orientation_tracking.OrientationTracker.simulation_element_tracks` (see
:doc:`tutorials/reorientation-fragmentation`).

Fragmentation & reorientation analysis
--------------------------------------

Reorientation-driven REI and grain fragmentation (see
:doc:`tutorials/reorientation-fragmentation`) add these options:

**Reorientation REI**
(:meth:`graintrace.fragmentation.FragmentationAnalyzer.write_reorientation_field`): ``step`` and
``ref_step`` (field-step indices; ``ref_step`` defaults to the first field step),
``symmetry`` (default ``"432"``), and ``extra_cols`` (feature columns to carry
through for a *combined* criterion). The rare-event scalar is compared with the
new :meth:`~graintrace.similarity_metric_library.SimilarityMetricLibrary.abs_scalar_diff`
metric (exactly one feature column).

**Orientation segmentation**
(:meth:`graintrace.fragmentation.FragmentationAnalyzer.segment`): ``misori_tol_deg``
(hard edge cutoff, default 5°), ``graph_mode`` (``"grid"`` for voxel grids,
``"knn"`` for scattered mesh elements), ``gamma`` (fixed Leiden resolution; when
``None`` a percolation-picked sweep over ``gamma_sweep`` runs), ``rbf_sigma_deg``
(fixed broad RBF sigma, default ≈ half the cutoff — **not** ``sigma_auto``, which
shatters grains), ``manhattan_radius``, ``reduce_topk``, ``grain_threshold_final``
(absorb threshold), ``remerge_miso_deg`` (adjacency re-merge), and
``percolation_max_frac``. Graph/Leiden is the default/better pathway; flood
over-merges into percolating grains. The gamma-by-percolation sweep is for large,
many-grain NF/EBSD; for few-grain or intragranular sub-grain segmentation pass a
fixed low ``gamma`` (~0.25 for the shipped CPFE data — ``gamma=0.1`` under-splits and
leaves most grains whole; raise ``gamma`` to resolve finer sub-grains).

For intragranular sub-grain segmentation,
:meth:`~graintrace.fragmentation.FragmentationAnalyzer.grain_fragments` adds
``min_subgrain_miso_deg`` (default 1°) — a **minimum sub-grain misorientation** applied after
Leiden: fragments whose symmetry-aware mean orientations are closer than this are merged
agglomeratively (closest-pair first, means recomputed each step, so no single-linkage chaining),
via :meth:`~graintrace.fragmentation.FragmentationAnalyzer.merge_fragments_by_misorientation`.
``gamma`` sets the raw resolution; ``min_subgrain_miso_deg`` sets the physical threshold so a
smooth intragranular gradient is not chopped into near-identical fragments (0 disables it). The
fragment means (and the branching-IPF fork endpoints) use
:meth:`~graintrace.fragmentation.FragmentationAnalyzer.mean_orientation_mrp`, a symmetry-aware
quaternion mean: pass ``symmetry`` (e.g. ``"432"``) so a set whose members are reported in
different symmetry variants still averages correctly (the default ``"1"`` is the plain mean).

**Split detection**
(:meth:`graintrace.fragmentation.FragmentationAnalyzer.detect_splits`): ``d_tol``
(cross-step centroid distance, at grain scale), ``theta_tol_deg`` (cross-step
misorientation link, generous ~15°), ``top_k``, and an optional ``adjacency_b`` to
reject a spurious split child. The same detector serves FF, NF, and EBSD; NF/EBSD
reach it through ``segment`` +
:meth:`graintrace.fragmentation.FragmentationAnalyzer.seg_to_grain_table`.

**Annotated Exodus**
(:meth:`~graintrace.ipf_postprocess.IPFProcessor.add_element_field_to_exodus` and
:meth:`~graintrace.ipf_postprocess.IPFProcessor.add_element_rgb_to_exodus`): write a
per-element (not block-constant) field into an Exodus copy, mapping values to elements
by a centroid KD-tree. The first writes a float scalar (encode the ``"{grain}.{k}"``
``fragment_label`` as a float); the second writes an RGB triple (``rgb_x``/``rgb_y``/``rgb_z``)
so ParaView can show the fragments in the **same colors as the branching-IPF figure** (map the
3-component array to color with scalar mapping off). Pass one color per element from the shared
fragment palette; for a consistent result annotate the Exodus of the *analyzed* mesh (a
co-registered loaded run's ``sim_output.e``/``mesh.e``), not a bare mesh of a different
microstructure.

**IPF trajectory plot styling**
(:func:`~graintrace.plot_postprocessing.plot_ipf_orientation_tracking` and
:func:`~graintrace.plot_postprocessing.plot_ipf_fragmentation_tracking`): both take a
shared ``ax`` (returning it instead of saving, for a side-by-side 1x2 figure) and the
styling knobs ``arrow_color`` (single color, a per-entity sequence, or the sentinel
``"ipf_final"`` — tint each track/fork by the IPF color of its final-stage orientation,
which reads as "where it ended up" and needs no legend), ``start_color`` (start-dot color;
``None`` = the initial-orientation IPF color, or e.g. ``"black"`` for a neutral origin),
``line_style`` / ``child_line_styles`` + ``parent_line_style`` (per-track / per-child
linestyles), ``child_colors`` + ``parent_color`` (per-child colors and the parent color on the
fragmentation plot, so a grain's sub-grains can be given contrasting categorical colors),
``line_width``, ``start_size``, ``step_size``, ``arrow_scale`` (arrowhead size), ``show_arrow``
(``False`` draws headless lines to declutter a crowded panel), and ``label_fontsize``. The
signature 1x2 figure links its panels on two channels keyed to a per-grain *fragment* index: a
dark categorical color (a grain's sub-grains contrast; grains told apart by IPF position) and a
shared *fragment -> line-style* map, with black start dots and a gray parent trajectory.

The ``solver_route`` selects one of two Executioner decks merged into the run.
``"timestep_optimized"`` (default) uses a **direct** LU factorization
(``superlu_dist``) with preconditioner reuse — robust and iteration-cheap, but the
factorization fill-in makes it memory-bound on large meshes; best for small/medium
problems. ``"hpc_memory"`` uses an **iterative** ``fgmres`` solve preconditioned by
**GAMG** algebraic multigrid — no global factorization, so the memory footprint stays
low and scales across ranks; it is the recommended partner of ``distributed_mesh=True``
for large HPC meshes, at the cost of looser and more numerous linear iterations. Both
decks keep ``residual_and_jacobian_together = false`` (required by the nodal-constraint
loading BC; MOOSE issue 33531).

Set ``distributed_mesh=True`` for large meshes that exhaust memory as a replicated
mesh: it pre-splits the mesh once (``--split-mesh ncore``) and runs ``--use-split``
so each rank reads only its partition. It is pre-split only (no in-situ option),
requires ``ncore >= 2``, and needs a ``puma-opt`` build with the
``EqualValueBoundaryConstraint`` distributed-mesh fix. Outputs are unchanged (a
single ``sim_output.e`` via gather, plus complete ``mesh_out/`` and ``grid_out/`` CSVs).

Running on HPC
~~~~~~~~~~~~~~

For a cluster run, combine the memory-lean solver with the scheduler launcher, and
add the distributed mesh only when the replicated mesh no longer fits:

.. code-block:: python

   sim.set_parameters(
       "simulation_parameters",
       solver_route="hpc_memory",   # iterative fgmres + GAMG (low, scalable memory)
       launcher="srun",             # Slurm/Cray; default "mpiexec" elsewhere
       distributed_mesh=True,       # large meshes only (~1M+ elements); needs ncore >= 2
       device="cuda:0 cuda:1 cuda:2 cuda:3",  # GPU: a cuda list, one entry per GPU on the node
       device_batch=20000,          # finite chunk caps per-GPU memory (0 risks OOM)
   )
   sim.run(ncore=4)                 # GPU: ncore == number of GPUs in the device list

- **GPU node:** ``device`` is a space-separated cuda list and ``ncore`` equals the
  number of GPUs; fewer, larger ranks give better GPU utilisation for the
  NEML2-dominated solve. Keep ``device_batch`` finite to bound per-GPU memory.
- **CPU node:** set ``device="cpu"`` and ``ncore`` to the number of MPI ranks
  (``srun -n``); ``distributed_mesh=True`` is what keeps per-rank memory bounded as
  the mesh grows.
- ``solver_route="hpc_memory"`` and ``distributed_mesh=True`` are independent levers
  but pair naturally: GAMG avoids the LU factorisation blow-up, the distributed mesh
  avoids holding the full mesh per rank. Enable ``distributed_mesh`` only for meshes
  that OOM as replicated (it needs the EVBC-fixed ``puma-opt``); ``hpc_memory`` is
  safe to use at any size.

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

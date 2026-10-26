Common pitfalls
===============

NEPER is bring-your-own
-----------------------

:class:`~graintrace.VoronoiMeshBuilder` and :class:`~graintrace.CrystalGenerator`
resolve a user-installed NEPER via ``graintrace/neper_env.py``. Precedence:
explicit ``neper_path=``/``env=`` → ``NEPER`` env var → ``graintrace_tools.json``
``"neper"`` key → ``neper`` on ``PATH``. If none resolve, the builder raises a
clear error. On Linux, ``auto_install=True`` performs a ``~/.local`` source build.

CrystalGenerator seed is random by default
------------------------------------------

:class:`~graintrace.CrystalGenerator` defaults to ``seed=None``, which draws a fresh
random seed on every instantiation (printed to stdout), so each unseeded run is a
*different* microstructure. Pass an explicit ``seed=<int>`` for a reproducible
tessellation; the printed value reproduces a given run. (NEPER honors ``-id``: different
seed gives different grains, the same seed is byte-identical. The FF *reconstruction* path
:class:`~graintrace.VoronoiMeshBuilder` ``centroid``/``centroidsize`` is data-seeded via
``-morphooptiini``, so its seed is inert — grains come from the measured centroids.)

Derive the cpfe_base path dynamically
-------------------------------------

Do not hardcode it:

.. code-block:: python

   import graintrace as _gt
   from pathlib import Path
   _cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")

FF output orientations are always in degrees
--------------------------------------------

:meth:`~graintrace.VoronoiMeshBuilder.build_voronoi` always writes
``orientations.dat`` in degrees, regardless of input units. When feeding to
:class:`~graintrace.VoxelMeshBuilder` afterward, set ``angle_type="degrees"``.

orientation_tolerance units must match ori_units
------------------------------------------------

When ``ori_units="radians"``, convert ``orientation_tolerance`` before passing it
to ``RegionBaseStitching``:

.. code-block:: python

   if ori_units == "radians":
       orientation_tolerance = np.deg2rad(orientation_tolerance)

grid_properties bounding box should be inset
--------------------------------------------

Inset ``grid_bb`` by ``0.0001`` on each face to avoid mesh-boundary issues, while
using the full box for the boundary conditions (see :doc:`configuration`).

CUDA material calibration: the model must be on the device
----------------------------------------------------------

``TaylorModel(device="cuda")`` works because ``taylor.py`` moves the whole
nonlinear system with ``nsys.to(device)`` before wrapping it in the pyzag factory.
Moving only the factory leaves the crystal-geometry buffers on CPU, which
surfaces as a silent ``loss=inf``.

Use the GPU when one is available
---------------------------------

The GPU-accelerated steps are CPFE, material calibration, and pole figures. When
a CUDA GPU is present, use it (``device="cuda:0"`` for CPFE,
``TaylorModel(device="cuda")`` for calibration); CPU is much slower for these
NEML2-dominated workloads.

Resuming a checkpoint validates the parameters that built it
------------------------------------------------------------

``GraphSpatialCluster.run(resume_from_checkpoint=True)`` loads the stored edges
and weights and skips graph building, distance filtering, sigma estimation and
top-k pruning. Anything that shaped those arrays is therefore compared against
the checkpoint metadata on resume, and a mismatch raises rather than returning
the old graph as the answer to a new question. That covers the data itself (a
``sha256`` of the coordinate and feature arrays), the metric, the graph mode and
its connectivity, ``max_edge_distance``, ``reduce_edges_topweights_k``, and the
weighting parameters each ``weight_cfg.mode`` actually reads: ``sigma`` and
``power`` for ``rbf``, ``sigma`` for ``exp``, ``eps`` for ``inverse`` and
``log_inv``.

``segmenter`` and the NetworKit options are deliberately *not* compared. They act
after the graph is loaded, which is what makes reusing one checkpoint across a
gamma sweep worthwhile.

One case needs care. When ``sigma`` is ``None`` the value is estimated from the
edge distances, and resume skips the step that computes them, so the resolved
sigma is not knowable without rebuilding the graph. Such a request is compared by
its ``sigma_auto`` quantile instead, which is exact because everything feeding
those distances is already compared. The consequence is that resuming a
``sigma``-pinned checkpoint with a ``sigma=None`` request is reported as a
mismatch. Pin the resolved sigma (it is recorded in ``<base>.meta.json`` as
``weight_sigma``) if you want to reuse such a checkpoint.

Checkpoints written before a given check existed lack its metadata key; those
warn instead of failing.

The multiprocessing guard
-------------------------

NF reconstruction (``graintrace/nf/convert.py``) uses ``multiprocess.Pool``, so
scripts that call it must be under an ``if __name__ == "__main__":`` guard. The
graph-clustering / REI pipeline does not require the guard, but keeping it is good
practice.

NF gridding is unmasked by default, and that is what percolates
---------------------------------------------------------------

:meth:`~graintrace.construct_nf_mesh.NearFieldMeshBuilder.reconstruct` defaults
``min_confidence=None``, which reproduces the historical grid: **every** grid point
inside a layer's Delaunay hull becomes crystal and takes the nearest measured
point's orientation. Air inside the hull, the sample surface, regions that failed to
index and low-confidence grain-boundary voxels therefore all reach the segmenter,
and the low-confidence voxels form a *connected network between grains* that
orientation segmentation leaks through. That is the percolation the graph path
patches after the fact with a size cap. Set ``min_confidence`` on real NF data
rather than relying on the cap alone, and use ``max_point_distance`` for concave
outlines, where the hull test fills notches and holes that a radius around the
measured points does not. See :ref:`config-nf-gridding`.

Two gridding details that look like bugs and are not:

* **Hull mode and radius mode pick the source point differently.** Under the hull
  test a grid point takes the nearest *vertex of its containing simplex*
  (:func:`graintrace.nf.convert.nearest_simplex_vertex`); under
  ``max_point_distance`` it takes the globally nearest measured point. These are not
  the same point — the nearest site need not be a vertex of the containing simplex
  (measured: 5 of 102 in-hull points on a 60-point random layer, by up to 0.16 µm).
  The legacy rule is kept in hull mode deliberately, so that an unmasked grid is
  identical to the pre-confidence one.
* :func:`graintrace.nf.convert.pointcloud_to_fixed_grid` and
  :func:`graintrace.nf.convert.process_layer_pointcloud` return a tuple,
  ``(grid, confidence)``, not a bare grid.

Unsegmented voxels must not reach the mesh writers
--------------------------------------------------

A segmented grid uses ``0`` for void, ``>= 1`` for a grain and ``-1`` for a voxel
that is material but belongs to no grain. ``-1`` is not only a flood-fill
leftover: :func:`graintrace.nf.segment.remove_small_segments` reassigns a
sub-threshold segment to ``-1`` when it has no kept neighbour, so a pipeline that
runs the infill *before* the removal, as
:meth:`~graintrace.construct_nf_mesh.NearFieldMeshBuilder.reconstruct` does, can
still hold orphans at the end.

``nf.mesh.write_spn`` raises ``ValueError`` on such a grid rather than meshing it.
Run :func:`graintrace.nf.segment.infill_nearest_neighbor` after
``remove_small_segments``, or mask the orphans to void (``0``) yourself —
``write_spn`` will not do it for you, because dropping them silently removes
material from the mesh. ``nf.mesh.write_voxel_exodus`` skips negative ids the same
way it skips the background id, so an orphan never becomes a block of its own.

Both builders now end with :func:`graintrace.nf.segment.void_unsegmented`: a voxel
still at ``-1`` after the final cleanup is an island with no solid neighbour (for
example a stray high-confidence point in an air notch under ``max_point_distance``),
which neither the infill nor the removal can place, so it becomes void. That is a
decision about a voxel with no grain to join, not a silent deletion of material
inside the sample.

A ``phase`` column is not a void marker
---------------------------------------

``VoxelMeshBuilder`` once read any ``phase <= 0`` as void, so an EBSD CSV with a
0-based ``.ang`` phase column became an all-void grid. Void detection is now
explicit: pass ``void_marker_col="phase"`` when ``0`` really means void. Only the
``cell_id == -1`` sentinel written by :class:`~graintrace.NFGridConversion` is read
automatically, because ``-1`` is never a grain id.

SCULPT block ids are sparse
---------------------------

SCULPT keeps the ``.spn`` material ids as Exodus block ids (``eb_prop1``) and may
drop small grains, so the ids come back with gaps, and the CPFE decks, which assume
blocks ``1..N``, abort at setup with "block id 5 does not exist".
:func:`graintrace.nf.mesh.mesh_sculpt` now renumbers them to ``1..N`` in file order
(:func:`graintrace.nf.mesh.renumber_blocks_contiguous`) after resolving each block's
grain from its original id, and writes ``<mesh stem>_block_map.csv`` mapping
``block_id`` back to ``sculpt_block_id``, ``grain_id`` and ``source_grain_id``. Row
``k`` of the orientations CSV belongs to block id ``k + 1``.

SCULPT meshes span voxel edges
------------------------------

The grid stores voxel *centres*; SCULPT's boundary faces lie on voxel *edges*.
Rescaling onto the centre extents shrank every mesh by one voxel per axis and
collapsed a single-layer grid to zero thickness, which puma-opt then fails to write
(``map_find() error: key 10 not found``). The mesh now spans
:func:`graintrace.nf.mesh.voxel_edge_bounds`, ``[min - h/2, max + h/2]`` per axis.
A single-voxel axis has no measurable spacing: z uses ``mesh(layer_spacing=...)``
(``NearFieldMeshBuilder`` defaults to its ``dz``), and otherwise the mean in-plane
spacing. :func:`graintrace.nf.mesh.write_voxel_exodus` uses the same rule, so both
meshers agree.

Grain orientation averages need the crystal symmetry
----------------------------------------------------

A cubic orientation has 24 equivalent rotation matrices, and a reconstruction may
report voxels of one grain in different variants. Averaging across variants gives
a mean close to no member of the set, so
:func:`graintrace.nf.metrics.average_rotations` takes a ``symmetry`` argument and
folds every member onto the variant closest to the first before averaging.

The default is ``symmetry="1"`` (the plain quaternion mean), which keeps direct
callers of the low-level ``graintrace.nf.mesh`` writers unchanged. The
:class:`~graintrace.construct_nf_mesh.NearFieldMeshBuilder` and
:class:`~graintrace.construct_voxel_mesh.VoxelMeshBuilder` pass their own
``symmetry`` (default ``"432"``) through to every orientation export, so the
builder path is symmetry-aware without any extra configuration. If you call
``nf.mesh.write_spn``, ``nf.mesh.mesh_sculpt`` or ``nf.mesh.write_voxel_exodus``
yourself, pass the crystal point group explicitly.

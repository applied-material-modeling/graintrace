Common pitfalls
===============

The percolation size cap is no longer on by default
---------------------------------------------------

**Behaviour change.** ``percolation_max_frac`` used to default to ``0.03`` in
:meth:`~graintrace.fragmentation.FragmentationAnalyzer.segment`,
:meth:`~graintrace.fragmentation.FragmentationAnalyzer.pick_gamma_by_percolation`
and :meth:`~graintrace.fragmentation.FragmentationAnalyzer.adjacency_remerge`. It
now defaults to ``None``, i.e. no cap, and the gamma pick is
``gamma_pick="lowest"`` rather than the percolation sweep. Grain size is not a
percolation signal: the cap forced any legitimately large grain below 3 % of the
solid and cut it into arbitrary pieces, which is wrong for few-grain data, a
small field of view, and intragranular sub-grain segmentation.

An existing script that relied on the old pick will segment differently. Pass
``percolation_max_frac=0.03`` explicitly to reproduce the previous behaviour
exactly; it is kept as an opt-in for unmasked data. The recommended replacement
is to mask low-confidence and void voxels so there is no bridge to percolate
along, and to use ``gamma_pick="spread"`` when a leak is still suspected --
see :doc:`configuration`.

The same opt-in also restores the legacy clean-up. On the default (no-cap) path
two further stages changed:

* **Absorption is orientation-gated.** ``absorb_max_miso_deg`` defaults to the
  segmentation cutoff: a small fragment joins only a touching neighbour within
  that misorientation (the closest), otherwise it stays a small grain. The old
  orientation-blind rule merged every small, differently oriented grain into its
  largest-contact neighbour; on Fe-9Cr NF the median grain then had 3.9 % of its
  voxels more than 5° from its mean. Expect more small grains than before; pass
  ``absorb_max_miso_deg=None`` for the old rule.
* **The re-merge is agglomerative and spread-checked.** The fixed-mean union-find
  chained a gradual orientation change (eight blocks 2.5° apart, 17.5° end to
  end) into one grain at ``remerge_miso_deg=3``. The default now merges the
  closest pair first, recomputes the union mean, and rejects a union that fails
  :meth:`~graintrace.fragmentation.FragmentationAnalyzer.cluster_leak_check`.

Labels below zero (``-1`` = material-but-unsegmented) are never clusters:
``absorb_fragments``, ``adjacency_remerge``, ``cluster_orientation_spread`` and
``cluster_leak_check`` leave them unchanged, and all take ``background_label``
(e.g. ``(0, -1)`` on a raw grid whose void is ``0``).

The orientation-spread percentile is not a leak detector
--------------------------------------------------------

:meth:`~graintrace.fragmentation.FragmentationAnalyzer.cluster_orientation_spread`
(the p95 misorientation to the mean) misses a leak of a few percent of the
points (1000 voxels of a grain plus 45 of one 30° away: p95 ≈ 1.9°) and flags a
real grain with a smooth 12° gradient (p95 ≈ 5.7° > 5°). Use
:meth:`~graintrace.fragmentation.FragmentationAnalyzer.cluster_leak_check`, which
looks for a separated orientation population instead; it is what ``pick_gamma``
and the re-merge guard use.

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

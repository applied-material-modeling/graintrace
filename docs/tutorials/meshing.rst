Meshing a grain grid into hex
=============================

Turn a voxel or tessellation grain grid into a hexahedral CPFE mesh with
:class:`~graintrace.VoxelMeshBuilder`. Two paths are available through
``VoxelMeshBuilder.mesh()``: ``mesher="sculpt"`` builds a conformal hex mesh
with CUBIT/SCULPT, and ``mesher="voxel"`` writes a direct voxel-to-Exodus dump
with no external tools. The recommendations below come from a 12-case
microstructure study.

**Needs:** CUBIT/SCULPT for the ``sculpt`` path (the ``voxel`` path needs no
external tools).

Inputs
------

A segmented grain grid (from a reconstruction or generation step) plus the
Euler columns and crystal symmetry. The same ``mesh()`` entry point is shared by
:class:`~graintrace.NearFieldMeshBuilder` and by
:class:`~graintrace.VoronoiMeshBuilder` (``build_voronoi(generate_mesh=True)``).
Grid generation lives in :doc:`../concepts`; see also :doc:`ff-reconstruction`
for producing ``reconstruction_reformatted.csv``.

SCULPT conformal mesh
---------------------

.. code-block:: python

   from graintrace.construct_voxel_mesh import VoxelMeshBuilder

   builder = VoxelMeshBuilder(
       file_path="voronoi.csv", save_dir="out/mesh",
       euler_cols=["Eul0", "Eul1", "Eul2"],
       angle_convention="bunge", angle_type="degrees", symmetry="432",
   )
   builder.mesh(
       sculpt_config={
           "psculpt": "/path/to/cubit/bin/psculpt",
           "epu":     "/path/to/cubit/bin/epu",
           "nprocs":  4,                    # <= physical cores
       },
       sculpt_options=SCULPT_OPTS,          # one of the two configs below
       merged_grid=merged_grid_path,
   )

Only two ``sculpt_options`` configs are recommended, both keeping ``-df 1``
(defeaturing) safe. See :doc:`../configuration` for the full ``sculpt_config``
and ``sculpt_options`` reference.

- ``adapt4``: ``("-A", "4", "-df", "1", "-S", "2", "-CS", "4")``. Grain
  preservation at or above 98 percent, positive Scaled Jacobian. Adapt-type 4
  refines without triggering aggressive small-grain absorption.
- ``df1``: ``("-df", "1", "-mvs", "2", "-S", "2", "-CS", "5")``. Grain
  preservation at or above 95 percent, positive Scaled Jacobian. ``-mvs 2``
  (minimum voxel size for defeaturing) prevents ``-df 1`` from swallowing
  larger grains.

Direct voxel dump
-----------------

.. code-block:: python

   builder.mesh(mesher="voxel", merged_grid=merged_grid_path)
   # -> mesh.e (HEX8, one block per grain, ids 1..N) + <orientations>.csv (MRP)

- One axis-aligned cube hex per voxel. Scaled Jacobian is 1 everywhere, so there
  are no inverted or sliver elements and grain preservation is 100 percent by
  construction. Needs no ``sculpt_config`` or CUBIT.
- Trade-offs: stair-stepped grain boundaries (not smoothed) and a fixed one hex
  per filled voxel (element count equals voxel count). Control resolution
  through the tesr/grid size.
- Use it when SCULPT smoothing produces bad elements (junction slivers) or when
  a guaranteed-clean mesh matters more than boundary smoothness.

Outputs
-------

- ``mesh.e`` — Exodus hex mesh, one block per grain.
- ``<mapped_orientations>.csv`` — per-element MRP orientations for CPFE.

Gotchas
-------

- Verify before running CPFE. For the SCULPT path, check all three; the voxel
  path guarantees the first two by construction, so only the last matters there:

  1. Grain preservation: ``N_mesh / N_tess`` at or above 98 percent for
     ``adapt4``, 95 percent for ``df1``. Lower means defeaturing has silently
     absorbed grains.
  2. Minimum Scaled Jacobian above 0, not just the mean. A single negative-SJ
     element will crash the MOOSE simulation.
  3. Mesh-vs-tess grain-size distribution check.

- ``-df 1`` on its own eats grains. Without ``-A 4`` or ``-mvs 2`` it silently
  absorbs a large fraction of grains at 100 cubed voxels.
- Do not override the smoothing method (``-S``) on irregular voxel meshes.
  Forcing smoothing after guaranteed-quality smoothing can diverge (minimum SJ
  driven to -1) and segfault SCULPT. Prefer the two configs above, or use
  ``mesher="voxel"``.
- Use the minimum number of grains that faithfully represents the target
  distribution. Fewer grains means fewer elements and a tractable CPFE
  wall-clock; add grains only when the distribution or the CPFE quantity of
  interest does not converge.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_synthetic_cpfe.py
   :language: python
   :caption: examples/demonstrate_synthetic_cpfe.py

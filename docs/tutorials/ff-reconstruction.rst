FF Voronoi reconstruction
=========================

Reconstruct a 3D microstructure from far-field (FF) HEDM grain centroids via
NEPER Voronoi/CVT tessellation, using :class:`~graintrace.VoronoiMeshBuilder`.
Use this to produce per-grain orientations, a per-grain initial elastic-strain
file, and a ``.tess`` reconstruction to feed CPFE or graph building.

**Needs:** NEPER (and gmsh if ``generate_mesh=True``).

Inputs
------

An FF grain CSV with ``X, Y, Z`` (and optional ``GrainRadius``), Euler columns
(``Eul0/1/2``), and a 9-component elastic strain (``eKen11..eKen33``). The repo
ships ``mwe_data/ff_calibration/0.csv`` (500 grains). See :doc:`../concepts` for
the raw FF format.

Recipe
------

.. code-block:: python

   from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder

   builder = VoronoiMeshBuilder(
       input_csv="mwe_data/ff_calibration/0.csv",
       output_dir="out/FF",
       bounding_box=[-477, 528, -487, 532, -1025, 625],   # xlo,xhi,ylo,yhi,zlo,zhi
       dim=3, weighted=False,
       auto_fix_bbox=True, bbox_fix_mode="remove_points",
       auto_rotate=False, rotate_angles=(0, 0, 0),
       angle_identifier=["Eul0", "Eul1", "Eul2"],
       orientation_descriptor="euler-bunge", orientation_active_convention=True,
       elastic_strain_identifier=[f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)],
       strain_unit="microstrain", unit="rad",             # unit of the Euler angles
   )
   builder.build_voronoi(
       generate_mesh=False,         # default; True -> NEPER/GMSH tet .msh (fallback only)
       option="centroid",           # "voronoi" | "centroid" | "centroidsize"
       CVT_iter=1000, morphoalgo="subplex",
       mesh_quality_min=0.7, relative_el_size=2.0,
   )

Key parameters
--------------

- ``bounding_box`` — ``[xlo,xhi,ylo,yhi,zlo,zhi]`` in micrometers. ``auto_fix_bbox``
  + ``bbox_fix_mode`` handle out-of-box points (``remove_points`` for production).
- ``unit`` (``"rad"``/``"deg"``) must match the actual Euler units; ``strain_unit``
  is the unit of the ee columns.
- ``option`` / ``CVT_iter`` / ``morphoalgo`` — tessellation morphology and CVT
  optimization.
- ``generate_mesh`` — keep ``False``. The default CPFE mesh is a SCULPT hex built
  from ``reconstruction_reformatted.csv`` via :class:`~graintrace.VoxelMeshBuilder`
  (see :doc:`meshing`); the NEPER/GMSH tet ``.msh`` is a fallback only.

Outputs (in ``output_dir``)
---------------------------

- ``reconstruction.tess`` / ``reconstruction.ori`` (9-col rotation matrix) /
  ``reconstruction.msh`` (GMSH tet, only if ``generate_mesh=True``).
- ``orientations.dat`` — per-grain Euler, always degrees after an FF build.
- ``reconstruction_cpfe_ee.csv`` — per-grain initial elastic strain (12 columns:
  ``x, y, z`` + 9).
- ``reconstruction_reformatted.csv`` — per-voxel grain IDs + Euler (FF → voxel input).

Gotchas
-------

- ``orientations.dat`` is in degrees regardless of input units; a downstream
  :class:`~graintrace.VoxelMeshBuilder` needs ``angle_type="degrees"``, and CPFE
  needs ``orientation_helper.euler_to_mrp(...)``.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_farfield.py
   :language: python
   :caption: examples/demonstrate_farfield.py

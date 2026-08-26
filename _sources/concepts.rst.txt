Concepts & data formats
=======================

The pipeline
------------

graintrace runs the grain-scale workflow in four stages:

1. **Reconstruct** a 3D microstructure from experimental data. Far-field (FF)
   grain centroids become a Voronoi/CVT tessellation (NEPER); near-field (NF)
   and EBSD voxel fields are segmented into grains (flood fill or graph/Leiden
   clustering). All three are meshed to conformal hex with CUBIT/SCULPT (a
   NEPER/gmsh tet mesh is a fallback). Multiple FF scan layers are stitched
   into one grain set by region matching.
2. **Calibrate** the crystal-plasticity material model to a macroscopic
   stress-strain curve with a pyzag analytic-adjoint Taylor model.
3. **Simulate** with MOOSE/PUMA CPFE using NEML2 v3 AOTI-compiled models.
4. **Analyze** the fields — distributions, macroscopic response, pole figures,
   IPF coloring — and identify rare events for targeted measurement.

Data types
----------

FF HEDM data
~~~~~~~~~~~~

Per-scan-layer CSV, whitespace-delimited with an 8-line header. Key columns:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``X, Y, Z``
     - grain centroid position (micrometers)
   * - ``GrainRadius``
     - equivalent-sphere radius (micrometers)
   * - ``Eul0, Eul1, Eul2``
     - Bunge Euler angles (degrees or radians)
   * - ``Confidence``
     - fit quality (typically filter to >= 0.7 or 0.9)
   * - ``eFab11..eFab33``
     - fabric/lattice strain tensor (row-major, 9 components)
   * - ``eKen11..eKen33``
     - Kenesei elastic strain tensor (row-major, 9 components), usually microstrain
   * - ``ScanID``
     - assigned during stitching

Units are detected automatically: if any Euler value exceeds 2π, the file is in
degrees, otherwise radians. Multiple scan layers are Z-shifted before stitching.

NF HEDM data
~~~~~~~~~~~~

Per-layer ``.mic`` files in a folder, tab-delimited with ``%`` header lines:

.. code-block:: text

   %OrientationRowNr OrientationID RunTime X Y TriEdgeSize UpDown Eul1 Eul2 Eul3 Confidence PhaseNr

:class:`~graintrace.NearFieldMeshBuilder` reads a folder of ``.mic`` files; the
``exp_file_token`` parameter is the filename prefix used to find them. If the
source is ``.ang`` files, convert them to ``.mic`` first. Pre-gridded NF data
can instead go through :class:`~graintrace.NFGridConversion`.

EBSD data
~~~~~~~~~

A flat CSV merged from per-layer ``.ang`` files, with columns
``x, y, z, Eul0, Eul1, Eul2`` where ``z = file_index * zstep_ebsd`` and the
Euler columns are Bunge angles. This feeds :class:`~graintrace.VoxelMeshBuilder`.

Orientations
------------

Orientations are interchanged as NEML2 v3 MRP (``tan(θ/4)·axis``), the canonical
on-disk format. FF ``orientations.dat`` is Euler-Bunge in degrees; convert it
with ``graintrace.orientation_helper`` (which delegates to NEML2). Note that
graintrace's ``"mrp"`` label elsewhere means Gibbs/Rodrigues (``tan θ/2``), which
is not the same as NEML2 v3 MRP — use ``euler_to_mrp`` to get the NEML2 form that
the CPFE model expects.

Meshing
-------

The recommended CPFE mesh is a conformal hex built with
:class:`~graintrace.VoxelMeshBuilder` (SCULPT). Hex elements behave better for
crystal plasticity. The NEPER/GMSH tet mesh (``generate_mesh=True`` on the FF
build) is a fallback for when CUBIT/SCULPT is unavailable.

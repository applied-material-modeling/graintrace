Conformal hex meshing
======================

Overview
--------
Conversion of a segmented voxel/grain grid into a hexahedral finite-element mesh for CPFE. The
stage is driven by :class:`~graintrace.VoxelMeshBuilder`, which produces an Exodus mesh
(``mesh.e``) with per-block grain assignments plus a per-element MRP orientation file. The
default backend is CUBIT/SCULPT (conformal, smoothed hex); a no-external-tools voxel dump is
available as a fallback.

Method
------
The input is a dense voxel grid where each voxel carries a grain (block) ID. Two meshing paths
produce hex elements from it:

- ``mesher="sculpt"`` (default): CUBIT/SCULPT reads the voxel field as an ``.spn`` file and
  builds an all-hex, grain-conformal mesh. SCULPT fits and smooths element faces to the
  grain-boundary surfaces implied by the voxel labels, controlled by adaptation, dilation, and
  smoothing flags. The result is a body-conforming mesh with far fewer elements than the raw
  voxel count and well-shaped (positive scaled-Jacobian) hexes.
- ``mesher="voxel"``: each voxel is emitted directly as one cube hex, straight to Exodus, with
  no external tools and no inverted elements. This is a fast, exact-but-blocky fallback when
  CUBIT/SCULPT is unavailable; boundaries are stair-stepped rather than smoothed.

graintrace performs the segmentation and voxel-grid assembly; CUBIT/SCULPT performs the hex
generation and smoothing on the SCULPT path.

Algorithm
---------
1. Load the gridded orientation CSV onto a dense voxel grid and (optionally) smooth it.
2. Segment the grid into grains via graph (Leiden) or flood-fill; remove small segments and
   infill (see :doc:`/algorithms/segmentation`).
3. SCULPT path: write the ``.spn`` voxel file and per-voxel orientations, invoke ``psculpt``
   under the configured launcher, and run ``epu`` to join the parallel Exodus parts.
   Voxel path: emit one cube hex per voxel directly to Exodus.
4. Write the per-element MRP orientation file and the Exodus ``mesh.e``.
5. Recommended check: verify grain preservation (block counts) and element scaled-Jacobian
   before using the mesh for CPFE.

Parameters that matter
----------------------
See :doc:`/configuration` for the full list.

- ``mesher``: ``sculpt`` (conformal hex) vs. ``voxel`` (one cube per voxel, no external
  tools).
- ``sculpt_config``: required keys ``psculpt``, ``epu``, ``nprocs``; plus ``launcher`` and
  ``environment`` for MPI execution.
- ``sculpt_options``: SCULPT CLI flags (adaptation ``-A``, dilation ``-df``, smoothing
  ``-S`` / ``-CS``, ``--void_mat``); use the vetted safe configs.
- Segmentation settings that set the grain field being meshed.

Further details
---------------
For the full hex-meshing algorithm, see the CUBIT/SCULPT documentation:
https://cubit.sandia.gov/ (SCULPT all-hex meshing).

See also
--------
- Tutorial: :doc:`/tutorials/meshing`
- API: :class:`~graintrace.VoxelMeshBuilder`

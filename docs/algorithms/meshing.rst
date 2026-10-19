Conformal hex meshing
======================

Overview
--------
Conversion of a segmented voxel/grain grid into a hexahedral finite-element mesh for CPFE. The
stage is driven by :class:`~graintrace.VoxelMeshBuilder`, which produces an Exodus mesh
(``mesh.e``) with per-block grain assignments plus a per-block MRP orientation file. The
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

Why conformal hexes
~~~~~~~~~~~~~~~~~~~
Crystal plasticity is a stiff, strongly anisotropic constitutive law evaluated at every
quadrature point, and grain boundaries are where its gradients are largest. Two properties
follow.

The elements must be **hexes**. Linear tets lock under the near-incompressible plastic flow
that crystal plasticity produces, and the locking is worst exactly where the strain
localises. The whole downstream pipeline -- the ``block_id`` convention, the per-block
orientation file, the REI element paths -- is built around hexes for this reason.

The mesh must be **grain-conformal**. An element straddling a grain boundary would need two
orientations, and it gets one, so the boundary is smeared across an element width. Making
element faces follow the boundary is what SCULPT contributes beyond a voxel dump, and it is
why the blocky ``mesher="voxel"`` fallback is a fallback: its stair-stepped boundaries are
geometrically conformal to the *voxel* boundary, which is itself an approximation of the
grain boundary at the reconstruction resolution.

The element-count argument points the same way. A grain-conformal SCULPT mesh resolves a
grain with far fewer, better-shaped elements than one cube per voxel, and CPFE cost is
roughly linear in quadrature points. At NF-HEDM resolutions the voxel path is usually
unaffordable for anything but a test run.

The label domain
~~~~~~~~~~~~~~~~
Three label values mean three different things on the segmented grid:

.. list-table::
   :header-rows: 1
   :widths: 14 24 62

   * - Label
     - Meaning
     - Treatment at mesh time
   * - ``0``
     - void
     - skipped; no block
   * - :math:`\ge 1`
     - a grain
     - becomes an Exodus block
   * - ``-1``
     - material, unsegmented
     - **raises**

The ``-1`` case is not a flood-fill leftover only. ``remove_small_segments`` reassigns a
sub-threshold segment to ``-1`` whenever it has no kept neighbour to merge into, so the
NF path -- which infills *then* removes -- can still be holding orphans when meshing starts.
``nf.mesh.write_spn`` raises a ``ValueError`` naming the remedy rather than meshing them,
and deliberately does **not** map them to void for you: that would silently delete material
that the reconstruction says is there. Run ``infill_nearest_neighbor`` after
``remove_small_segments``, or mask the orphans to void explicitly if that is what you mean.

The ``.spn`` writer renumbers labels out of place, so it is collision-proof for any label
set rather than only a contiguous ``1..N``. See :doc:`/file-formats` for the on-disk ordering
(:eq:`spn-order`) and the axis-convention caveat.

Algorithm
---------
1. Load the gridded orientation CSV onto a dense voxel grid and (optionally) smooth it.
2. Segment the grid into grains via graph (Leiden) or flood-fill; remove small segments and
   infill (see :doc:`/algorithms/segmentation`).
3. SCULPT path: write the ``.spn`` voxel file and per-voxel orientations, invoke ``psculpt``
   under the configured launcher, and run ``epu`` to join the parallel Exodus parts.
   Voxel path: emit one cube hex per voxel directly to Exodus.
4. Write the per-block MRP orientation file and the Exodus ``mesh.e``.
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

NF mesh reconstruction
======================

Reconstruct a high-resolution 3D mesh from near-field (NF) HEDM ``.mic`` layers via
voxel segmentation and CUBIT/SCULPT hex meshing, using
:class:`~graintrace.NearFieldMeshBuilder`. Use this to produce an Exodus ``mesh.e``
and per-element MRP orientations from NF data to drive CPFE, typically paired with
an FF residual-strain initial condition.

**Needs:** CUBIT/SCULPT (via ``sculpt_config``). The driver must run under an
``if __name__ == "__main__"`` guard: NF reconstruction uses ``multiprocess.Pool``.

Inputs
------

A folder of per-layer ``.mic`` files (tab-delimited, ``%``-prefixed headers). If the
source is ``.ang`` files (8-column, no header), convert them to ``.mic`` first. The
``exp_file_token`` argument is the filename prefix used to find the layers. See
:doc:`../concepts` for the ``.mic`` format. For the synthetic path,
``SyntheticHEDMGenerator`` writes the NF folder (see
``examples/demonstrate_cpfe_nfff.py``).

Recipe
------

.. code-block:: python

   from graintrace.construct_nf_mesh import NearFieldMeshBuilder
   import numpy as np

   def main():
       builder_nf = NearFieldMeshBuilder(
           input_folder="experiment_data/NF", save_dir="out/NF",
           exp_file_token="layer", angle_convention="bunge", angle_type="radians",
           symmetry="432", prefix="reconstructed", write_intermediate=True, write_vtk=True,
       )
       merged_grid = builder_nf.reconstruct(
           dz=5.0, nx=200, ny=300,
           segmentation={   # flat dict (radians) for NearFieldMeshBuilder
               "misorientation_tol": 5.0 / 180 * np.pi, "connectivity": 6,
               "batch_norm": 200_000, "grain_threshold": 1000, "stop_count": 500,
               "grain_threshold_final": 10000,
           },
       )
       mesh_path = builder_nf.mesh(sculpt_config=sculpt_config, sculpt_options=sculpt_options,
                                   merged_grid=merged_grid)
       # per-element MRP orientations -> builder_nf.mapped_orientations_path + ".csv"

   if __name__ == "__main__":
       main()

Key parameters
--------------

- ``dz`` — layer thickness in micrometers, must match the data; ``nx``, ``ny`` set
  the in-plane grid resolution.
- ``segmentation`` — a flat dict (no ``method``/``params`` nesting) with
  ``misorientation_tol`` in radians and ``connectivity`` of 6 or 26.
- ``sculpt_config`` — required keys ``psculpt``, ``epu``, ``nprocs``, plus
  ``launcher`` and ``environment`` for MPI. ``sculpt_options`` is a tuple of CLI
  flags, e.g. ``("--adapt", "-S", "2", "-CS", "4", "--void_mat", "0")``. See
  :doc:`../configuration` and :doc:`meshing`.

Outputs (in ``save_dir``)
-------------------------

- ``merged_segmented_fixed_grid.npy`` — the segmented voxel grid, also a restart
  checkpoint.
- ``mesh.e`` — the Exodus mesh for CPFE.
- ``orientations.csv`` — per-element neml2 MRP orientations; use as the CPFE
  ``ori_file``.

Gotchas
-------

- Restart: if segmentation already ran, load ``merged_segmented_fixed_grid.npy``
  directly and pass it as ``merged_grid`` to ``mesh(...)`` instead of re-running
  ``reconstruct(...)``.
- Derive the NF bounding box from the saved grid coordinates for the CPFE boundary
  conditions; see :doc:`../pitfalls`.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_cpfe_nfff.py
   :language: python
   :caption: examples/demonstrate_cpfe_nfff.py

Voxel segmentation and meshing
==============================

Segment a voxel or grid orientation field (EBSD or gridded NF) into grains via
graph clustering (Leiden) or flood-fill, then build a conformal hex mesh with
CUBIT/SCULPT, using :class:`~graintrace.VoxelMeshBuilder`. Use this when you have a
gridded Euler-angle CSV or an FF ``reconstruction_reformatted.csv`` and want a
segmented mesh.

**Needs:** CUBIT/SCULPT (via ``sculpt_config``); graph segmentation uses networkit
(Leiden). Wrap the driver in ``if __name__ == "__main__": main()``.

Inputs
------

A merged CSV with ``x,y,z,Eul0,Eul1,Eul2`` (EBSD), or an FF
``reconstruction_reformatted.csv`` (pass ``cell_id_col`` and
``angle_type="degrees"``). See :doc:`../concepts`.

Recipe
------

.. code-block:: python

   from graintrace.construct_voxel_mesh import VoxelMeshBuilder

   builder = VoxelMeshBuilder(
       file_path="out/ebsd/EBSD_merged.csv", save_dir="out/ebsd/mesh",
       euler_cols=["Eul0", "Eul1", "Eul2"], angle_convention="bunge",
       angle_type="radians", symmetry="432",
   )
   merged_grid = builder.reconstruct(
       apply_smoothing=True,
       segmentation={
           "method": "graph",                       # "graph" or "flood"
           "params": {"misorientation_tol": 5.0,    # deg if angle_type="degrees" else rad
                      "connectivity": 26, "grain_threshold_final": 100},
           "graph_params": {
               "segmenter": "leiden", "graph_mode": "grid", "manhattan_radius": 2,
               "grid_tol": 1e-6, "n_jobs": 10, "weight_chunk_size": 1_000_000,
               "reduce_edges_topweights_k": 8, "nodes_chunk": 500_000, "seed": 42,
               "networkit_kwargs": {"gamma": 0.001},   # lower gamma = fewer clusters
               "weight_cfg": {"mode": "rbf", "sigma": None,
                              "sigma_auto": {"sample_size": 20_000, "random_state": 42,
                                             "quantile": 0.5}, "power": 2.0},
               "plot": True,
           },
       },
   )
   mesh_path = builder.mesh(sculpt_config=sculpt_config, sculpt_options=sculpt_options,
                            merged_grid=merged_grid)

Key parameters
--------------

- ``segmentation.method`` — ``"graph"`` (Leiden; better for complex textures) or
  ``"flood"`` (simpler, adds ``batch_norm`` / ``grain_threshold`` / ``stop_count``).
  See :doc:`../configuration`.
- ``misorientation_tol`` units follow ``angle_type``; ``connectivity`` is 6 or 26.
- Graph tuning: ``networkit_kwargs["gamma"]`` (lower gives fewer, larger grains) and
  the ``weight_cfg`` RBF settings.
- ``sculpt_config`` / ``sculpt_options`` — CUBIT hex meshing; see
  :doc:`../configuration` and :doc:`meshing`.

Outputs (in ``save_dir``)
-------------------------

- the segmented voxel grid ``.npy``
- ``mesh.e`` — the Exodus mesh
- per-element MRP ``orientations.csv``
- optional VTK and diagnostic plots

Gotchas
-------

- FF-to-voxel path: an input ``reconstruction_reformatted.csv`` holds Euler angles
  in degrees, so pass ``angle_type="degrees"``. See :doc:`../pitfalls`.
- Large grids: tune ``n_jobs``, ``weight_chunk_size``, and ``nodes_chunk`` for
  memory.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_grid_segmentation_mesh.py
   :language: python
   :caption: examples/demonstrate_grid_segmentation_mesh.py

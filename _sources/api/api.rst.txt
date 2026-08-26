API reference
=============

The API documentation for graintrace, grouped by workflow stage. The public
top-level names (importable as ``graintrace.<Name>``) are defined by
``_LAZY_EXPORTS`` in ``graintrace/__init__.py``.

.. toctree::
   :maxdepth: 1
   :caption: Reconstruction & meshing

   construct_voronoi_mesh
   construct_nf_mesh
   construct_voxel_mesh
   nf_grid_conversion
   generate_random_crystal
   synthetic_hedm_generator
   neper_env

.. toctree::
   :maxdepth: 1
   :caption: Stitching

   scan_stitching_comparison
   hedm_stitching_techniques

.. toctree::
   :maxdepth: 1
   :caption: Simulation & calibration

   run_cpfe_simulation
   grid_resampling
   material_calibration
   taylor
   base_material_approximation
   experiment_rotation_helper
   orientation_helper

.. toctree::
   :maxdepth: 1
   :caption: Post-processing

   simulation_postprocessing
   experiment_postprocessing
   plot_postprocessing
   ipf_postprocess

.. toctree::
   :maxdepth: 1
   :caption: Rare-event identification

   rare_cluster_indicator
   graph_spatial_cluster
   cluster_indicator
   similarity_metric_library
   rare_criteria_selection_library
   rei_comparison
   user_data_class

.. toctree::
   :maxdepth: 1
   :caption: Grain tracking

   grain_graph_matching
   tess_to_gnn

.. toctree::
   :maxdepth: 1
   :caption: Near-field subpackage

   nf

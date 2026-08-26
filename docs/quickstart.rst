Quickstart
==========

This walks through a pure-Python path that runs from ``pip install graintrace``
alone (no NEML2, MOOSE, NEPER, or CUBIT). It uses the sample data under
``mwe_data/``, so clone the repo (the PyPI package does not ship the datasets):

.. code-block:: bash

   git clone https://github.com/applied-material-modeling/graintrace.git
   cd graintrace
   pip install -e .

Load CPFE output
----------------

:class:`~graintrace.SimulationResults` loads the block CSV and the per-time
field CSVs written by a CPFE run. The repo ships a small example set
(``mwe_data/out.csv`` and ``mwe_data/grid_out/``):

.. code-block:: python

   from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming

   field_naming = FieldFileNaming(
       prefix="out_element_centroid", index_width=4, sep="_", suffix=".csv"
   )
   res = SimulationResults(
       block_csv="mwe_data/out.csv",
       field_dir="mwe_data/grid_out",
       field_naming=field_naming,
   )

Plot a macroscopic stress-strain curve
--------------------------------------

.. code-block:: python

   from graintrace import plot_postprocessing as postprocess

   postprocess.plot_macroscopic_stress_strain(
       res,
       stress_tensor_prefix="cauchy_stress",
       strain_tensor_prefix="strain",
       volume_prefix="volume",
       output_folder="out/postprocess",
   )

Identify rare events
--------------------

Rare-event identification (REI) groups a field into spatially coherent clusters
and flags the rare ones. It is pure Python. The
``examples/demonstrate_rei_pipeline.py`` script runs the full pipeline on a
shipped seed dataset:

.. code-block:: bash

   python examples/demonstrate_rei_pipeline.py

See :doc:`tutorials/rare-event-identification` for the full walk-through.

Next steps
----------

- :doc:`install`: add the native stack for calibration, meshing, and CPFE.
- :doc:`concepts`: the data formats and the reconstruct → simulate → analyze pipeline.
- :doc:`tutorials/tutorials`: one guide per workflow stage.

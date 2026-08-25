Registering FF data to the sim frame
====================================

Rotate raw far-field experiment CSVs into the simulation frame before
calibration, using ``graintrace.experiment_rotation_helper``. For each file it
applies a sample tilt, builds a Voronoi reconstruction, and appends the rotated
orientation matrix ``O11..O33``. Use this to prepare per-stress-level FF CSVs for
a physically registered :doc:`material-calibration`.

**Needs:** NEPER (a Voronoi build per file supplies the rotated ``O``).

Inputs
------

A folder of numeric-named per-stress-level FF CSVs (``0.csv``, ``50.csv``, ...),
each with ``X, Y, Z, GrainRadius``, Euler columns (``Eul0/1/2``), and the
9-component elastic strain (``eKen11..eKen33``). Raw FF files may already carry
``O11..O33``. The repo ships ``mwe_data/ff_calibration/``. See :doc:`../concepts`
for the raw FF format.

Recipe
------

.. code-block:: python

   import numpy as np
   from graintrace.experiment_rotation_helper import (
       update_experiments, collect_experiment_files,
   )

   files, stress_levels = collect_experiment_files("mwe_data/ff_calibration")
   update_experiments(
       input_files=files,
       output_root="out/rotated_experiments",
       bounding_box=[-477, 528, -487, 532, -1025, 625],
       auto_fix_bbox=True, bbox_fix_mode="remove_points",
       rotate_angles=(0, 0, -3.6 / 180 * np.pi),   # sample tilt; unit must match `unit`
       unit="rad",
       angle_identifier=["Eul0", "Eul1", "Eul2"],
       orientation_descriptor="euler-bunge", orientation_active_convention=True,
       elastic_strain_identifier=[f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)],
   )

Key parameters
--------------

- ``rotate_angles`` + ``unit`` — the sample-to-sim tilt (radians here), applied
  to both positions and orientations.
- ``bounding_box`` + ``auto_fix_bbox`` / ``bbox_fix_mode`` — drop out-of-box
  grains (``remove_points`` for production).
- ``elastic_strain_identifier`` — the 9 ``eKen`` columns, scaled by 1e6 if the
  strain unit is microstrain.

Outputs
-------

``out/rotated_experiments/<name>.csv`` per input, each with rotated ``O11..O33``
(from ``reconstruction.ori``) plus coordinates, Euler angles, and ``eKen``
columns. These feed :doc:`material-calibration` as its ``data_dir``.

Gotchas
-------

- The helper replaces any pre-existing ``O`` columns with the freshly rotated
  ones (it drops the raw ``O`` before concatenating), so there are no duplicate
  ``O11.1`` columns.
- Each file triggers a full NEPER tessellation; a few hundred grains is quick,
  thousands are slower.
- For a quick, non-registered calibration you can skip this step and point
  calibration straight at raw CSVs that already contain ``O11..O33`` (as in
  ``mwe_data/ff_calibration``).

Full example
------------

.. literalinclude:: ../../examples/demonstrate_farfield.py
   :language: python
   :caption: examples/demonstrate_farfield.py

Microstructure generation
==========================

Generate synthetic grain structures with NEPER via
:class:`~graintrace.CrystalGenerator`. This page collects vetted morpho-string and
iteration-budget recipes distilled from 12 studied cases spanning equiaxed,
lamellar, columnar, and elongated grains and lognormal, normal, and bimodal size
distributions. For meshing the generated grid, see :doc:`meshing`.

**Needs:** NEPER.

Inputs
------

None. Generation is driven entirely by a morpho string and a bounding box. The
``examples/demonstrate_hedm_anisotropic.py`` example runs the 12-case anisotropic
study; ``examples/demonstrate_synthetic_cpfe.py`` generates, meshes, and runs CPFE.

Recipe
------

.. code-block:: python

   from graintrace.generate_random_crystal import CrystalGenerator

   cg = CrystalGenerator(
       output_dir="out",
       bounding_box=[-500, 500, -500, 500, -500, 500],   # 1 mm cube
       dim=3,
       seed=42,
   )

   # Equiaxed, normal size distribution
   cg.generate_tessellation(
       morpho_args={"type": "raw", "morpho_str": "diameq:normal(100, 20)"},
       iterations=5000,
       extra_neper_args=["-reg", "1"],
   )

For elongated or anisotropic grains, append ``aspratio(x, y, z)`` to the morpho
string. The following elongates roughly 3x along z, matching
``examples/demonstrate_hedm_anisotropic.py``. Use the higher iteration budget and
keep ``-reg 1``:

.. code-block:: python

   z_aspect = 3.0
   cg.generate_tessellation(
       morpho_args={
           "type": "raw",
           "morpho_str": f"diameq:lognormal(130, 5),aspratio(1, 1, {z_aspect})",
       },
       iterations=20000,            # elongated -> use the 20k budget
       extra_neper_args=["-reg", "1"],
   )

``aspratio`` sets grain shape only. Far-field HEDM cannot recover true grain
morphology from centroids; NF-HEDM is the genuine source of grain morphology. Use
the elongated generator for benchmarks or ground truth, not to infer shape from FF.
See :doc:`../pitfalls`.

Key parameters
--------------

- ``seed`` — a fixed integer gives a bit-identical ``.tess`` across runs.
- ``morpho_args={"type": "raw", ...}`` — pass ``morpho_str`` directly. This is the
  only path that exposes NEPER's full recipe grammar (mixtures, coupled
  constraints, columnar, lamellar, ``aspratio``).
- ``extra_neper_args=["-reg", "1"]`` — removes sub-resolution artifact cells; use
  for equiaxed, columnar, and elongated, skip for lamellar.
- ``iterations`` — 5000 for normal, 15000 for bimodal, 20000 for lognormal-tail or
  elongated. More iterations reduce the loss and improve agreement with the
  prescribed distribution; beyond about 50,000 the loss decreases very slowly.
- ``n`` — the grain count is derived automatically (``-n from_morpho``) when the
  morpho string carries a size scale (``diameq:...``); otherwise pass ``n=<int>``.

The API applies the ``-reg 1`` cleanup plus an optional post-hoc ``r >= 5 um``
filter, then writes the canonical outputs (``voronoi.tess``, ``.geo``, ``.ori``,
``.stcell``, ``.csv``).

Known limitations (NEPER 4.10.2-45)
-----------------------------------

- **Lamellar takes a single scalar width only.** ``lamellar(w=<distribution>)``
  inline aborts; ``lamellar(w=file(...))`` reads per-grain values from a file.
  Varying-width lamellar requires multiple steps.
- **Columnar takes a single axis letter only.** ``columnar(v=<axis_letter>)`` (e.g.
  ``v=z``) is the only working form; the vector form ``columnar(v=(0,0,1))`` hangs
  NEPER indefinitely.
- **Columnar cross-section sigma has a Poisson floor.** For single-mode columnar
  recipes NEPER hits the mean but delivers a coefficient of variation near 0.36
  regardless of the target sigma. Bimodal columnar fits well because the mixture
  gives the optimizer more variance.
- **Bimodal mixture weights are by count, not volume.** Two peaks at 40 and 100 um
  with 50/50 weights means 50/50 grain counts, not equal volume fractions.

Outputs (in ``output_dir``)
---------------------------

- ``voronoi.tess`` — the tessellation.
- ``voronoi.csv`` — per-grain centroids, sizes, and orientations.
- ``voronoi.stcell`` — per-cell statistics used for the distribution check.
- ``.geo`` / ``.ori`` — geometry and orientation files.

Gotchas
-------

- After generation, confirm that the achieved grain-size distribution matches the
  prescribed one: plot the histogram from ``voronoi.stcell`` against the target PDF.
  Non-agreement means too few iterations or one of the NEPER limitations above.
- Use the minimum number of grains that faithfully represents the target
  distribution. Fewer grains means fewer mesh elements and shorter CPFE wall-clock;
  add grains only when the histogram or the CPFE quantity of interest fails to
  converge.

Full example
------------

.. literalinclude:: ../../examples/demonstrate_hedm_anisotropic.py
   :language: python
   :caption: examples/demonstrate_hedm_anisotropic.py

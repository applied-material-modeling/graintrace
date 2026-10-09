Tutorials
=========

One notebook per workflow stage, each linking to the matching
:doc:`algorithm page </algorithms/index>` and API reference. Read a tutorial for
*how to do it*; read the algorithm page for *why it works*.

Reading order
-------------
The stages chain, and the tutorials are ordered to match. A complete
experimental study runs left to right:

.. code-block:: text

   raw HEDM scans  ->  stitch  ->  reconstruct  ->  mesh  ->  CPFE  ->  analyse

If you are evaluating graintrace rather than using it, start at the analysis
end: :doc:`post-processing` and :doc:`rare-event-identification` run on shipped
sample data with nothing but ``pip install graintrace``, so you can see what the
package produces before installing anything heavy.

Two tiers
---------
Whether a tutorial shows you real output depends on what it needs to run.

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Tier
     - What it means
   * - **A**
     - Runs with no external binary. **Every number and figure on the page was
       produced by running the code above it** — nothing is hand-written.
   * - **B**
     - Needs NEPER, CUBIT/SCULPT, or MOOSE/PUMA. None of those exist in CI or
       Colab, so these notebooks are annotated code listings with no stored
       output. Each one names its external requirement at the top, and the
       table below says exactly which binary.

Tier A splits again by *where* the execution happens, which is worth knowing if
you are editing one:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Tutorial
     - How its output is produced
   * - :doc:`rei-example-2d`, :doc:`rei-example-3d`,
       :doc:`rei-comparison`, :doc:`rare-event-identification`
     - No stored output; ``nbsphinx`` **executes them during the documentation
       build**, so a broken cell breaks CI. The first three generate their own
       synthetic data and therefore carry an "Open in Colab" badge.
   * - :doc:`post-processing`, :doc:`reorientation-fragmentation`
     - Outputs executed locally and **committed**, because both need NEML2 v3
       for the orientation maths and NEML2 is not pip-installable in CI. Re-run
       them after editing: ``jupyter nbconvert --execute --inplace`` from
       ``docs/tutorials/``.

Neither of the last two gets a Colab badge: they read ``../../mwe_data/``, which
only exists in a checkout.

What each Tier B tutorial needs
-------------------------------
Reproduce these by running the matching script from :doc:`/examples` in a
configured environment. The required capability is the *only* thing standing
between the listing and a figure:

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Tutorial
     - Needs
     - Why
   * - :doc:`hedm-stitching`
     - ``neper``
     - Generates the synthetic crystal and the z-scans being stitched.
   * - :doc:`microstructure-generation`
     - ``neper``
     - The tessellation *is* the output.
   * - :doc:`ff-reconstruction`
     - ``neper``
     - Voronoi/CVT reconstruction from FF centroids.
   * - :doc:`experiment-rotation`
     - ``neper``
     - One Voronoi build per file supplies the rotated ``O11..O33``.
   * - :doc:`grain-tracking`
     - ``neper``
     - Builds the per-step grain graphs that are matched.
   * - :doc:`nf-reconstruction`
     - ``psculpt`` + ``epu`` (CUBIT)
     - Hexahedral meshing of the segmented voxel grid.
   * - :doc:`voxel-segmentation-mesh`
     - ``psculpt`` + ``epu`` (CUBIT)
     - Same; the segmentation half is pure Python.
   * - :doc:`meshing`
     - ``psculpt`` + ``epu`` (CUBIT)
     - The ``mesher="voxel"`` path needs nothing, the ``"sculpt"`` path needs CUBIT.
   * - :doc:`cpfe-simulation`
     - ``puma-opt`` + ``neml2-compile``, CUDA recommended
     - Runs the finite-element solve.
   * - :doc:`cpfe-nf-ff`
     - ``neper`` + CUBIT + ``puma-opt``
     - Spans the whole pipeline in one notebook.
   * - :doc:`material-calibration`
     - NEML2 v3 + ``pyzag``
     - No binary needed, but NEML2 v3 is not pip-installable, so CI cannot
       build it. This is the one Tier B tutorial that could become Tier A on a
       machine with NEML2 installed.

.. note::

   Tier A notebooks address data as ``../../mwe_data/...``, relative to
   ``docs/tutorials/``, because that is the working directory ``nbsphinx`` gives
   the kernel. Tier B notebooks use repository-root-relative paths and are meant
   to be run from the root of a checkout. If a path does not resolve, that is
   which convention you are on.

.. toctree::
   :maxdepth: 1
   :caption: Data & microstructure

   hedm-stitching
   microstructure-generation
   ff-reconstruction
   nf-reconstruction
   voxel-segmentation-mesh
   meshing
   experiment-rotation

.. toctree::
   :maxdepth: 1
   :caption: Calibration & simulation

   material-calibration
   cpfe-simulation
   cpfe-nf-ff

.. toctree::
   :maxdepth: 1
   :caption: Analysis

   post-processing
   reorientation-fragmentation
   rare-event-identification
   rei-example-2d
   rei-example-3d
   rei-comparison
   grain-tracking

.. toctree::
   :maxdepth: 1
   :caption: Reference

   /examples

Benchmarks
==========

The ``benchmark/`` folder holds standalone performance / scaling scripts for the
three heaviest graintrace paths. They measure wall-clock timing and scaling on
your machine.

Results are written to ``benchmark/results/`` (gitignored) and are **not**
regression gates — timings are machine-dependent. Each script skips cleanly
(``SKIP: …``, exit 0) when a dependency is missing.

What each measures
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Measures
   * - ``bench_graph_segmentation.py``
     - thread scaling of the graph-segmentation kernels: numba top-k prune
       (``n_jobs``) and NetworKit Leiden (``n_threads``)
   * - ``bench_cpfe.py``
     - CPFE wall time vs mesh resolution (element count) and ``device_batch`` on
       the GPU
   * - ``bench_calibration.py``
     - calibration wall time vs ``device`` (cpu/cuda), ``n_grains``, ``npoints``

Running
-------

Run from the repo root:

.. code-block:: bash

   # graph segmentation — pure Python, no external tools
   python benchmark/bench_graph_segmentation.py --sizes 30 --jobs 1,2,4,8 --threads 1,2,4,8

   # calibration — needs a working neml2 v3 + pyzag env; --probe checks it (exit 0/1)
   python benchmark/bench_calibration.py --probe
   python benchmark/bench_calibration.py --device auto --n-grains 50,100,250,500 --npoints 30

   # CPFE — point --puma-bin at your built puma-opt (generates its own cube mesh)
   python benchmark/bench_cpfe.py \
       --puma-bin /path/to/puma-opt \
       --resolution 16,24,32 --device-batch 5000,20000,50000

Run each ``bench_*.py --help`` for the full flag list.

Output
------

Each run creates ``benchmark/results/<host>_<timestamp>/<name>/`` with
``<name>.csv`` and ``<name>.json`` (rows plus host/GPU/git metadata).
``bench_graph_segmentation.py --plot`` also writes a speedup PNG.

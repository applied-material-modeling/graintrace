Benchmarks
==========

The ``benchmark/`` folder holds standalone performance / scaling scripts for the
three heaviest graintrace paths — graph segmentation, CPFE, and material
calibration. They measure wall-clock timing and scaling **on your machine**, so
you can size a run, pick thread counts and device batches, and see how cost grows
with problem size before committing to a large job.

They are **not** regression gates: timings are hardware- and build-dependent, and
the results are written under ``benchmark/results/`` (gitignored). Each script
**skips cleanly** (prints ``SKIP: …`` and exits 0) when a dependency is missing,
so running the whole set on a partial install reports only what it can.

Run everything from the repo root, and pass ``--help`` to any script for the full
flag list and defaults.

Graph segmentation
------------------

``bench_graph_segmentation.py`` — **pure Python (numba / networkit / numpy); no
NEPER, SCULPT, MOOSE, or GPU, so it runs anywhere.**

*What it measures.* Thread scaling of the two parallel kernels that dominate the
graph-segmentation path (shared by rare-event identification, NF reconstruction,
and voxel meshing):

- the numba top-k edge prune (``prune_topk_per_node_parallel``), swept via
  ``--jobs`` (the ``n_jobs`` thread count), and
- the NetworKit Leiden partition (``segment_graph_networkit``), swept via
  ``--threads`` (the NetworKit thread count).

*Why it matters.* These two kernels set the wall time of segmentation on large
grids; the sweep shows where added threads stop paying off on your CPU, which is
the value to set for ``n_jobs`` in :doc:`/algorithms/segmentation` and the REI
pipeline.

*Run.*

.. code-block:: bash

   python benchmark/bench_graph_segmentation.py \
       --sizes 30 --jobs 1,2,4,8 --threads 1,2,4,8 --repeat 3 --plot

``--sizes`` is the per-side voxel count of the synthetic grid (30 → a 30³ grid);
``--repeat`` averages several runs; ``--manhattan-radius`` sets the grid graph
connectivity; ``--plot`` also writes a speedup PNG.

*Read the results.* Each row reports the kernel, the thread count, and the mean
time; speedup is time at one thread divided by time at N. Near-linear speedup at
low thread counts that flattens out marks the useful maximum for your machine.

CPFE
----

``bench_cpfe.py`` — **requires the full PUMA stack:** ``puma-opt``
(``--puma-bin``), ``neml2-compile`` plus a C/C++ toolchain, ``mpiexec``, and a
CUDA GPU. Skips cleanly if any is missing.

*What it measures.* It generates a cube microstructure in memory, dumps it
straight to an Exodus hex mesh with the voxel mesher (one HEX8 per voxel — no
SCULPT/CUBIT/NEPER), then runs CPFE on the GPU and times it along two axes:

- **resolution** → element count (``nx·ny·nz`` HEX8 elements), via
  ``--resolution``; and
- **device_batch** (the per-device NEML2 chunk, i.e. quad points per call), via
  ``--device-batch``.

Per resolution the AOTI model is compiled with ``neml2-compile`` **once** (for
the first device batch) and then reused, so the device-batch sweep isolates NEML2
solve throughput from the one-time compile.

*Why it matters.* Element count drives memory and solve time; ``device_batch``
trades GPU memory for throughput. The sweep finds the largest batch that fits and
the point where more elements stop scaling on your GPU — the inputs to
``number_of_elements`` and ``device_batch`` in :doc:`/tutorials/cpfe-simulation`.

*Run.*

.. code-block:: bash

   python benchmark/bench_cpfe.py \
       --puma-bin /path/to/puma-opt \
       --resolution 16,24,32 --device-batch 5000,20000,50000 \
       --device cuda:0 --ncore 1

Other knobs mirror a real run: ``--n-grains``, ``--spacing``, ``--total-strain``,
``--dt``, ``--total-time``, ``--initialize-time``, ``--timeout``, ``--seed``.

*Read the results.* Each row separates ``setup_s`` (bake + compile + launch,
measured synchronously) from ``solve_s`` (the asynchronous MOOSE solve), so the
one-time compile cost does not contaminate the per-solve timing. Compare
``solve_s`` across ``device_batch`` at fixed resolution to pick a batch, and
across ``resolution`` at fixed batch to see how solve time grows with element
count.

Material calibration
--------------------

``bench_calibration.py`` — **in-process only (NEML2 v3 + pyzag + torch); no
external binaries, but the env must have a working NEML2 v3 + pyzag.** Use
``--probe`` first to check whether the current environment can run a calibration
at all (exit 0 = ok, 1 = broken) before a full sweep.

*What it measures.* LBFGS calibration wall time as a function of ``--device``
(``cpu`` vs ``cuda``), ``--n-grains`` (the per-step state size), and ``--npoints``
(the number of pyzag time steps). It uses a **fixed** LBFGS budget
(``--maxiter`` / ``--inner``), so it measures per-solve cost, not convergence.

*Why it matters.* Calibration cost scales with grains × time steps × iterations;
the sweep shows the GPU speedup and how cost grows with ``n_grains`` and
``npoints``, guiding the settings in :doc:`/tutorials/material-calibration` and
:doc:`/algorithms/calibration`.

*Run.*

.. code-block:: bash

   python benchmark/bench_calibration.py --probe
   python benchmark/bench_calibration.py \
       --device cuda --n-grains 50,100,250,500 --npoints 15,30 --nchunk 2

*Read the results.* Each row reports the device, ``n_grains``, ``npoints``, and
the calibration wall time for the fixed iteration budget. Comparing ``cpu`` vs
``cuda`` rows gives the GPU speedup for your problem size; the ``n_grains`` and
``npoints`` sweeps show how cost grows so you can extrapolate to a full fit.

Output
------

Each run creates ``benchmark/results/<host>_<timestamp>/<name>/`` containing:

- ``<name>.csv`` — one row per swept configuration (the timing columns above),
- ``<name>.json`` — the same rows plus run metadata (host, CPU/GPU, git commit,
  and the resolved arguments), so a result is self-describing, and
- a speedup PNG for ``bench_graph_segmentation.py --plot``.

Because the metadata records the host and git commit, results from different
machines or builds stay comparable only within the same environment — treat
absolute numbers as machine-specific and compare trends, not headline figures.

Common pitfalls
===============

NEPER is bring-your-own
-----------------------

:class:`~graintrace.VoronoiMeshBuilder` and :class:`~graintrace.CrystalGenerator`
resolve a user-installed NEPER via ``graintrace/neper_env.py``. Precedence:
explicit ``neper_path=``/``env=`` → ``NEPER`` env var → ``graintrace_tools.json``
``"neper"`` key → ``neper`` on ``PATH``. If none resolve, the builder raises a
clear error. On Linux, ``auto_install=True`` performs a ``~/.local`` source build.

CrystalGenerator seed is random by default
------------------------------------------

:class:`~graintrace.CrystalGenerator` defaults to ``seed=None``, which draws a fresh
random seed on every instantiation (printed to stdout), so each unseeded run is a
*different* microstructure. Pass an explicit ``seed=<int>`` for a reproducible
tessellation; the printed value reproduces a given run. (NEPER honors ``-id``: different
seed gives different grains, the same seed is byte-identical. The FF *reconstruction* path
:class:`~graintrace.VoronoiMeshBuilder` ``centroid``/``centroidsize`` is data-seeded via
``-morphooptiini``, so its seed is inert — grains come from the measured centroids.)

Derive the cpfe_base path dynamically
-------------------------------------

Do not hardcode it:

.. code-block:: python

   import graintrace as _gt
   from pathlib import Path
   _cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")

FF output orientations are always in degrees
--------------------------------------------

:meth:`~graintrace.VoronoiMeshBuilder.build_voronoi` always writes
``orientations.dat`` in degrees, regardless of input units. When feeding to
:class:`~graintrace.VoxelMeshBuilder` afterward, set ``angle_type="degrees"``.

orientation_tolerance units must match ori_units
------------------------------------------------

When ``ori_units="radians"``, convert ``orientation_tolerance`` before passing it
to ``RegionBaseStitching``:

.. code-block:: python

   if ori_units == "radians":
       orientation_tolerance = np.deg2rad(orientation_tolerance)

grid_properties bounding box should be inset
--------------------------------------------

Inset ``grid_bb`` by ``0.0001`` on each face to avoid mesh-boundary issues, while
using the full box for the boundary conditions (see :doc:`configuration`).

CUDA material calibration: the model must be on the device
----------------------------------------------------------

``TaylorModel(device="cuda")`` works because ``taylor.py`` moves the whole
nonlinear system with ``nsys.to(device)`` before wrapping it in the pyzag factory.
Moving only the factory leaves the crystal-geometry buffers on CPU, which
surfaces as a silent ``loss=inf``.

Use the GPU when one is available
---------------------------------

The GPU-accelerated steps are CPFE, material calibration, and pole figures. When
a CUDA GPU is present, use it (``device="cuda:0"`` for CPFE,
``TaylorModel(device="cuda")`` for calibration); CPU is much slower for these
NEML2-dominated workloads.

The multiprocessing guard
-------------------------

NF reconstruction (``graintrace/nf/convert.py``) uses ``multiprocess.Pool``, so
scripts that call it must be under an ``if __name__ == "__main__":`` guard. The
graph-clustering / REI pipeline does not require the guard, but keeping it is good
practice.

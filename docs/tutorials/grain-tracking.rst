Grain tracking across load steps
================================

Track and match grains across two load steps by building a grain graph from each
FF reconstruction and matching them via message passing, using
:meth:`~graintrace.VoronoiMeshBuilder.build_graph` and
:class:`~graintrace.GraphGrainMatcher`. Use this when you have two FF grain CSVs
of the same sample at different loads or times and want a grain correspondence.

**Needs:** NEPER (tessellation for the graph).

Inputs
------

Two FF grain CSVs of the same sample at two load steps, each with ``X, Y, Z``,
Euler columns (``Eul0/1/2``), and a 9-component elastic strain
(``eKen11..eKen33``). The repo ships
``mwe_data/synthetic_load_exp/expsyn_146time.csv`` and ``expsyn_160time.csv``.
See :doc:`../concepts` for the raw FF format.

Recipe
------

.. code-block:: python

   from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
   from graintrace.grain_graph_matching import GraphGrainMatcher

   eKen = [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]

   def make(csv, out, bbox):
       b = VoronoiMeshBuilder(
           input_csv=csv, output_dir=out, bounding_box=bbox, dim=3, weighted=False,
           auto_fix_bbox=False, auto_rotate=False,
           angle_identifier=["Eul0", "Eul1", "Eul2"], orientation_descriptor="euler-bunge",
           orientation_active_convention=True, elastic_strain_identifier=eKen,
           strain_unit="microstrain",
       )
       return b.build_graph(CVT_iter=10)

   ga = make("mwe_data/synthetic_load_exp/expsyn_146time.csv", "out/track_a",
             [-200, 200, -173.205, 173.205, 0, 650])
   gb = make("mwe_data/synthetic_load_exp/expsyn_160time.csv", "out/track_b",
             [-200, 200, -173.205, 173.205, 0, 680])

   GraphGrainMatcher(graph_a=ga, graph_b=gb, output_dir="out/grain_tracking").match_grains(
       message_passing_iter=3,
       neighbor_selection_param={"lambda": 0.00125, "iterations": 100, "tolerance": 1e-6},
   )

Key parameters
--------------

- ``bounding_box`` — per load step; each step can have a different z-extent.
- ``build_graph(CVT_iter=...)`` — CVT iterations for the tessellation the graph is
  built on.
- ``match_grains(message_passing_iter, neighbor_selection_param)`` — the
  message-passing depth and the neighbor-selection optimization (``lambda``,
  ``iterations``, ``tolerance``).

Outputs
-------

The grain correspondence between the two graphs (matched IDs and mapping, plus
diagnostics) is written to ``output_dir``.

Gotchas
-------

- Orientations are read as Euler-Bunge; keep ``angle_identifier`` and
  ``orientation_descriptor`` consistent with the data.
- This uses ``build_graph`` rather than a full mesh, so no GMSH or CUBIT is
  needed. NEPER must be resolvable (see :doc:`../install`).

Full example
------------

.. literalinclude:: ../../examples/demonstrate_graintracking.py
   :language: python
   :caption: examples/demonstrate_graintracking.py

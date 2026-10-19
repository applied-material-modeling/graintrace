Tessellation to graph
=====================

Overview
--------
Conversion of a NEPER ``.tess`` tessellation into a graph whose **nodes are grains** and whose
**edges are shared faces**, as a ``torch_geometric`` ``Data`` object. The stage is driven by
:class:`~graintrace.NeperTessToGraphNN`. It exists so that a reconstructed microstructure can
be fed to a graph neural network -- predicting a per-grain quantity from the grain's own
attributes *and* its neighbourhood, which is the natural data structure for a problem where
what happens in a grain depends on who it is next to.

This stage requires the ``gnn`` extra (``pip install "graintrace[gnn]"``).

Method
------
A tessellation is already a graph. NEPER's ``.tess`` file stores the full cell complex --
vertices, edges, faces, and which faces bound which cells -- so the grain adjacency structure
needs no geometric inference: two grains are neighbours exactly when they share a face.

The construction inverts the cell-to-face lists into a face-to-cell map and keeps the faces
with two owners:

.. math::
   :label: tess-adjacency

   \mathcal{E} = \bigl\{\, (a, b) \;:\; \exists\, f \in \mathcal{F},\;
       \mathrm{cells}(f) = \{a, b\} \,\bigr\}.

Faces with a single owner lie on the domain boundary and contribute **no edge**: there is no
"exterior" node. Grains on the surface of the domain therefore have systematically lower
degree than interior grains, which is a real feature of the data -- a surface grain genuinely
has fewer neighbours inside the reconstruction -- but it also means degree is confounded with
boundary membership. A model that learns from degree will partly be learning "am I on the
edge of the sample".

A face owned by zero cells, or by more than two, means the tessellation is not a valid
manifold complex. ``validate_topology`` counts all four cases and raises on the bad ones
before any graph is built, rather than letting a malformed ``.tess`` produce a quietly wrong
adjacency.

.. warning::

   ``edge_index`` is built with **one directed edge per internal face**, source in row 0 and
   target in row 1. It is not symmetrised. Most ``torch_geometric`` message-passing layers
   propagate only along the given direction, so used as-is each pair of neighbouring grains
   exchanges information one way only, chosen by whichever cell NEPER happened to list first.
   Symmetrise before training unless you specifically want a directed graph:

   .. code-block:: python

      from torch_geometric.utils import to_undirected

      graph = parser.build_cell_graph()
      graph.edge_index = to_undirected(graph.edge_index)

   Note that symmetrising after the fact invalidates ``edge_attr``, which is indexed
   edge-for-edge against the original ``edge_index``. Duplicate the edge features to match if
   you have registered any.

Features
~~~~~~~~
Node and edge features are supplied by *registries*: a name mapped to a callable, with the
active subset concatenated column-wise at build time. ``graph.feature_slices`` records the
column range each feature occupies, so a block can be recovered from the assembled matrix.

Two defaults are worth knowing before use:

- The only pre-registered node feature is ``seed_centroid``, and it is a documented
  placeholder. It returns the raw ``.tess`` ``*seed`` rows, which are **four** columns --
  :math:`(x, y, z, w)`, the seed position and its Laguerre weight -- not a three-column
  centroid. A seed is also not a cell centroid in general (see :doc:`ff-tessellation`).
- **No edge features are registered by default**, so ``edge_attr`` comes back with shape
  :math:`(E, 0)`. The graph carries topology, not geometry.

``register_dataframe_features`` is the practical route to real node features: it attaches
columns of a per-grain dataframe (orientation, size, strain, anything from the reconstruction)
as named node features, row-aligned with the cell ordering.

.. note::

   The geometry hooks -- ``load_geometry_information`` and
   ``run_neper_for_geometry_information`` -- are **unimplemented placeholders**. The
   ``cell_centroid``, ``cell_vol``, ``face_centroid`` and ``face_area`` attributes they would
   fill stay empty lists, so face area is not available as an edge weight out of the box.
   Compute the geometry separately (``neper -T -statcell``/``-statface``) and attach it
   through the feature registries.

Orientations
~~~~~~~~~~~~
NEPER writes orientations in whatever descriptor the tessellation was generated with, named
on the first line of the ``*ori`` block. The parser reads that descriptor and converts to
canonical neml2 MRP, :math:`\tan(\theta/4)\,\hat{\mathbf{e}}`, so the graph's orientations
match every other orientation in the package regardless of how the ``.tess`` was produced.
See :doc:`/notation` for the representations and the conversion rules.

Algorithm
---------
1. Parse the ``.tess`` sections: cell seeds and orientations, vertices, edges, face vertices
   and face edges, and the cell-to-face lists. IDs are 1-based in the file and 0-based in
   memory; signed face and edge references are preserved.
2. Convert the ``*ori`` rows from NEPER's descriptor to neml2 MRP.
3. Register the default node feature and activate all registered features.
4. Invert cell-to-face into face-to-cell, validate that every face has one or two owners, and
   emit one edge per two-owner face as :eq:`tess-adjacency`.
5. Concatenate the active node features into ``x`` and the active edge features into
   ``edge_attr``, recording ``feature_names``, ``feature_slices`` and ``edge_feature_names``
   on the returned ``Data``.

Parameters that matter
----------------------
- ``tess_path``: the NEPER ``.tess`` file; the only required input.
- ``device`` / ``dtype``: where the tensors are built. The default is ``cpu`` and
  ``torch.float64``; most GNN stacks expect ``float32``, so cast before training.
- ``geometry_cell_file`` / ``geometry_face_file``: intended inputs for the geometry hooks;
  currently inert (see the note above).

Further details
---------------
For the ``.tess`` file format, see the NEPER documentation: https://neper.info/doc/ . For the
``Data`` object, message passing, and the graph utilities referenced here, see
https://pytorch-geometric.readthedocs.io/ .

See also
--------
- :doc:`ff-tessellation` -- where the ``.tess`` comes from
- :doc:`/notation` -- orientation representations
- API: :class:`~graintrace.NeperTessToGraphNN`

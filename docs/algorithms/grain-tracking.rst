Grain tracking
==============

Overview
--------
Grain tracking establishes a correspondence between the grains of the same sample reconstructed
at two load steps (or times). Each far-field reconstruction is turned into a grain graph
(``VoronoiMeshBuilder.build_graph``), and :class:`~graintrace.GraphGrainMatcher` matches the two
graphs by propagating node features with message passing and then selecting mutual best
neighbours under a graph-consistency cost.

Method
------
Each grain is a graph node with position and orientation features; edges come from tessellation
adjacency (neighbouring grains). Node features are refined by :math:`T` rounds of message passing
over the undirected edge set,

.. math::

    F^{(k+1)}_v = F^{(k)}_v + \sum_{u \in \mathcal{N}(v)} \Phi\!\left(F^{(k)}_u, F^{(k)}_v, k\right).

In the default scheme the message aggregates local structure into a scalar channel: at
:math:`k = 0` it carries the symmetry-aware misorientation between the two endpoints, and at later
iterations the :math:`L_2` distance between endpoint feature vectors. Because a node accumulates
messages from its neighbourhood, after :math:`T` rounds :math:`F_v` encodes a :math:`T`-hop
structural signature that is comparable across the two graphs.

Given the refined features :math:`F^A` and :math:`F^B`, each grain :math:`i` in A takes its
top-:math:`k` candidates in B by squared feature distance (``torch.cdist``). The match cost
combines a feature term with a neighbour-consistency term,

.. math::

    c(i, j) = \lVert F^A_i - F^B_j \rVert^2
            - \frac{\lambda}{\lvert M_i \rvert}
              \sum_{\substack{p \in \mathcal{N}(i)\\ q = \pi(p)\, \in\, \mathcal{N}(j)}}
              \lVert F^A_p - F^B_q \rVert^2,

where :math:`\pi` is the current A→B assignment and :math:`M_i` counts the matched neighbours of
:math:`i` that map into neighbours of :math:`j`. The neighbour term (weighted by :math:`\lambda`
and degree-normalized) lowers the cost when a candidate preserves the local adjacency of already
matched grains. Assignment is iterative: each grain proposes its lowest-cost candidate, each B
grain is claimed by the lowest-cost proposer, and the mapping is updated until it stabilizes.

Algorithm
---------
1. Build a grain graph for each load step from its FF reconstruction (nodes = grains with
   position/orientation features, edges = tessellation adjacency).
2. Run :math:`T = ` ``message_passing_iter`` rounds of message passing on each graph to produce
   refined node features :math:`F^A`, :math:`F^B`.
3. For every A grain, precompute its top-:math:`k` B candidates by squared feature distance.
4. Iterate: score each candidate with the feature + neighbour-consistency cost, let each grain
   pick its best candidate, resolve conflicts by lowest cost (mutual claiming), and update the
   A→B map; repeat up to ``iterations`` or until it converges.
5. Emit the matched pairs with their final costs, the full A→B mapping, the node embeddings, and a
   metadata JSON.

Parameters that matter
----------------------
- ``message_passing_iter`` (:math:`T`): number of hops encoded into each node signature; deeper
  passing spreads structural context further but costs more.
- ``neighbor_selection_param["lambda"]`` (:math:`\lambda`): weight of the neighbour-consistency
  term; larger values favour matches that preserve local adjacency over pure feature similarity.
- ``neighbor_selection_param["topk"]``: candidate pool size per A grain.
- ``neighbor_selection_param["iterations"]`` / ``["tolerance"]``: iteration budget and
  convergence control of the assignment loop.
- ``build_graph(CVT_iter=...)``: CVT iterations for the tessellation the graph is built on; a
  consistent bounding box per load step keeps the two graphs comparable.

See :doc:`/configuration` for the surrounding reconstruction options.

References
----------
- Neural message passing on graphs: Gilmer et al. (2017), *Neural Message Passing for Quantum
  Chemistry*, https://arxiv.org/abs/1704.01212 (the update rule here follows the same
  message/aggregate/update pattern).
- The neighbour-consistency term is a graph-matching regularizer that rewards assignments
  preserving local adjacency; the assignment loop is a mutual best-neighbour selection rather than
  a global optimum.

See also
--------
- Tutorial: :doc:`/tutorials/grain-tracking`
- API: :class:`~graintrace.GraphGrainMatcher`

Notation and conventions
========================

Every equation in these pages uses the symbols defined here, and every
orientation in graintrace obeys the conventions stated here. If a number coming
out of the package looks wrong by a factor of 57.3, by :math:`\pi/2`, or by a
crystal symmetry operation, this page is the first place to look.

.. contents::
   :local:
   :depth: 2

Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 16 20 18 46

   * - Symbol
     - Type
     - Units
     - Meaning
   * - :math:`\varphi_1,\ \Phi,\ \varphi_2`
     - scalars
     - rad internally; deg or rad at the boundary
     - Bunge Euler angles
   * - :math:`\psi,\ \theta,\ \phi`
     - scalars
     - rad
     - Kocks / Roe Euler triple, remapped to Bunge on entry
   * - :math:`g`
     - :math:`3\times3`, :math:`g \in SO(3)`
     - –
     - orientation matrix, **passive**, sample :math:`\to` crystal
   * - :math:`A`
     - :math:`3\times3`, :math:`SO(3)`
     - –
     - the corresponding *active* rotation; :math:`g = A^{\mathsf T}`
   * - :math:`R_z(a),\ R_x(a)`
     - :math:`3\times3`
     - –
     - active right-handed rotations about :math:`\hat z` / :math:`\hat x`
   * - :math:`q = (w, x, y, z)`
     - length-4 vector
     - –
     - **scalar-first** unit quaternion
   * - :math:`\mathbf v = (x,y,z)`
     - 3-vector
     - –
     - vector part of :math:`q`
   * - :math:`\hat n`
     - unit 3-vector
     - –
     - rotation axis
   * - :math:`\theta`
     - scalar
     - rad internally, deg on output by default
     - rotation or misorientation angle
   * - :math:`\mathbf r`
     - 3-vector
     - –
     - Rodrigues–Gibbs vector, :math:`\mathbf r = \tan(\theta/2)\,\hat n`
   * - :math:`\mathbf p`
     - 3-vector
     - –
     - **NEML2 v3 MRP**, :math:`\mathbf p = \tan(\theta/4)\,\hat n`
   * - :math:`[\mathbf a]_\times`
     - :math:`3\times3` skew
     - –
     - cross-product matrix, :math:`[\mathbf a]_\times \mathbf b = \mathbf a \times \mathbf b`
   * - :math:`\Delta g`
     - :math:`3\times3`, :math:`SO(3)`
     - –
     - misorientation matrix between two orientations
   * - :math:`O_i`
     - :math:`3\times3`, :math:`SO(3)`
     - –
     - :math:`i`-th crystal symmetry operator (proper rotation)
   * - :math:`S_c,\ S_s`
     - integers
     - –
     - number of crystal / sample symmetry operators
   * - :math:`\mathbf d`
     - unit 3-vector
     - –
     - sample direction projected in an inverse pole figure
   * - :math:`\mathbf c`
     - unit 3-vector
     - –
     - crystal-frame direction
   * - :math:`M`
     - :math:`4\times4` symmetric PSD
     - –
     - Markley quaternion-mean matrix, :math:`\sum_n q_n q_n^{\mathsf T}`
   * - :math:`\sigma`
     - scalar
     - units of the edge metric
     - kernel width; see :ref:`notation-sigma`
   * - :math:`R_r`
     - :math:`3\times3`, :math:`SO(3)`
     - –
     - sample-tilt correction rotation
   * - :math:`\boldsymbol\varepsilon`
     - :math:`3\times3` symmetric
     - strain or microstrain
     - elastic strain tensor
   * - :math:`\gamma`
     - scalar
     - –
     - Leiden resolution parameter
   * - :math:`w_{ij}`
     - scalar
     - –
     - graph edge weight between nodes :math:`i` and :math:`j`
   * - :math:`d_{ij}`
     - scalar
     - units of the edge metric
     - graph edge distance between nodes :math:`i` and :math:`j`

.. _notation-rotation-convention:

The rotation convention
-----------------------

graintrace uses **Bunge Z–X–Z Euler angles** and the **passive**
(sample :math:`\to` crystal) orientation matrix. Stated once, as settled fact:

.. math::
   :label: bunge-matrix

   g(\varphi_1, \Phi, \varphi_2)
   \;=\; \bigl[\, R_z(\varphi_1)\, R_x(\Phi)\, R_z(\varphi_2) \,\bigr]^{\mathsf T}

The bracketed product is the active rotation :math:`A`; the orientation matrix
is its transpose, so that :math:`\mathbf c = g\,\mathbf s` takes a direction
from sample coordinates to crystal coordinates. Written out, with
:math:`c_1 = \cos\varphi_1`, :math:`s_1 = \sin\varphi_1`,
:math:`c_2 = \cos\varphi_2`, :math:`s_2 = \sin\varphi_2`,
:math:`c_\Phi = \cos\Phi`, :math:`s_\Phi = \sin\Phi`:

.. math::
   :label: bunge-matrix-explicit

   g =
   \begin{bmatrix}
   c_1 c_2 - s_1 c_\Phi s_2 & s_1 c_2 + c_1 c_\Phi s_2 & s_\Phi s_2 \\
   -c_1 s_2 - s_1 c_\Phi c_2 & -s_1 s_2 + c_1 c_\Phi c_2 & s_\Phi c_2 \\
   s_1 s_\Phi & -c_1 s_\Phi & c_\Phi
   \end{bmatrix}

:func:`~graintrace.orientation_helper.euler_to_matrix` implements
:eq:`bunge-matrix`. The inverse,
:func:`~graintrace.orientation_helper.matrix_to_euler`, reads
:eq:`bunge-matrix-explicit` back off the entries it predicts:

.. math::

   \Phi = \arccos\bigl(\mathrm{clamp}(g_{33}, -1, 1)\bigr), \quad
   \varphi_1 = \operatorname{atan2}(g_{31},\, -g_{32}), \quad
   \varphi_2 = \operatorname{atan2}(g_{13},\, g_{23})

When :math:`|\sin\Phi| < 10^{-8}` the triple is degenerate (gimbal lock). The
code then returns :math:`\varphi_2 = 0` and
:math:`\varphi_1 = \operatorname{atan2}(g_{12}, g_{11})`, which recovers
:math:`\varphi_1 + \varphi_2` at :math:`\Phi = 0` and
:math:`\varphi_1 - \varphi_2` at :math:`\Phi = \pi`. The matrix round-trip is
exact there; the individual angles are not unique.

Other conventions are accepted at the boundary and remapped to Bunge before
anything else happens:

.. math::

   \text{bunge: } (a, b, c), \qquad
   \text{kocks: } \left(a + \tfrac{\pi}{2},\ b,\ \tfrac{\pi}{2} - c\right), \qquad
   \text{roe: } \left(a + \tfrac{\pi}{2},\ b,\ c - \tfrac{\pi}{2}\right)

Passing Kocks angles while the function reads Bunge therefore shifts two of the
three angles by :math:`\pi/2`. Nothing downstream can detect it, because every
Euler triple is a valid Euler triple.

.. _notation-representations:

The six representations
-----------------------

The **rotation matrix is the pivot**: every conversion in the package goes
through it. **MRP is the interchange format**: it is what lands on disk and what
the CPFE model reads and writes.

**Euler angles** :math:`(\varphi_1, \Phi, \varphi_2)`, Bunge Z–X–Z, as above.

**Quaternion**, scalar-first :math:`q = (w, x, y, z)` with
:math:`w = \cos(\theta/2)` and :math:`(x,y,z) = \sin(\theta/2)\,\hat n`:

.. math::
   :label: quat-matrix

   R(q) = \bigl(w^2 - \lVert \mathbf v \rVert^2\bigr)\mathbf I
        + 2\,\mathbf v \mathbf v^{\mathsf T}
        + 2w\,[\mathbf v]_\times

No hemisphere convention is imposed anywhere in the package. Where the
:math:`q \equiv -q` ambiguity could matter — the quaternion mean below — it
cancels, because that estimator is quadratic in :math:`q`.

.. warning::

   :eq:`quat-matrix` is **homogeneous of degree two** in :math:`q`: a quaternion
   of norm :math:`N` yields :math:`N^2 R`, which is not a rotation.
   :func:`~graintrace.orientation_helper.quat_to_matrix` documents that it wants
   a unit quaternion and does not enforce it, so an unnormalised input produces
   a plausible, wrong matrix with no error. Normalise before you call it.

**Rodrigues–Gibbs vector** :math:`\mathbf r = \tan(\theta/2)\,\hat n`:

.. math::

   q = \frac{(1,\ \mathbf r)}{\sqrt{1 + \lVert \mathbf r \rVert^2}},
   \qquad
   R(\mathbf r) = \mathbf I
       + \frac{2}{1 + \lVert \mathbf r \rVert^2}
         \Bigl([\mathbf r]_\times + [\mathbf r]_\times^2\Bigr)

**NEML2 v3 MRP** :math:`\mathbf p = \tan(\theta/4)\,\hat n`:

.. math::
   :label: mrp-def

   \mathbf p = \frac{\mathbf v}{1 + w},
   \qquad
   w = \frac{1 - \lVert \mathbf p \rVert^2}{1 + \lVert \mathbf p \rVert^2},
   \qquad
   \mathbf v = \frac{2\mathbf p}{1 + \lVert \mathbf p \rVert^2}

and the two vector forms are related by

.. math::

   \mathbf r = \frac{2\mathbf p}{1 - \lVert \mathbf p \rVert^2},
   \qquad
   \mathbf p = \frac{\mathbf r}{1 + \sqrt{1 + \lVert \mathbf r \rVert^2}}

**Rotation matrix as a storage format.** Nine columns named ``O11``–``O33`` are
read **row-major**, so ``O11..O33`` map to :math:`g_{11}..g_{33}`. A producer
that writes column-major silently yields the transpose; nothing checks.

**Conversions as implemented.** Every entry below lives in
:mod:`graintrace.orientation_helper` unless noted:

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - From :math:`\to` To
     - Function
     - Route
   * - Euler :math:`\to` matrix
     - :func:`~graintrace.orientation_helper.euler_to_matrix`
     - :eq:`bunge-matrix`, via NEML2
   * - matrix :math:`\to` Euler
     - :func:`~graintrace.orientation_helper.matrix_to_euler`
     - closed form, pure torch
   * - matrix :math:`\to` MRP
     - :func:`~graintrace.orientation_helper.matrix_to_mrp`
     - NEML2 ``MRP.from_matrix``
   * - MRP :math:`\to` matrix
     - :func:`~graintrace.orientation_helper.mrp_to_matrix`
     - NEML2 ``euler_rodrigues``
   * - Euler :math:`\to` MRP
     - :func:`~graintrace.orientation_helper.euler_to_mrp`
     - composition of the two above
   * - MRP :math:`\to` Euler
     - :func:`~graintrace.orientation_helper.mrp_to_euler`
     - composition of the two above
   * - quaternion :math:`\to` matrix
     - :func:`~graintrace.orientation_helper.quat_to_matrix`
     - :eq:`quat-matrix`, via NEML2
   * - matrix :math:`\to` quaternion
     - :func:`~graintrace.orientation_helper.matrix_to_quat`
     - NEML2 ``to_quaternion``
   * - dataframe :math:`\to` matrix
     - :func:`~graintrace.orientation_helper.load_orientation_matrices`
     - row-major read of ``O11..O33``
   * - dataframe :math:`\to` MRP
     - :func:`~graintrace.orientation_helper.load_orientations`
     - the above, then ``matrix_to_mrp``

There is no direct Euler :math:`\leftrightarrow` quaternion or
Euler :math:`\leftrightarrow` MRP path; both go through the matrix.

.. _notation-ori-rodrigues:

``ori_rodrigues_*`` holds MRP, not Rodrigues
--------------------------------------------

.. warning::

   The CPFE output columns ``ori_rodrigues_x``, ``ori_rodrigues_y`` and
   ``ori_rodrigues_z`` contain the **NEML2 v3 MRP**
   :math:`\mathbf p = \tan(\theta/4)\,\hat n`, **not** the classical Rodrigues
   vector :math:`\mathbf r = \tan(\theta/2)\,\hat n`. The name is a misnomer.

The two differ by roughly a factor of two at small angles and diverge from
there, so feeding these columns to a Rodrigues-expecting tool produces a
plausible wrong answer rather than an error. The quantity really is MRP: the
MOOSE deck declares ``unknowns_MRP = 'orientation'`` and the three aux variables
are components of that same state variable.

Every in-repo consumer already decodes it correctly as MRP — the fragmentation
analyser, the pole-figure plotter and the orientation trackers all call
:func:`~graintrace.orientation_helper.mrp_to_matrix`. The name is wrong in four
places (the aux variables, the initial-condition functions, the grid transfers
and the column names) and the value is right everywhere. Renaming is a
cross-cutting change that has not been made; until it is, read the name as
``ori_mrp_*``.

.. _notation-units:

The three angle-unit switches
-----------------------------

Three differently-named keyword arguments select an angle unit, and they are not
interchangeable.

.. list-table::
   :header-rows: 1
   :widths: 16 22 20 42

   * - Argument
     - Where
     - Default
     - Meaning
   * - ``angle_type``
     - orientation helpers, mesh builders, fragmentation
     - ``"degrees"``
     - unit of Euler input **and** of angle output; see the warning below
   * - ``unit``
     - :class:`~graintrace.VoronoiMeshBuilder` and the rotation helper
     - ``"deg"``
     - unit of the Euler columns in the input CSV, and of ``rotate_angles``
   * - ``orientation_units``
     - :class:`~graintrace.hedm_stitching_techniques.region_base_stitching.RegionBaseStitching`
     - ``"degrees"``
     - unit of the Euler columns **and** of ``orientation_tolerance``

.. warning::

   ``angle_type`` is **dual-purpose**. It sets the unit in which Euler angles
   are read *and* the unit in which an angle is returned, from the same switch.
   ``misorientation(e1, e2, angle_type="radians")`` therefore reads radian Euler
   input and returns a radian misorientation; there is no way to ask for radian
   input with degree output. Convert explicitly if you need the mixed case, as
   :meth:`~graintrace.fragmentation.FragmentationAnalyzer.segment` does
   internally.

.. warning::

   ``orientation_tolerance`` is **not** converted for you. With
   ``orientation_units="radians"`` you must pass a tolerance already in radians::

      if ori_units == "radians":
          orientation_tolerance = np.deg2rad(orientation_tolerance)

   Passing ``5.0`` meaning five degrees to a radians-configured stitcher asks
   for a 286-degree tolerance, which matches everything.

Validation is asymmetric.
:func:`~graintrace.orientation_helper.euler_to_matrix` raises on an unrecognised
``angle_type``; :func:`~graintrace.orientation_helper.matrix_to_euler` and
:func:`~graintrace.orientation_helper.misorientation_matrix` test only for the
literal string ``"degrees"`` and silently return radians for anything else. A
typo such as ``"degree"`` or ``"deg"`` is not an error there — it is radians.

.. _notation-symmetry:

Symmetry, and the default that bites
------------------------------------

Point groups are named in orbifold notation (``"432"`` for cubic, ``"1"`` for
none) and expanded by NEML2 into **proper rotations only** — 24 operators for
``"432"``, one for ``"1"``.

.. warning::

   The symmetry default is **not** uniform across the package. The low-level
   orientation layer defaults to ``symmetry="1"``, meaning *no reduction*; the
   analysis layer defaults to ``"432"``.

   .. list-table::
      :header-rows: 1
      :widths: 50 50

      * - Defaults to ``"1"`` (no reduction)
        - Defaults to ``"432"``
      * - ``misorientation_matrix``
        - ``FragmentationAnalyzer``
      * - ``misorientation``
        - ``merge_fragments_by_misorientation``
      * - ``move_to_fundamental_zone``
        - ``adjacency_remerge``
      * - ``average_orientation``
        - ``SimilarityMetricLibrary.misorientation``
      * - ``nf.metrics.misorientation``
        - ``misorientation_distance``
      * - ``nf.metrics.average_rotations``
        -
      * - ``FragmentationAnalyzer.mean_orientation_mrp``
        -

   With ``"1"`` the returned "misorientation" is the raw rotation angle, which
   for a cubic crystal reaches :math:`180^\circ` instead of the
   :math:`62.8^\circ` disorientation maximum. Calling
   ``orientation_helper.misorientation(e1, e2)`` with no keyword arguments
   returns a number that looks like a misorientation and is not one. Pass the
   point group explicitly.

**Inversion is included in exactly one place, and that is correct.** Inversion
is not a symmetry of an orientation — improper operators would take
:math:`SO(3)` out of itself — so disorientation, folding and fundamental-zone
reduction all use the 24 proper operators. Inverse pole figures compare
*directions*, which are axial (:math:`\mathbf c \equiv -\mathbf c`), so the IPF
path uses the full 48-element point group. Without that, a deformed orientation
can have no representative in the fundamental sector at all.

**One-sided versus two-sided.** Misorientation applies operators on both sides,
:math:`O_i\,\Delta g\,O_j^{\mathsf T}`. Fundamental-zone reduction, variant
folding and mean-orientation alignment apply them on the **left only**,
:math:`O_i g`, which is the correct side given that :math:`g` maps sample to
crystal and the crystal index is the row index.

Misorientation
--------------

For two orientation matrices :math:`g_A` and :math:`g_B`, the misorientation
matrix is

.. math::
   :label: misorientation-matrix

   \Delta g = g_A\, g_B^{\mathsf T}

and the symmetry-reduced (disorientation) angle minimises over **both** crystals'
symmetry orbits:

.. math::
   :label: disorientation

   \theta_{\min} = \min_{i,j \in \{1 \dots S_c\}}
   \arccos\!\left(
     \mathrm{clamp}\!\left(
       \frac{\operatorname{tr}\bigl(O_i\, \Delta g\, O_j^{\mathsf T}\bigr) - 1}{2},
       \,-1,\, 1
     \right)
   \right)

For ``"432"`` that is :math:`24^2 = 576` candidates per pair. The clamp is what
makes numerically identical orientations return exactly zero instead of ``NaN``.

The angle of :math:`\Delta g` equals the angle of :math:`\Delta g^{-1}`, so the
returned *angle* does not depend on which orientation is passed first — but the
matrix in :eq:`misorientation-matrix` does, so an axis would.

Fundamental-zone reduction picks the symmetry variant of a single orientation
closest to the identity:

.. math::

   g_{\mathrm{FZ}} = O_{i^\star} g, \qquad
   i^\star = \arg\max_i \operatorname{tr}(O_i g)

Maximising the trace minimises the angle, since
:math:`\theta = \arccos\bigl((\operatorname{tr} - 1)/2\bigr)` decreases in the
trace.

Mean orientation
----------------

graintrace ships **two** mean-orientation estimators with different objective
functions and different symmetry references. They agree on tight clusters and
diverge on dispersed ones, so it matters which one you are calling.

**Quaternion (Markley) mean** —
:meth:`~graintrace.fragmentation.FragmentationAnalyzer.mean_orientation_mrp` and
``nf.metrics.average_rotations``. Members are folded onto the symmetry variant
closest to the **first** member, converted to scalar-first unit quaternions
stacked row-wise into :math:`Q \in \mathbb{R}^{N \times 4}`, and the mean is the
dominant eigenvector:

.. math::
   :label: markley-mean

   M = Q^{\mathsf T} Q = \sum_{n=1}^{N} q_n q_n^{\mathsf T},
   \qquad
   \bar q = \arg\max_{\lVert q \rVert = 1} q^{\mathsf T} M q

Equivalently :math:`\bar q` minimises
:math:`\sum_n \bigl(1 - (q^{\mathsf T} q_n)^2\bigr)`, the sum of squared
quaternion chordal distances. Because that is quadratic in :math:`q_n`, the
:math:`q \equiv -q` sign ambiguity cancels and no hemisphere alignment is needed.
All members are weighted equally.

**Matrix (polar/SVD) mean** —
:func:`~graintrace.orientation_helper.average_orientation`. Members are folded
onto the variant closest to the **highest-weight** member, and the mean is the
orthogonal factor of the weighted arithmetic mean:

.. math::
   :label: polar-mean

   \bar M = \frac{\sum_n w_n\, \tilde g_n}{\sum_n w_n},
   \qquad
   \bar M = U \Sigma V^{\mathsf T},
   \qquad
   \bar g = U V^{\mathsf T}

with a reflection guard that flips the last column of :math:`U` when
:math:`\det \bar g < 0`. This one accepts weights.

.. note::

   Folding is skipped entirely when ``symmetry="1"``, which is the default on
   both estimators. On cubic data whose members are reported in different
   symmetry variants, the unfolded mean is close to **no** member of the set.
   Pass the point group.

.. _notation-sigma:

:math:`\sigma` is in the units of the metric
---------------------------------------------

:attr:`~graintrace.user_data_class.WeightConfig.sigma` sets the width of the
weight kernel that turns an edge *distance* into an edge *weight*. Its unit is
whatever the edge metric produces, which depends entirely on which metric you
chose:

* a misorientation metric :math:`\Rightarrow` :math:`\sigma` is in **degrees**
  (or radians, if you configured the metric that way);
* a von Mises stress metric :math:`\Rightarrow` :math:`\sigma` is in **MPa**;
* ``abs_scalar_diff`` on an arbitrary column :math:`\Rightarrow` :math:`\sigma`
  is in that column's units.

There is no universal good value. The vetted starting point for orientation
segmentation is :math:`\sigma \approx` half the misorientation cutoff; see
:doc:`algorithms/segmentation`.

Units used throughout
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Quantity
     - Unit
     - Notes
   * - Length, position
     - micrometre (µm)
     - all coordinates, bounding boxes, tolerances and radii
   * - Stress
     - megapascal (MPa)
     - CPFE material constants and all stress output
   * - ``burger_scale``
     - ångström (Å)
     - the sole exception to the MPa/µm pair
   * - Strain
     - dimensionless
     - except FF elastic-strain columns, often microstrain
   * - Microstrain
     - :math:`10^{-6}`
     - selected with ``strain_unit="microstrain"``
   * - Angle
     - degrees by default
     - see :ref:`notation-units`; always check the switch
   * - Time
     - model time units
     - ``dt``, ``total_time``, ``initialize_time`` share one arbitrary scale
   * - Voxel counts
     - voxels or elements
     - ``manhattan_radius`` is in voxels; ``grain_threshold_final`` counts
       voxels on a grid and elements on a mesh

See also
--------

* :doc:`file-formats` — the on-disk schema of every file graintrace reads or writes, with the unit of each column.
* :doc:`configuration` — per-class option tables carrying type, default and unit.
* :doc:`algorithms/segmentation` — where :eq:`disorientation` and :math:`\sigma` are actually used.
* :doc:`api/orientation_helper` — the function-level reference for every conversion listed above.

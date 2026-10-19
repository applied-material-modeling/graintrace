File formats
============

Every file graintrace reads or writes, with its delimiter, header convention,
exact column order and units. Angle and length conventions are defined in
:doc:`notation`; this page gives the schemas that carry them.

Column order matters in several of these formats because the file has no header
row, and tensor components are stored **row-major** throughout
(:math:`11, 12, 13, 21, \dots, 33`) except in the CPFE field output, which mixes
row-major and Voigt in the same file. Each entry says which.

.. contents::
   :local:
   :depth: 2

Experimental input
------------------

FF HEDM raw scan CSV
~~~~~~~~~~~~~~~~~~~~

One file per scan layer. Whitespace-delimited. Lines 0–7 are a header block,
**line 8 is the column-name row**, and data starts at line 9.

.. important::

   The column-name row may carry a leading ``%``, which must be stripped before
   the names parse. It is optional, so a reader must handle both forms::

      with open(fpath) as fh:
          lines = fh.readlines()
      data_lines = lines[8:]
      if data_lines[0].lstrip().startswith("%"):
          line = data_lines[0]
          idx = line.find("%")
          data_lines[0] = line[:idx] + line[idx + 1:]
      df = pd.read_csv(StringIO("".join(data_lines)), sep=r"\s+")

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Column
     - Unit
     - Meaning
   * - ``X``, ``Y``, ``Z``
     - µm
     - grain centroid position
   * - ``GrainRadius``
     - µm
     - equivalent-sphere radius
   * - ``Eul0``, ``Eul1``, ``Eul2``
     - deg or rad
     - Bunge Euler angles; detect the unit by testing whether any value exceeds :math:`2\pi`
   * - ``Confidence``
     - –
     - fit quality; typically filtered to :math:`\ge 0.7` or :math:`\ge 0.9`
   * - ``eFab11``–``eFab33``
     - microstrain
     - fabric / lattice strain tensor, **row-major** 9 components
   * - ``eKen11``–``eKen33``
     - microstrain
     - Kenesei elastic strain tensor, **row-major** 9 components
   * - ``ScanID``
     - –
     - assigned during stitching; **not present** in a raw file

FF calibration CSV
~~~~~~~~~~~~~~~~~~

This is a **different format** from the raw scan file above, and the two are
easy to confuse. One file per applied stress level, named for that stress
(``0.csv``, ``100.csv``, …). Comma-delimited, **ordinary header row, no
preamble**:

.. code-block:: text

   O11,O12,O13,O21,O22,O23,O31,O32,O33,X,Y,Z,GrainRadius,Eul0,Eul1,Eul2,eKen11,…,eKen33

The ``O11``–``O33`` block is the row-major rotation matrix. It is written by
:func:`~graintrace.experiment_rotation_helper.update_experiments`, which drops
any pre-existing ``O*`` columns and re-derives them from the ``Eul*`` columns in
the simulation frame. This is the format
:meth:`~graintrace.taylor.TaylorModel.load_experiment_data` reads. Shipped
example: ``mwe_data/ff_calibration/``.

``strain-stress.csv``
~~~~~~~~~~~~~~~~~~~~~

The macroscopic curve for calibration. Two columns, **no header row**:

.. code-block:: text

   0,0
   0.000107941,18.6042

.. warning::

   The first row is **data**, not a label. Adding a header row silently drops
   the first point or raises, depending on the reader.

Column 0 is strain (dimensionless), column 1 is stress (MPa).

NF ``.mic``
~~~~~~~~~~~

One file per layer, tab-delimited, with **exactly three** ``%`` metadata lines
followed by the column-name row:

.. code-block:: text

   %TriEdgeSize 2.000000
   %NumPhases 1
   %GlobalPosition 0.000000
   %OrientationRowNr	NrMatches	RunTime	X	Y	TriEdgeSize	UpDown	Eul1	Eul2	Eul3	Confidence	PhaseNr

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Column
     - Unit
     - Meaning
   * - ``%OrientationRowNr``
     - –
     - row index; the reader renames it to ``OrientationRowNr``
   * - ``NrMatches``
     - –
     - number of matched reflections
   * - ``RunTime``
     - s
     - per-point fit time
   * - ``X``, ``Y``
     - µm
     - in-plane position
   * - ``TriEdgeSize``
     - µm
     - triangular-mesh edge length of the reconstruction
   * - ``UpDown``
     - –
     - triangle orientation flag
   * - ``Eul1``, ``Eul2``, ``Eul3``
     - deg or rad
     - Euler angles; the unit is set by the builder's ``angle_type``
   * - ``Confidence``
     - –
     - fit confidence
   * - ``PhaseNr``
     - –
     - phase index

The reader adds ``Z = layer_index * dz`` and a ``layer`` column. Layer numbers
must be **contiguous**; a gap raises ``ValueError``. The reader accepts ``.csv``
files in the input folder as well as ``.mic``, with the same schema.

EDAX/TSL ``.ang``
~~~~~~~~~~~~~~~~~

The native EDAX/TSL OIM acquisition format. No builder reads it directly;
:class:`~graintrace.AngConverter` converts it to either of the two schemas that
are read -- the NF ``.mic`` above, or the merged EBSD CSV below.

Two layouts are accepted. The minimal one is headerless and eight columns:

.. code-block:: text

   0.174540  0.349660  0.523050   0.00000   0.00000   122.3   0.690  1

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Column
     - Unit
     - Meaning
   * - ``phi1``, ``PHI``, ``phi2``
     - rad or deg
     - Euler angles; the convention and unit are **not** inferred (below)
   * - ``x``, ``y``
     - µm
     - in-plane position
   * - ``IQ``
     - –
     - image quality
   * - ``CI``
     - –
     - confidence index; TSL writes a negative value for an unindexed point
   * - ``phase``
     - –
     - phase index

The second layout prefixes a ``#`` comment block and usually carries ten
columns, appending ``SEM`` and ``fit``. Columns past the tenth are read and
named ``col10``, ``col11`` and so on. A ``COLUMN_HEADERS`` key in the header
overrides the count-based naming, and a count that disagrees with the data
raises rather than silently misaligning the columns.

Header keys the reader acts on:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Effect
   * - ``ANGLE_UNITS``
     - pins the unit: ``radians`` or ``degrees``
   * - ``EULER_CONVENTION``
     - pins the convention: ``bunge``, ``kocks`` or ``roe``
   * - ``COLUMN_HEADERS``
     - names the columns, overriding the count-based default
   * - ``XSTEP``, ``YSTEP``
     - in-plane step; ``XSTEP`` becomes the default ``TriEdgeSize`` on ``.mic`` output
   * - ``GRID``
     - recorded on the parsed header, not acted on

Key matching ignores case and separators, and both ``KEY: value`` and
``KEY   value`` forms are read. Every other key is kept verbatim on
``AngHeader.keys``. A key present but empty -- TSL does write those -- raises
``ValueError`` for ``XSTEP``, ``YSTEP`` and ``GRID`` rather than a bare
``IndexError``.

.. note::

   ``ANGLE_UNITS`` and ``EULER_CONVENTION`` are a **graintrace** convention,
   not a TSL one. Stock OIM writes neither, so an unmodified export pins
   nothing and both ``angle_convention`` and ``angle_type`` must be passed
   explicitly. Add the two keys to the header only if you want the file to
   carry its own provenance.

.. warning::

   **The convention and the unit are never guessed.** They come from the header
   when it pins them, otherwise from the explicit ``angle_convention`` and
   ``angle_type`` arguments, and a file that supplies neither raises
   ``ValueError``. EDAX/TSL OIM writes Bunge angles in radians, so that pair is
   right for an unmodified export -- but a degrees file read as radians gives a
   microstructure that segments without complaint and is wrong, which is why
   the default is a refusal rather than the common case.

Converting to either target:

.. code-block:: python

   from graintrace import AngConverter

   converter = AngConverter(
       input_path="ebsd/ang",          # one .ang file, or a folder of layers
       save_dir="out/ebsd",
       file_token="layer",             # layer index follows this in the filename
       angle_convention="bunge",       # omit only if the header pins it
       angle_type="radians",           # omit only if the header pins it
       output_angle_type="degrees",    # must match the builder's angle_type
       confidence_min=0.0,             # drops TSL's negative-CI unindexed points
   )

   merged = converter.to_ebsd_csv(zstep=2.0)   # -> VoxelMeshBuilder
   layers = converter.to_mic(zstep=2.0)        # -> NearFieldMeshBuilder

``z`` is the layer's **index in the filename** minus the lowest index in the
folder, times ``zstep``, so the first layer sits at ``z = 0`` and so does a
single-file input. Folder inputs are ordered by that integer, numerically --
``layer10`` sorts after ``layer9``, not between ``layer1`` and ``layer2``.
Output angles are always Bunge, in ``output_angle_type``; pass that same unit
to the downstream builder's ``angle_type``. Shipped example:
``mwe_data/ang_ebsd/`` (three layers, ten columns, header-pinned).

A gap in the layer numbering is preserved rather than closed up. ``layer1`` and
``layer3`` at ``zstep=2.0`` give ``z = 0`` and ``z = 4``, and ``to_mic`` names
them ``layer1.mic`` and ``layer3.mic``. Both matter:
:func:`~graintrace.nf.convert.nf_to_pointcloud` recomputes ``z`` from the file
order and raises on a non-contiguous layer set, so renumbering the outputs
contiguously would place the stack at the wrong ``z`` *and* hide the missing
layer from the only check that catches it.

EBSD merged CSV
~~~~~~~~~~~~~~~

A single flat CSV covering all layers, with a header row:

.. code-block:: text

   x,y,z,Eul0,Eul1,Eul2

``z`` is the layer index times the EBSD z-step. The Euler columns are Bunge
angles in degrees or radians, selected by the builder's ``angle_type``. Merging
per-layer source files into this one CSV is the user's job. Shipped example:
``mwe_data/synthetic_split_ebsd/``.

Grain table
~~~~~~~~~~~

The de-facto interchange format for per-grain data across load steps, consumed
by :meth:`~graintrace.fragmentation.FragmentationAnalyzer.detect_splits` (as
``nodes_a`` / ``nodes_b``) and by the grain matcher. Header row, comma-delimited:

.. code-block:: text

   grain_id,X,Y,Z,GrainRadius,Eul0,Eul1,Eul2

Positions and radius in µm. FF data supplies this directly; NF and EBSD
segmentations reach it through
:meth:`~graintrace.fragmentation.FragmentationAnalyzer.seg_to_grain_table`.
Shipped example: ``mwe_data/synthetic_split_ff/``.

.. note::

   ``seg_to_grain_table``'s ``background_label`` depends on which label domain you
   feed it. Output of :meth:`~graintrace.fragmentation.FragmentationAnalyzer.segment`
   is **0-based grain labels** over solid points only, so the default ``-1`` is right
   and ``background_label=0`` would delete a grain. A raw segmented voxel grid instead
   uses ``0`` void / ``>= 1`` grain / ``-1`` unsegmented, so pass
   ``background_label=(0, -1)`` to drop both sentinels.

Reconstruction output
---------------------

``reconstruction_reformatted.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The per-voxel grain field, and the input to the default hex mesher. Header row:

.. code-block:: text

   X,Y,Z,CellID,Eul0,Eul1,Eul2

Four properties of this file are silent traps when post-processing it by hand:

* Coordinates are **voxel centres**, :math:`x_{\min} + (i + \tfrac12)\,\Delta x`,
  not voxel corners.
* Row order is **column-major**: ``i`` varies fastest, then ``j``, then ``k``.
* ``CellID == 0`` means **void**.
* Void voxels carry Euler angles ``-1.0, -1.0, -1.0``, **not** zeros.

Euler angles here are **always degrees**, whatever the input unit was.

``reconstruction_cpfe_ee.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-grain initial elastic strain for the CPFE initial condition. Comma-delimited,
**no header**, **12 columns**:

.. code-block:: text

   x_seed, y_seed, z_seed, ee11, ee12, ee13, ee21, ee22, ee23, ee31, ee32, ee33

Columns 0–2 are the seed position in µm; columns 3–11 are the **row-major**
elastic strain tensor. The reader accepts whitespace *or* comma delimiting even
though the writer emits commas.

Shifting this field into another coordinate frame — the NF+FF workflow — means
offsetting columns 0–2 only, which is only discoverable from this schema.

``.ee`` and ``zero_initial_strain.ee``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Space-delimited, no header, ``%.8e`` formatting, one row per cell.

.. note::

   The two writers disagree on width. The reconstruction writes **9** columns
   (the strain tensor alone). When ``CPFESimulation(eeres_file=None)``, the
   simulation writes ``zero_initial_strain.ee`` with **12** zero columns, which
   is what the MOOSE ``PropertyReadFile`` declares via ``nprop=12``. The
   discrepancy is real; the 12-column form is the one MOOSE reads.

``orientations.dat`` (FF)
~~~~~~~~~~~~~~~~~~~~~~~~~

Whitespace-delimited, no header, three columns, one row per grain in NEPER cell
order. Euler-Bunge, and **always degrees** regardless of the input unit.

Convert to NEML2 MRP before handing it to
:class:`~graintrace.CPFESimulation`::

   mrp = oh.euler_to_mrp(torch.tensor(euler, dtype=torch.float64), "bunge", "degrees")

``orientations.csv`` (NF / voxel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comma-delimited, three columns of **NEML2 MRP**, **one row per Exodus block —
that is, one row per grain**, not per element.

The writer emits a sibling file alongside it, ``<name>_sameconv.csv``, holding
the same orientations in the input Euler convention rather than MRP. Feed
``<name>.csv`` to :class:`~graintrace.CPFESimulation`; ``_sameconv`` exists for
comparison against the source data.

``mrps_orientation.csv``
~~~~~~~~~~~~~~~~~~~~~~~~

The normalised orientation file the CPFE run actually consumes.
:class:`~graintrace.CPFESimulation` accepts **either** 9 columns (a flattened
row-major rotation matrix) **or** 3 columns (NEML2 MRP) and rewrites whichever
it got into this 3-column MRP file.

.. tip::

   NEML2 MRP is :math:`\tan(\theta/4)\,\hat n`, so every component satisfies
   :math:`|p_k| \le 1` inside the fundamental zone. Components much larger than
   one almost always mean Euler angles — in degrees or radians — were passed by
   mistake. This is the cheapest sanity check on an orientation file.

Mesh
----

``.spn`` (SCULPT input)
~~~~~~~~~~~~~~~~~~~~~~~

Flat, space-delimited integer material ids, one value per voxel, no header.
Stream order is **z fastest**:

.. math::
   :label: spn-order

   \mathrm{line}(i, j, k) = (i\, n_y + j)\, n_z + k

.. warning::

   graintrace **writes** z-fastest, per :eq:`spn-order`. What ``psculpt``
   expects is not verified here; its own documentation describes x-fastest. On
   a cubic grid (:math:`n_x = n_y = n_z`) a transposition is undetectable, so
   validate on a deliberately anisotropic grid with
   :math:`n_x \ne n_y \ne n_z` before trusting a new SCULPT version.

Two behaviours of the writer matter downstream:

* It **raises** ``ValueError`` on any voxel with a negative label rather than
  meshing it or quietly folding it into void, and the message names the remedy.
  See the label domain in :doc:`algorithms/meshing`.
* It **relabels** to a collision-proof contiguous set, so ``.spn`` material ids
  are *not* the grid's grain ids. Block-to-grain mapping therefore goes through
  the Exodus ``eb_prop1`` property, never by assuming the identity map.

Exodus ``mesh.e`` and ``sim_output.e``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary Exodus II. On the SCULPT path the element **block id is the grain id**.
Use ``element_order="FIRST"`` for a SCULPT or voxel hex mesh and ``"SECOND"``
for a GMSH tet ``.msh``.

.. note::

   CPFE fields are order-FIRST ``MONOMIAL`` variables, which MOOSE stores in the
   Exodus as **nodal projections**. Anything resampled from the Exodus is
   therefore smoothed relative to the element values — see
   :doc:`algorithms/cpfe` for the magnitude of that effect and when it matters.

NEPER ``.tess`` and ``.tesr``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NEPER's own formats; see the `NEPER documentation
<https://neper.info/doc/fileformat.html>`_ for the full specification.
graintrace depends on a subset: the cell list and per-cell face list (the
tessellation graph), the seed positions and weights, and the ``*ori`` block with
its descriptor string. The stitching path additionally requests per-cell
statistics and parses ``Zmin``/``Zmax`` and the volume centroid from them.

The reconstruction also merges NEPER's ``.stcell`` and ``.stseed`` statistics
into ``<base>_out.csv``, with ``_cell`` and ``_seed`` column suffixes
distinguishing the two sources.

Simulation output
-----------------

``out.csv`` (per-grain block CSV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One row per output time. A ``time`` column plus one column per
``<field>_<grainId>`` pair, matched by the regular expression ``.+_(\d+)$``.
The shipped ``mwe_data/out.csv`` is 50 rows by 1127 columns covering 45 grains,
with field prefixes ``cauchy_stress_{xx,xy,xz,yy,yz,zz}``, ``centroid_{x,y,z}``,
``ee_*``, ``ori_rodrigues_{x,y,z}``, ``strain_*`` and ``volume``.

The validation contract is: a ``time`` column must exist, and at least one
``<field>_<integer>`` column must exist.

``grid_out/out_element_centroid_NNNN.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields sampled onto a regular grid. **43 columns, sorted alphabetically by
MOOSE**, which puts ``x, y, z`` last and buries ``id`` in the middle:

.. code-block:: text

   Fe_11 … Fe_33,
   cauchy_stress_xx, cauchy_stress_xy, cauchy_stress_xz,
   cauchy_stress_yy, cauchy_stress_yz, cauchy_stress_zz,
   ee_xx, ee_xy, ee_xz, ee_yy, ee_yz, ee_zz,
   id,
   nye_tensor_11 … nye_tensor_33,
   ori_rodrigues_x, ori_rodrigues_y, ori_rodrigues_z,
   strain_xx, strain_xy, strain_xz, strain_yy, strain_yz, strain_zz,
   x, y, z

.. warning::

   **Component naming is mixed within the same file.** Stress and strain use the
   six Voigt components ``xx/xy/xz/yy/yz/zz``; ``Fe_`` and ``nye_tensor_`` use
   the nine full-tensor components ``11``–``33``, row-major. A metric that
   assumes one naming scheme for the whole file will silently select the wrong
   columns.

``ori_rodrigues_*`` holds MRP, not Rodrigues — see
:ref:`notation-ori-rodrigues`.

The ``0000``-indexed file is **header-only, with no data rows**. Code that
iterates the directory meets it first and must tolerate an empty frame.

``mesh_out/out_element_centroid_NNNN.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields sampled on the true CPFE mesh, one row per element, no interpolation and
no smoothing. **Variable width**: ``id, x, y, z, block_id`` plus whichever aux
variables the run emitted. The shipped
``mwe_data/cpfe_ff_fragmentation/mesh_out/`` files have eight columns, adding
``ori_rodrigues_{x,y,z}``.

``block_id`` is the element subdomain, which on the SCULPT path **is the grain
id** — that is what lets per-grain selection work without opening the mesh.

.. warning::

   **File indices are not step numbers and are not contiguous.** The shipped
   example runs ``0001, 0004, 0007, 0010, …`` because the indices follow the
   solver's output cadence. Field-step arguments elsewhere in the API are
   positions in ``sorted(res.field_files.keys())``, **not** the number in the
   filename. Index into that sorted list rather than constructing a filename.

Analysis output
---------------

REI rare-point CSV
~~~~~~~~~~~~~~~~~~

Header row; the coordinate names come from the instance's ``coord_cols``:

.. code-block:: text

   x,y,z,rare_cluster_id

This is the input contract for :class:`~graintrace.rei_comparison.REIComparison`.
Shipped example: ``mwe_data/rei_comparison/``.

REI rare-cluster stats CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~

One row per rare cluster, with ``mean``, ``var`` and ``std`` per feature, plus
``min`` and ``max`` where available. Column names are derived by suffixing the
feature name: feature ``f`` yields ``f_mean``, ``f_var``, ``f_std``.

.. important::

   That suffix convention is **load-bearing**. The indicator stage's
   ``SimilarityMetric`` must name its feature columns with the same suffixes the
   reduction produced::

      spec_reduced = SimilarityMetric(
          name=spec.name + "_mean",
          feature_cols=[f"{c}_mean" for c in spec.feature_cols],
          func=spec.func,
      )

   Getting this wrong is the most common REI configuration error, because the
   mismatch surfaces as a missing-column error several stages later.

REI ``.vtk`` outputs
~~~~~~~~~~~~~~~~~~~~

ASCII legacy VTK, written by hand so the write path needs no ``vtk`` or
``pyvista`` dependency. ``export_control="auto"`` chooses between
``DATASET POLYDATA`` (one vertex per point) and ``DATASET STRUCTURED_GRID``
(with ``DIMENSIONS nx ny nz``). Point data carries the block-id array and,
with ``also_write_final_label=True``, a merged ``final_label`` array.

The comparison writes ``overlap_cloud.vtk`` with three scalars:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Scalar
     - Meaning
   * - ``membership``
     - ``1`` = in region 1 only, ``2`` = in region 2 only, ``3`` = in both
   * - ``cluster_id_1``
     - the matching cluster id in region 1, or ``-1``
   * - ``cluster_id_2``
     - the matching cluster id in region 2, or ``-1``

``overlap_metrics.json`` and ``cluster_match.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``overlap_metrics.json`` holds the global agreement figures — IoU, Dice,
``containment_1``, ``containment_2`` — with the point counts and volumes they
were computed from. ``cluster_match.csv`` holds the one-to-one Hungarian pairing
by overlap volume, with unmatched clusters flagged ``-1`` and the split and merge
counts. The definitions are in :doc:`algorithms/rei-comparison`.

Shipped datasets
----------------

Small datasets ship with the repository under ``mwe_data/`` so the examples run
without external data.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Contents
   * - ``ff_calibration/``
     - 9 stress levels of FF calibration CSV plus ``strain-stress.csv``; drives calibration and FF reconstruction
   * - ``cpfe_ff/``
     - a 10-grain ``reconstruction.msh`` and its ``orientations.dat``
   * - ``out.csv`` and ``grid_out/``
     - a finished CPFE run's block CSV and 4 regular-grid field files, for post-processing
   * - ``cpfe_ff_fragmentation/``
     - a true-mesh ``mesh_out/`` with ``block_id`` over 10 grains and 9 steps; drives reorientation tracking and fragmentation
   * - ``cpfe_hex_fragmentation/mesh.e``
     - a real SCULPT hex mesh, for the annotated-Exodus writer
   * - ``synthetic_split_{ff,nf,ebsd}/``
     - synthetic 1→1, 1→2 and 1→3 grain splits with ``ground_truth.json``
   * - ``synthetic_load_exp/``
     - per-load-step FF grain tables with elastic strain, for grain tracking
   * - ``synthetic_vms.csv``
     - a stress point cloud (``id, x, y, z, sxx, syy, szz, sxy, sxz, syz``), the REI seed dataset

Each ``synthetic_split_*/ground_truth.json`` carries ``multiplicities``,
``n_splitting_grains``, ``split_step``, ``thetas_deg``,
``misorientation_tol_deg``, ``step_csvs`` and ``parent_children``.

See also
--------

* :doc:`notation` — the orientation conventions and units these schemas are expressed in.
* :doc:`configuration` — which option selects which of these files.
* :doc:`algorithms/meshing` — the label domain and renumbering behind the ``.spn`` writer.

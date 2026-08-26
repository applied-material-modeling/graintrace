MCP server
==========

graintrace ships a `Model Context Protocol <https://modelcontextprotocol.io>`_
(MCP) server that exposes each workflow segment as a tool, so the whole
HEDM → CPFE → rare-event workflow can be driven from a chat client. The server
contains no LLM: the client you connect (Claude Desktop, Claude Code, or any
MCP-capable client) supplies the model and your API key.

.. code-block:: text

   your chat client (brings the LLM + key)
           |  MCP
           v
     graintrace-mcp  -->  graintrace public API  -->  NEPER / MOOSE-PUMA / CUBIT / NEML2

What it exposes
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Tool
     - Segment
     - External tools it needs
   * - ``generate_synthetic_hedm``
     - synthetic FF+NF test data
     - NEPER
   * - ``stitch_scans``
     - merge FF z-scan layers
     - NEPER if ``refine_extents``
   * - ``compare_stitching``
     - validate a stitch vs. truth
     - none
   * - ``ff_reconstruct``
     - FF Voronoi reconstruction
     - NEPER (GMSH if meshing)
   * - ``nf_reconstruct``
     - NF voxel reconstruction + mesh
     - CUBIT/SCULPT (mesh only)
   * - ``voxel_mesh``
     - EBSD/gridded segmentation + mesh
     - CUBIT/SCULPT (mesh only)
   * - ``calibrate_material``
     - Taylor-model parameter fit
     - NEML2 v3 + pyzag
   * - ``run_cpfe``
     - MOOSE/PUMA CPFE run
     - ``puma-opt`` + NEML2 AOTI
   * - ``postprocess``
     - distributions / stress-strain / pole figure
     - NEML2 for pole figure
   * - ``identify_rare_events``
     - REI graph clustering → VTK
     - none
   * - ``track_grains``
     - grain matching across load steps
     - NEPER + torch-geometric

Introspection and job tools (``dependency_status``,
``list_recommended_recipes``, ``get_recommended_parameters``, ``job_status``,
``list_jobs``, ``job_log``, ``list_outputs``) need no external tools.

Two safety behaviors
--------------------

1. **Nothing runs without confirmation.** Every computing tool takes a
   ``confirm`` flag. With ``confirm=false`` (the default) it returns a preview
   (resolved parameters, which external tools it needs, whether they are present,
   and what it will write) and runs nothing. Only ``confirm=true`` executes.
2. **Missing external tools fail gracefully.** If a required tool is not built,
   the tool reports exactly what is missing and how to build it, with no
   traceback. Check the whole stack with ``dependency_status``.

Heavy runs execute as background jobs: the tool returns a ``job_id``; poll
``job_status(job_id)`` for progress.

Recommendation recipes
----------------------

``graintrace/mcp/recipes/*.md`` hold vetted parameter presets and prose guidance
per setup. The model reads them (via ``get_recommended_parameters`` or the
``recipe://<name>`` resources) before proposing parameters. They are plain
Markdown; edit or extend them to encode your lab's conventions without touching
code.

Install
-------

.. code-block:: bash

   pip install "graintrace[mcp]"      # or: pip install -e ".[mcp]" from a checkout

The compiled stack is not on PyPI; build it separately (see :doc:`install`). The
MCP server runs without it; tools that need a missing piece report so. Run it
standalone to sanity-check:

.. code-block:: bash

   graintrace-mcp            # stdio transport (Claude Desktop / Code use this)
   graintrace-mcp --http     # streamable-http on 127.0.0.1:8000

Outputs land under ``$GRAINTRACE_MCP_WORKDIR`` (default ``./graintrace_mcp_out``).

Connect a client
----------------

For Claude Desktop / Claude Code, add to your MCP config (Claude Desktop:
``claude_desktop_config.json``; Claude Code: ``claude mcp add``):

.. code-block:: json

   {
     "mcpServers": {
       "graintrace": {
         "command": "graintrace-mcp",
         "env": {
           "GRAINTRACE_MCP_WORKDIR": "/abs/path/to/where/outputs/go",
           "PATH": "/home/you/.local/bin:/usr/bin:/bin"
         }
       }
     }
   }

Use the conda env's interpreter if ``graintrace-mcp`` is not on the default PATH,
e.g. ``"command": "/home/you/miniconda3/envs/<puma-env>/bin/graintrace-mcp"``.
Make sure ``PATH`` in ``env`` includes NEPER and any built binaries, since the
client launches the server as a subprocess.

Any other MCP-capable client can connect over the stdio or streamable-http
transport; point it at the ``graintrace-mcp`` command (stdio) or
``http://127.0.0.1:8000`` (``--http``).

Notes
-----

- The job registry is in-memory and lives with the server process, fine for a
  single-user session driving one machine; it is not a distributed queue.
- ``run_cpfe`` distinguishes ``neml2`` (importable) from ``neml2-aoti`` (the
  compiled runtime CPFE needs to load ``.pt2`` models); calibration only needs
  the former.

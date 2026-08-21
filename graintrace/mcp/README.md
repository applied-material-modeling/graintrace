# graintrace MCP server

Drive the whole graintrace **HEDM → CPFE → rare-event** workflow from a chat UI.

This package ships a [Model Context Protocol](https://modelcontextprotocol.io)
(MCP) server that exposes each graintrace workflow segment as a tool. The server
itself contains **no LLM**: the chat client you connect (Claude Desktop or
Claude Code, or any MCP-capable client) supplies the reasoning model and *your*
API key. So the
distribution story is simply: **install graintrace, point your client at it.**

```
your chat client (brings the LLM + key)
        │  MCP
        ▼
  graintrace-mcp  ──►  graintrace public API  ──►  NEPER / MOOSE-PUMA / CUBIT / NEML2
```

## What it exposes

| Tool | Segment | External tools it needs |
|---|---|---|
| `generate_synthetic_hedm` | synthetic FF+NF test data | NEPER |
| `stitch_scans` | merge FF z-scan layers | (NEPER if `refine_extents`) |
| `compare_stitching` | validate a stitch vs. truth | none |
| `ff_reconstruct` | FF Voronoi reconstruction | NEPER (GMSH if meshing) |
| `nf_reconstruct` | NF voxel reconstruction + mesh | CUBIT/SCULPT (mesh only) |
| `voxel_mesh` | EBSD/gridded segmentation + mesh | CUBIT/SCULPT (mesh only) |
| `calibrate_material` | Taylor-model parameter fit | NEML2 v3 + pyzag |
| `run_cpfe` | MOOSE/PUMA CPFE run | `puma-opt` + NEML2 AOTI |
| `postprocess` | distributions / stress-strain / pole figure | (NEML2 for pole figure) |
| `identify_rare_events` | REI graph clustering → VTK | none |
| `track_grains` | grain matching across load steps | NEPER + torch-geometric |
| `dependency_status`, `list_recommended_recipes`, `get_recommended_parameters`, `job_status`, `list_jobs`, `job_log`, `list_outputs` | introspection / recipes / jobs | none |

Two built-in safety behaviors:

1. **Nothing runs without confirmation.** Every computing tool takes a `confirm`
   flag. Called with `confirm=false` (the default) it returns a *preview* (the
   resolved parameters, which external tools it needs, whether they're present,
   and what it will write) and runs nothing. Only `confirm=true` executes.
2. **Missing external tools fail gracefully.** If NEPER / MOOSE-PUMA / CUBIT /
   NEML2 isn't built, the tool says exactly what's missing and how to build it,
   with no traceback. Check the whole stack any time with `dependency_status`.

Heavy runs (CPFE, reconstruction, meshing, calibration) execute as **background
jobs**: the tool returns a `job_id`; poll `job_status(job_id)` for progress.

## Recommendation recipes

`graintrace/mcp/recipes/*.md` hold vetted parameter presets + prose guidance per
setup. The model reads them (via `get_recommended_parameters` or the
`recipe://<name>` resources) before proposing parameters. **They're plain
Markdown; edit/extend them to encode your lab's conventions without touching
code.**

## Install

```bash
pip install "graintrace[mcp]"      # or: pip install -e ".[mcp]" from a checkout
```

The compiled stack (NEPER, MOOSE/PUMA, CUBIT/SCULPT, NEML2, pyzag) is **not** on
PyPI; build it separately (see the repo README). The MCP server
runs without it; tools that need a missing piece just report so.

Run it standalone to sanity-check:

```bash
graintrace-mcp            # stdio transport (what Claude Desktop / Code use)
graintrace-mcp --http     # streamable-http on 127.0.0.1:8000
```

Outputs land under `$GRAINTRACE_MCP_WORKDIR` (default `./graintrace_mcp_out`).

## Connect a client

### Claude Desktop / Claude Code (native MCP, simplest)

Claude *is* the model, so there's nothing else to configure. Add to your MCP
config (Claude Desktop: `claude_desktop_config.json`; Claude Code:
`claude mcp add`):

```json
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
```

Use the conda env's interpreter if `graintrace-mcp` isn't on the default PATH,
e.g. `"command": "/home/you/miniconda3/envs/<your-puma-env>/bin/graintrace-mcp"`.
Make sure `PATH` in `env` includes NEPER and any built binaries, since the client
launches the server as a subprocess.

Then just chat: *"reconstruct mwe_data/ff_calibration/0.csv"* → Claude reads the
recipe, previews parameters, you approve, it runs.

### Other MCP clients

Any MCP-capable client can connect over the stdio or streamable-http transport
above; point it at the `graintrace-mcp` command (stdio) or
`http://127.0.0.1:8000` (`--http`). The client must supply a reasoning model that
supports tool-calling.

## Notes

- The job registry is in-memory and lives with the server process, fine for a
  single-user session driving one machine; it is not a distributed queue.
- `run_cpfe` distinguishes `neml2` (importable) from `neml2-aoti` (the compiled
  runtime CPFE needs to load `.pt2` models); calibration only needs the former.

---
name: grain-tracking
description: >
  Track/match grains across two load steps by building a grain graph from each FF
  reconstruction and matching via message passing (VoronoiMeshBuilder.build_graph +
  GraphGrainMatcher). Use when the user has two FF grain CSVs (different loads/times)
  and wants a grain correspondence.
---

# Grain tracking across load steps

Uses `VoronoiMeshBuilder.build_graph` + `grain_graph_matching.GraphGrainMatcher`. Env:
`conda activate graintrace_env`. External tool: **NEPER** (tessellation for the graph).

## Inputs
Two FF grain CSVs (same sample, two load steps) with `X,Y,Z,Eul0/1/2,eKen11..33`.
Self-contained: `mwe_data/synthetic_load_exp/expsyn_146time.csv` and `expsyn_160time.csv`.

## Recipe (mirrors demonstrate_graintracking.py)
```python
from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
from graintrace.grain_graph_matching import GraphGrainMatcher

eKen = [f"eKen{i}{j}" for i in (1,2,3) for j in (1,2,3)]

def make(csv, out, bbox):
    b = VoronoiMeshBuilder(
        input_csv=csv, output_dir=out, bounding_box=bbox, dim=3, weighted=False,
        auto_fix_bbox=False, auto_rotate=False,
        angle_identifier=["Eul0","Eul1","Eul2"], orientation_descriptor="euler-bunge",
        orientation_active_convention=True, elastic_strain_identifier=eKen,
        strain_unit="microstrain",
    )
    return b.build_graph(CVT_iter=10)

ga = make("mwe_data/synthetic_load_exp/expsyn_146time.csv", "out/track_a",
          [-200,200,-173.205,173.205,0,650])
gb = make("mwe_data/synthetic_load_exp/expsyn_160time.csv", "out/track_b",
          [-200,200,-173.205,173.205,0,680])

GraphGrainMatcher(graph_a=ga, graph_b=gb, output_dir="out/grain_tracking").match_grains(
    message_passing_iter=3,
    neighbor_selection_param={"lambda": 0.00125, "iterations": 100, "tolerance": 1e-6},
)
```

## Key parameters
- Per-step `bounding_box` (each load step can have a different z-extent).
- `build_graph(CVT_iter=...)` — CVT iterations for the tessellation the graph is built on.
- `match_grains(message_passing_iter, neighbor_selection_param)` — message-passing depth and
  the neighbor-selection optimization (`lambda`, `iterations`, `tolerance`).

## Outputs
`output_dir` — grain correspondence between the two graphs (matched IDs / mapping + any
diagnostics).

## Gotchas
- Orientations are read as Euler-bunge; keep `angle_identifier`/`orientation_descriptor`
  consistent with the data. NEPER must be on PATH.
- Uses `build_graph` (not a full mesh) — no GMSH/CUBIT needed.

## See also
`examples/demonstrate_graintracking.py`; CLAUDE.md §3 Step 3 (graph from tessellation).

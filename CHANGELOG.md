# Changelog

## 0.3.0

### Breaking changes
- **Orientation convention.** graintrace's rotation matrix is now the *active* `R`
  (crystal → sample), the same matrix as MIDAS `O11..O33` and the neml2 `orientation`.
  Euler angles are unchanged (standard Bunge). Crystal symmetry acts on the right
  (`R S`), sample-frame rotations on the left (`Q R`). graintrace ≤ 0.2 used the passive
  `g = Rᵀ`, which handed neml2 the inverse orientation and made Euler-input IPF plots
  project the wrong direction.
  - MRP / `O` files written by graintrace ≤ 0.2 hold the inverse rotation; convert MRP
    with `orientation_helper.legacy_mrp_to_mrp` (`-p`).
  - CPFE runs driven by those files must be **rerun**; negating their outputs is not
    equivalent.
  - Calibration (`TaylorModel` / `MaterialCalibration`) reads the texture from the Euler
    columns (`angle_identifier`, `angle_convention`, `angle_type="auto"`); `O11..O33`
    is only a fallback. `experiment_rotation_helper` writes rotated Euler and drops `O`.
- **NEPER 7.x is required.** graintrace checks the version whenever it resolves NEPER
  (`neper_env.require_neper_version`). NEPER is driven with explicit descriptors
  (`euler-bunge:passive`, `rotmat:active`); the default `morphoalgo` is `"subplex"`
  (NEPER 7 removed `"praxis"`); `auto_install=True` builds NEPER 7.0.0.
- **Laguerre weights now take effect.** `VoronoiMeshBuilder(weighted=True)` and
  `RegionBaseStitching(tess_weighted=True)` (the default when `refine_extents=True`)
  write the NEPER 7 power weight `GrainRadius**2`. Before, the weights were normalised to
  sum to 1 and had no visible effect, so weighted results change when rerun.
- **SCULPT meshes are quality-checked.** `VoxelMeshBuilder.mesh` / `NearFieldMeshBuilder.mesh`
  (SCULPT path) raise on inverted hexes (scaled Jacobian ≤ 0) and write
  `<mesh>_quality.json`; pass `quality_check="warn"` or `"off"` to keep the old behaviour.

### New
- `slip_systems=` on `CPFESimulation` and `TaylorModel` (and the MCP `run_cpfe` /
  `calibrate_material` tools): presets `"fcc"` (default), `"bcc"`, `"bcc_112"`,
  `"bcc_110_112"`, `"bcc_110_112_123"`, or explicit cubic `(direction, plane)` Miller
  families (`graintrace.slip_systems`). Previously FCC slip was hard-coded.
- `UniaxialTaylorModel.forward(..., return_orientation=True)` /
  `TaylorModel.simulate(..., return_orientation=True)`: per-grain neml2-MRP orientation
  histories.
- `IPFProcessor.get_ipf_track_points`: variant-consistent IPF trajectories (tracks no
  longer jump across the fundamental-sector edges).
- `nf.mesh.hex_scaled_jacobian` / `nf.mesh.check_mesh_quality`.
- The MCP server runs on the `mcp` SDK 1.x (`FastMCP`) and 2.x (`MCPServer`);
  `graintrace.mcp` loads the server lazily, so `tool_paths` / `deps` import without the
  `mcp` extra.

### Fixes
- Simulated HEDM z-scans (`CrystalGenerator.hedm_zscan`) were empty under NEPER 7 (the
  `cut(hspace)` sense flipped and cut-away cells are kept as zero-volume entries), which
  produced all-zero radii and NaN stitching merges.
- `orientation_helper.average_orientation` falls back to uniform weights when the weights
  sum to zero, and rejects negative or non-finite weights.
- `scan_tessellation.compute_cell_geometry` raises a clear error on an empty cell.
- Regenerated `mwe_data` (`out.csv` + `grid_out/`, `synthetic_load_exp/` with true Bunge
  Euler, `cpfe_ff_fragmentation/`) and added `mwe_data/ff_stitched_grains/` so the
  experiment-post-processing tests run.
- Examples read tool paths from `tools.json`; `demonstrate_synthetic_cpfe.py` uses a
  vetted SCULPT config and fewer CPFE ranks per GPU; the grain-tracking recipe uses the
  data's units (`unit="rad"`, `strain_unit="strain"`).

# CLAUDE.md: graintrace Working Reference

This file is for Claude's use when helping write or debug experiment scripts. It is not a user README.

---

## 1. Package Overview

`graintrace` is a Python toolkit that links experimental grain-scale characterization data (FF HEDM, NF HEDM, EBSD) to crystal plasticity finite element (CPFE) simulations. It reconstructs 3D microstructure meshes from raw experimental data, sets up and runs MOOSE/PUMA CPFE simulations with NEML2 material models, and post-processes results for rare-event identification (REI).

**Key external dependencies:**

| Dependency | Role |
|---|---|
| **NEPER** | Voronoi/CVT tessellation from FF centroids; produces `.tess` (and an optional `.msh` tet mesh — a fallback; the default FF mesh is SCULPT hex) |
| **CUBIT/SCULPT** (`psculpt`) | Hexahedral mesh generation from voxel `.spn` files (NF/EBSD meshes) |
| **MOOSE/PUMA** (`puma-opt`) | Runs CPFE simulations; outputs CSV field data at grid points |
| **neml2** | Crystal plasticity material model (Taylor model, CPFE model); Python bindings used for orientation math and calibration |

---

## 2. Data Types

### FF HEDM data (per scan layer, CSV format)
Raw files are whitespace-delimited with an 8-line header (skip lines 0-7). Column names appear on line 8, possibly prefixed with `%`. Key columns:

```
X, Y, Z           : grain centroid position (micrometers)
GrainRadius       : equivalent sphere radius (micrometers)
Eul0, Eul1, Eul2 : Bunge Euler angles (degrees or radians; detect via "auto")
Confidence        : fit quality, filter to >= 0.7 or 0.9
eFab11..eFab33    : fabric/lattice strain tensor (row-major, 9 components)
eKen11..eKen33    : Kenesei elastic strain tensor (row-major, 9 components), typically in microstrain
ScanID            : assigned during stitching
```

To read raw FF files:
```python
with open(fpath) as fh:
    lines = fh.readlines()
data_lines = lines[8:]
if data_lines[0].lstrip().startswith("%"):
    line = data_lines[0]
    idx = line.find("%")
    data_lines[0] = line[:idx] + line[idx+1:]
df = pd.read_csv(StringIO("".join(data_lines)), sep=r"\s+")
```

Detect units automatically: if any Euler value > 2π, units are degrees, else radians.

Multiple scan layers are Z-shifted before stitching:
```python
df["Z"] = df["Z"] + scan_idx * Zheight_per_file * (1 - overlap_fraction)
```

### NF HEDM data (per-layer `.mic` files in a folder)
Expected `.mic` format (tab-delimited, with `%` header lines):
```
%OrientationRowNr  OrientationID  RunTime  X  Y  TriEdgeSize  UpDown  Eul1  Eul2  Eul3  Confidence  PhaseNr
```
`NearFieldMeshBuilder` reads a folder of `.mic` files. The `exp_file_token` parameter is the filename prefix token used to find files. If the source data is `.ang` files (8-column, no header), convert them to `.mic` format first (see `run_experiment_afrl.py` for the conversion pattern).

For the alternate path using `NFGridConversion` (pre-gridded NF data):
```python
from graintrace.nf_grid_conversion import NFGridConversion
nf_converter = NFGridConversion(
    input_folder="...",
    save_dir="...",
    exp_file_token="layer_prefix",
    prefix="reconstructed",
)
nf_grid_csv = nf_converter.convert(dz=nf_dz, nx=nx, ny=ny)
```

### EBSD data (CSV, merged from per-layer `.ang` files)
After converting and merging from `.ang` files, the expected format is a flat CSV:
```
x, y, z, Eul0, Eul1, Eul2
```
`z` is `file_index * zstep_ebsd`. Columns `Eul0/1/2` hold Bunge Euler angles in degrees or radians. The file is then fed to `VoxelMeshBuilder`.

---

## 3. FF-Only Workflow

Full pipeline: stitch → reconstruct → (optional) voxel mesh → CPFE simulation.

### Step 1: Stitch multiple scan layers
```python
from graintrace.hedm_stitching_techniques.region_base_stitching import RegionBaseStitching

stitcher = RegionBaseStitching(
    scan_files=file_list,           # list of pre-processed CSVs (Z already shifted)
    output_csv="out/stitched_output.csv",
    position_tolerance=50,          # micrometers
    orientation_tolerance=5.0,      # degrees (convert to radians if ori_units="radians")
    radius_tolerance=-1,            # -1 disables the radius gate; weights["rad"]=0 drops it from the cost
    weights={"pos": 0.1, "ori": 1.0, "rad": 0},
    min_neighbors=5,
    orientation_convention="bunge",
    orientation_units="degrees",    # or "radians"
    symmetry="432",
    output_column=important_columns,
)
stitched = stitcher.run(zlo=bounding_box[4], zhi=bounding_box[5],
                        overlap_fraction=overlap_percentage/100.0)
```

**Region classification** (which duplicate to trust/merge) uses each grain's z-extent.
By default this is the equivalent-sphere approximation `z ± GrainRadius`. For elongated /
anisotropic grains that mis-estimates the extent; enable the **opt-in NEPER tessellation**
refinement (`refine_extents=True`) to use the true per-cell `[zmin, zmax]` instead. It
re-tessellates the accumulator + next scan at each fold step (needs NEPER; slower). Extra kwargs
(all in `RegionBaseStitching.__init__`, overlap path only):
```python
stitcher = RegionBaseStitching(
    ...,
    refine_extents=True,      # default False -> spherical z ± GrainRadius (unchanged)
    tess_weighted=True,       # Laguerre weight = effective grain volume (4/3)πr³; False = plain Voronoi
    update_centroid=False,    # True -> replace X,Y,Z with the cell volume-centroid (degrades equiaxed matching; experimental)
    tess_dir=None,            # scratch dir for NEPER I/O (temp dir per call if None)
    neper_env=None,           # defaults to scan_tessellation.default_neper_env, which resolves a user-installed NEPER (NEPER env var -> tools.json -> PATH)
    xy_bounding_box=None,     # [xlo,xhi,ylo,yhi]; inferred from data (+2% pad) if None
)
```
Helper: `graintrace/hedm_stitching_techniques/scan_tessellation.py::compute_cell_geometry`
(runs `neper -T ... -morphooptistop iter=0` at the measured centroids (no CVT relaxation) and
parses per-cell `Zmin/Zmax` + volume centroid). Note: on ~equiaxed grains the space-filling
Voronoi extent slightly *overestimates* vs a sphere, so the benefit is specific to elongated
grains; validate with `ScanStitchingComparison` before trusting it on new data.

**Reality check (measured):** the extent is fundamentally under-determined by FF observables.
FF measures the grain volume-centroid + equivalent size, not shape or a tessellation *seed*;
reconstructing extents from centroids is ~⅓-radius noisy (verified: exact only from the true
seeds; centroid/centroidsize NEPER optimization does not recover it). And even for strongly
elongated grains (`aspratio(1,1,3)`), per-scan tessellation clips the cell at the scan FOV so it
can't recover the full z-extent either; `examples/demonstrate_hedm_anisotropic.py` is a
benchmark that shows tessellation does **not** reliably beat the sphere. For true grain
morphology use NF-HEDM, not an FF tessellation. Generate anisotropic test microstructures with
`CrystalGenerator` via the `raw` morpho type: `morpho_str="diameq:lognormal(130,5),aspratio(1,1,3)"`.

### Step 2: Build Voronoi reconstruction (NEPER)
```python
from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder

builder_ff = VoronoiMeshBuilder(
    input_csv="out/stitched_output.csv",
    output_dir="out/FF",
    bounding_box=[-560, 580, -360, 410, -120, 0],  # [xlo,xhi,ylo,yhi,zlo,zhi] micrometers
    dim=3,
    weighted=False,
    auto_fix_bbox=True,
    bbox_fix_mode="remove_points",   # "remove_points" for production; "extend_bounding_box" for debug
    bbox_tolerance=2.5,
    auto_rotate=False,
    rotate_angles=[0, 0, 8.1],       # sample tilt correction, must match ori_units
    rotate_convention="xyz",
    angle_identifier=["Eul0", "Eul1", "Eul2"],
    orientation_descriptor="euler-bunge",
    orientation_active_convention=True,
    elastic_strain_identifier=["eKen11","eKen12","eKen13",
                               "eKen21","eKen22","eKen23",
                               "eKen31","eKen32","eKen33"],
    strain_unit="microstrain",
    unit="deg",                      # "deg" or "rad", must match actual data units
)

builder_ff.build_voronoi(
    generate_mesh=False,             # keep False (default). True = NEPER/GMSH tet .msh — a FALLBACK only, see note below
    option="centroid",               # "voronoi" | "centroid" | "centroidsize"
    CVT_iter=1000,                   # CVT optimization iterations
    morphoalgo="subplex",            # "subplex" | "lloyd" | "praxis"
    mesh_quality_min=0.7,
    relative_el_size=2.0,            # mesh element size relative to grain size
    tesr_size=[100, 100, 100],       # voxel grid resolution [nx, ny, nz]
)
```

Key outputs in `output_dir`:
- `reconstruction_reformatted.csv`: per-voxel grain IDs and orientations (input to the default hex mesher)
- `reconstruction_cpfe_ee.csv`: per-grain elastic strain for CPFE initial conditions
- `orientations.dat`: Euler angles for each grain (always degrees after FF build)
- `reconstruction.msh`: NEPER/GMSH tet mesh — **fallback only**, written only if `generate_mesh=True`

**Default FF meshing = SCULPT hex, not GMSH.** The recommended CPFE mesh is a conformal hex
built from `reconstruction_reformatted.csv` via `VoxelMeshBuilder` (`mesher="sculpt"` default,
§5), or `mesher="voxel"` for a no-CUBIT one-cube-per-voxel hex dump. Hex elements behave better
for crystal plasticity and the rest of the pipeline is built around them. The NEPER/GMSH tet
`.msh` (`generate_mesh=True`) is a **fallback** for when CUBIT/SCULPT is unavailable.

### Step 3 (optional): Build graph from tessellation
```python
from graintrace.tess_to_gnn import NeperTessToGraphNN
parser = NeperTessToGraphNN(tess_path="out/FF/voronoi.tess", device="cpu")
graph = parser.build_cell_graph()
```

### Step 4: Run CPFE simulation
All orientations communicate as **neml2 v3 MRP** (`tan(θ/4)·axis`), the canonical
on-disk/interchange format. The FF `orientations.dat` is Euler-Bunge (degrees); convert it
via `orientation_helper` (which delegates to neml2; do NOT hand-roll the math):
```python
import torch, numpy as np
from graintrace import orientation_helper as oh
euler = np.loadtxt("out/FF/orientations.dat")            # Euler-Bunge, degrees
mrp = oh.euler_to_mrp(torch.tensor(euler, dtype=torch.float64), "bunge", "degrees")
np.savetxt("out/FF/orientations_MRP.dat", mrp.numpy(), fmt="%.12g")
```
graintrace "mrp" is Gibbs/Rodrigues (`tan θ/2`) ≠ neml2 v3 MRP; `euler_to_mrp` returns true
neml2 MRP, which is what the CPFE model's `orientation` IC and `ori_rodrigues` output expect.

```python
from graintrace.run_cpfe_simulation import CPFESimulation

sim = CPFESimulation(
    mesh_file="out/FF/mesh.e",       # default: SCULPT hex from VoxelMeshBuilder (§5). Fallback: "out/FF/reconstruction.msh" (GMSH tet)
    save_simulation_folder="out/simulation",
    element_order="FIRST",           # FIRST for SCULPT/voxel hex; SECOND for a GMSH tet .msh
    eeres_file="out/FF/reconstruction_cpfe_ee.csv",
    ori_file="out/FF/orientations_MRP.dat",
    dim=3,
    moose_run_file="external/puma/puma-opt",   # your built PUMA binary (see README/submodules)
    use_ff_initial_field=True,       # True when mesh and ee file are co-registered FF
)
# eeres_file=None -> a 12-col zero zero_initial_strain.ee is written (no residual strain).

sim.set_parameters("material",
    slip_constant_strength=100.0,
    voce_hardening_initial_slope=1650.0,
    voce_hardening_saturation=220.0,
    power_slip_n=25,
    power_slip_g0=1e-4,
    elastic_E=109000.0,
    elastic_nu=0.307,
    elastic_G=41700.0,
    burger_scale=2.54,
)

sim.set_parameters("simulation_parameters",
    dt=0.2, total_time=5.0,
    initialize_time=1.0,             # load ramps from initialize_time -> total_time
    device="cuda:0",                 # "cpu", "cuda:N", or space-sep list "cuda:0 cuda:1"
    device_batch=20000,              # per-device NEML2 chunk (quad pts/call); 0 = whole batch
    sync_times="2.0 3.0 4.0 5.0",    # space-separated string of MOOSE grid-output times
    grid_transfer="final",           # regular-grid MultiApp transfer: "final" (default) |
                                     # "per_step" (REI at every step) | "off" (resample offline)
    exodus_output="sync",            # native-mesh Exodus writes: "sync" (default, only at
                                     # sync_times) | "per_step"
    mesh_csv="sync",                 # CPFE-mesh element CSV -> mesh_out/: "sync" (default) |
                                     # "per_step" | "off". Crisp full-mesh REI (no smoothing)
    # AOTI (v3): material params are BAKED into the model .i and neml2-compile'd on run().
    # recompile=True (default) rebuilds the .pt2 when params change; compile_devices/
    # neml2_load_files/extra_ld_library_paths auto-derive from moose_run_file's repo layout.
)
# grid_transfer/exodus_output default to the CHEAP settings (transfer only at the last step,
# Exodus only at sync_times) — the per-step grid MultiApp transfer dominates wall time.
# Three ways to get REI field data:
#   1. mesh_csv (default "sync") -> mesh_out/out_element_centroid_*.csv on the TRUE mesh:
#      crisp (samples aux vars directly, no transfer/no smoothing), full-fidelity; REI runs
#      on it via the kNN path (one row per element). Best default for REI.
#   2. grid_transfer="per_step" -> grid_out/ crisp regular grid every step (fast grid path,
#      but pays the per-step transfer cost).
#   3. GridResampler (§7) -> regenerate grid_out/ offline from the Exodus (cheap but smoothed).
# v3 has no runtime [NEML2] cli_args or [Schedulers]: no scheduler_name/hybrid_batch_sizes.
# Multi-GPU = MPI ranks over a device list (ncore == mpiexec -n); fewer ranks (~#GPUs) give
# bigger per-rank NEML2 batches + better GPU utilization for neml2-dominated CPFE.

displace_amount = total_strain * (bounding_box[5] - bounding_box[4])
sim.set_parameters("boundary",
    bounding_box=bounding_box,
    bc={
        "x": {"negative": "stress_free", "positive": "stress_free"},
        "y": {"negative": "stress_free", "positive": "stress_free"},
        "z": {"negative": 0, "positive": displace_amount},
    },
)

sim.set_parameters("grid_properties",
    number_of_elements=[100, 100, 100],
    bounding_box=grid_bb,            # bounding_box shrunk by 0.0001 on each face
)

sim.run(ncore=8)
```

Converting a `sync_strain` list to MOOSE sync times (load ramps over `initialize_time`→`total_time`):
```python
sync_times = np.asarray(sync_strain) / total_strain * (total_time - initialize_time) + initialize_time
string_sync_times = " ".join(map(str, sync_times))
```

---

## 4. NF-Only Workflow

### Using NearFieldMeshBuilder (`.mic` files)
```python
from graintrace.construct_nf_mesh import NearFieldMeshBuilder

builder_nf = NearFieldMeshBuilder(
    input_folder="experiment_data/NF",   # folder containing .mic files
    save_dir="out/NF",
    exp_file_token="layer",              # filename token to match files in folder
    angle_convention="bunge",
    angle_type="radians",                # "radians" or "degrees"
    symmetry="432",
    prefix="reconstructed",
    write_intermediate=True,
    write_vtk=True,
)

merged_grid_path = builder_nf.reconstruct(
    dz=5.0,          # layer thickness in micrometers
    nx=200,          # in-plane grid resolution x
    ny=300,          # in-plane grid resolution y
    segmentation={   # flat dict for NearFieldMeshBuilder (no method/params nesting)
        "misorientation_tol": 5.0/180*np.pi,   # radians
        "connectivity": 6,
        "batch_norm": 200_000,
        "grain_threshold": 1000,
        "stop_count": 500,
        "grain_threshold_final": 10000,
    },
)

mesh_path = builder_nf.mesh(
    sculpt_config=sculpt_config,
    sculpt_options=sculpt_options,
    merged_grid=merged_grid_path,
)
# Orientations saved to: builder_nf.mapped_orientations_path + ".csv"
# Use output_nf + "/orientations.csv" for CPFESimulation ori_file
```

Key outputs in `save_dir`:
- `merged_segmented_fixed_grid.npy`: segmented voxel grid (restart checkpoint)
- `mesh.e`: Exodus mesh file for CPFE
- `orientations.csv`: per-element MRP orientations

---

## 5. EBSD Workflow

Use `VoxelMeshBuilder` with a merged CSV (columns: `x, y, z, Eul0, Eul1, Eul2`):

```python
from graintrace.construct_voxel_mesh import VoxelMeshBuilder

ebsd_voxel_builder = VoxelMeshBuilder(
    file_path="out/ebsd/EBSD_merged_downsampled_5x.csv",
    save_dir="out/ebsd/mesh",
    euler_cols=["Eul0", "Eul1", "Eul2"],
    angle_convention="bunge",
    angle_type="radians",            # or "degrees"
    symmetry="432",
)

merged_grid_path = ebsd_voxel_builder.reconstruct(
    apply_smoothing=True,
    segmentation={
        "method": "graph",           # "graph" or "flood"
        "params": {
            "misorientation_tol": 5.0,   # degrees if angle_type="degrees", radians otherwise
            "connectivity": 26,
            "grain_threshold_final": 100,
        },
        "graph_params": {
            "segmenter": "leiden",
            "graph_mode": "grid",
            "manhattan_radius": 2,
            "grid_tol": 1e-6,
            "n_jobs": 10,
            "weight_chunk_size": 1_000_000,
            "reduce_edges_topweights_k": 8,
            "nodes_chunk": 500_000,
            "seed": 42,
            "networkit_kwargs": {"gamma": 0.001},   # lower gamma = fewer clusters
            "weight_cfg": {
                "mode": "rbf",
                "sigma": None,
                "sigma_auto": {"sample_size": 20_000, "random_state": 42, "quantile": 0.5},
                "power": 2.0,
            },
            "plot": True,
        },
    },
)

mesh_path = ebsd_voxel_builder.mesh(
    sculpt_config=sculpt_config,
    sculpt_options=sculpt_options,
    merged_grid=merged_grid_path,
)
```

Note: `VoxelMeshBuilder` also accepts an optional `cell_id_col` parameter if the input CSV has a pre-existing grain ID column (used when processing FF `reconstruction_reformatted.csv`).

---

## 6. Combined NF+FF Workflow

The typical use case: NF provides the high-resolution mesh geometry; FF provides initial elastic strain field.

```python
# 1. Build NF mesh (as in Section 4)
builder_nf = NearFieldMeshBuilder(...)
merged_grid_path = builder_nf.reconstruct(...)
mesh_path = builder_nf.mesh(...)

# 2. Build FF reconstruction for initial strain (build_voronoi only, no mesh generation)
builder_ff = VoronoiMeshBuilder(...)
builder_ff.build_voronoi(generate_mesh=False, ...)
# produces: out/FF/reconstruction_cpfe_ee.csv

# 3. Spatially shift FF ee data to align with NF coordinate frame
ff_translation = (dx, dy, dz)   # determined by experiment geometry
ee_data = pd.read_csv("out/FF/reconstruction_cpfe_ee.csv", header=None, index_col=False)
ee_data.iloc[:, 0] += ff_translation[0]
ee_data.iloc[:, 1] += ff_translation[1]
ee_data.iloc[:, 2] += ff_translation[2]
ee_data.to_csv("out/FF/reconstruction_cpfe_ee_shifted.csv", index=False, header=False)

# 4. Run CPFE using NF mesh, NF orientations, FF initial strain
sim = CPFESimulation(
    mesh_file="out/NF/mesh.e",
    save_simulation_folder="out/simulation",
    element_order="FIRST",           # NF mesh typically uses FIRST order
    eeres_file="out/FF/reconstruction_cpfe_ee_shifted.csv",
    ori_file="out/NF/orientations.csv",
    dim=3,
    moose_run_file="/path/to/puma-opt",
    use_ff_initial_field=False,      # False when eeres_file is from a different mesh
)
# ... set_parameters and run as normal
```

Derive NF bounding box from the saved voxel grid:
```python
data = np.load("out/NF/merged_segmented_fixed_grid.npy")
coords = data[..., 4:7].reshape(-1, 3)
x_min, y_min, z_min = coords.min(axis=0)
x_max, y_max, z_max = coords.max(axis=0)
bounding_box_nf = [x_min, x_max, y_min, y_max, z_min, z_max]
```

---

## 7. Post-CPFE Analysis Workflow

### Where REI field data comes from (three sources)
- **`mesh_out/out_element_centroid_*.csv`** — the DEFAULT (`mesh_csv="sync"`). Crisp
  per-element fields sampled on the true CPFE mesh (no transfer, no smoothing). Point REI /
  `SimulationResults(field_dir=".../mesh_out")` here for full-fidelity full-mesh REI (kNN
  path; one row per element).
- **`grid_out/…`** — a regular grid; written during the run only if `grid_transfer="per_step"`
  (crisp, fast grid path, but pays the per-step transfer), or regenerated offline (below).
- **GridResampler** — regenerate `grid_out/` from the Exodus after a cheap run.

### (If you want a grid) regenerate grid CSVs offline: GridResampler
With `grid_transfer="final"`/`"off"`, regenerate the regular-grid CSVs afterward from the
native-mesh Exodus via MOOSE's shape-evaluation transfer — a **cheap approximation**, not a
bit-exact copy of the online per-step grid (see caveat below). For full fidelity prefer
`mesh_out/` above.
```python
from graintrace.grid_resampling import GridResampler
gr = GridResampler(
    cpfe_exodus="out/simulation/simulation_out/sim_output.e",
    save_dir="out/simulation/simulation_out",   # writes grid_out/out_element_centroid_<idx4>.csv
    number_of_elements=[100, 100, 100],
    bounding_box=grid_bb,                        # same inset grid box used for the run
    moose_run_file="external/puma/puma-opt",
    launcher="mpiexec",                          # or "srun" on Cray/Slurm
)
gr.resample(timesteps="all")   # or a list of 1-based Exodus timestep indices
```
Then point `SimulationResults(field_dir=".../grid_out", ...)` at the output as usual. The
resampler needs no NEML2/AOTI (no `[NEML2]` block) and no device; `sim_output.e` must
contain the fields (so pair it with `exodus_output="sync"`/`"per_step"`).

**Fidelity caveat.** The CPFE fields are order-FIRST `MONOMIAL`, which MOOSE stores in the
Exodus as *nodal projections* (element→node averaging). Resampling from that Exodus is
therefore **smoothed** — grain-boundary extremes are compressed vs the online per-step grid
at the *same* resolution (e.g. cauchy_stress range shrank ~2-3x in an 8³ check). A denser
resample grid recovers spatial detail but NOT the extremes lost to the nodal projection. For
extreme-sensitive REI use `grid_transfer="per_step"` (crisp online grid) or run REI on the
true mesh; use the resampler for cheap/approximate grids.

### Load simulation results
```python
from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming
from graintrace import plot_postprocessing as postprocess

field_naming = FieldFileNaming(
    prefix="out_element_centroid",
    index_width=4,
    sep="_",
    suffix=".csv",
)
res = SimulationResults(
    block_csv="out/simulation/simulation_out/out.csv",
    field_dir="out/simulation/simulation_out/grid_out",
    field_naming=field_naming,
)
```

### Plot distributions at sync times
```python
postprocess.plot_block_properties_distribution(
    res, time=sync_times[i], tensor_prefix="ee", order=2, output_folder="out/postprocess"
)
postprocess.plot_block_properties_distribution(
    res, time=sync_times[i], tensor_prefix="nye_tensor", order=2, output_folder="out/postprocess"
)
```

### Plot macroscopic stress-strain
```python
postprocess.plot_macroscopic_stress_strain(
    res,
    stress_tensor_prefix="cauchy_stress",
    strain_tensor_prefix="strain",
    volume_prefix="volume",
    output_folder="out/postprocess",
)
```

### Plot pole figure (requires updated neml2)
```python
postprocess.plot_pole_figure(
    res,
    tensor_prefix="ori_rodrigues",
    time=sync_times[i],
    direction=[0, 0, 1],
    crystal_symmetry="432",
    device="cpu",
    output_folder="out/postprocess",
    construct_odf=False,
)
```

### IPF coloring of mesh
```python
from graintrace.ipf_postprocess import IPFProcessor

ipf = IPFProcessor(crystal_symmetry="432", sample_symmetry="432", save_dir="out/mesh")
ipf.ipf_color_chart(savefig_name="ipf_color_chart.png")
ipf.add_block_rgb_to_exodus(
    mesh_file="out/mesh/mesh.e",
    orientations_csv="out/mesh/orientations.csv",
    output_file="mesh_rgb.e",
    direction=[0.0, 0.0, 1.0],
    angle_convention="mrp",
)
ipf.add_block_rgb_to_vtk(
    vtk_file="out/mesh/fixed_grid.vtk",
    output_file="merged_segmented_fixed_grid_rgb.vtk",
    direction=[0.0, 0.0, 1.0],
    angle_convention="bunge",
    angle_type="degrees",
    orientation_fields=("Eul1", "Eul2", "Eul3"),
)
```

### Rare event identification (REI): IdentifyRareClusters
Takes the last grid output CSV (from `grid_out/`) and identifies spatially coherent rare regions.

```python
from graintrace.rare_cluster_indicator import IdentifyRareClusters
from graintrace.similarity_metric_library import SimilarityMetricLibrary
from graintrace.user_data_class import SimilarityMetric, WeightConfig, RareCriteria
from graintrace import rare_criteria_selection_library as rcs

metric_lib = SimilarityMetricLibrary()
spec = metric_lib.nye_tensor_norm(cols=["nye_tensor_11", ..., "nye_tensor_33"])
spec_reduced = SimilarityMetric(
    name=spec.name + "_mean",
    feature_cols=[f"{c}_mean" for c in spec.feature_cols],
    func=spec.func,
)
rare_criteria = RareCriteria(
    selector=lambda df: rcs.select_highest_scalar(
        df, k=5, required_cols="nye_tensor_norm_mean_mean", min_size=1
    )
)

weight_cfg = WeightConfig(
    mode="rbf", power=2.0, sigma=None,
    sigma_auto={"sample_size": 500_000, "random_state": 42, "quantile": 0.5},
)

irc = IdentifyRareClusters(
    input_csv_path=filename,
    id_col="id",
    coord_cols=("x", "y", "z"),
)
gsc, indicator = irc.make_stage_objects(graph_cluster_out=base + "_reduced.csv")

bundle = irc.run_clustering(
    gsc=gsc,
    indicator=indicator,
    reduced_csv_path=base + "_reduced.csv",
    gsc_run_kwargs=dict(
        spec=spec,
        graph_mode="grid",           # "grid" | "knn" | "auto"
        manhattan_radius=4,
        grid_tol=1e-6,
        n_jobs=12,
        weight_chunk_size=500_000,
        segmenter="leiden",
        seed=42,
        weight_cfg=weight_cfg,
        reduce_edges_topweights_k=20,
        networkit_kwargs={"gamma": 10.0},
        checkpoint_base_path=base + "_gsc_ckpt",
        resume_from_checkpoint=False,
    ),
    indicator_run_kwargs=dict(
        method_type="scipy_hierarchical",
        spec=spec_reduced,
        threshold=0.0005,
        method="average",
        criterion="distance",
        dendrogram_path="out/rei/dendrogram.png",
    ),
)
pd.to_pickle(bundle, base + "_bundle.pkl")

out = irc.run_get_rare_cluster(
    bundle=bundle,
    criteria=rare_criteria,
    output_vtk_path="out/rei/rare_clusters.vtk",
    export_control="auto",
    background_block_id=1,
    first_rare_block_id=2,
    also_write_final_label=True,
    rare_reduced_stats_csv_path="out/rei/rare_cluster_stats.csv",
    use_sample_std=False,
)
```

### Compare two REIs: REIComparison
Compare two rare-event point clouds (two metrics/thresholds/methods, or prediction vs.
reference). First have each REI emit a point-cloud CSV: pass `rare_points_csv_path=...` to
`run_get_rare_cluster` above (writes `x,y,z,rare_cluster_id` for the rare points). Then:

```python
from graintrace.rei_comparison import REIComparison

comp = REIComparison(
    rei_csv_1="out/rei_A.csv", rei_csv_2="out/rei_B.csv",
    output_dir="out/rei_comparison",
    spacing_1=1.0, spacing_2=2.0,      # scalar or [dx,dy,dz]; None -> auto-detect (sparse-unsafe)
    coord_cols=("x", "y", "z"),
    cluster_col="rare_cluster_id",     # None -> global overlap only, no cluster matching
    supersample=1,                     # >1 -> sub-voxel boundary accuracy
)
result = comp.run_comparison()
```

Pure Python (numpy + scipy). Voxel model: each rare point is its cube; both regions are
resampled onto the finer lattice `s_ref = min(spacing_1, spacing_2)` and overlap is an
integer-index hash intersection (no KD-tree / alpha-shape / marching-cubes). Grids may have
**different spacing** but must **share an origin** (no rotation/translation is applied).
Outputs in `output_dir`: `overlap_metrics.json` (IoU/Dice/`containment_1`/`containment_2` +
counts/volumes), `overlap_cloud.vtk` (scalar `membership` 1=only-1/2=only-2/3=both, plus
`cluster_id_1`/`cluster_id_2`), and `cluster_match.csv` (1-to-1 Hungarian cluster pairing by
overlap volume, label-agnostic; unmatched flagged `-1`; split/merge counts).

---

## 8. Material Calibration

Fits 6 crystal-plasticity parameters to a macroscopic stress-strain curve (+ full-field
elastic strains) with a **neml2 v3 + pyzag analytic-adjoint** Taylor model driven by LBFGS.
Self-contained data ships in `mwe_data/ff_calibration/` (9 load steps × 500 grains, columns
`O11..O33`, coords, `Eul0/1/2`, `eKen11..33`, + `strain-stress.csv`).

```python
import graintrace as _gt
from pathlib import Path
from graintrace.material_calibration import MaterialCalibration
from graintrace.taylor import TaylorModel

_cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")

calib = MaterialCalibration(
    model_class=TaylorModel,
    model_args=dict(
        neml2_path=_cpfe_base + "/neml2_cpfe_calibration.i",
        npoints=30,          # pyzag time steps = resampled stress-strain points
        nchunk=2,            # pyzag chunk size for the bidiagonal-in-time solve
        device="cuda",       # or "cpu"; cuda works (model moved via nsys.to(device))
        compile=False,
    ),
    data_args=dict(
        data_dir="mwe_data/ff_calibration",                       # per-stress-level CSVs
        strain_stress_file="mwe_data/ff_calibration/strain-stress.csv",
        npoints=30,
        full_field_strain_units="microstrain",
        straintype="eKen",   # full-field strain column prefix ("eKen" or "eFab")
        max_strain=0.006,    # cap the macro curve to a convergent regime
        n_grains=100,        # subsample grains per load step (None = all)
        seed=42,
    ),
    save_dir="out/material_calibration",
    apply_elastic_correction=False,
    strain_window=(0.0, 0.0015),
)

calib.plot_texture(direction=[1, 1, 1])
calib.plot_stress_strain()
# maxiter is an upper bound; the plateau guard stops early once the relative loss
# improvement over `plateau_window` steps drops below `plateau_rtol`.
calib.calibrate(maxiter=15, lr=0.3, max_iter_per_step=6,
                line_search_fn="strong_wolfe", plateau_rtol=1e-3, plateau_window=2)
calib.load("out/material_calibration/calibrated_material.json")
calib.plot_stress_strain(include_model=True)
calib.plot_strain_histogram(include_initial_strain=True)
```

The 6 opt vars (`TaylorModel.opt_vars`) map to CPFE material names via:
`elastic_tensor_E→elastic_E`, `elastic_tensor_G→elastic_G`, `elastic_tensor_nu→elastic_nu`,
`slip_strength_constant_strength→slip_constant_strength`, `voce_hardening_initial_slope`,
`voce_hardening_saturated_hardening→voce_hardening_saturation`.

For a physically registered fit, first rotate the raw FF CSVs into the simulation frame with
`experiment_rotation_helper` (reads `Eul*`, (re)writes the rotated `O11..O33` as
`reconstruction.ori`, dropping any pre-existing `O`):
```python
from graintrace.experiment_rotation_helper import update_experiments, collect_experiment_files

files, stress_levels = collect_experiment_files(exp_data_dir)
update_experiments(
    input_files=files,
    output_root="out/rotated_experiments",
    bounding_box=bounding_box,
    auto_fix_bbox=True,
    bbox_fix_mode="remove_points",
    rotate_angles=rotate_angles,
    unit="rad",
    angle_identifier=["Eul0", "Eul1", "Eul2"],
    orientation_descriptor="euler-bunge",
    orientation_active_convention=True,
    elastic_strain_identifier=["eKen11", ..., "eKen33"],
)
```

---

## 9. Key Config Patterns

### `sculpt_config` dict
Required for any step calling `builder.mesh(sculpt_config=...)`:
```python
sculpt_config = {
    "launcher": "/path/to/cubit/bin/mpi/bin/mpiexec",
    "psculpt":  "/path/to/cubit/bin/psculpt",
    "epu":      "/path/to/cubit/bin/epu",
    "nprocs":   10,
    "environment": {
        "OPAL_LIBDIR": "/path/to/cubit/bin/mpi/lib",
        "OPAL_PREFIX": "/path/to/cubit/bin/mpi",
    },
}
```

Required keys: `psculpt`, `epu`, `nprocs`. `launcher` and `environment` are needed for MPI-based execution.

### `sculpt_options` tuple
Passed as a tuple of CLI flag strings. Common options:
```python
sculpt_options = (
    "--adapt", "-A", "7",    # mesh adaptation level
    "-df", "1",              # dilation factor
    "-S", "2",               # smoothing passes
    "-CS", "4",              # curve smoothing
    "--void_mat", "0",       # void material ID
)
```
For FF Voronoi meshes without adaptation, use just `("--void_mat", "0")`.

### `segmentation_prop` dict (VoxelMeshBuilder / EBSD / NF-as-voxel)
Two methods:
```python
# Flood fill (simpler, faster)
segmentation = {
    "method": "flood",
    "params": {
        "misorientation_tol": 5.0/180*np.pi,  # always radians for VoxelMeshBuilder
        "connectivity": 26,                    # 6 or 26
        "grain_threshold_final": 1000,
        "batch_norm": 200_000,                 # flood-only
        "grain_threshold": 1000,               # flood-only
        "stop_count": 500,                     # flood-only
    },
}

# Graph-based (better for complex textures)
segmentation = {
    "method": "graph",
    "params": {
        "misorientation_tol": 5.0,    # degrees when angle_type="degrees", else radians
        "connectivity": 26,
        "grain_threshold_final": 100,
    },
    "graph_params": {
        "segmenter": "leiden",
        "graph_mode": "grid",
        "manhattan_radius": 2,
        "grid_tol": 1e-6,
        "n_jobs": 10,
        "weight_chunk_size": 1_000_000,
        "reduce_edges_topweights_k": 8,
        "nodes_chunk": 500_000,
        "seed": 42,
        "networkit_kwargs": {"gamma": 0.001},   # lower = fewer clusters
        "weight_cfg": {
            "mode": "rbf",
            "sigma": None,
            "sigma_auto": {"sample_size": 20_000, "random_state": 42, "quantile": 0.5},
            "power": 2.0,
        },
        "plot": True,
    },
}
```

Note: For `NearFieldMeshBuilder.reconstruct()`, the `segmentation` argument is a flat dict (no `method`/`params` nesting), using `misorientation_tol` directly in radians.

### `bc` dict format
```python
bc = {
    "x": {"negative": "stress_free", "positive": "stress_free"},
    "y": {"negative": "stress_free", "positive": "stress_free"},
    "z": {"negative": 0, "positive": displace_amount},  # 0 = fixed, float = displacement
}
```
`"stress_free"` means traction-free (no constraint). Integer/float means prescribed displacement.

### `WeightConfig` dataclass
```python
from graintrace.user_data_class import WeightConfig

weight_cfg = WeightConfig(
    mode="rbf",           # "rbf" | "inverse"
    power=2.0,
    sigma=None,           # if None, use sigma_auto
    sigma_auto={
        "sample_size": 500_000,
        "random_state": 42,
        "quantile": 0.5,
    },
)
```

---

## 10. Common Pitfalls

### NEPER is bring-your-own
`VoronoiMeshBuilder` and `CrystalGenerator` resolve a **user-installed** NEPER via
`graintrace/neper_env.py` (`resolve_neper_env`): precedence is explicit
`neper_path=`/`env=` → `NEPER` env var → `graintrace_tools.json` `"neper"` key →
`neper` on PATH. If none resolve, the builder raises a clear `RuntimeError` linking
neper.info. An opt-in `auto_install=True` performs a Linux `~/.local` source build
(GSL + OpenBLAS + NEPER). gmsh is a pip dependency (auto-installed with graintrace),
so the builders take no gmsh arguments. `scan_tessellation.default_neper_env()`
delegates to the same resolver.

### cpfe_base path
Always derive the cpfe_base path dynamically, do not hardcode it:
```python
import graintrace as _gt
from pathlib import Path
_cpfe_base = str(Path(_gt.__file__).parent / "cpfe_base")
# Contains: neml2_cpfe.i, neml2_cpfe_calibration.i, run_cpfe.i, etc.
```

### `if __name__ == "__main__"` (not required for clustering)
The graph segmentation / REI clustering pipeline (`GraphSpatialCluster`,
`IdentifyRareClusters`) does not use `multiprocessing`: the prune step is
numba-threaded (`nogil`, with a single-threaded numpy fallback) and edge-distance
computation is single-process. So the `if __name__ == "__main__"` guard is **not
required** for clustering/REI scripts.

It **is required** for any script that calls **NF reconstruction**
(`graintrace/nf/convert.py` `pointcloud_to_fixed_grid` uses `multiprocess.Pool`
for GIL-bound scipy Delaunay work). It is also good practice generally, so the
example scripts keep the pattern:
```python
def main():
    # ... all experiment logic ...

if __name__ == "__main__":
    main()
```

### Pole figures use neml2.texture (v3)
`plot_pole_figure` uses `neml2.texture` (`polefigure`, `odf`, IPF helpers). If the
import fails the bindings are outdated; reinstall neml2 v3 from the PUMA neml2 submodule
(`external/puma/neml2`, `pip install . --no-deps`) into the `moose-src` env.

### GPU: if available, always use it
The GPU-accelerated steps are **CPFE** and **material calibration** (and pole
figures). Policy: whenever a CUDA GPU is present, use it; CPU is much slower for
these neml2-dominated workloads. Detect with `torch.cuda.is_available()` /
`torch.cuda.device_count()`, then:
- CPFE: `sim.set_parameters("simulation_parameters", device="cuda:0")` (or a
  space-separated list `"cuda:0 cuda:1"` for multi-GPU = MPI ranks; `ncore` ==
  number of GPUs).
- Calibration: `TaylorModel(device="cuda")` (see the note below).
- Pole figures: `plot_pole_figure(..., device="cuda")`.
Only fall back to `"cpu"` when no GPU exists. The MCP server enforces this: its
tools default the device to the GPU when one is available (`run_cpfe` auto-fills
`cuda:0`, `calibrate_material` defaults to `cuda`), and `dependency_status`
reports the visible GPUs.

### cuda material calibration = model must be on device
`TaylorModel(device="cuda")` works because `taylor.py` moves the whole nonlinear system with
`nsys.to(device)` before wrapping it in the pyzag factory. `factory.to(device)` alone leaves the
model's crystal-geometry buffers (Schmid tensors) on CPU → a cuda/cpu mismatch that surfaces as
a silent `loss=inf`.

### `orientation_tolerance` units must match `ori_units`
When `ori_units="radians"`, convert `orientation_tolerance` before passing to `RegionBaseStitching`:
```python
if ori_units == "radians":
    orientation_tolerance = np.deg2rad(orientation_tolerance)
    sample_rotate_angle = np.deg2rad(sample_rotate_angle)
```

### FF output orientations are always in degrees
`VoronoiMeshBuilder.build_voronoi()` always writes `orientations.dat` in degrees regardless of input units. When feeding to `VoxelMeshBuilder` afterward:
```python
ff_voxel_builder = VoxelMeshBuilder(
    file_path=os.path.join(output_ff, "reconstruction_reformatted.csv"),
    ...
    angle_type="degrees",   # always degrees after FF build
)
```

### `bounding_box` for grid_properties should be inset
Set `grid_bb` as `bounding_box ± 0.0001` to avoid mesh boundary issues:
```python
grid_bb = bounding_box.copy()
for i in range(0, 6, 2):   # xlo, ylo, zlo
    grid_bb[i] += 0.0001
for i in range(1, 6, 2):   # xhi, yhi, zhi
    grid_bb[i] -= 0.0001
sim.set_parameters("grid_properties", number_of_elements=[nx, ny, nz], bounding_box=grid_bb)
sim.set_parameters("boundary", bounding_box=bounding_box, bc={...})  # full box for BCs
```

### NF reconstruction restart
If segmentation already ran, skip it with a restart flag and load the `.npy` directly:
```python
if NF_RECONSTRUCTION_MESH_RESTART:
    merged_grid_path = os.path.join(output_nf, "merged_segmented_fixed_grid.npy")
else:
    merged_grid_path = builder_nf.reconstruct(...)
```

### Auto-detection of orientation units
Use `ori_units = "auto"` and detect from values: if any Euler component exceeds `2π`, units are degrees.

### REI checkpoint pattern
Three checkpointing levels in order of priority:
1. `PICK_CLUSTER_RESTART`: load bundle pickle (fastest restart)
2. `FINAL_CLUSTERING_RESTART`: load reduced CSV + GSC labels numpy
3. `GRAPH_SEGMENTATION_RESTART`: load graph edges/weights/meta from `_gsc_ckpt.*`

---

## 11. Module Map

```
graintrace/
  neper_env.py                 resolve_neper_env     : locate a user-installed NEPER (env var/tools.json/PATH); opt-in ~/.local build
  construct_voronoi_mesh.py    VoronoiMeshBuilder    : FF HEDM reconstruction (NEPER)
  construct_nf_mesh.py         NearFieldMeshBuilder  : NF HEDM reconstruction (SCULPT)
  construct_voxel_mesh.py      VoxelMeshBuilder      : EBSD/NF voxel meshing (SCULPT)
  nf_grid_conversion.py        NFGridConversion      : Pre-gridded NF data to CSV
  run_cpfe_simulation.py       CPFESimulation        : MOOSE/PUMA CPFE runner
  grid_resampling.py           GridResampler         : offline resample of a CPFE Exodus -> grid CSV
  simulation_postprocessing.py SimulationResults     : Load/query CPFE output CSVs
  experiment_postprocessing.py ExperimentResults     : Load experimental data
  plot_postprocessing.py       plot_*                : Distribution/stress-strain/pole figures
  ipf_postprocess.py           IPFProcessor          : IPF coloring on mesh
  rare_cluster_indicator.py    IdentifyRareClusters  : REI full pipeline
  graph_spatial_cluster.py     GraphSpatialCluster   : Graph segmentation of field data
  cluster_indicator.py         ClusterAnalysisIndicator : Hierarchical clustering stage
  similarity_metric_library.py SimilarityMetricLibrary  : Built-in feature metrics
  rare_criteria_selection_library.py                 : select_highest_scalar, etc.
  material_calibration.py      MaterialCalibration   : Taylor model parameter fitting
  taylor.py                    TaylorModel           : NEML2 Taylor model wrapper
  experiment_rotation_helper.py                      : Rotate experiment CSVs
  hedm_stitching_techniques/
    region_base_stitching.py   RegionBaseStitching   : Multi-scan Z-stitching
  scan_stitching_comparison.py ScanStitchingComparison : Compare stitching results
  tess_to_gnn.py               NeperTessToGraphNN    : .tess → graph data structure
  user_data_class.py           SimilarityMetric, WeightConfig, RareCriteria
  cpfe_base/                   MOOSE/NEML2 input templates (.i files)
    neml2_cpfe.i               : CPFE material model definition
    neml2_cpfe_calibration.i   : Taylor model for calibration
    run_cpfe.i                 : Main MOOSE simulation input
    grid_file.i                : Grid output configuration (online regular-grid sub-app)
    resample_grid.i            : GridResampler main app (grid + shape-eval transfer)
    resample_source.i          : GridResampler sub-app (loads CPFE Exodus at one timestep)
    initial_conditions.i       : Initial condition setup (NF orientation)
    initial_conditions_ff.i    : Initial condition setup (FF-registered)
    transfer.i                 : Variable transfer between meshes (grid_transfer_execute_on)
```

---

## 12. Examples & Skills

Each workflow segment has a runnable example (`examples/demonstrate_*.py`, flat top-level
`## INPUT` style) and an invocable Skill (`/<name>`, in `.claude/skills/<name>/SKILL.md`) that
distills the recipe. Run examples from the `moose-src` env (`conda activate moose-src`), the
standard graintrace env (graintrace pip-installed editable; neml2 v3.0.7 + pyzag 2.0.0).

| `/skill` | Segment | Example | Self-contained data |
|---|---|---|---|
| `graintrace-overview` | segment → skill → example index + env/tool matrix | none | none |
| `hedm-stitching` | synthetic crystal + z-scan + stitch + compare | demonstrate_hedm_study.py | self-generates (NEPER) |
| `ff-reconstruction` | FF Voronoi reconstruction (`VoronoiMeshBuilder`) | demonstrate_farfield.py | mwe_data/ff_calibration |
| `nf-reconstruction` | NF mesh (`NearFieldMeshBuilder`) | demonstrate_cpfe_nfff.py | synthetic (NEPER/CUBIT) |
| `voxel-segmentation-mesh` | EBSD/NF voxel graph-seg + sculpt (`VoxelMeshBuilder`) | demonstrate_grid_segmentation_mesh.py | synthetic gen |
| `microstructure-generation` | NEPER morpho (incl. `aspratio`) generation param guidance from a 12-case study | demonstrate_hedm_anisotropic.py | none |
| `meshing` | SCULPT `sculpt_options` (2 safe configs) + `mesher="voxel"` direct-to-Exodus dump | demonstrate_synthetic_cpfe.py | synthetic (NEPER/CUBIT) |
| `experiment-rotation` | rotate raw FF CSVs into sim frame | demonstrate_farfield.py | mwe_data/ff_calibration |
| `material-calibration` | pyzag-adjoint Taylor calibration | demonstrate_material_calibration.py | mwe_data/ff_calibration |
| `cpfe-simulation` | FF CPFE via AOTI (`CPFESimulation`) | demonstrate_cpfe.py | mwe_data/cpfe_ff |
| `cpfe-nf-ff` | NF geometry + FF initial strain CPFE | demonstrate_cpfe_nfff.py | synthetic (NEPER/CUBIT/MOOSE) |
| `post-processing` | distributions / stress-strain / IPF | demonstrate_postprocess.py | mwe_data/out.csv + grid_out |
| `rare-event-identification` | graph cluster → hierarchical → rare VTK | demonstrate_rei_pipeline.py (+2D/3D) | mwe_data/synthetic_vms.csv (regen) |
| `rei-comparison` | compare two REI point clouds → overlap metrics + classified VTK | demonstrate_rei_comparison.py | synthetic (generated on demand) |
| `grain-tracking` | grain graph matching across load steps | demonstrate_graintracking.py | mwe_data/synthetic_load_exp |

**Shippable `mwe_data/` datasets:** `ff_calibration/` (calibration + FF recon), `cpfe_ff/`
(10-grain `reconstruction.msh` + `orientations.dat`), `out.csv` + `grid_out/` (CPFE post-proc),
`synthetic_load_exp/` (grain tracking; load-step FF CSVs), `synthetic_vms.csv` (REI seed).

**External-tool matrix:** NEPER → ff-reconstruction, nf/ff/hedm synthetic; CUBIT/`sculpt_config`
→ nf-reconstruction, voxel-segmentation-mesh, cpfe-nf-ff; MOOSE `puma-opt` + `neml2-compile` →
cpfe-simulation, cpfe-nf-ff; NEML2 v3 only → material-calibration; none → post-processing, REI.

---

## 13. Code Standards

Follows the pyzag project's conventions: **black** + **pylint** (not ruff/pyright).

- **Formatting:** `black` (pinned `24.3.0` in `[dev]`); run `black graintrace tests`.
- **Linting:** `pylint --rcfile=.pylintrc graintrace` must be clean (0 messages). `.pylintrc`
  sets `max-line-length=240` and disables `C0103,E1101,E1102,R0903,R0801` (pyzag base) plus the
  `too-many-*` complexity family (`R0902,R0904,R0911,R0912,R0913,R0914,R0915,R0916,R0917,C0302`);
  complexity refactors of the numerical routines are deliberately out of scope. Docstring and
  import-placement rules stay ON and are fixed in code.
- **Copyright:** every `.py` carries the MIT header (see any existing file).
- **Lazy imports:** heavy/optional deps (neml2, pyzag, torch, torch_geometric, gmsh, matplotlib,
  pyvista, vtk, …) are imported inside functions with an inline
  `# pylint: disable=import-outside-toplevel`; `graintrace/__init__.py` uses a PEP 562 lazy
  `__getattr__` so `import graintrace` never pulls the compiled stack.
- **Tests:** pytest; `torch.float64`; `torch.manual_seed(42)` for reproducibility; finite-difference
  checks for gradients. NEML2/pyzag-dependent tests `pytest.importorskip` so a plain checkout skips
  them. `pylint` gate is on `graintrace/` only; `black --check` on `graintrace` + `tests`.
- Keep comments succinct.

---

## 14. Definition of Done (adding or changing code)

Any new feature or code change is **not complete** until all five of these are
done in the same change. Treat it as the checklist for every PR.

1. **Tests pass.** Add or extend tests under `tests/` for the new behavior, and
   keep the suite green: `pytest`, `black --check graintrace tests`, and
   `pylint --rcfile=.pylintrc graintrace` (0 messages). Tests that need the
   external stack must self-skip via `pytest.importorskip`/`skipif`.
2. **Docstrings updated.** Every new/changed public class, method, and function
   has a docstring (Google or NumPy style; napoleon renders both). Document new
   constructor kwargs — they surface directly in the API reference.
3. **Docs updated.** Update `docs/` for the change: the relevant tutorial under
   `docs/tutorials/`, `docs/configuration.rst` for new options, and add an
   `automodule` page under `docs/api/` (+ `docs/api/api.rst`) for a new public
   module. The strict build must pass: `sphinx-build -W --keep-going -b html
   docs docs/_build/html`, plus `make -C docs doctest`.
4. **CLAUDE.md updated.** Reflect the change in this file — the relevant workflow
   section (§3–§9), the option tables (§9), the Module Map (§11), the
   examples/skills table (§12), or the pitfalls (§10) — so this reference stays
   the source of truth.
5. **MCP updated.** If the change is a workflow segment or adds/changes
   user-facing parameters, update the MCP layer: the tool in
   `graintrace/mcp/tools/*.py`, the vetted recipe in `graintrace/mcp/recipes/*.md`,
   and the tool table in `graintrace/mcp/README.md`. New external-tool
   dependencies must be reported by `dependency_status`.

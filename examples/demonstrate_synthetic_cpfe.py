# Copyright 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: graintrace
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""End-to-end synthetic CPFE demo: generate a microstructure and drive it to CPFE.

Pipeline:
  1. CrystalGenerator     -> voronoi.tess + voronoi.ori + voronoi.tesr (voxelized in
                             the same neper call by adding tesr to -format)
  2. reformat_tesr_file   -> voronoi_reformatted.csv (X,Y,Z,CellID,Eul0,Eul1,Eul2)
  3. VoxelMeshBuilder     -> hex mesh + per-element MRP orientations, either:
                             3a. mesher="sculpt" -> conformal (smoothed) SCULPT mesh
                             3b. mesher="voxel"  -> one cube hex per voxel, no SCULPT,
                                 zero inverted elements (stair-stepped boundaries)
Then two extensions off the same microstructure:
  A. NeperTessToGraphNN   -> cell graph (nodes = grains, edges = shared faces)
  B. CPFESimulation       -> MOOSE/PUMA crystal-plasticity run (NEML2 v3 AOTI)

Because NEPER hands us exact per-voxel grain IDs (CellID), we pass cell_id_col and
VoxelMeshBuilder SKIPS segmentation entirely. Drop cell_id_col to instead segment the
raster Euler field (see the commented segmentation_prop block below).
"""

import os

from graintrace.generate_random_crystal import CrystalGenerator
from graintrace.construct_voronoi_mesh import VoronoiMeshBuilder
from graintrace.construct_voxel_mesh import VoxelMeshBuilder


## INPUTS ---------------------------------------------------
def main():

    box = 1000.0
    grains = 100
    z_aspect = 1.0

    tesr_res = [100, 100, 100]
    ncore = 10

    output_dir = "uniform"
    bounding_box = [0, box, 0, box, 0, box]

    sculpt_config = {
        "launcher": "/path/to/cubit/bin/mpi/bin/mpiexec",
        "psculpt": "/path/to/cubit/bin/psculpt",
        "epu": "/path/to/cubit/bin/epu",
        "nprocs": int(ncore),
        "environment": {
            "OPAL_LIBDIR": "/path/to/cubit/bin/mpi/lib",
            "OPAL_PREFIX": "/path/to/cubit/bin/mpi",
        },
    }
    sculpt_options = ("--void_mat", "0")

    ## 1. Generate the tessellation
    cg = CrystalGenerator(
        output_dir=output_dir,
        bounding_box=bounding_box,
        dim=3,
        seed=42,
        env=None,
    )
    cg.generate_tessellation(
        morpho_args={
            "type": "raw",
            "morpho_str": f"diameq:lognormal(130, 5),aspratio(1, 1, {z_aspect})",
        },
        iterations=20000,
        extra_neper_args=[
            "-reg",
            "1",
            "-format",
            "tess,geo,ori,tesr",
            "-tesrsize",
            f"{tesr_res[0]},{tesr_res[1]},{tesr_res[2]}",
            "-tesrformat",
            "ascii",
        ],
    )
    print(
        f"(nominal target grains ~= {grains}; actual count set by diameq distribution)"
    )

    ori_file = os.path.join(output_dir, "voronoi.ori")  # per-cell Euler-bunge, degrees
    tesr_file = os.path.join(output_dir, "voronoi.tesr")

    ## 2. Reformat .tesr -> per-voxel CSV (X,Y,Z,CellID,Eul0,Eul1,Eul2)
    reformatter = VoronoiMeshBuilder(
        input_csv=None,
        output_dir=output_dir,
        bounding_box=bounding_box,
        dim=3,
        env=cg.env,
    )
    reformatter.reformat_tesr_file(tesr_file=tesr_file, orientation_file=ori_file)
    voxel_csv = os.path.join(output_dir, "voronoi_reformatted.csv")

    ## 3. Voxel meshing pathway
    builder = VoxelMeshBuilder(
        file_path=voxel_csv,
        save_dir=os.path.join(output_dir, "mesh"),
        euler_cols=("Eul0", "Eul1", "Eul2"),
        cell_id_col="CellID",
        angle_convention="bunge",
        angle_type="degrees",
        symmetry="432",
    )

    merged_grid = builder.reconstruct(apply_smoothing=False)
    print(f"\nReconstruction complete: {merged_grid}\n")

    mesh_path = builder.mesh(
        sculpt_config=sculpt_config,
        sculpt_options=sculpt_options,
        merged_grid=merged_grid,
    )
    ori_csv = f"{builder.mapped_orientations_path}.csv"  # per-element MRP orientations
    print(f"Meshing complete: {mesh_path}")
    print(f"Per-element orientations (MRP): {ori_csv}")

    ## 3b. ALTERNATIVE mesh: direct voxel -> Exodus (no SCULPT, zero bad elements)
    # mesher="voxel" dumps one perfect cube hex per voxel: stair-stepped boundaries
    # but scaled Jacobian = 1 everywhere (no inverted/sliver elements). Use this
    # mesh + orientations for CPFE if the SCULPT mesh has bad elements.
    voxel_mesh = builder.mesh(
        mesher="voxel",
        merged_grid=merged_grid,
        mesh_path=os.path.join(output_dir, "mesh", "mesh_voxel.e"),
        mapped_orientations_path=os.path.join(output_dir, "mesh", "orientations_voxel"),
    )
    voxel_ori = os.path.join(output_dir, "mesh", "orientations_voxel.csv")
    print(f"Voxel Exodus: {voxel_mesh}  (orientations: {voxel_ori})")

    ## EXTENSION A: build a cell graph directly from the tessellation -----------
    # Independent of the mesh; reads voronoi.tess (nodes = grains, edges = shared
    # faces, node orientations returned as neml2 v3 MRP -> needs neml2 + torch_geometric).
    from graintrace.tess_to_gnn import (
        NeperTessToGraphNN,
    )  # pylint: disable=import-outside-toplevel

    parser = NeperTessToGraphNN(
        tess_path=os.path.join(output_dir, "voronoi.tess"), device="cpu"
    )
    graph = parser.build_cell_graph()
    print(f"\nGraph built: {graph}")

    ## EXTENSION B: run a CPFE simulation on the meshed microstructure ----------
    # Consumes the SCULPT mesh + the per-element MRP orientations written above.
    # (eeres_file=None -> zero initial elastic strain; this is a purely synthetic
    #  microstructure with no far-field residual-strain field.)
    import torch  # pylint: disable=import-outside-toplevel
    from graintrace.run_cpfe_simulation import (
        CPFESimulation,
    )  # pylint: disable=import-outside-toplevel

    # EDIT: path to your built PUMA binary (e.g. external/puma/puma-opt).
    moose_run_file = "external/puma/puma-opt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # GPU if present

    total_strain = 0.002  # applied axial (z) engineering strain
    grid_elements = [50, 50, 50]  # field-output grid resolution

    sim = CPFESimulation(
        mesh_file=str(mesh_path),
        save_simulation_folder=os.path.join(output_dir, "simulation"),
        moose_run_file=moose_run_file,
        element_order="FIRST",  # SCULPT hex meshes are linear (FIRST) order
        eeres_file=None,  # no FF residual strain -> zero_initial_strain.ee
        ori_file=ori_csv,  # already MRP from the mesh step (no conversion needed)
        dim=3,
        use_ff_initial_field=False,  # voxel mesh, not co-registered FF strain
    )

    sim.set_parameters(
        "material",
        slip_constant_strength=130.0,
        voce_hardening_initial_slope=1556.09,
        voce_hardening_saturation=100.0,
        power_slip_n=20,
        power_slip_g0=0.0001,
        elastic_E=209016.0,
        elastic_nu=0.307,
        elastic_G=60355.0,
        burger_scale=2.22,
    )
    sim.set_parameters(
        "simulation_parameters",
        device=device,
        device_batch=20000,
        dt=0.5,
        total_time=2.0,
        initialize_time=1.0,
        sync_times="2.0",
    )

    displace = total_strain * (bounding_box[5] - bounding_box[4])
    sim.set_parameters(
        "boundary",
        bounding_box=bounding_box,
        bc={
            "x": {"negative": "stress_free", "positive": "stress_free"},
            "y": {"negative": "stress_free", "positive": "stress_free"},
            "z": {"negative": 0, "positive": displace},
        },
    )

    grid_bb = list(bounding_box)  # inset the grid box to avoid boundary voxels
    for i in range(0, 6, 2):
        grid_bb[i] += 1e-4
    for i in range(1, 6, 2):
        grid_bb[i] -= 1e-4
    sim.set_parameters(
        "grid_properties", number_of_elements=grid_elements, bounding_box=grid_bb
    )

    # Bakes material params -> neml2-compile -> AOTI package -> launches puma-opt.
    sim.run(ncore=ncore)
    print(f"Simulation complete: {sim.save_simulation_folder}")


if __name__ == "__main__":
    main()

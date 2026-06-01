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

import numpy as np
import pandas as pd
import os
from graintrace.construct_voxel_mesh import VoxelMeshBuilder


## INPUTS ---------------------------------------------------
def main():  ## i dont like this but somehow this is what needed for multiprocessing to work

    filename = "mwe_data/synthetic_vms.csv"
    generate_synthetic = False
    ncore = 24

    save_folder = "test_demo_grid_segmentation"
    filename = "test_graph_segmentation_2d.csv"
    nx = 10000
    ny = 1000
    radius_elements_range = (300, 700)

    segmentation_prop = {
        "method": "graph",
        "params": {
            "misorientation_tol": 5.0,  # degrees
            "connectivity": 6,
            "grain_threshold_final": 10,
            # flood params
            "batch_norm": 1000,
            "grain_threshold": 100,
            "stop_count": 100,
        },
        "graph_params": {
            "segmenter": "leiden",
            "graph_mode": "grid",
            "manhattan_radius": 3,
            "grid_tol": 1e-6,
            "n_jobs": ncore,
            "weight_chunk_size": 1_000_000,
            "nodes_chunk": 1_000_000,
            "reduce_edges_topweights_k": 8,
            "seed": 42,
            "networkit_kwargs": {"gamma": 0.001},  # lower gamma less clusters
            "weight_cfg": {
                "mode": "rbf",
                "sigma": None,
                "sigma_auto": {
                    "sample_size": 20_000,
                    "random_state": 42,
                    "quantile": 0.5,
                },
                "power": 2.0,
            },
            "plot": True,
        },
    }

    sculpt_config = {
        "launcher": "/home/tranh/Progs/cubit_gov/bin/mpi/bin/mpiexec",
        "psculpt": "/home/tranh/Progs/cubit_gov/bin/psculpt",
        "epu": "/home/tranh/Progs/cubit_gov/bin/epu",
        "nprocs": int(ncore),
        "environment": {
            "OPAL_LIBDIR": "/home/tranh/Progs/cubit_gov/bin/mpi/lib",
            "OPAL_PREFIX": "/home/tranh/Progs/cubit_gov/bin/mpi",
        },
    }

    sculpt_options = (
        "--void_mat",
        "0",
    )

    def generate_layered_euler_csv(
        path: str,
        nx: int = 80,
        ny: int = 80,
        n_layers: int = 2,
        rare_patches: int = 6,
        random_state: int = 0,
        radius_range: tuple[int, int] = (4, 10),  # number of elements
        base_euler_spread_deg: float = 2.0,
        patch_euler_spread_deg: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(random_state)

        # regular 2D grid embedded in 3D
        x_coords = np.linspace(0.0, 1.0, nx)
        y_coords = np.linspace(0.0, 1.0, ny)

        # baseline layer orientations in degrees
        # each row-band gets one base orientation
        layer_bases = []
        for _ in range(n_layers):
            layer_bases.append(
                np.array(
                    [
                        rng.uniform(0.0, 360.0),  # phi1
                        rng.uniform(0.0, 180.0),  # Phi
                        rng.uniform(0.0, 360.0),  # phi2
                    ],
                    dtype=float,
                )
            )

        layer_height = ny // n_layers
        layer_index = np.zeros(ny, dtype=int)
        for i in range(n_layers):
            start = i * layer_height
            end = ny if i == n_layers - 1 else (i + 1) * layer_height
            layer_index[start:end] = i

        # base Euler field
        euler_field = np.zeros((ny, nx, 3), dtype=float)
        for j in range(ny):
            base = layer_bases[layer_index[j]]
            noise = rng.normal(0.0, base_euler_spread_deg, size=(nx, 3))
            euler_field[j, :, :] = base[None, :] + noise

        # add rare circular patches with distinct orientations
        yy, xx = np.ogrid[:ny, :nx]
        for _ in range(rare_patches):
            cx = int(rng.integers(0, nx))
            cy = int(rng.integers(0, ny))
            radius = int(rng.integers(radius_range[0], radius_range[1]))

            patch_base = np.array(
                [
                    rng.uniform(0.0, 360.0),
                    rng.uniform(0.0, 180.0),
                    rng.uniform(0.0, 360.0),
                ],
                dtype=float,
            )

            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            n_mask = int(mask.sum())
            patch_noise = rng.normal(0.0, patch_euler_spread_deg, size=(n_mask, 3))
            euler_field[mask] = patch_base[None, :] + patch_noise

        # wrap angles to conventional degree ranges
        euler_field[..., 0] = np.mod(euler_field[..., 0], 360.0)  # phi1
        euler_field[..., 1] = np.clip(euler_field[..., 1], 0.0, 180.0)  # Phi
        euler_field[..., 2] = np.mod(euler_field[..., 2], 360.0)  # phi2

        rows = []
        element_id = 0
        for j in range(ny):
            for i in range(nx):
                element_id += 1
                rows.append(
                    {
                        "id": element_id,
                        "x": float(x_coords[i]),
                        "y": float(y_coords[j]),
                        "z": 0.0,
                        "phi1": float(euler_field[j, i, 0]),
                        "Phi": float(euler_field[j, i, 1]),
                        "phi2": float(euler_field[j, i, 2]),
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    ## MAIN ---------------------------------------------------

    os.makedirs(save_folder, exist_ok=True)
    filename = os.path.join(save_folder, filename)

    if generate_synthetic:

        print("Generating synthetic Euler-angle data...")
        generate_layered_euler_csv(
            path=filename,
            nx=nx,
            ny=ny,
            n_layers=3,
            rare_patches=12,
            random_state=42,
            radius_range=radius_elements_range,
            base_euler_spread_deg=2.0,
            patch_euler_spread_deg=0.5,
        )

    import time

    start = time.time()

    builder = VoxelMeshBuilder(
        file_path=filename,
        save_dir=save_folder,
        euler_cols=("phi1", "Phi", "phi2"),
        angle_convention="bunge",
        angle_type="degrees",
        symmetry="432",
    )

    merged_path = builder.reconstruct(
        segmentation=segmentation_prop,
        apply_smoothing=False,
    )

    end = time.time()
    elapsed = end - start
    print(f"segmentaion took {elapsed:.2f} seconds")

    fda

    print(f"\nReconstruction complete: {merged_path}\n")

    mesh_path = builder.mesh(
        sculpt_config=sculpt_config,
        sculpt_options=sculpt_options,
        merged_grid=merged_path,
    )
    print(f"Meshing complete: {mesh_path}")

    # plot 2D original and segmentation to check
    import matplotlib.pyplot as plt

    df_in = pd.read_csv(filename)
    merged = np.load(merged_path)

    # reconstruct x-y index maps from the original csv
    xs = np.sort(df_in["x"].unique())
    ys = np.sort(df_in["y"].unique())
    x_map = {v: i for i, v in enumerate(xs)}
    y_map = {v: i for i, v in enumerate(ys)}

    nx_plot = len(xs)
    ny_plot = len(ys)

    phi1_grid = np.full((ny_plot, nx_plot), np.nan, dtype=float)
    Phi_grid = np.full((ny_plot, nx_plot), np.nan, dtype=float)
    phi2_grid = np.full((ny_plot, nx_plot), np.nan, dtype=float)

    for _, row in df_in.iterrows():
        i = x_map[row["x"]]
        j = y_map[row["y"]]
        phi1_grid[j, i] = float(row["phi1"])
        Phi_grid[j, i] = float(row["Phi"])
        phi2_grid[j, i] = float(row["phi2"])

    # merged grid has shape (nx, ny, nz, 7); for this synthetic case nz=1
    phase_grid = merged[:, :, 0, 0].T
    phi1_seg_grid = merged[:, :, 0, 1].T
    Phi_seg_grid = merged[:, :, 0, 2].T
    phi2_seg_grid = merged[:, :, 0, 3].T

    plot_dir = os.path.join(save_folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # segmentation map
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        phase_grid,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        aspect="equal",
    )
    ax.set_title("Final segmented phase IDs")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="phase id")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "segmentation_phase_map.png"), dpi=300)
    plt.close(fig)

    print("identify {} unique segments".format(np.unique(phase_grid).size))

    for arr, name in [
        (phi1_grid, "phi1"),
        (Phi_grid, "Phi"),
        (phi2_grid, "phi2"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            aspect="equal",
        )
        ax.set_title(f"Original {name} (deg)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label=f"{name} (deg)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"original_{name}.png"), dpi=300)
        plt.close(fig)

    for arr, name in [
        (phi1_seg_grid, "phi1"),
        (Phi_seg_grid, "Phi"),
        (phi2_seg_grid, "phi2"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            aspect="equal",
        )
        ax.set_title(f"Segmented {name} stored on grid")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label=f"{name} (deg)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"segmented_{name}.png"), dpi=300)
        plt.close(fig)

    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()

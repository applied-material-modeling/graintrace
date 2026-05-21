import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from graintrace.construct_voxel_mesh import VoxelMeshBuilder


def make_unique_eulers(n, rng):
    return rng.uniform(0.0, np.pi / 2.0, size=(n, 3))


def dilate_mask(grid_mask, radius_cells):
    if radius_cells <= 0:
        return grid_mask.copy()

    out = grid_mask.copy()
    ny, nx = grid_mask.shape

    ys, xs = np.where(grid_mask)
    for y0, x0 in zip(ys, xs):
        y1 = max(0, y0 - radius_cells)
        y2 = min(ny, y0 + radius_cells + 1)
        x1 = max(0, x0 - radius_cells)
        x2 = min(nx, x0 + radius_cells + 1)

        for yy in range(y1, y2):
            for xx in range(x1, x2):
                if (yy - y0) ** 2 + (xx - x0) ** 2 <= radius_cells**2:
                    out[yy, xx] = True
    return out


def assign_top_anisotropic_grains(
    df,
    top_mask,
    y_interface_top,
    total_height,
    nx_total,
    dl,
    n_grains,
    t_interface,
    rng_seed=0,
    ax=1.0,
    ay=6.0,
):
    rng = np.random.default_rng(rng_seed)

    x_min = df["x"].min()
    x_max = df["x"].max()
    top_height = total_height - y_interface_top

    # 2 or 3 random seed layers in y
    n_y_layers = int(rng.integers(2, 4))

    layer_counts = np.full(n_y_layers, n_grains // n_y_layers, dtype=int)
    layer_counts[: n_grains % n_y_layers] += 1
    rng.shuffle(layer_counts)

    layer_edges = np.linspace(y_interface_top, total_height, n_y_layers + 1)

    seed_x_all = []
    seed_y_all = []

    for j in range(n_y_layers):
        n_j = layer_counts[j]
        if n_j == 0:
            continue

        y0 = layer_edges[j]
        y1 = layer_edges[j + 1]

        # evenly spaced in x with small jitter
        pitch = (x_max + dl - x_min) / n_j
        seed_x = np.linspace(
            x_min + 0.5 * pitch,
            x_max + dl - 0.5 * pitch,
            n_j,
            endpoint=True,
        )
        seed_x += rng.uniform(-0.2 * pitch, 0.2 * pitch, size=n_j)
        seed_x = np.clip(seed_x, x_min, x_max)

        seed_y = rng.uniform(y0, y1, size=n_j)

        seed_x_all.append(seed_x)
        seed_y_all.append(seed_y)

    seed_x = np.concatenate(seed_x_all)
    seed_y = np.concatenate(seed_y_all)

    seeds = np.column_stack([seed_x, seed_y])
    grain_eulers = make_unique_eulers(len(seeds), rng)
    grain_cell_ids = 4 + np.arange(len(seeds))

    # build top-layer grid
    top_ny = int(np.ceil(top_height / dl))
    x_vals = np.arange(nx_total) * dl
    y_vals = y_interface_top + np.arange(top_ny) * dl

    xx, yy = np.meshgrid(x_vals, y_vals, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    dx = pts[:, None, 0] - seeds[None, :, 0]
    dy = pts[:, None, 1] - seeds[None, :, 1]
    dist2 = (dx / ax) ** 2 + (dy / ay) ** 2

    labels = np.argmin(dist2, axis=1).reshape(top_ny, nx_total)

    # detect interface here
    boundary_core = np.zeros_like(labels, dtype=bool)
    boundary_core[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary_core[1:, :] |= labels[1:, :] != labels[:-1, :]

    radius_cells = 0
    interface_mask_grid = dilate_mask(boundary_core, radius_cells)

    # map top points back from grid
    top_df = df.loc[top_mask, ["x", "y"]].copy()

    ix = np.round(top_df["x"].to_numpy() / dl).astype(int)
    iy = np.floor((top_df["y"].to_numpy() - y_interface_top) / dl).astype(int)

    ix = np.clip(ix, 0, nx_total - 1)
    iy = np.clip(iy, 0, top_ny - 1)

    point_labels = labels[iy, ix]
    point_is_interface = interface_mask_grid[iy, ix]

    top_indices = df.index[top_mask]

    # assign unique top grains
    df.loc[top_indices, "CellID"] = grain_cell_ids[point_labels]
    df.loc[top_indices, ["Eul0", "Eul1", "Eul2"]] = grain_eulers[point_labels]

    # overwrite grain-boundary interface
    interface_indices = top_indices[point_is_interface]
    df.loc[interface_indices, "CellID"] = 2
    df.loc[interface_indices, ["Eul0", "Eul1", "Eul2"]] = [0.1, 0.5, 0.5]

    return df, seeds, labels, interface_mask_grid, n_y_layers


def generate_point_cloud(
    t_base,
    t_top,
    t_interface,
    nx_total=50,
    dl=1.0,
    n_top_grains=8,
    rng_seed=0,
    ax=1.0,
    ay=6.0,
):
    total_height = t_base + t_interface + t_top
    ny_total = int(np.ceil(total_height / dl)) + 1

    x = np.arange(nx_total) * dl
    y = np.arange(ny_total) * dl
    z_levels = np.arange(2) * dl

    xx, yy, zz = np.meshgrid(x, y, z_levels, indexing="ij")

    df = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
        }
    )

    df = df[df["y"] < total_height].copy()

    df["Eul0"] = 0.0
    df["Eul1"] = 0.0
    df["Eul2"] = 0.0
    df["CellID"] = 0

    y_base_top = t_base
    y_interface_top = t_base + t_interface

    base_mask = df["y"] < y_base_top
    interface_mask = (df["y"] >= y_base_top) & (df["y"] < y_interface_top)
    top_mask = (df["y"] >= y_interface_top) & (df["y"] < total_height)

    # 3 = base
    df.loc[base_mask, ["Eul0", "Eul1", "Eul2"]] = [0.5, 0.1, 0.5]
    df.loc[base_mask, "CellID"] = 3

    # 1 = base-to-top horizontal interface
    df.loc[interface_mask, ["Eul0", "Eul1", "Eul2"]] = [0.1, 0.5, 0.5]
    df.loc[interface_mask, "CellID"] = 1

    # top grains and top grain-boundary interface
    df, seeds, top_labels, top_interface_grid, n_y_layers = (
        assign_top_anisotropic_grains(
            df=df,
            top_mask=top_mask,
            y_interface_top=y_interface_top,
            total_height=total_height,
            nx_total=nx_total,
            dl=dl,
            n_grains=n_top_grains,
            t_interface=t_interface,
            rng_seed=rng_seed,
            ax=ax,
            ay=ay,
        )
    )

    return df, seeds, top_labels, top_interface_grid, n_y_layers


os.makedirs("test_pwmesh", exist_ok=True)

df, seeds, top_labels, top_interface_grid, n_y_layers = generate_point_cloud(
    t_base=4.0,
    t_interface=1.0,
    t_top=20.0,
    nx_total=100,
    dl=0.5,
    n_top_grains=11,
    rng_seed=32,
    ax=1.0,
    ay=4.0,
)

print(f"Used {n_y_layers} seed layers in y")

df.to_csv("test_pwmesh/point_cloud_with_attrs.csv", index=False)

plt.figure(figsize=(10, 6))
for cid in sorted(df["CellID"].unique()):
    mask = df["CellID"] == cid
    plt.scatter(
        df.loc[mask, "x"],
        df.loc[mask, "y"],
        s=10,
        alpha=0.5,
        label=f"CellID {cid}",
    )

# plt.scatter(seeds[:, 0], seeds[:, 1], marker="x", s=80, label="Seeds")
plt.title("Generated Point Cloud")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.legend(ncol=2, fontsize=8)
plt.savefig("test_pwmesh/point_cloud.png")
plt.close()

### Called Sculpt for meshing

ncore = 4
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
    # "--adapt",
    # "-A",
    # "7",
    # "-df",
    # "1",
    # "-S",
    # "3",
    # "-CS",
    # "4",
    "--void_mat",
    "0",
)

voxel_builder = VoxelMeshBuilder(
    file_path="test_pwmesh/point_cloud_with_attrs.csv",
    save_dir="test_pwmesh",
    euler_cols=["Eul0", "Eul1", "Eul2"],
    cell_id_col="CellID",
    angle_convention="bunge",
    angle_type="radians",
    symmetry="432",
)

grid_path = voxel_builder.reconstruct(apply_smoothing=False)
print(f"\nReconstruction complete: {grid_path}\n")

mesh_path = voxel_builder.mesh(
    sculpt_config=sculpt_config,
    sculpt_options=sculpt_options,
    merged_grid=grid_path,
)
print(f"Meshing complete: {mesh_path}")

import shutil

# Copy the mesh to the output folder.
destination_mesh_path = os.path.join("test_pwmesh", "simulations")
os.makedirs(destination_mesh_path, exist_ok=True)
shutil.copy2(mesh_path, destination_mesh_path)

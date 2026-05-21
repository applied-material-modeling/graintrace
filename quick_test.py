import os
import time
import functools
import numpy as np
import pandas as pd

from graph_spatial_cluster import GraphSpatialCluster
from similarity_metric_library import (
    make_misorientation_dist_edges,
    misorientation_distance,
)
from user_data_class import SimilarityMetric, WeightConfig


def generate_layered_euler_csv(
    path: str,
    nx: int = 80,
    ny: int = 80,
    n_layers: int = 2,
    rare_patches: int = 6,
    random_state: int = 0,
    radius_range: tuple[int, int] = (4, 10),
    base_euler_spread_deg: float = 2.0,
    patch_euler_spread_deg: float = 1.0,
) -> None:
    rng = np.random.default_rng(random_state)

    x_coords = np.linspace(0.0, 1.0, nx)
    y_coords = np.linspace(0.0, 1.0, ny)

    layer_bases = []
    for _ in range(n_layers):
        layer_bases.append(
            np.array(
                [
                    rng.uniform(0.0, 360.0),
                    rng.uniform(0.0, 180.0),
                    rng.uniform(0.0, 360.0),
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

    euler_field = np.zeros((ny, nx, 3), dtype=float)
    for j in range(ny):
        base = layer_bases[layer_index[j]]
        noise = rng.normal(0.0, base_euler_spread_deg, size=(nx, 3))
        euler_field[j, :, :] = base[None, :] + noise

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

    euler_field[..., 0] = np.mod(euler_field[..., 0], 360.0)
    euler_field[..., 1] = np.clip(euler_field[..., 1], 0.0, 180.0)
    euler_field[..., 2] = np.mod(euler_field[..., 2], 360.0)

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

    pd.DataFrame(rows).to_csv(path, index=False)


# -----------------------------
# test setup
# -----------------------------
os.makedirs("test_graph_direct", exist_ok=True)
csv_path = "test_graph_direct/synthetic_euler.csv"

generate_layered_euler_csv(
    path=csv_path,
    nx=1000,  # change this
    ny=1000,  # change this
    n_layers=3,
    rare_patches=20,
    random_state=42,
    radius_range=(30, 70),
    base_euler_spread_deg=2.0,
    patch_euler_spread_deg=0.5,
)

spec = SimilarityMetric(
    name="misorientation",
    feature_cols=["phi1", "Phi", "phi2"],
    func=functools.partial(
        misorientation_distance,
        angle_convention="bunge",
        input_angle_type="degrees",
        symmetry="432",
        output_unit="degrees",
    ),
    dist_edges=make_misorientation_dist_edges(
        angle_convention="bunge",
        input_angle_type="degrees",
        symmetry="432",
        output_unit="degrees",
    ),
)

gsc = GraphSpatialCluster(
    csv_path=csv_path,
    id_col="id",
    coord_cols=("x", "y", "z"),
)

t0 = time.time()
res = gsc.run(
    spec=spec,
    graph_mode="grid",
    manhattan_radius=2,
    grid_tol=1e-6,
    n_jobs=10,  # test 1, then try 2, 4, 8
    weight_chunk_size=1_000_000,  # test 10_000 / 50_000 / 100_000
    segmenter="leiden",
    seed=42,
    return_labels=True,
    max_edge_distance=5.0,  # degrees
    weight_cfg=WeightConfig(
        mode="rbf",
        power=2.0,
        sigma_auto={"sample_size": 20_000, "random_state": 42, "quantile": 0.5},
    ),
    networkit_kwargs={"gamma": 1.0},
    reduce_edges_topweights_k=None,
)
t1 = time.time()

print(f"\nElapsed: {t1 - t0:.2f} s")
print("Extras:", res["extras"])
print(res["clusters"].head())

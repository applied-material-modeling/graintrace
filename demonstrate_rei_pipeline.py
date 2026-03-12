import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

from similarity_metric_library import SimilarityMetricLibrary
from user_data_class import SimilarityMetric, WeightConfig, RareCriteria

from rare_cluster_indicator import IdentifyRareClusters
from cluster_indicator import ClusterAnalysisIndicator

# INPUT
filename = "test_rei_pipeline/synthetic_vms.csv"

# graph spatial cluster parameters
gsc_csv_path = filename
gsc_id_col = "id"
gsc_coord_cols = ("x", "y", "z")

# graph build
gsc_graph_mode = "grid"  # "grid" | "knn" | "auto"
gsc_k = 120  # used only if graph_mode="knn"
gsc_grid_radius = 4  # manhattan radius for grid connectivity if graph_mode="grid"
gsc_grid_tol = 1e-6
reduce_edges_topweights_k = (
    20  # keep only top k edges per node by weight before clustering
)

# edge weights
gsc_eps = 1e-8
gsc_n_jobs = 12
gsc_weight_chunk_size = 500_000
weight_cfg = WeightConfig(
    mode="rbf",
    power=2.0,
    sigma=None,
    sigma_auto={"sample_size": 500_000, "random_state": 42, "quantile": 0.5},
)

# segmentation
gsc_segmenter: str = "leiden"  # "leiden" | "plm" | "plp"
gsc_seed: int = 42
graph_cluster_arguments = {"gamma": 1.0}
# for all parameters: https://networkit.github.io/dev-docs/python_api/community.html

# synthetic generation
generate_synthetic = True
nx = 20
ny = 20
nz = 20

# second-stage merge params (same semantics as your script)
threshold = 0.05
radius_elements_range = (3, 8)

# exporter / rarity selection
vtk_out = "test_rei_pipeline/rare_clusters.vtk"
rare_criteria = RareCriteria(
    selector=None,  # use size_quantile logic
    size_quantile=0.05,  # bottom 5% by merged cluster size (stage 2)
    min_size=1,
    max_rare=None,
)

os.makedirs(os.path.dirname(vtk_out), exist_ok=True)


# Synthetic data generation
def _slice_to_array_worker(args):
    k0, k1, x_coords, y_coords, z_coords, vm_field, seed = args
    rng = np.random.default_rng(seed)

    nx_ = x_coords.size
    ny_ = y_coords.size
    nk = k1 - k0
    n = nk * ny_ * nx_

    X = np.broadcast_to(x_coords[None, None, :], (nk, ny_, nx_))
    Y = np.broadcast_to(y_coords[None, :, None], (nk, ny_, nx_))
    Z = np.broadcast_to(z_coords[k0:k1, None, None], (nk, ny_, nx_))

    vm = vm_field[k0:k1, :, :].astype(np.float64, copy=False)

    sxx = vm + rng.normal(0.0, vm * 0.05, size=vm.shape)
    syy = rng.normal(0.0, vm * 0.02, size=vm.shape)
    szz = rng.normal(0.0, vm * 0.02, size=vm.shape)
    sxy = rng.normal(0.0, vm * 0.02, size=vm.shape)
    sxz = rng.normal(0.0, vm * 0.02, size=vm.shape)
    syz = rng.normal(0.0, vm * 0.02, size=vm.shape)

    out = np.empty((n, 9), dtype=np.float64)
    out[:, 0] = X.reshape(-1)
    out[:, 1] = Y.reshape(-1)
    out[:, 2] = Z.reshape(-1)
    out[:, 3] = sxx.reshape(-1)
    out[:, 4] = syy.reshape(-1)
    out[:, 5] = szz.reshape(-1)
    out[:, 6] = sxy.reshape(-1)
    out[:, 7] = sxz.reshape(-1)
    out[:, 8] = syz.reshape(-1)
    return k0, out


def generate_layered_vms_csv(
    path: str,
    nx: int = 20,
    ny: int = 20,
    nz: int = 20,
    n_layers: int = 2,
    rare_patches: int = 3,
    vm_low: float = 50.0,
    vm_high: float = 200.0,
    random_state: int = 0,
    radius_range: tuple[int, int] = (4, 10),
    gsc_n_jobs: int = 1,
    z_chunk: int = 4,
) -> None:

    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    if radius_range[0] < 1 or radius_range[1] <= radius_range[0]:
        raise ValueError("radius_range must be (min_radius>=1, max_radius>min_radius)")

    rng = np.random.default_rng(random_state)

    x_coords = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    y_coords = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    z_coords = np.linspace(0.0, 1.0, nz, dtype=np.float64)

    layer_vm_levels = np.linspace(vm_low, vm_high, n_layers, dtype=np.float64)

    layer_thickness = max(1, nz // n_layers)
    layer_index = np.empty(nz, dtype=np.int64)
    for i in range(n_layers):
        start = i * layer_thickness
        end = nz if i == n_layers - 1 else min(nz, (i + 1) * layer_thickness)
        layer_index[start:end] = i

    vm_field = layer_vm_levels[layer_index][:, None, None] * np.ones(
        (nz, ny, nx), dtype=np.float64
    )

    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    for _ in range(rare_patches):
        cx = int(rng.integers(low=0, high=nx))
        cy = int(rng.integers(low=0, high=ny))
        cz = int(rng.integers(low=0, high=nz))
        radius = int(rng.integers(low=radius_range[0], high=radius_range[1]))
        vm_boost = float(rng.uniform(1.5, 2.5))

        mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius**2
        vm_field[mask] *= vm_boost

    tasks = []
    for k0 in range(0, nz, z_chunk):
        k1 = min(nz, k0 + z_chunk)
        tasks.append(
            (k0, k1, x_coords, y_coords, z_coords, vm_field, random_state + 10_000 + k0)
        )

    if gsc_n_jobs is None or gsc_n_jobs < 2:
        parts = [_slice_to_array_worker(t) for t in tasks]
    else:
        ctx = mp.get_context()
        with ctx.Pool(processes=gsc_n_jobs) as pool:
            parts = list(pool.imap(_slice_to_array_worker, tasks, chunksize=1))

    parts.sort(key=lambda t: t[0])
    data = np.concatenate([p[1] for p in parts], axis=0)  # (N,9)

    N = data.shape[0]
    ids = np.arange(1, N + 1, dtype=np.int64)

    df = pd.DataFrame(
        {
            "id": ids,
            "x": data[:, 0],
            "y": data[:, 1],
            "z": data[:, 2],
            "sxx": data[:, 3],
            "syy": data[:, 4],
            "szz": data[:, 5],
            "sxy": data[:, 6],
            "sxz": data[:, 7],
            "syz": data[:, 8],
        }
    )
    df.to_csv(path, index=False)


# Main

os.makedirs(os.path.dirname(filename), exist_ok=True)

if generate_synthetic:
    print("Generating synthetic VMS data...")
    generate_layered_vms_csv(
        path=filename,
        nx=nx,
        ny=ny,
        nz=nz,
        n_layers=1,
        rare_patches=8,
        vm_low=50.0,
        vm_high=200.0,
        random_state=42,
        radius_range=radius_elements_range,
        gsc_n_jobs=gsc_n_jobs,
        z_chunk=gsc_weight_chunk_size,
    )


PICK_CLUSTER_RESTART = False
FINAL_CLUSTERING_RESTART = False
GRAPH_SEGMENTATION_RESTART = True


metric_lib = SimilarityMetricLibrary()
spec = metric_lib.von_mises_stress()

spec_reduced = SimilarityMetric(
    name=spec.name + "_mean",
    feature_cols=[f"{c}_mean" for c in spec.feature_cols],
    func=spec.func,
)

irc = IdentifyRareClusters(
    input_csv_path=filename,
    id_col=gsc_id_col,
    coord_cols=gsc_coord_cols,
)

base = os.path.splitext(filename)[0]
graph_cluster_out = base + "_reduced.csv"
bundle_checkpoint = base + "_bundle.pkl"
gsc_labels_path = base + "_reduced_gsc_labels.npy"
gsc_ckpt_base = base + "_gsc_ckpt"

# warning messages
if (
    sum([PICK_CLUSTER_RESTART, FINAL_CLUSTERING_RESTART, GRAPH_SEGMENTATION_RESTART])
    > 1
):
    print(
        "\nMultiple restart flags enabled. Priority order:\n"
        "   1) PICK_CLUSTER_RESTART\n"
        "   2) FINAL_CLUSTERING_RESTART\n"
        "   3) GRAPH_SEGMENTATION_RESTART\n"
        "The first valid checkpoint found will be used.\n"
        "Valid checkpoint = all required files for that stage exist.\n"
        "Required files status:\n"
    )

    print("  PICK_CLUSTER_RESTART (bundle):")
    print(
        f"     {bundle_checkpoint}  "
        f"[{'FOUND' if os.path.exists(bundle_checkpoint) else 'MISSING'}]"
    )

    print("  FINAL_CLUSTERING_RESTART (reduced CSV + GSC labels):")
    print(
        f"     {graph_cluster_out}  "
        f"[{'FOUND' if os.path.exists(graph_cluster_out) else 'MISSING'}]"
    )
    print(
        f"     {gsc_labels_path}  "
        f"[{'FOUND' if os.path.exists(gsc_labels_path) else 'MISSING'}]"
    )

    print("  GRAPH_SEGMENTATION_RESTART (edges/weights/meta):")
    edges_f = gsc_ckpt_base + ".edges.npy"
    weights_f = gsc_ckpt_base + ".weights.npy"
    meta_f = gsc_ckpt_base + ".meta.json"

    print(f"     {edges_f}  " f"[{'FOUND' if os.path.exists(edges_f) else 'MISSING'}]")
    print(
        f"     {weights_f}  " f"[{'FOUND' if os.path.exists(weights_f) else 'MISSING'}]"
    )
    print(f"     {meta_f}  " f"[{'FOUND' if os.path.exists(meta_f) else 'MISSING'}]\n")

if not any(
    [PICK_CLUSTER_RESTART, FINAL_CLUSTERING_RESTART, GRAPH_SEGMENTATION_RESTART]
):
    print("No restart requested → full pipeline will run from scratch.")


start_time = time.time()

bundle = None

if PICK_CLUSTER_RESTART and os.path.exists(bundle_checkpoint):
    print(f"Loading bundle checkpoint: {bundle_checkpoint}")
    bundle = pd.read_pickle(bundle_checkpoint)

if bundle is None and FINAL_CLUSTERING_RESTART:
    if os.path.exists(graph_cluster_out) and os.path.exists(gsc_labels_path):
        print("Resume from final clustering checkpoint")

        input_df = pd.read_csv(filename)
        gsc_labels = np.load(gsc_labels_path, mmap_mode="r")

        indicator = ClusterAnalysisIndicator(
            csv_path=graph_cluster_out,
            id_col="cluster_id",
            coord_cols=gsc_coord_cols,
        )

        ind_out = indicator.run(
            method_type="scipy_hierarchical",
            spec=spec_reduced,
            threshold=threshold,
            method="average",
            criterion="distance",
            dendrogram_path="test_rei_pipeline/dendrogram.png",
            minimal_return=False,
        )

        bundle = {
            "input_df": input_df,
            "gsc_labels": np.asarray(gsc_labels, dtype=np.int64),
            "reduced_csv_path": graph_cluster_out,
            "gsc_extras": {"labels_path": gsc_labels_path},
            "indicator_points_df": ind_out["points"],
            "indicator_clusters_df": ind_out["clusters"],
            "indicator_extras": ind_out.get("extras", {}),
        }
    else:
        print(
            "FINAL_CLUSTERING_RESTART requested, but reduced CSV or GSC labels missing."
        )

if bundle is None:
    gsc, indicator = irc.make_stage_objects(graph_cluster_out=graph_cluster_out)

    weights_ckpt_exists = (
        os.path.exists(gsc_ckpt_base + ".edges.npy")
        and os.path.exists(gsc_ckpt_base + ".weights.npy")
        and os.path.exists(gsc_ckpt_base + ".meta.json")
    )

    bundle = irc.run_clustering(
        gsc=gsc,
        indicator=indicator,
        reduced_csv_path=graph_cluster_out,
        gsc_run_kwargs=dict(
            spec=spec,
            graph_mode=gsc_graph_mode,
            k=gsc_k,
            manhattan_radius=gsc_grid_radius,
            grid_tol=gsc_grid_tol,
            n_jobs=gsc_n_jobs,
            weight_chunk_size=gsc_weight_chunk_size,
            segmenter=gsc_segmenter,
            seed=gsc_seed,
            weight_cfg=weight_cfg,
            reduce_edges_topweights_k=reduce_edges_topweights_k,
            networkit_kwargs=graph_cluster_arguments,
            checkpoint_base_path=gsc_ckpt_base,
            resume_from_checkpoint=(GRAPH_SEGMENTATION_RESTART and weights_ckpt_exists),
        ),
        indicator_run_kwargs=dict(
            method_type="scipy_hierarchical",
            spec=spec_reduced,
            threshold=threshold,
            method="average",
            criterion="distance",
            dendrogram_path="test_rei_pipeline/dendrogram.png",
        ),
    )

    print(f"Saving bundle checkpoint to: {bundle_checkpoint}")
    pd.to_pickle(bundle, bundle_checkpoint)

out = irc.run_get_rare_cluster(
    bundle=bundle,
    criteria=rare_criteria,
    output_vtk_path=vtk_out,
    export_control="auto",
    background_block_id=1,
    first_rare_block_id=2,
    also_write_final_label=True,
)

elapsed = time.time() - start_time

print("\n--- Done ---")
print(f"Total elapsed time: {elapsed:.2f} seconds")
print("VTK exported:", out["output_vtk_path"])
print("Export mode:", out["export_mode"])
print("Rare clusters (merged labels):", out["rare_super_labels"])

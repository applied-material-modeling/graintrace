import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram
import multiprocessing as mp

from cluster_indicator import ClusterAnalysisIndicator
from similarity_metric_library import SimilarityMetricLibrary
from user_data_class import SimilarityMetric, WeightConfig

from graph_spatial_cluster import GraphSpatialCluster

# set code timer
import time

## INPUTS ---------------------------------------------------

filename = "test_speed_gsc/synthetic_vms.csv"

if_plot = False
second_step = False

## graph spatial cluster parameters
gsc_csv_path = filename
gsc_id_col = "id"
gsc_coord_cols = ("x", "y", "z")

# graph build
gsc_graph_mode = "grid"          # "grid" | "knn" | "auto"
gsc_k = 120                       # used only if graph_mode="knn"
gsc_grid_radius = 4        # manhattan radius for grid connectivity if graph_mode="grid"
gsc_grid_tol = 1e-6
reduce_edges_topweights_k = 20   # if not None, keep only top k edges per node by weight before clustering          

# edge weights
gsc_eps = 1e-8
gsc_n_jobs = 12
gsc_weight_chunk_size = 500_000 # how many edges to process in a chunk when computing weights
weight_cfg = WeightConfig(
    mode="rbf",
    power=2.0,
    sigma=None,  # if None, will be set to quantile distance of the graph edges
    sigma_auto={"sample_size": 500_000, 
                "random_state": 42,
                "quantile": 0.5},  # if sigma is None, sample this many edges to estimate quantile distance
)

# segmentation
gsc_segmenter: str = "leiden"           # leiden" | "plm" | "plp"
gsc_seed: int = 42
graph_cluster_arguments = {"gamma": 1.0}
## for all parameters: https://networkit.github.io/dev-docs/python_api/community.html


generate_synthetic = True
nx = 300
ny = 1000
nz = 300

threshold = 0.1
radius_elements_range = (30, 80) # (60,160) #(30, 80)

if generate_synthetic:
    ## generate synthetic cluster data for testing purpose
    def _slice_to_array_worker(args):
        k0, k1, x_coords, y_coords, z_coords, vm_field, seed = args
        rng = np.random.default_rng(seed)

        nx = x_coords.size
        ny = y_coords.size
        nk = k1 - k0
        n = nk * ny * nx

        # Broadcast coordinate grids for this slab: (nk, ny, nx)
        X = np.broadcast_to(x_coords[None, None, :], (nk, ny, nx))
        Y = np.broadcast_to(y_coords[None, :, None], (nk, ny, nx))
        Z = np.broadcast_to(z_coords[k0:k1, None, None], (nk, ny, nx))

        vm = vm_field[k0:k1, :, :].astype(np.float64, copy=False)

        # Approx uniaxial: sxx ≈ vm_target, small noise elsewhere
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
        radius_range: tuple[int, int] = (4, 10),  # in grid cells
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

        # Regular grid in x,y,z
        x_coords = np.linspace(0.0, 1.0, nx, dtype=np.float64)
        y_coords = np.linspace(0.0, 1.0, ny, dtype=np.float64)
        z_coords = np.linspace(0.0, 1.0, nz, dtype=np.float64)

        # Baseline layer VM levels (evenly spaced between vm_low and vm_high)
        layer_vm_levels = np.linspace(vm_low, vm_high, n_layers, dtype=np.float64)

        # Assign each z-slice to a layer (layers along z)
        layer_thickness = max(1, nz // n_layers)
        layer_index = np.empty(nz, dtype=np.int64)
        for i in range(n_layers):
            start = i * layer_thickness
            end = nz if i == n_layers - 1 else min(nz, (i + 1) * layer_thickness)
            layer_index[start:end] = i

        # Base VM field: shape (nz, ny, nx)
        vm_field = layer_vm_levels[layer_index][:, None, None] * np.ones((nz, ny, nx), dtype=np.float64)

        # Add rare high-VM spherical patches (in index space)
        zz, yy, xx = np.ogrid[:nz, :ny, :nx]
        for _ in range(rare_patches):
            cx = int(rng.integers(low=0, high=nx))
            cy = int(rng.integers(low=0, high=ny))
            cz = int(rng.integers(low=0, high=nz))
            radius = int(rng.integers(low=radius_range[0], high=radius_range[1]))
            vm_boost = float(rng.uniform(1.5, 2.5))

            mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2
            vm_field[mask] *= vm_boost

        # Convert VM field to stress components (approx uniaxial), chunked over z
        tasks = []
        for k0 in range(0, nz, z_chunk):
            k1 = min(nz, k0 + z_chunk)
            # stable per-chunk seed for reproducibility
            tasks.append((k0, k1, x_coords, y_coords, z_coords, vm_field, random_state + 10_000 + k0))

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

        df = pd.DataFrame({
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
        })

        df.to_csv(path, index=False)


    ## main ---------------------------------------------------------- ##
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
        gsc_n_jobs = gsc_n_jobs,
        z_chunk = gsc_weight_chunk_size,
    )

print("Running clustering analysis...")

start_time = time.time()

## perform clustering
metric_lib = SimilarityMetricLibrary()
spec = metric_lib.von_mises_stress()

graph_cluster_out = os.path.splitext(filename)[0] + "_reduced.csv"
gsc = GraphSpatialCluster(
    csv_path=gsc_csv_path,
    id_col=gsc_id_col,
    coord_cols=gsc_coord_cols,
)

gsc_res = gsc.run(
    spec=spec,
    graph_mode=gsc_graph_mode,
    k=gsc_k,
    manhattan_radius=gsc_grid_radius,
    grid_tol=gsc_grid_tol,
    n_jobs=gsc_n_jobs,
    weight_chunk_size=gsc_weight_chunk_size,
    segmenter=gsc_segmenter,
    seed=gsc_seed,
    output_csv_path=graph_cluster_out,
    return_labels=if_plot,
    weight_cfg=weight_cfg,
    reduce_edges_topweights_k=reduce_edges_topweights_k,
    ## networkit_kwargs can be used to pass additional arguments to the chosen networkit community detection algorithm (e.g. resolution parameter for Leiden) --- IGNORE ---
    networkit_kwargs=graph_cluster_arguments,
)

print("Reduced CSV saved:", gsc_res["csv_path"])
print("GSC extras:", gsc_res["extras"])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"GSC elapsed time: {elapsed_time:.2f} seconds")

if second_step:
    # This goes through the second steps to merge the clusters globally
    indicator = ClusterAnalysisIndicator(gsc_res["csv_path"],
                                        id_col="cluster_id",
                                        coord_cols=("x", "y", "z"))

    spec_reduced = SimilarityMetric(
        name=spec.name + "_mean",
        feature_cols=[f"{c}_mean" for c in spec.feature_cols],
        func=spec.func,
    )

    out = indicator.run(
        method_type="scipy_hierarchical",
        spec=spec_reduced,
        threshold=threshold,
        method = "average",
        criterion = "distance",
        dendrogram_path="test_speed_gsc/dendrogram.png",
        minimal_return=False,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")

### PLOTTING TO CHECK
# plot to check:
if if_plot:
    labels_gsc = gsc_res["extras"]["labels"]          # point-level cluster id from NetworKit
    points_df = pd.read_csv(filename)

    # dendrogram info (from CAI scipy_hierarchical extras)
    dinfo = out["extras"]["dendrogram"]
    leaf_order = dinfo["leaves"]                      # indices of samples in dendrogram order (0..n_reduced-1)
    leaf_colors = dinfo["leaves_color_list"]          # same length as leaves; strings like 'C0', '#rrggbb', etc.

    reduced_points = out["points"]                    # reduced CSV rows + final cluster_label (for scipy_hierarchical too)
    # IMPORTANT: leaf indices refer to row positions used in linkage input.
    # In your CAI scipy_hierarchical implementation, linkage is built from X in df row order.
    # So leaf indices map to reduced_points row positions.

    # ---------------- von Mises stress image ----------------
    sxx = points_df["sxx"].to_numpy()
    syy = points_df["syy"].to_numpy()
    szz = points_df["szz"].to_numpy()
    sxy = points_df["sxy"].to_numpy()
    sxz = points_df["sxz"].to_numpy()
    syz = points_df["syz"].to_numpy()

    term1 = ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) / 2
    term2 = 3 * (sxy**2 + sxz**2 + syz**2)
    vm = np.sqrt(term1 + term2)

    nx = len(np.unique(points_df["x"]))
    ny = len(np.unique(points_df["y"]))
    vm_img = vm.reshape(ny, nx)

    # ---------------- panel 2: GSC labels ----------------
    unique_gsc = np.unique(labels_gsc)
    K_gsc = unique_gsc.size
    map_gsc = {lab: i for i, lab in enumerate(unique_gsc)}
    mapped_gsc = np.vectorize(map_gsc.get)(labels_gsc).reshape(ny, nx)

    cmap_gsc = ListedColormap(plt.cm.hsv(np.linspace(0, 1, K_gsc, endpoint=False)))
    norm_gsc = BoundaryNorm(np.arange(K_gsc + 1) - 0.5, K_gsc)

    # ---------------- panel 3: FINAL merged labels, colored by dendrogram leaf colors ----------------
    # Step A: build mapping from reduced row index -> dendrogram leaf color (RGBA)
    rowidx_to_rgba = {}
    for row_idx, c in zip(leaf_order, leaf_colors):
        rowidx_to_rgba[int(row_idx)] = to_rgba(c)

    # Step B: map reduced "cluster_id" -> its dendrogram color via the reduced row index
    # reduced_points index is the row order used for linkage if unchanged; enforce position index explicitly:
    reduced_points = reduced_points.reset_index(drop=True)

    cluster_id_to_rgba = {}
    for row_idx in range(reduced_points.shape[0]):
        cid = int(reduced_points.loc[row_idx, "cluster_id"])
        cluster_id_to_rgba[cid] = rowidx_to_rgba.get(row_idx, (0.0, 0.0, 0.0, 1.0))  # fallback black

    # Step C: build an RGBA image for the original points by looking up each point's gsc cluster id
    final_rgba = np.empty((labels_gsc.size, 4), dtype=np.float32)
    for i, cid in enumerate(labels_gsc):
        final_rgba[i] = cluster_id_to_rgba[int(cid)]

    final_img = final_rgba.reshape(ny, nx, 4)

    # ---------------- plot ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(vm_img, origin="lower", cmap="viridis")
    axes[0].set_title("Von Mises Stress")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(mapped_gsc, origin="lower", cmap=cmap_gsc, norm=norm_gsc)
    axes[1].set_title("GSC Labels (pre-merge)")
    plt.colorbar(im2, ax=axes[1])

    axes[2].imshow(final_img, origin="lower")
    axes[2].set_title("Final (post-merge) colored by dendrogram leaves")

    plt.tight_layout()
    plt.savefig("test_speed_gsc/gsc_cluster_labels_check.png", dpi=300)
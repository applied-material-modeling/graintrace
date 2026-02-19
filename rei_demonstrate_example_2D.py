import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram

from cluster_indicator import ClusterAnalysisIndicator
from user_data_class import SimilarityMetricLibrary

## INPUTS ---------------------------------------------------

filename = "mwe_data/synthetic_vms.csv"
generate_synthetic = True

if generate_synthetic:
    ## generate synthetic cluster data for testing purpose
    def generate_layered_vms_csv(
        path: str,
        nx: int = 80,
        ny: int = 80,
        n_layers: int = 2,
        rare_patches: int = 3,
        vm_low: float = 50.0,
        vm_high: float = 200.0,
        random_state: int = 0,
        radius_range: tuple[int, int] = (4, 10), # number of elements
    ) -> None:

        rng = np.random.default_rng(random_state)

        # Regular grid in x,y
        x_coords = np.linspace(0.0, 1.0, nx)
        y_coords = np.linspace(0.0, 1.0, ny)

        # Define baseline layer VM levels (evenly spaced between vm_low and vm_high)
        layer_vm_levels = np.linspace(vm_low, vm_high, n_layers)

        # Assign each row (in y) to a layer
        layer_height = ny // n_layers
        layer_index = np.zeros(ny, dtype=int)
        for i in range(n_layers):
            start = i * layer_height
            end = ny if i == n_layers - 1 else (i + 1) * layer_height
            layer_index[start:end] = i

        # Base VM field
        vm_field = np.zeros((ny, nx), dtype=float)
        for j in range(ny):
            vm_field[j, :] = layer_vm_levels[layer_index[j]]

        # Add rare high-VM patches
        for _ in range(rare_patches):
            
            cx = rng.integers(low=0, high=nx)
            cy = rng.integers(low=0, high=ny)

            radius = rng.integers(low=radius_range[0], high=radius_range[1])

            vm_boost = rng.uniform(1.5, 2.5)

            yy, xx = np.ogrid[:ny, :nx]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

            vm_field[mask] *= vm_boost

        # Now convert VM field to actual stress components (approx uniaxial)
        rows = []
        element_id = 0
        for j in range(ny):
            for i in range(nx):
                element_id += 1
                x = x_coords[i]
                y = y_coords[j]
                z = 0.0

                vm_target = vm_field[j, i]

                # Approx uniaxial: sxx ≈ vm_target, small noise elsewhere
                sxx = vm_target + rng.normal(0.0, vm_target * 0.05)
                syy = rng.normal(0.0, vm_target * 0.02)
                szz = rng.normal(0.0, vm_target * 0.02)
                sxy = rng.normal(0.0, vm_target * 0.02)
                sxz = rng.normal(0.0, vm_target * 0.02)
                syz = rng.normal(0.0, vm_target * 0.02)

                rows.append(
                    {
                        "id": element_id,
                        "x": x,
                        "y": y,
                        "z": z,
                        "sxx": sxx,
                        "syy": syy,
                        "szz": szz,
                        "sxy": sxy,
                        "sxz": sxz,
                        "syz": syz,
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)


    ## main ---------------------------------------------------------- ##
    nx = 30
    ny = 30

    threshold = 0.01
    radius_elements_range = (2, 5)

    print("Generating synthetic VMS data...")

    generate_layered_vms_csv(
        path=filename,
        nx=nx,
        ny=ny,
        n_layers=1,
        rare_patches=10,
        vm_low=50.0,
        vm_high=200.0,
        random_state=42,
        radius_range=radius_elements_range,
    )

    print("Running clustering analysis...")

## perform clustering
indicator = ClusterAnalysisIndicator(filename, coord_cols=("x", "y", "z"))

metric_lib = SimilarityMetricLibrary()
spec = metric_lib.von_mises_stress()

result, linkage = indicator.run(
    method_type="scipy_hierarchical",
    spec=spec,
    threshold=threshold,
    method = "average",
    criterion = "distance",
)

# result = indicator.run(
#     method_type = "sklearn_dbscan",
#     spec = spec,
#     eps = 0.01,
#     min_samples = 2,
#     leaf_size = 10,
# )

print("Plotting results...")

def prepare_cluster_colormap(labels):

    unique_labels = np.unique(labels)
    K = len(unique_labels)

    # remap labels
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    label_norm = np.vectorize(lambda x: label_to_idx[x])(labels)

    # discrete K-color colormap
    base_cmap = plt.get_cmap("tab20")
    colors = base_cmap(np.linspace(0, 1, K))
    cmap = ListedColormap(colors)

    bounds = np.arange(K + 1) - 0.5
    norm = BoundaryNorm(bounds, K)

    return label_norm, unique_labels, cmap, norm

## quick visualization
def compute_von_mises(df: pd.DataFrame) -> np.ndarray:
    sxx = df["sxx"].to_numpy()
    syy = df["syy"].to_numpy()
    szz = df["szz"].to_numpy()
    sxy = df["sxy"].to_numpy()
    sxz = df["sxz"].to_numpy()
    syz = df["syz"].to_numpy()

    term1 = ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) / 2.0
    term2 = 3.0 * (sxy**2 + sxz**2 + syz**2)
    return np.sqrt(term1 + term2)

# compute fields
vm = compute_von_mises(result)                  # shape (nx*ny,)
labels = result["cluster_label"].to_numpy()     # shape (nx*ny,)

# reshape to 2D grid (row-major order: j over y, i over x)
vm_grid = vm.reshape(ny, nx)

# prepare colormap / normalized labels once
label_norm, unique_labels, cmap, norm = prepare_cluster_colormap(labels)
label_grid = label_norm.reshape(ny, nx)


###--------------- plotting the image and clusters----------------- ###
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# left: field
im0 = axes[0].imshow(vm_grid, origin="lower", aspect="equal", cmap="viridis")
axes[0].set_title("Von Mises Stress")
axes[0].set_xlabel("x index")
axes[0].set_ylabel("y index")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# right: cluster labels (discrete)
im1 = axes[1].imshow(
    label_grid,
    origin="lower",
    aspect="equal",
    cmap=cmap,
    norm=norm,
)
axes[1].set_title("Cluster Labels")
axes[1].set_xlabel("x index")
axes[1].set_ylabel("y index")

cbar = fig.colorbar(im1, ax=axes[1], ticks=np.arange(len(unique_labels)))
cbar.ax.set_yticklabels(unique_labels)  # show original cluster IDs

plt.tight_layout()
plt.savefig(
    "testing_during_code_not_upload_to_github/rei_demonstrate_example_clusters.png",
    dpi=300,
)
plt.close()

### ------------------- Plotting MDS ----------------------------- ###
fig, ax = plt.subplots(figsize=(6, 3))
X_1d = result["mds_1d"].to_numpy()
y = np.zeros_like(X_1d)

# use the SAME label_norm / cmap / norm
sc = ax.scatter(X_1d, y, c=label_norm, cmap=cmap, norm=norm, s=50)

ax.set_yticks([])
ax.set_xlabel("MDS coordinate (1D)")

fig.subplots_adjust(bottom=0.25)

cbar = fig.colorbar(
    sc,
    ax=ax,
    ticks=np.arange(len(unique_labels)),
    orientation="horizontal",
    pad=0.25
)
cbar.ax.set_xticklabels(unique_labels)
cbar.set_label("Cluster Labels")
plt.tight_layout()

fig.savefig(
    "testing_during_code_not_upload_to_github/rei_demonstrate_mds.png",
    dpi=300,
)
plt.close(fig)
### ------------------------------------------------------------- ###

### ------------------- Plotting Denogram ----------------------- ###

###--------------- plotting the image and clusters----------------- ###
# vm_grid: (ny, nx)
# color_grid will be derived from dendrogram colors
n = result.shape[0]
assert n == nx * ny

fig = plt.figure(figsize=(6.5, 5.5))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.8], width_ratios=[1.0, 0.9], wspace=0.4)

ax_vm      = fig.add_subplot(gs[0, 0])   # top-left
ax_cluster = fig.add_subplot(gs[0, 1])   # top-right
ax_dend    = fig.add_subplot(gs[1, :])   # bottom, spans both columns


im0 = ax_vm.imshow(vm_grid, origin="lower", aspect="equal", cmap="viridis")
ax_vm.set_title("Von Mises Stress")
fig.colorbar(im0, ax=ax_vm, fraction=0.046, pad=0.04)


dinfo = dendrogram(linkage, color_threshold=threshold, ax=ax_dend, no_labels=True)

## add horizontal line for threshold
ax_dend.axhline(y=threshold, color="k", linestyle="--", label="Threshold")

ax_dend.set_ylabel("Ultrametric distance")

leaf_order  = dinfo["leaves"]             # indices of samples (0..n-1)
leaf_colors = dinfo["leaves_color_list"]  # colors for each leaf

assert len(leaf_order) == n

# Map observation index -> RGBA
idx_to_color = {
    idx: mcolors.to_rgba(col)
    for idx, col in zip(leaf_order, leaf_colors)
}

colors_rgba = np.vstack([idx_to_color[i] for i in range(n)])  # (n, 4)
color_grid  = colors_rgba.reshape(ny, nx, 4)                  # row-major reshape

im2 = ax_cluster.imshow(color_grid, origin="lower", aspect="equal")
ax_cluster.set_title("Clusters (dendrogram colors)")

# fig.tight_layout()
fig.savefig(
    "testing_during_code_not_upload_to_github/rei_demonstrate_clusters_from_dendro.png",
    dpi=300,
)
plt.close(fig)


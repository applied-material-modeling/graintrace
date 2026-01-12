import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram

from cluster_indicator import ClusterAnalysisIndicator
from similarity_metric_library import SimilarityMetricLibrary

## INPUTS ---------------------------------------------------

save_folder = "cpfe_ff_nf_demonstrate_v2/rei"
filename = "cpfe_ff_nf_demonstrate_v2/simulation/simulation_out/grid_out/out_element_centroid_0005.csv"
threshold = 0.1

print("Running clustering analysis...")


## MAIN CODE ------------------------------------------------

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

## perform clustering
indicator = ClusterAnalysisIndicator(filename, coord_cols=("x", "y", "z"))

metric_lib = SimilarityMetricLibrary()
spec = metric_lib.misorientation()

# result, linkage = indicator.run(
#     method_type="scipy_hierarchical",
#     spec=spec,
#     threshold=threshold,
#     method = "average",
#     criterion = "distance",
# )

result = indicator.run(
    method_type = "sklearn_dbscan",
    spec = spec,
    eps = 0.01,
    min_samples = 2,
    leaf_size = 10,
)

asdf

print("Plotting results...")

###--------------- plotting the image and clusters----------------- ###
# probably save to mesh file somehow and use paraview for visualization
### ------------------------------------------------------------- ###

### ------------------- Plotting MDS ----------------------------- ###
fig, ax = plt.subplots(figsize=(6, 3))
X_1d = result["mds_1d"].to_numpy()
y = np.zeros_like(X_1d)

ax.set_yticks([])
ax.set_xlabel("MDS coordinate (1D)")

fig.subplots_adjust(bottom=0.25)

cbar = fig.colorbar(
    ax=ax,
    orientation="horizontal",
    pad=0.25
)

cbar.set_label("Cluster Labels")
plt.tight_layout()

fig.savefig(
    save_folder + "/rei_demonstrate_mds.png",
    dpi=300,
)
plt.close(fig)
### ------------------------------------------------------------- ###


###--------------- plotting the image and clusters----------------- ###
fig, ax_dend = plt.subplots(figsize=(6, 3))
dinfo = dendrogram(linkage, color_threshold=threshold, ax=ax_dend, no_labels=True)

## add horizontal line for threshold
ax_dend.axhline(y=threshold, color="k", linestyle="--", label="Threshold")
ax_dend.set_ylabel("Ultrametric distance")

leaf_order  = dinfo["leaves"]             # indices of samples (0..n-1)
leaf_colors = dinfo["leaves_color_list"]  # colors for each leaf

# Map observation index -> RGBA
idx_to_color = {
    idx: mcolors.to_rgba(col)
    for idx, col in zip(leaf_order, leaf_colors)
}

fig.tight_layout()
fig.savefig(
    save_folder + "/dendro.png",
    dpi=300,
)
plt.close(fig)


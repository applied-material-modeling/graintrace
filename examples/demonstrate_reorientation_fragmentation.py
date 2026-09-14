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

# One coherent story: reorientation -> fragmentation -> rare events.
#  (A) Simulation: grains REORIENT under load (IPF orientation tracking).
#  (B) Simulation: that reorientation FRAGMENTS grains into orientation sub-grains.
#      The 1x2 figure (real CPFE data) puts them side by side: LEFT every element of
#      each grain (how its orientation spreads), RIGHT the resulting fragmentation forks.
#  (C) Experiment: detect grain SPLITS across load steps for FF/NF/EBSD (synthetic,
#      mixed 1->1 / 1->2 / 1->3 with ground truth).
#  (D) REI capstone: flag the rare grains -- "the grains that split the most".
#
# OrientationTracker builds the tracks; FragmentationAnalyzer does the segmentation,
# split detection, and per-grain fragment counts; plot_postprocessing renders the IPF.

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from graintrace.simulation_postprocessing import SimulationResults, FieldFileNaming
from graintrace.ipf_orientation_tracking import OrientationTracker
from graintrace.fragmentation import FragmentationAnalyzer
from graintrace import orientation_helper as oh
from graintrace import plot_postprocessing as pp
from graintrace import rare_criteria_selection_library as rcs

## INPUT
## ------------------------------------
# Simulation = a REAL deforming CPFE run (true-mesh per-element mesh_out w/ block_id).
sim_block_csv = "mwe_data/cpfe_ff_fragmentation/out.csv"
sim_field_dir = "mwe_data/cpfe_ff_fragmentation/mesh_out"
hex_mesh = "mwe_data/cpfe_hex_fragmentation/mesh.e"  # SCULPT hex, for annotated Exodus
# Optional co-registered loaded hex run (its mesh.e + mesh_out with block_id + ori) for a
# FRAGMENT-colored Exodus matching the figure exactly. Leave None to instead color the bare
# shipped hex mesh by block id (a per-grain demo of the same RGB writer + palette).
hex_run_mesh = None
hex_run_block_csv = None
hex_run_field_dir = None
sim_grains = list(range(1, 11))
tol_deg = 5.0  # sub-grain misorientation tolerance
seg_gamma = 0.25  # Leiden resolution for intragranular sub-grain segmentation
min_subgrain_miso_deg = (
    1.0  # merge sub-grains closer than this (min sub-grain boundary)
)

# Experiment = synthetic FF/NF/EBSD split series (mixed multiplicity + ground truth).
ff_dir = "mwe_data/synthetic_split_ff"
nf_dir = "mwe_data/synthetic_split_nf"
ebsd_dir = "mwe_data/synthetic_split_ebsd"
d_tol = 20.0  # cross-step centroid link
theta_tol_deg = 15.0  # cross-step misorientation link (generous)
misori_tol_deg = 5.0  # segmentation cutoff (NF/EBSD)
voxel_gamma = 0.1  # fixed low gamma for few-grain voxel segmentation

output_folder = "reorientation_fragmentation_out"
k_rare = 3  # number of rare (most-fragmented) grains to flag
## ------------------------------------

os.makedirs(output_folder, exist_ok=True)
naming = FieldFileNaming(
    prefix="out_element_centroid", index_width=4, sep="_", suffix=".csv"
)
tracker = OrientationTracker()
analyzer = FragmentationAnalyzer()
ori_cols = ["ori_rodrigues_x", "ori_rodrigues_y", "ori_rodrigues_z"]

res = SimulationResults(sim_block_csv, sim_field_dir, field_naming=naming)
field_steps = sorted(res.field_files.keys())
ref_step, last_step = field_steps[0], field_steps[-1]
plot_steps = [
    field_steps[i]
    for i in np.unique(np.linspace(0, len(field_steps) - 1, 5).astype(int))
]

# (A) Simulation reorientation tracking -----------------------------------------
tracks = tracker.simulation_grain_tracks(res, grain_ids=sim_grains, steps=plot_steps)
pp.plot_ipf_orientation_tracking(
    tracks,
    angle_convention="mrp",
    output_folder=output_folder,
    savefig_name="A_sim_reorientation.png",
)
print(f"(A) reorientation tracking: {len(tracks)} grains -> A_sim_reorientation.png")

# (B) Simulation fragmentation: 1x2 reorientation (left) vs segmentation (right) --
frags_df, per_elem = analyzer.grain_fragments(
    res,
    last_step,
    tol_deg=tol_deg,
    min_subgrain_miso_deg=min_subgrain_miso_deg,
    seg_kwargs={"gamma": seg_gamma},
)
frags_df.to_csv(os.path.join(output_folder, "fragment_counts.csv"), index=False)

# Both panels reference each other via a shared per-grain fragment map, on two channels:
#  - COLOR: a dark categorical palette keyed by the sub-grain (fragment) index within a
#    grain, so a grain's fragments CONTRAST sharply (grains are told apart by IPF position,
#    sub-grains by color). Start dots stay black.
#  - LINE STYLE: the same fragment index, reinforcing the color in the crowded LEFT panel.
START_SIZE, STEP_SIZE = 80.0, 7.0
ARROW_SCALE, LABEL_FS = 12.0, 18.0  # small arrowheads on the fork panel
LEFT_LW, RIGHT_LW = 0.8, 2.0
FRAG_STYLES = ["-", "--", "-.", ":"]
# dark, high-contrast qualitative palette (ColorBrewer Dark2) for the fragments
FRAG_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"]
PARENT_COLOR = "0.35"  # neutral dark gray for the pre-split parent trajectory


def _grain_mean_track(gid, steps):
    out = []
    for s in steps:
        df = res.load_field_data(int(s))
        m = np.rint(df["block_id"].to_numpy()).astype(int) == gid
        if m.any():
            out.append(
                analyzer.mean_orientation_mrp(
                    df.loc[m, ori_cols].to_numpy(), analyzer.symmetry
                )
            )
    return np.asarray(out)


# per-grain fragment-label -> (linestyle, color) -- the single source both panels draw from
frag_style, frag_color = {}, {}
for g in sim_grains:
    labs = sorted(lab for lab in per_elem.values() if lab.startswith(f"{g}."))
    frag_style[g] = {
        lab: FRAG_STYLES[i % len(FRAG_STYLES)] for i, lab in enumerate(labs)
    }
    frag_color[g] = {
        lab: FRAG_COLORS[i % len(FRAG_COLORS)] for i, lab in enumerate(labs)
    }

# LEFT: EVERY element of each grain, colored + styled by the fragment it belongs to
df0 = res.load_field_data(plot_steps[0])
blk0 = np.rint(df0["block_id"].to_numpy()).astype(int)
left_tracks, left_styles, left_colors = [], [], []
for g in sim_grains:
    eids = df0.loc[blk0 == g, "id"].astype(int).tolist()
    g_tracks = tracker.simulation_element_tracks(
        res, element_ids=eids, steps=plot_steps
    )
    for eid, tr in zip(eids, g_tracks):
        lab = per_elem.get(eid)
        left_tracks.append(tr)
        left_styles.append(frag_style[g].get(lab, "-"))
        left_colors.append(frag_color[g].get(lab, "0.35"))

# RIGHT: parent (pre-last) forks into per-fragment children at the last step
last_df = res.load_field_data(last_step).set_index("id")
branches, child_styles, child_colors = [], [], []
for g in sim_grains:
    parent = _grain_mean_track(g, plot_steps[:-1])
    labs = sorted(frag_style[g])
    children = [
        analyzer.mean_orientation_mrp(
            last_df.loc[
                [e for e, l in per_elem.items() if l == lab], ori_cols
            ].to_numpy(),
            analyzer.symmetry,
        )[None, :]
        for lab in labs
    ]
    branches.append((parent, children))
    child_styles.append([frag_style[g][lab] for lab in labs])
    child_colors.append([frag_color[g][lab] for lab in labs])

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 7))
# color = dark categorical fragment color (sub-grains of one grain contrast), black start
# dots, no legend; line style reinforces the fragment in the crowded LEFT panel.
pp.plot_ipf_orientation_tracking(
    left_tracks,
    angle_convention="mrp",
    arrow_color=left_colors,
    start_color="black",
    start_size=START_SIZE,
    step_size=STEP_SIZE,
    line_width=LEFT_LW,
    arrow_scale=ARROW_SCALE,
    line_style=left_styles,
    show_arrow=False,
    label_fontsize=LABEL_FS,
    ax=ax_l,
)
pp.plot_ipf_fragmentation_tracking(
    branches,
    angle_convention="mrp",
    child_colors=child_colors,
    parent_color=PARENT_COLOR,
    start_color="black",
    start_size=START_SIZE,
    step_size=STEP_SIZE,
    line_width=RIGHT_LW,
    arrow_scale=ARROW_SCALE,
    child_line_styles=child_styles,
    show_arrow=True,
    label_fontsize=LABEL_FS,
    ax=ax_r,
)
fig.tight_layout()
fig.savefig(
    os.path.join(output_folder, "B_sim_reorientation_vs_fragmentation.png"), dpi=200
)
plt.close(fig)
n_frag = int((frags_df["n_fragments"] > 1).sum())
print(
    f"(B) 1x2 reorientation-vs-fragmentation ({n_frag}/{len(frags_df)} grains split) "
    f"-> B_sim_reorientation_vs_fragmentation.png"
)

# Annotate an Exodus with the fragments, colored by the SAME dark palette as the figure,
# via IPFProcessor.add_element_rgb_to_exodus (per-element RGB; in ParaView map rgb_x/y/z to
# color with scalar-mapping off). Point hex_run_* at a co-registered loaded hex run for a
# fully consistent fragment-colored mesh; otherwise color the bare shipped hex by block id.
from graintrace.ipf_postprocess import IPFProcessor
from matplotlib.colors import to_rgb

_ipf = IPFProcessor(crystal_symmetry="432", sample_symmetry="1", save_dir=output_folder)
if hex_run_mesh and hex_run_field_dir and os.path.isdir(hex_run_field_dir):
    hres = SimulationResults(hex_run_block_csv, hex_run_field_dir, field_naming=naming)
    hlast = sorted(hres.field_files)[-1]
    _, hpe = analyzer.grain_fragments(
        hres,
        hlast,
        tol_deg=tol_deg,
        min_subgrain_miso_deg=min_subgrain_miso_deg,
        seg_kwargs={"gamma": seg_gamma},
    )
    hdf = hres.load_field_data(hlast)
    hblk = np.rint(hdf["block_id"].to_numpy()).astype(int)
    hcolor = {}  # per-grain fragment-label -> palette color (matches the figure scheme)
    for g in np.unique(hblk):
        labs = sorted(l for l in hpe.values() if l.startswith(f"{int(g)}."))
        hcolor[int(g)] = {
            lab: FRAG_COLORS[i % len(FRAG_COLORS)] for i, lab in enumerate(labs)
        }
    rgb = np.array(
        [
            to_rgb(hcolor[int(b)].get(hpe.get(int(i)), "0.35"))
            for i, b in zip(hdf["id"].to_numpy(), hblk)
        ]
    )
    out_e = _ipf.add_element_rgb_to_exodus(
        hex_run_mesh,
        hdf[["x", "y", "z"]].to_numpy(float),
        rgb,
        output_file="sim_fragments_rgb.e",
    )
    print(f"    fragment-colored Exodus (ParaView, map rgb_* to color): {out_e}")
elif os.path.exists(hex_mesh):
    import scipy.io as sio

    with sio.netcdf_file(hex_mesh, "r") as fh:
        coords = np.stack(
            [
                fh.variables["coordx"][:],
                fh.variables["coordy"][:],
                fh.variables["coordz"][:],
            ],
            axis=1,
        )
        cents, rgb = [], []
        for eb in range(int(fh.dimensions["num_el_blk"])):
            conn = fh.variables[f"connect{eb + 1}"][:]
            cents.append(coords[conn - 1, :].mean(axis=1))
            col = to_rgb(FRAG_COLORS[eb % len(FRAG_COLORS)])
            rgb.append(np.tile(col, (conn.shape[0], 1)))
    out_e = _ipf.add_element_rgb_to_exodus(
        hex_mesh, np.vstack(cents), np.vstack(rgb), output_file="sim_fragments_rgb.e"
    )
    print(f"    block-colored Exodus demo (no loaded hex run; ParaView): {out_e}")


# (C) Experiment: detect grain splits across load steps (FF / NF / EBSD) ---------
def _voxel_grain_table(coords, eul_deg):
    mrp = (
        oh.euler_to_mrp(torch.tensor(eul_deg, dtype=torch.float64), "bunge", "degrees")
        .cpu()
        .numpy()
    )
    labels = analyzer.segment(
        coords,
        mrp,
        misori_tol_deg=misori_tol_deg,
        graph_mode="grid",
        manhattan_radius=2,
        grain_threshold_final=10,
        gamma=voxel_gamma,
    )
    return analyzer.seg_to_grain_table(labels, coords, mrp=mrp)


def _read_mic(path):
    zval = 0.0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("%GlobalPosition"):
                zval = float(line.split()[1])
                break
    df = pd.read_csv(path, sep="\t", skiprows=3)
    return pd.DataFrame(
        {
            "x": df["X"],
            "y": df["Y"],
            "z": np.full(len(df), zval),
            "Eul0": df["Eul1"],
            "Eul1": df["Eul2"],
            "Eul2": df["Eul3"],
        }
    )


# FF: grain tables feed the detector directly (mixed 1->1/1->2/1->3).
ff_gt = json.load(open(os.path.join(ff_dir, "ground_truth.json"), encoding="utf-8"))
s = ff_gt["split_step"]
ff_steps = sorted(glob.glob(os.path.join(ff_dir, "ff_step*.csv")))
res_ff = analyzer.detect_splits(
    ff_steps[s - 1],
    ff_steps[s],
    d_tol=d_tol,
    theta_tol_deg=theta_tol_deg,
    angle_type="degrees",
)
print(
    f"(C-FF) n_splits={res_ff['n_splits']} (ground truth {ff_gt['n_splitting_grains']}; "
    f"multiplicities {ff_gt['multiplicities']})"
)

# branching IPF: one fork per detected split, colored by parent id
dfs = [pd.read_csv(p) for p in ff_steps]
eul3 = ["Eul0", "Eul1", "Eul2"]


def _geul(df, gid):
    r = df.loc[df["grain_id"] == gid, eul3]
    return r.to_numpy()[0] if len(r) else None


ff_branches, ff_colors = [], []
for j, (_, row) in enumerate(res_ff["split_correspondences"].iterrows()):
    pid = int(row["parent_a"])
    cids = list(map(int, str(row["children_b"]).split(";")))
    parent = np.asarray(
        [_geul(dfs[k], pid) for k in range(0, s) if _geul(dfs[k], pid) is not None]
    )
    child_tracks = [
        np.asarray(
            [e for e in (_geul(dfs[k], c) for k in range(s, len(dfs))) if e is not None]
        )
        for c in cids
    ]
    ff_branches.append((parent, child_tracks))
    ff_colors.append(cm.tab10(j % 10))
pp.plot_ipf_fragmentation_tracking(
    ff_branches,
    angle_convention="bunge",
    angle_type="degrees",
    arrow_color=ff_colors,
    output_folder=output_folder,
    savefig_name="C_experiment_ff_splits.png",
)
print(f"(C-FF) {len(ff_branches)} branching forks -> C_experiment_ff_splits.png")

# NF / EBSD: segment each step -> grain table -> SAME detector (first vs last step).
for tag, folder, reader in [("EBSD", ebsd_dir, "csv"), ("NF", nf_dir, "mic")]:
    gt = json.load(open(os.path.join(folder, "ground_truth.json"), encoding="utf-8"))
    tables = []
    if reader == "csv":
        steps_csv = sorted(glob.glob(os.path.join(folder, "ebsd_step*.csv")))
        picks = [steps_csv[0], steps_csv[-1]]
        for p in picks:
            df = pd.read_csv(p)
            tables.append(
                _voxel_grain_table(
                    df[["x", "y", "z"]].to_numpy(float),
                    df[["Eul0", "Eul1", "Eul2"]].to_numpy(float),
                )
            )
    else:
        sdirs = [gt["step_dirs"][0], gt["step_dirs"][-1]]
        for sd in sdirs:
            rows = pd.concat(
                [_read_mic(p) for p in sorted(glob.glob(os.path.join(sd, "*.mic")))],
                ignore_index=True,
            )
            tables.append(
                _voxel_grain_table(
                    rows[["x", "y", "z"]].to_numpy(float),
                    rows[["Eul0", "Eul1", "Eul2"]].to_numpy(float),
                )
            )
    r = analyzer.detect_splits(
        tables[0],
        tables[1],
        d_tol=d_tol,
        theta_tol_deg=theta_tol_deg,
        angle_type="degrees",
    )
    print(
        f"(C-{tag}) grains {len(tables[0])}->{len(tables[1])}, n_splits={r['n_splits']} "
        f"(ground truth {gt['n_splitting_grains']})"
    )

# (D) REI capstone: the grains that split the most --------------------------------
# reuse select_highest_scalar on a per-grain table (scalar = fragment count).
rank = frags_df.rename(columns={"grain_id": "cluster_label", "n_elements": "n"})
rare = rcs.select_highest_scalar(
    rank, required_cols="n_fragments", k=k_rare, min_size=1
)
print(f"(D) rare grains (most fragmented, top {k_rare}): {list(map(int, rare))}")
print(
    "    -> reorientation drives fragmentation; the most-split grains are the rare events."
)

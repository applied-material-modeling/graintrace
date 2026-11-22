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

"""Layer-B baseline for the orientation-convention flip (generated on 0544c83).

Runs the orientation-consuming pipelines (fragmentation segmentation / absorb /
re-merge / merge / spreads / leak check, ``seg_to_grain_table``, ``detect_splits``
on the shipped ``mwe_data/synthetic_split_*`` data, ``grain_fragments`` on
``mwe_data/cpfe_ff_fragmentation``, ``GraphGrainMatcher`` and
``RegionBaseStitching``) on the code as it stood BEFORE the convention change
(``development`` @ 0544c83) and stores inputs + outputs in
``orientation_convention_baseline.npz``.

``tests/test_convention_invariance.py`` re-runs the same pipelines on the
current code and requires identical results. Euler-input pipelines get the same
Euler input. MRP-input pipelines get ``-p``: at 0544c83 graintrace's MRP of an
Euler triplet was the MRP of the passive ``g``, now it is the MRP of the active
``R = g^T``, and ``mrp(g^T) == -mrp(g)``. So the stored legacy MRP inputs are fed
negated, and MRP outputs are compared as ``-new`` vs legacy.

Regenerate ONLY on a checkout of 0544c83 (never on the flipped code -- the
whole point is that it records the old behaviour)::

    git worktree add /tmp/gt0544 0544c83
    cd /tmp/gt0544 && python <this repo>/tests/data/make_orientation_baseline.py \
        --out <this repo>/tests/data/orientation_convention_baseline.npz
"""

# pylint: disable=import-outside-toplevel
import argparse
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MWE = REPO / "mwe_data"

# shared knobs (the invariance test imports these)
GRID_SHAPE = (14, 10, 5)
SEG_KW = dict(
    misori_tol_deg=5.0,
    graph_mode="grid",
    manhattan_radius=2,
    grain_threshold_final=10,
    gamma=1.0,
    verbose=False,
)
VOXEL_SEG_KW = dict(
    misori_tol_deg=5.0,
    graph_mode="grid",
    manhattan_radius=2,
    grain_threshold_final=10,
    gamma=0.1,
    verbose=False,
)
SPLIT_KW = dict(d_tol=20.0, theta_tol_deg=15.0, angle_type="degrees")
FRAG_KW = dict(tol_deg=5.0, min_subgrain_miso_deg=1.0, seg_kwargs={"gamma": 0.25})
FRAG_STEP_FILE = "out_element_centroid_0026.csv"
# Frozen copy of the pre-regeneration mwe_data/cpfe_ff_fragmentation inputs (the shipped dataset
# was regenerated under the active-R convention, so the baseline no longer reads it).
FRAG_DIR = Path(__file__).resolve().parent / "frag_legacy"


def grid_case(seed: int = 7):
    """Synthetic voxel grid: 5 grains, gradients, noise and symmetry-variant voxels.

    Returns ``coords (N,3)``, ``euler (N,3)`` degrees (Bunge), ``truth (N,)``.
    Every Euler row is a valid Bunge triplet; the variant voxels are built from the
    crystal symmetry group (so their Euler differs, the orientation does not), which
    exercises every fold-to-reference code path.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    nx, ny, nz = GRID_SHAPE
    xs, ys, zs = np.meshgrid(
        np.arange(nx, dtype=float), np.arange(ny, dtype=float), np.arange(nz) * 1.0
    )
    coords = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1) * 2.0
    truth = np.zeros(len(coords), dtype=np.int64)
    truth[coords[:, 0] >= 10.0] = 1
    truth[coords[:, 0] >= 18.0] = 2
    truth[(coords[:, 0] >= 18.0) & (coords[:, 1] >= 10.0)] = 3
    truth[(coords[:, 0] < 10.0) & (coords[:, 1] >= 12.0)] = 4

    base = Rotation.random(5, random_state=seed)
    # grain 1 sits next to grain 0 at a low angle (3 deg) -> remerge candidates
    base = Rotation.concatenate(
        [base[0], base[0] * Rotation.from_rotvec([0.0, 0.0, np.deg2rad(3.0)])]
        + [base[k] for k in range(2, 5)]
    )
    # intragranular gradient (up to ~4 deg along x) + 0.3 deg noise
    grad = Rotation.from_rotvec(
        np.outer(coords[:, 0] / coords[:, 0].max(), [np.deg2rad(4.0), 0.0, 0.0])
    )
    noise = Rotation.from_rotvec(
        rng.normal(0.0, np.deg2rad(0.3), size=(len(coords), 3))
    )
    rot = base[truth] * grad * noise
    euler = rot.as_euler("ZXZ", degrees=True)  # intrinsic ZXZ == Bunge angles
    euler[:, 0] %= 360.0
    euler[:, 2] %= 360.0
    # symmetry-variant voxels: same crystal orientation, different Euler triplet.
    # Cubic ops are 90-deg turns about cube axes; applied on the crystal side.
    variant = rng.random(len(coords)) < 0.15
    ops = Rotation.from_euler(
        "xyz", [[90, 0, 0], [0, 90, 0], [0, 0, 90], [180, 0, 0]], degrees=True
    )
    pick = rng.integers(0, len(ops), size=len(coords))
    rot_var = rot * ops[pick]
    eul_var = rot_var.as_euler("ZXZ", degrees=True)
    eul_var[:, 0] %= 360.0
    eul_var[:, 2] %= 360.0
    euler[variant] = eul_var[variant]
    return coords, euler, truth


def oversplit(truth: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Split every true grain into halves along y -> re-merge / merge inputs."""
    lab = truth * 2 + (coords[:, 1] >= np.median(coords[:, 1])).astype(np.int64)
    # a few tiny islands -> absorb inputs
    lab = lab.copy()
    lab[::97] = lab.max() + 1 + np.arange(len(lab[::97]))
    return lab


def read_mic(path):
    """Read one synthetic-split ``.mic`` layer as x, y, z, Eul0..2 (degrees)."""
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


def voxel_steps():
    """First/last step voxel tables of the shipped EBSD and NF split datasets."""
    out = {}
    ebsd = sorted(glob.glob(str(MWE / "synthetic_split_ebsd" / "ebsd_step*.csv")))
    out["ebsd"] = [pd.read_csv(ebsd[0]), pd.read_csv(ebsd[-1])]
    nf = sorted(glob.glob(str(MWE / "synthetic_split_nf" / "step*")))
    out["nf"] = [
        pd.concat(
            [read_mic(p) for p in sorted(glob.glob(os.path.join(sd, "*.mic")))],
            ignore_index=True,
        )
        for sd in (nf[0], nf[-1])
    ]
    return out


def table_array(df):
    """Numeric grain table -> (n, 8) float array (grain_id, X..Z, R, Eul0..2)."""
    cols = ["grain_id", "X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]
    return df[cols].to_numpy(dtype=float)


def split_summary(res):
    """Order-stable numeric summary of a ``detect_splits`` result."""
    sc = res["split_correspondences"]
    corr = sorted(
        f"{int(r.parent_a)}:{r.children_b}" for r in sc.itertuples(index=False)
    )
    summ = {"n_splits": np.array(res["n_splits"]), "corr": np.array(corr, dtype=str)}
    if "one_to_one" in res:
        o2o = res["one_to_one"].sort_values(["a_id", "b_id"])
        summ["o2o"] = o2o[["a_id", "b_id", "miso_deg", "dist"]].to_numpy(dtype=float)
    if "null" in res:
        summ["null"] = np.array(
            [res["null"]["mean_links"], res["null"]["std_links"]], dtype=float
        )
    return summ


def stitch_inputs(seed: int = 3):
    """Two overlapping scans whose overlap grains are perturbed duplicates."""
    from graintrace.orientation_helper import perturb_orientation

    rng = np.random.default_rng(seed)
    n = 60
    df = pd.DataFrame(
        {
            "X": rng.uniform(0, 300, n),
            "Y": rng.uniform(0, 300, n),
            "Z": rng.uniform(0, 200, n),
            "GrainRadius": rng.uniform(10, 25, n),
            "Eul0": rng.uniform(0, 360, n),
            "Eul1": rng.uniform(0, 180, n),
            "Eul2": rng.uniform(0, 360, n),
        }
    )
    s0 = df[df["Z"] < 110].copy()
    s1 = df[df["Z"] > 90].copy()
    eul = s1[["Eul0", "Eul1", "Eul2"]].to_numpy()
    s1[["Eul0", "Eul1", "Eul2"]] = perturb_orientation(
        eul, 1.0, "bunge", "degrees", rng=np.random.default_rng(seed + 1)
    ).numpy()
    s1[["X", "Y", "Z"]] += rng.normal(0, 2.0, size=(len(s1), 3))
    return s0.reset_index(drop=True), s1.reset_index(drop=True)


def run_stitch(s0, s1, workdir):
    """RegionBaseStitching overlap path (no NEPER refinement)."""
    from graintrace.hedm_stitching_techniques.region_base_stitching import (
        RegionBaseStitching,
    )

    p0, p1 = os.path.join(workdir, "s0.csv"), os.path.join(workdir, "s1.csv")
    s0.to_csv(p0, index=False)
    s1.to_csv(p1, index=False)
    out = os.path.join(workdir, "stitched.csv")
    RegionBaseStitching(
        scan_files=[p0, p1],
        output_csv=out,
        position_tolerance=40,
        orientation_tolerance=10,
        radius_tolerance=-1,
        weights={"pos": 0.1, "ori": 1.0, "rad": 0},
        min_neighbors=5,
    ).run(zlo=0, zhi=200, overlap_fraction=0.1)
    res = pd.read_csv(out)
    cols = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]
    return res[cols].to_numpy(dtype=float)


def matcher_frames(seed: int = 11, n: int = 20):
    """Two 20-grain chain graphs: B = perturbed, permuted copy of A."""
    from graintrace.orientation_helper import perturb_orientation

    rng = np.random.default_rng(seed)
    xyz = np.cumsum(rng.uniform(5, 15, size=(n, 3)), axis=0)
    eul = np.stack(
        [rng.uniform(0, 360, n), rng.uniform(0, 180, n), rng.uniform(0, 360, n)], 1
    )
    perm = rng.permutation(n)
    eul_b = perturb_orientation(
        eul, 2.0, "bunge", "degrees", rng=np.random.default_rng(seed + 1)
    ).numpy()
    xyz_b = xyz + rng.normal(0, 0.5, size=xyz.shape)
    return xyz, eul, xyz_b, eul_b, perm


def run_matcher(xyz, eul, xyz_b, eul_b, perm, workdir):
    """GraphGrainMatcher default (misorientation) scheme; returns a_to_b + features."""
    import torch
    from torch_geometric.data import Data

    from graintrace.grain_graph_matching import GraphGrainMatcher

    n = len(xyz)
    feats = ("X", "Y", "Z", "Eul0", "Eul1", "Eul2")
    edges = [(i, i + 1) for i in range(n - 1)] + [(i, i + 2) for i in range(n - 2)]

    def frame(pos, ang, order):
        rows = [None] * n
        for g, node in enumerate(order):
            rows[node] = list(pos[g]) + list(ang[g])
        graph = Data(
            x=torch.tensor(rows, dtype=torch.float64),
            edge_index=torch.tensor(
                [(order[u], order[v]) for u, v in edges], dtype=torch.long
            ).T,
        )
        graph.num_nodes = n
        graph.feature_names = list(feats)
        graph.feature_slices = {f: (k, k + 1) for k, f in enumerate(feats)}
        return graph

    matcher = GraphGrainMatcher(
        graph_a=frame(xyz, eul, list(range(n))),
        graph_b=frame(xyz_b, eul_b, list(perm)),
        output_dir=workdir,
    )
    res = matcher.match_grains(
        message_passing_iter=2,
        neighbor_selection_param={
            "lambda": 0.125,
            "iterations": 5,
            "tolerance": 1e-6,
            "topk": 6,
        },
        angle_convention="bunge",
        angle_type="degrees",
        symmetry="432",
    )
    return (
        res["match"]["a_to_b"].cpu().numpy(),
        res["graph_a_features"].cpu().numpy(),
        res["graph_b_features"].cpu().numpy(),
    )


def run_grain_fragments(field_dir, block_csv):
    """``grain_fragments`` on the last cpfe_ff_fragmentation step."""
    from graintrace.fragmentation import FragmentationAnalyzer
    from graintrace.simulation_postprocessing import FieldFileNaming, SimulationResults

    res = SimulationResults(
        block_csv=block_csv,
        field_dir=field_dir,
        field_naming=FieldFileNaming(
            prefix="out_element_centroid", index_width=4, sep="_", suffix=".csv"
        ),
    )
    step = sorted(res.field_files.keys())[-1]
    frags, per_elem = FragmentationAnalyzer(symmetry="432").grain_fragments(
        res, step, **FRAG_KW
    )
    ids = np.array(sorted(per_elem), dtype=np.int64)
    codes = np.array([per_elem[i] for i in ids], dtype=str)
    return (
        frags[["grain_id", "n_fragments", "n_elements"]].to_numpy(dtype=np.int64),
        ids,
        codes,
    )


def main():
    """Generate the baseline npz (must run on 0544c83)."""
    import torch

    from graintrace import orientation_helper as oh
    from graintrace.fragmentation import FragmentationAnalyzer

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(Path(__file__).with_name("orientation_convention_baseline.npz")),
    )
    args = ap.parse_args()

    out = {}
    an = FragmentationAnalyzer(symmetry="432")

    # --- synthetic voxel grid (MRP-input fragmentation pipeline) -----------------
    coords, euler, truth = grid_case()
    mrp = oh.euler_to_mrp(torch.as_tensor(euler), "bunge", "degrees").numpy()
    over = oversplit(truth, coords)
    out["grid_coords"], out["grid_euler"], out["grid_truth"] = coords, euler, truth
    out["grid_mrp_legacy"], out["grid_oversplit"] = mrp, over

    seg = an.segment(coords, mrp, **SEG_KW)
    out["grid_segment"] = seg
    out["grid_table_mrp"] = table_array(an.seg_to_grain_table(seg, coords, mrp=mrp))
    out["grid_table_euler"] = table_array(
        an.seg_to_grain_table(
            seg, coords, euler=euler, angle_convention="bunge", angle_type="degrees"
        )
    )
    out["grid_table_trim"] = table_array(
        an.seg_to_grain_table(seg, coords, mrp=mrp, trim_miso_deg=5.0)
    )
    spread = an.cluster_orientation_spread(mrp, over, "432", max_points=0)
    out["grid_spread"] = np.array([[k, v] for k, v in sorted(spread.items())])
    leak = an.cluster_leak_check(mrp, truth * 0 + (truth >= 2), 5.0, "432")
    out["grid_leak"] = np.array(
        [
            [k, v["flagged"], v["spread_deg"], v["leak_frac"], v["leak_sep_deg"]]
            for k, v in sorted(leak.items())
        ],
        dtype=float,
    )
    out["grid_absorb"] = an.absorb_fragments(
        coords, over, 30, mrp=mrp, max_miso_deg=5.0, symmetry="432"
    )
    out["grid_remerge_agg"] = an.adjacency_remerge(coords, mrp, over, 3.5, "432")
    out["grid_remerge_spread"] = an.adjacency_remerge(
        coords, mrp, over, 3.5, "432", spread_tol_deg=5.0
    )
    out["grid_remerge_uf"] = an.adjacency_remerge(
        coords, mrp, over, 3.5, "432", percolation_max_frac=0.5
    )
    out["grid_merge_frag"] = an.merge_fragments_by_misorientation(mrp, over, 3.5, "432")
    out["grid_mean_mrp_legacy"] = np.stack(
        [an.mean_orientation_mrp(mrp[truth == g], "432") for g in range(5)]
    )

    # --- shipped synthetic split datasets (Euler input) --------------------------
    ff = sorted(glob.glob(str(MWE / "synthetic_split_ff" / "ff_step*.csv")))
    with open(MWE / "synthetic_split_ff" / "ground_truth.json", encoding="utf-8") as fh:
        s = json.load(fh)["split_step"]
    res_ff = an.detect_splits(
        ff[s - 1], ff[s], assignment="one_to_one", n_null=5, seed=0, **SPLIT_KW
    )
    for k, v in split_summary(res_ff).items():
        out[f"ff_split_{k}"] = v
    for tag, steps in voxel_steps().items():
        tables = []
        for i, df in enumerate(steps):
            c = df[["x", "y", "z"]].to_numpy(float)
            e = df[["Eul0", "Eul1", "Eul2"]].to_numpy(float)
            m = oh.euler_to_mrp(torch.as_tensor(e), "bunge", "degrees").numpy()
            lab = an.segment(c, m, **VOXEL_SEG_KW)
            out[f"{tag}_segment_{i}"] = lab
            tab = an.seg_to_grain_table(lab, c, mrp=m)
            out[f"{tag}_table_{i}"] = table_array(tab)
            tables.append(tab)
        res = an.detect_splits(tables[0], tables[1], **SPLIT_KW)
        for k, v in split_summary(res).items():
            out[f"{tag}_split_{k}"] = v

    # --- simulation intragranular fragments (MRP from the CPFE run) --------------
    frag_dir = FRAG_DIR
    if frag_dir.is_dir():
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "mesh_out"))
            shutil.copy(
                frag_dir / "mesh_out" / FRAG_STEP_FILE,
                os.path.join(tmp, "mesh_out", FRAG_STEP_FILE),
            )
            fr, ids, codes = run_grain_fragments(
                os.path.join(tmp, "mesh_out"), str(frag_dir / "out.csv")
            )
        out["frag_table"], out["frag_ids"], out["frag_codes"] = fr, ids, codes

    # --- GraphGrainMatcher (Euler input) -----------------------------------------
    try:
        import torch_geometric  # noqa: F401  pylint: disable=unused-import

        frames = matcher_frames()
        for key, arr in zip(("xyz", "eul", "xyz_b", "eul_b", "perm"), frames):
            out[f"match_in_{key}"] = arr
        with tempfile.TemporaryDirectory() as tmp:
            a2b, fa, fb = run_matcher(*frames, tmp)
        out["match_a_to_b"], out["match_fa"], out["match_fb"] = a2b, fa, fb
    except ImportError:
        print("torch_geometric missing: GraphGrainMatcher baseline skipped")

    # --- RegionBaseStitching (Euler input) ---------------------------------------
    s0, s1 = stitch_inputs()
    out["stitch_s0"] = s0.to_numpy(dtype=float)
    out["stitch_s1"] = s1.to_numpy(dtype=float)
    with tempfile.TemporaryDirectory() as tmp:
        out["stitch_out"] = run_stitch(s0, s1, tmp)

    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {sorted(out)}")


if __name__ == "__main__":
    main()

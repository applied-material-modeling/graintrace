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

"""Tests for the shared fragmentation core (reorientation + segmentation bridge)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _write_field_step(path, ids, coords, mrp, block_ids=None):
    df = pd.DataFrame(
        {
            "id": ids,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "ori_rodrigues_x": mrp[:, 0],
            "ori_rodrigues_y": mrp[:, 1],
            "ori_rodrigues_z": mrp[:, 2],
        }
    )
    if block_ids is not None:
        df["block_id"] = block_ids
    df.to_csv(path, index=False)


class _FakeResults:
    """Minimal SimulationResults stand-in for reorientation tests."""

    def __init__(self, field_files):
        self.field_files = field_files

    def load_field_data(self, idx):
        return pd.read_csv(self.field_files[idx])


class TestReorientationFromReference:
    def test_identity_zero(self, tmp_path):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        ids = np.arange(5)
        coords = np.random.default_rng(0).normal(size=(5, 3))
        mrp = np.zeros((5, 3))  # identity orientation
        p0 = tmp_path / "s0.csv"
        p1 = tmp_path / "s1.csv"
        _write_field_step(p0, ids, coords, mrp)
        _write_field_step(p1, ids, coords, mrp)
        res = _FakeResults({0: str(p0), 1: str(p1)})
        out_ids, deg = FragmentationAnalyzer().reorientation_from_reference(
            res, 1, ref_step=0
        )
        assert np.array_equal(np.sort(out_ids), ids)
        assert np.allclose(deg, 0.0, atol=1e-6)

    def test_known_rotation(self, tmp_path):
        pytest.importorskip("neml2")
        import torch

        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.orientation_helper import euler_to_mrp

        ids = np.arange(3)
        coords = np.zeros((3, 3))
        mrp0 = np.zeros((3, 3))
        # 10 deg about z (Bunge phi1) -> misorientation 10 deg under cubic sym.
        mrp10 = (
            euler_to_mrp(
                torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64),
                "bunge",
                "degrees",
            )
            .cpu()
            .numpy()
        )
        p0 = tmp_path / "s0.csv"
        p1 = tmp_path / "s1.csv"
        _write_field_step(p0, ids, coords, mrp0)
        _write_field_step(p1, ids, coords, mrp10)
        res = _FakeResults({0: str(p0), 1: str(p1)})
        _, deg = FragmentationAnalyzer().reorientation_from_reference(
            res, 1, ref_step=0
        )
        assert np.allclose(deg, 10.0, atol=1e-3)

    def test_write_reorientation_field_schema(self, tmp_path):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        ids = np.arange(4)
        coords = np.random.default_rng(1).normal(size=(4, 3))
        mrp = np.zeros((4, 3))
        p0 = tmp_path / "s0.csv"
        p1 = tmp_path / "s1.csv"
        _write_field_step(p0, ids, coords, mrp)
        _write_field_step(p1, ids, coords, mrp)
        res = _FakeResults({0: str(p0), 1: str(p1)})
        out = FragmentationAnalyzer().write_reorientation_field(
            res, 1, str(tmp_path / "reo.csv"), ref_step=0
        )
        df = pd.read_csv(out)
        assert list(df.columns) == ["id", "x", "y", "z", "reorientation_deg"]
        assert len(df) == 4


class TestSegGridToGrainCsv:
    def test_two_clusters_reduce(self, tmp_path):
        pytest.importorskip("neml2")
        import torch

        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.orientation_helper import euler_to_mrp

        # Two labelled blocks with distinct orientations.
        coords = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 10, 10], [11, 10, 10]],
            dtype=float,
        )
        labels = np.array([0, 0, 0, 1, 1])
        eul = np.array([[10.0, 20.0, 30.0]] * 3 + [[70.0, 10.0, 5.0]] * 2)
        mrp = (
            euler_to_mrp(torch.tensor(eul, dtype=torch.float64), "bunge", "degrees")
            .cpu()
            .numpy()
        )
        df = FragmentationAnalyzer().seg_to_grain_table(labels, coords, mrp=mrp)
        assert list(df.columns) == [
            "grain_id",
            "X",
            "Y",
            "Z",
            "GrainRadius",
            "Eul0",
            "Eul1",
            "Eul2",
        ]
        assert len(df) == 2
        assert (df["GrainRadius"] > 0).all()

    def test_background_skipped(self):
        pytest.importorskip("neml2")
        import torch

        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.orientation_helper import euler_to_mrp

        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        labels = np.array([-1, 0, 0])
        eul = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        mrp = (
            euler_to_mrp(torch.tensor(eul, dtype=torch.float64), "bunge", "degrees")
            .cpu()
            .numpy()
        )
        df = FragmentationAnalyzer().seg_to_grain_table(
            labels, coords, mrp=mrp, background_label=-1
        )
        assert len(df) == 1
        assert int(df.iloc[0]["grain_id"]) == 0


class TestSegmentByOrientation:
    def test_recovers_two_clusters(self, tmp_path):
        pytest.importorskip("neml2")
        pytest.importorskip("networkit")
        import torch

        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.orientation_helper import euler_to_mrp

        # Two spatially separated 3x3x3 voxel blocks, ~40 deg apart in orientation.
        def block(origin):
            pts = []
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        pts.append([origin[0] + i, origin[1] + j, origin[2] + k])
            return np.array(pts, dtype=float)

        c0 = block((0, 0, 0))
        c1 = block((10, 0, 0))
        coords = np.vstack([c0, c1])
        eul = np.vstack(
            [
                np.tile([10.0, 20.0, 30.0], (len(c0), 1)),
                np.tile([60.0, 30.0, 15.0], (len(c1), 1)),
            ]
        )
        mrp = (
            euler_to_mrp(torch.tensor(eul, dtype=torch.float64), "bunge", "degrees")
            .cpu()
            .numpy()
        )
        labels = FragmentationAnalyzer().segment(
            coords,
            mrp,
            misori_tol_deg=5.0,
            graph_mode="grid",
            manhattan_radius=2,
            grain_threshold_final=5,
            workdir=str(tmp_path),
        )
        assert len(np.unique(labels)) == 2


class TestMeanOrientationAndMerge:
    def test_mean_symmetry_aware_vs_variant_mixed(self):
        pytest.importorskip("neml2")
        import numpy as np
        import torch

        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.orientation_helper import (
            matrix_to_mrp,
            misorientation_matrix,
            mrp_to_matrix,
            symmetry_operators,
        )

        an = FragmentationAnalyzer(symmetry="432")
        rng = np.random.default_rng(0)
        # a tight cluster of orientations (~1.4 deg spread) around a base
        base = mrp_to_matrix(torch.tensor([[0.1, -0.05, 0.2]], dtype=torch.float64))
        pert = mrp_to_matrix(
            torch.tensor(rng.normal(0, 0.006, size=(200, 3)), dtype=torch.float64)
        )
        rmat = torch.matmul(base, pert)
        mrp = matrix_to_mrp(rmat).numpy()

        # contaminate half with a cubic symmetry variant (same physical orientation):
        # a left-multiplied symmetry op is a zero-misorientation equivalent that the
        # symmetry-aware mean folds back, and the plain mean does not.
        ops = symmetry_operators("432")
        var = rmat.clone()
        var[: len(var) // 2] = torch.matmul(ops[1].unsqueeze(0), var[: len(var) // 2])
        mrp_var = matrix_to_mrp(var).numpy()

        def miso(a, b):
            return float(
                misorientation_matrix(
                    mrp_to_matrix(torch.tensor(a[None, :])),
                    mrp_to_matrix(torch.tensor(b[None, :])),
                    symmetry="432",
                ).item()
            )

        # symmetry-aware mean is robust to variant mixing; the plain mean is not
        m_sym = an.mean_orientation_mrp(mrp_var, "432")
        m_plain = an.mean_orientation_mrp(mrp_var, "1")
        m_clean = an.mean_orientation_mrp(mrp, "432")
        assert miso(m_sym, m_clean) < 0.5  # folded back to the coherent mean
        assert miso(m_plain, m_clean) > 5.0  # naive mean is corrupted

    def test_merge_fragments_by_misorientation(self):
        pytest.importorskip("neml2")
        import numpy as np
        import torch

        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.orientation_helper import euler_to_mrp

        # three fragments: 0 and 1 ~0.5 deg apart, 2 is ~8 deg away
        def mrp_of(rows):
            return (
                euler_to_mrp(
                    torch.tensor(rows, dtype=torch.float64), "bunge", "degrees"
                )
                .cpu()
                .numpy()
            )

        mrp = np.vstack(
            [
                mrp_of([[10.0, 20.0, 30.0]] * 10),
                mrp_of([[10.5, 20.0, 30.0]] * 10),
                mrp_of([[18.0, 20.0, 30.0]] * 10),
            ]
        )
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        # tol below the 0-1 gap: nothing merges
        keep = FragmentationAnalyzer.merge_fragments_by_misorientation(
            mrp, labels, 0.2, "432"
        )
        assert len(np.unique(keep)) == 3
        # tol above the 0-1 gap but below the 8 deg gap: 0 and 1 merge -> 2 clusters
        merged = FragmentationAnalyzer.merge_fragments_by_misorientation(
            mrp, labels, 1.0, "432"
        )
        assert len(np.unique(merged)) == 2


class TestSegmentGrainFragments:
    def test_counts_two_fragments_in_one_grain(self, tmp_path):
        pytest.importorskip("neml2")
        pytest.importorskip("networkit")
        import torch

        from graintrace.fragmentation import FragmentationAnalyzer
        from graintrace.orientation_helper import euler_to_mrp

        # One grain (block 1) split into two spatially separated orientation
        # sub-domains ~40 deg apart -> two fragments. Block 2 is a whole grain.
        def block(origin, n=3):
            pts = []
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        pts.append([origin[0] + i, origin[1] + j, origin[2] + k])
            return np.array(pts, dtype=float)

        sub_a = block((0, 0, 0))
        sub_b = block((10, 0, 0))
        g2 = block((0, 30, 0))
        coords = np.vstack([sub_a, sub_b, g2])
        blocks = np.array(
            [1] * len(sub_a) + [1] * len(sub_b) + [2] * len(g2), dtype=int
        )
        eul = np.vstack(
            [
                np.tile([10.0, 20.0, 30.0], (len(sub_a), 1)),
                np.tile([55.0, 30.0, 15.0], (len(sub_b), 1)),
                np.tile([80.0, 10.0, 5.0], (len(g2), 1)),
            ]
        )
        mrp = (
            euler_to_mrp(torch.tensor(eul, dtype=torch.float64), "bunge", "degrees")
            .cpu()
            .numpy()
        )
        ids = np.arange(len(coords))
        path = tmp_path / "s0.csv"
        _write_field_step(path, ids, coords, mrp, block_ids=blocks)
        res = _FakeResults({0: str(path)})

        frag_df, per_elem = FragmentationAnalyzer().grain_fragments(
            res, 0, tol_deg=5.0, min_fragment_elems=3
        )
        g1 = frag_df[frag_df["grain_id"] == 1].iloc[0]
        assert int(g1["n_fragments"]) == 2
        g2row = frag_df[frag_df["grain_id"] == 2].iloc[0]
        assert int(g2row["n_fragments"]) == 1
        # labels are "grain.fragment"
        assert all("." in v for v in per_elem.values())
        assert {v.split(".")[0] for v in per_elem.values()} == {"1", "2"}


class TestBranchingPlot:
    def test_branching_plot_writes_png(self, tmp_path):
        pytest.importorskip("neml2")
        import matplotlib

        matplotlib.use("Agg")
        import torch

        from graintrace.orientation_helper import euler_to_mrp
        from graintrace.plot_postprocessing import plot_ipf_fragmentation_tracking

        def mrp_of(eul_rows):
            return (
                euler_to_mrp(
                    torch.tensor(eul_rows, dtype=torch.float64), "bunge", "degrees"
                )
                .cpu()
                .numpy()
            )

        parent = mrp_of([[10.0, 20.0, 30.0], [12.0, 21.0, 30.0]])
        child1 = mrp_of([[16.0, 22.0, 31.0]])
        child2 = mrp_of([[8.0, 19.0, 29.0]])
        branches = [(parent, [child1, child2])]
        out = plot_ipf_fragmentation_tracking(
            branches,
            angle_convention="mrp",
            output_folder=str(tmp_path),
            savefig_name="branch.png",
        )
        assert out.exists()

    def test_per_entity_colors_and_shared_ax(self, tmp_path):
        # both plotters accept a per-entity color list and draw into a supplied ax
        # (returning the ax, not saving) -- the 1x2 reorientation-vs-fragmentation figure.
        pytest.importorskip("neml2")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch

        from graintrace.orientation_helper import euler_to_mrp
        from graintrace.plot_postprocessing import (
            plot_ipf_fragmentation_tracking,
            plot_ipf_orientation_tracking,
        )

        def mrp_of(rows):
            return (
                euler_to_mrp(
                    torch.tensor(rows, dtype=torch.float64), "bunge", "degrees"
                )
                .cpu()
                .numpy()
            )

        tracks = [
            mrp_of([[10.0, 20.0, 30.0], [12.0, 21.0, 30.0]]),
            mrp_of([[50.0, 40.0, 15.0], [55.0, 42.0, 15.0]]),
        ]
        branches = [
            (mrp_of([[10.0, 20.0, 30.0]]), [mrp_of([[16.0, 22.0, 31.0]])]),
            (mrp_of([[50.0, 40.0, 15.0]]), [mrp_of([[52.0, 42.0, 16.0]])]),
        ]
        colors = ["tab:red", "tab:blue"]
        fig, (ax_l, ax_r) = plt.subplots(1, 2)
        ret_l = plot_ipf_orientation_tracking(
            tracks, angle_convention="mrp", arrow_color=colors, ax=ax_l
        )
        ret_r = plot_ipf_fragmentation_tracking(
            branches, angle_convention="mrp", arrow_color=colors, ax=ax_r
        )
        # when an ax is supplied the plotters return the ax (no save/close)
        assert ret_l is ax_l and ret_r is ax_r
        out = tmp_path / "compare.png"
        fig.savefig(out)
        plt.close(fig)
        assert out.exists()

    def test_arrow_color_length_mismatch_raises(self):
        from graintrace.plot_postprocessing import _resolve_arrow_colors

        assert _resolve_arrow_colors("black", 3) == ["black"] * 3
        assert _resolve_arrow_colors((0.1, 0.2, 0.3), 2) == [(0.1, 0.2, 0.3)] * 2
        with pytest.raises(ValueError):
            _resolve_arrow_colors(["red", "blue"], 3)

    def test_resolve_line_styles(self):
        from graintrace.plot_postprocessing import _resolve_line_styles

        assert _resolve_line_styles(None, 2) == ["-", "-"]
        assert _resolve_line_styles("--", 3) == ["--"] * 3
        # a single (offset, dashes) tuple is repeated, not treated as per-entity
        assert _resolve_line_styles((0, (3, 1)), 2) == [(0, (3, 1))] * 2
        assert _resolve_line_styles(["-", "--", ":"], 3) == ["-", "--", ":"]
        with pytest.raises(ValueError):
            _resolve_line_styles(["-", "--"], 3)

    def test_per_fragment_line_styles_and_show_arrow(self, tmp_path):
        # per-track / per-child line styles + show_arrow toggle draw into a shared ax
        pytest.importorskip("neml2")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch

        from graintrace.orientation_helper import euler_to_mrp
        from graintrace.plot_postprocessing import (
            plot_ipf_fragmentation_tracking,
            plot_ipf_orientation_tracking,
        )

        def mrp_of(rows):
            return (
                euler_to_mrp(
                    torch.tensor(rows, dtype=torch.float64), "bunge", "degrees"
                )
                .cpu()
                .numpy()
            )

        tracks = [
            mrp_of([[10.0, 20.0, 30.0], [12.0, 21.0, 30.0]]),
            mrp_of([[50.0, 40.0, 15.0], [55.0, 42.0, 15.0]]),
        ]
        branches = [
            (
                mrp_of([[10.0, 20.0, 30.0]]),
                [mrp_of([[16.0, 22.0, 31.0]]), mrp_of([[8.0, 18.0, 29.0]])],
            )
        ]
        fig, (ax_l, ax_r) = plt.subplots(1, 2)
        ret_l = plot_ipf_orientation_tracking(
            tracks,
            angle_convention="mrp",
            arrow_color="black",
            line_style=["-", "--"],
            show_arrow=False,
            ax=ax_l,
        )
        ret_r = plot_ipf_fragmentation_tracking(
            branches,
            angle_convention="mrp",
            arrow_color="black",
            child_line_styles=[["-", "--"]],
            show_arrow=True,
            ax=ax_r,
        )
        assert ret_l is ax_l and ret_r is ax_r
        out = tmp_path / "styled.png"
        fig.savefig(out)
        plt.close(fig)
        assert out.exists()

    def test_ipf_final_color_and_black_start(self, tmp_path):
        # arrow_color="ipf_final" tints by the endpoint orientation; start_color="black"
        pytest.importorskip("neml2")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch

        from graintrace.orientation_helper import euler_to_mrp
        from graintrace.plot_postprocessing import (
            plot_ipf_fragmentation_tracking,
            plot_ipf_orientation_tracking,
        )

        def mrp_of(rows):
            return (
                euler_to_mrp(
                    torch.tensor(rows, dtype=torch.float64), "bunge", "degrees"
                )
                .cpu()
                .numpy()
            )

        tracks = [mrp_of([[10.0, 20.0, 30.0], [14.0, 22.0, 31.0]])]
        branches = [(mrp_of([[10.0, 20.0, 30.0]]), [mrp_of([[16.0, 22.0, 31.0]])])]
        fig, (ax_l, ax_r) = plt.subplots(1, 2)
        ret_l = plot_ipf_orientation_tracking(
            tracks,
            angle_convention="mrp",
            arrow_color="ipf_final",
            start_color="black",
            ax=ax_l,
        )
        ret_r = plot_ipf_fragmentation_tracking(
            branches,
            angle_convention="mrp",
            arrow_color="ipf_final",
            start_color="black",
            ax=ax_r,
        )
        assert ret_l is ax_l and ret_r is ax_r
        out = tmp_path / "ipf_final.png"
        fig.savefig(out)
        plt.close(fig)
        assert out.exists()

    def test_per_child_colors_and_parent_color(self, tmp_path):
        # per-child categorical colors (a grain's sub-grains contrast) + a gray parent
        pytest.importorskip("neml2")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch

        from graintrace.orientation_helper import euler_to_mrp
        from graintrace.plot_postprocessing import plot_ipf_fragmentation_tracking

        def mrp_of(rows):
            return (
                euler_to_mrp(
                    torch.tensor(rows, dtype=torch.float64), "bunge", "degrees"
                )
                .cpu()
                .numpy()
            )

        branches = [
            (
                mrp_of([[10.0, 20.0, 30.0]]),
                [mrp_of([[16.0, 22.0, 31.0]]), mrp_of([[6.0, 18.0, 29.0]])],
            )
        ]
        fig, ax = plt.subplots()
        ret = plot_ipf_fragmentation_tracking(
            branches,
            angle_convention="mrp",
            child_colors=[["#1b9e77", "#d95f02"]],
            parent_color="0.35",
            start_color="black",
            child_line_styles=[["-", "--"]],
            ax=ax,
        )
        assert ret is ax
        out = tmp_path / "child_colors.png"
        fig.savefig(out)
        plt.close(fig)
        assert out.exists()


class TestAddElementFieldToExodus:
    def test_writes_per_element_field(self, tmp_path):
        pytest.importorskip("neml2")
        import glob

        # Use a shipped hex mesh if present; skip otherwise (needs a real .e).
        candidates = glob.glob("mwe_data/**/mesh.e", recursive=True)
        if not candidates:
            pytest.skip("no shipped Exodus hex mesh available")
        import scipy.io as sio

        from graintrace.ipf_postprocess import IPFProcessor

        mesh_file = candidates[0]
        with sio.netcdf_file(mesh_file, "r", mmap=False) as f:
            coords = np.stack(
                [
                    np.array(f.variables["coordx"][:]),
                    np.array(f.variables["coordy"][:]),
                    np.array(f.variables["coordz"][:]),
                ],
                axis=1,
            )
            centroids = []
            for eb in range(int(f.dimensions["num_el_blk"])):
                connect = np.array(f.variables[f"connect{eb + 1}"][:])
                centroids.append(coords[connect - 1, :].mean(axis=1))
        centroids = np.vstack(centroids)
        values = np.arange(len(centroids), dtype=float)

        ipf = IPFProcessor(
            crystal_symmetry="432", sample_symmetry="1", save_dir=str(tmp_path)
        )
        out = ipf.add_element_field_to_exodus(
            mesh_file,
            centroids,
            values,
            field_name="fragment_label",
            output_file="field.e",
        )
        assert out is not None
        with sio.netcdf_file(str(out), "r") as f:
            assert "name_elem_var" in f.variables

    def test_writes_per_element_rgb(self, tmp_path):
        pytest.importorskip("neml2")
        import glob

        candidates = glob.glob("mwe_data/**/mesh.e", recursive=True)
        if not candidates:
            pytest.skip("no shipped Exodus hex mesh available")
        import scipy.io as sio

        from graintrace.ipf_postprocess import IPFProcessor

        mesh_file = candidates[0]
        with sio.netcdf_file(mesh_file, "r", mmap=False) as f:
            coords = np.stack(
                [
                    np.array(f.variables["coordx"][:]),
                    np.array(f.variables["coordy"][:]),
                    np.array(f.variables["coordz"][:]),
                ],
                axis=1,
            )
            centroids = []
            for eb in range(int(f.dimensions["num_el_blk"])):
                connect = np.array(f.variables[f"connect{eb + 1}"][:])
                centroids.append(coords[connect - 1, :].mean(axis=1))
        centroids = np.vstack(centroids)
        rng = np.random.default_rng(0)
        rgb = rng.random((len(centroids), 3))

        ipf = IPFProcessor(
            crystal_symmetry="432", sample_symmetry="1", save_dir=str(tmp_path)
        )
        out = ipf.add_element_rgb_to_exodus(
            mesh_file, centroids, rgb, output_file="rgb.e"
        )
        assert out is not None
        with sio.netcdf_file(str(out), "r") as f:
            names = f.variables["name_elem_var"][:]
            assert int(f.dimensions["num_elem_var"]) == 3
            assert names.shape[0] == 3
        # an RGBA array (4 cols) is accepted; mismatched lengths raise
        ipf.add_element_rgb_to_exodus(
            mesh_file,
            centroids,
            np.hstack([rgb, np.ones((len(centroids), 1))]),
            output_file="rgba.e",
        )
        with pytest.raises(ValueError):
            ipf.add_element_rgb_to_exodus(mesh_file, centroids, rgb[:-1])

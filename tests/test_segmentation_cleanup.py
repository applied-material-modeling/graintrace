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

"""Regression tests for the segmentation clean-up stages (PR #64 follow-ups).

Covers the orientation-gated fragment absorption, the agglomerative (non-chaining)
adjacency re-merge, the leak-aware percolation check, label ``-1`` / background
handling, the robust grain-table mean, the ``grain_fragments`` gamma default and
the per-gamma diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _mrp(euler_deg):
    import torch

    from graintrace.orientation_helper import euler_to_mrp

    return (
        euler_to_mrp(
            torch.tensor(np.asarray(euler_deg, dtype=float), dtype=torch.float64),
            "bunge",
            "degrees",
        )
        .cpu()
        .numpy()
    )


def _grid(nx, ny, nz):
    g = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return np.stack(g, -1).reshape(-1, 3).astype(float)


# --------------------------------------------------------------------------- #
# Reference (pre-vectorisation) implementations, for bit-exact legacy checks
# --------------------------------------------------------------------------- #
def _legacy_adjacency(coords, labels, k_neighbors):
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    kq = min(k_neighbors + 1, len(coords))
    _, idx = tree.query(coords, k=kq)
    idx = np.atleast_2d(idx)
    contacts = {}
    for i in range(coords.shape[0]):
        li = labels[i]
        for j in idx[i, 1:]:
            lj = labels[j]
            if li == lj:
                continue
            key = (int(min(li, lj)), int(max(li, lj)))
            contacts[key] = contacts.get(key, 0) + 1
    return contacts


def _legacy_absorb(coords, labels, min_size, k_neighbors=8, max_passes=25):
    labels = np.asarray(labels, dtype=np.int64).copy()
    for _ in range(max_passes):
        uniq, counts = np.unique(labels, return_counts=True)
        small = set(int(u) for u, c in zip(uniq, counts) if c < min_size)
        if not small or len(uniq) == 1:
            break
        contacts = _legacy_adjacency(coords, labels, k_neighbors)
        best = {}
        for (a, b), n in contacts.items():
            for x, y in ((a, b), (b, a)):
                if n > best.get(x, (0, -1))[0]:
                    best[x] = (n, y)
        remap = {}
        for s in small:
            if s in best:
                remap[s] = best[s][1]
        if not remap:
            break
        for s in list(remap):
            seen = {s}
            tgt = remap[s]
            while tgt in remap and tgt not in seen:
                seen.add(tgt)
                tgt = remap[tgt]
            remap[s] = tgt
        labels = np.array([remap.get(int(l), int(l)) for l in labels], dtype=np.int64)
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int64)


class TestVectorisedAdjacencyMatchesLegacy:
    """The vectorised kNN contact/absorb must reproduce the old loops exactly."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_contacts_same_pairs_counts_and_order(self, seed):
        from graintrace.fragmentation import _knn_cluster_adjacency

        rng = np.random.default_rng(seed)
        coords = _grid(9, 8, 3) + rng.normal(0, 0.05, (216, 3))
        labels = rng.integers(0, 12, len(coords))
        new = _knn_cluster_adjacency(coords, labels, 8)
        old = _legacy_adjacency(coords, labels, 8)
        assert list(new.items()) == list(old.items())

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_ungated_absorb_is_bit_identical(self, seed):
        from graintrace.fragmentation import FragmentationAnalyzer

        rng = np.random.default_rng(seed)
        coords = _grid(12, 10, 4)
        # a few big blocks plus scattered tiny fragments, incl. small->small chains
        labels = (coords[:, 0] // 4).astype(np.int64) * 3 + (coords[:, 1] // 4)
        noise = rng.random(len(coords)) < 0.08
        labels[noise] = rng.integers(20, 60, noise.sum())
        new = FragmentationAnalyzer.absorb_fragments(coords, labels, 6)
        old = _legacy_absorb(coords, labels, 6)
        assert np.array_equal(new, old)


class TestLabelHandling:
    """Labels < 0 (unsegmented) and background labels are never clusters."""

    def test_absorb_leaves_unsegmented_points_alone(self):
        from graintrace.fragmentation import FragmentationAnalyzer

        labels = np.array([0] * 50 + [-1] * 5 + [1] * 50)
        coords = np.c_[np.arange(105.0), np.zeros(105), np.zeros(105)]
        out = FragmentationAnalyzer.absorb_fragments(coords, labels, 10)
        assert np.array_equal(out[50:55], [-1] * 5)
        assert set(np.unique(out[:50])) == {0} and set(np.unique(out[55:])) == {1}

    def test_absorb_raw_grid_void_is_background(self):
        from graintrace.fragmentation import FragmentationAnalyzer

        # raw grid: 0 void, 7/9 grains, 3 a tiny fragment, -1 unsegmented
        labels = np.array([0] * 20 + [7] * 30 + [3] * 3 + [9] * 30 + [-1] * 2)
        coords = np.c_[np.arange(85.0), np.zeros(85), np.zeros(85)]
        out = FragmentationAnalyzer.absorb_fragments(
            coords, labels, 5, background_label=(0, -1)
        )
        assert np.array_equal(out[:20], [0] * 20)  # void untouched
        assert np.array_equal(out[-2:], [-1, -1])
        grains = out[20:83]
        assert 0 not in grains  # cluster ids never collide with the void value
        assert len(np.unique(grains)) == 2  # the fragment joined a grain

    def test_remerge_leaves_unsegmented_points_alone(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        coords = _grid(10, 4, 4)
        labels = (coords[:, 0] >= 5).astype(np.int64)
        labels[:6] = -1
        mrp = _mrp(np.tile([10.0, 20.0, 30.0], (len(coords), 1)))
        mrp[:6] = _mrp([[60.0, 40.0, 5.0]] * 6)
        for frac in (None, 1.0):
            out = FragmentationAnalyzer.adjacency_remerge(
                coords, mrp, labels, 3.0, percolation_max_frac=frac
            )
            assert np.array_equal(out[:6], [-1] * 6)
            assert len(np.unique(out[6:])) == 1

    def test_spread_skips_background_void(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        eul = np.vstack(
            [
                np.tile([10.0, 20.0, 30.0], (20, 1)),
                np.random.default_rng(0).uniform(0, 90, (10, 3)),
            ]
        )
        labels = np.array([1] * 20 + [0] * 10)
        spread = FragmentationAnalyzer.cluster_orientation_spread(
            _mrp(eul), labels, background_label=(0, -1)
        )
        assert set(spread) == {1}
        chk = FragmentationAnalyzer.cluster_leak_check(
            _mrp(eul), labels, 5.0, background_label=0
        )
        assert set(chk) == {1}


class TestLeakCheck:
    """The percolation flag must catch small leaks but not real gradients."""

    def test_small_leak_is_flagged_though_p95_is_low(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        rng = np.random.default_rng(0)
        eul = np.vstack(
            [np.tile([10, 40, 0], (1000, 1)), np.tile([40, 40, 0], (45, 1))]
        )
        mrp = _mrp(eul + rng.normal(0, 0.2, eul.shape))
        labels = np.zeros(len(mrp), dtype=np.int64)
        spread = FragmentationAnalyzer.cluster_orientation_spread(mrp, labels)[0]
        assert spread < 5.0  # the p95 alone misses this 4 % leak
        chk = FragmentationAnalyzer.cluster_leak_check(mrp, labels, 5.0)[0]
        assert chk["flagged"]
        assert chk["leak_frac"] == pytest.approx(45 / 1045, abs=0.01)
        assert chk["leak_sep_deg"] > 25.0

    @pytest.mark.parametrize("noise", [0.0, 0.3])
    def test_smooth_gradient_is_not_flagged_though_p95_is_high(self, noise):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        eul = np.zeros((2000, 3))
        eul[:, 0] = np.linspace(0, 12, 2000)
        eul[:, 1] = 30
        eul += np.random.default_rng(1).normal(0, noise, eul.shape)
        mrp = _mrp(eul)
        labels = np.zeros(2000, dtype=np.int64)
        assert FragmentationAnalyzer.cluster_orientation_spread(mrp, labels)[0] > 5.0
        chk = FragmentationAnalyzer.cluster_leak_check(mrp, labels, 5.0)[0]
        assert not chk["flagged"]

    def test_a_few_outlier_voxels_do_not_flag(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        rng = np.random.default_rng(2)
        eul = np.vstack([np.tile([10, 20, 30], (1000, 1)), rng.uniform(0, 90, (6, 3))])
        chk = FragmentationAnalyzer.cluster_leak_check(
            _mrp(eul + rng.normal(0, 0.2, eul.shape)), np.zeros(1006, dtype=int), 5.0
        )[0]
        assert not chk["flagged"]

    def test_speckled_bridge_does_not_join_two_grains(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        rng = np.random.default_rng(3)
        eul = np.vstack(
            [
                np.tile([10, 20, 30], (500, 1)),
                np.tile([60, 30, 15], (500, 1)),
                rng.uniform(0, 60, (30, 3)),  # isolated noisy bridge voxels
            ]
        )
        chk = FragmentationAnalyzer.cluster_leak_check(
            _mrp(eul + rng.normal(0, 0.2, eul.shape)), np.zeros(1030, dtype=int), 5.0
        )[0]
        assert chk["flagged"] and chk["leak_frac"] > 0.4

    def test_backstop_flags_an_overly_wide_cluster(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        eul = np.zeros((2000, 3))
        eul[:, 0] = np.linspace(0, 40, 2000)  # 40 deg "gradient": not one grain at 5
        eul[:, 1] = 30
        chk = FragmentationAnalyzer.cluster_leak_check(
            _mrp(eul), np.zeros(2000, dtype=int), 5.0
        )[0]
        assert chk["flagged"] and chk["leak_frac"] == 0.0


class TestGatedAbsorb:
    """Small fragments join only an orientation-compatible neighbour."""

    @staticmethod
    def _case():
        # 12x6x4 grid: grain A (x<6), grain B (x>=6, 25 deg from A); a 2x2x2 grain C
        # inside A, 30 deg off; a 2x2x2 fragment D inside A, 1 deg off A.
        coords = _grid(12, 6, 4)
        lab = np.where(coords[:, 0] < 6, 0, 1).astype(np.int64)
        c = (coords[:, 0] < 2) & (coords[:, 1] < 2) & (coords[:, 2] < 2)
        d = (coords[:, 0] >= 3) & (coords[:, 0] < 5) & (coords[:, 1] >= 3)
        d &= (coords[:, 1] < 5) & (coords[:, 2] < 2)
        lab[c], lab[d] = 2, 3
        table = {
            0: [10.0, 20.0, 30.0],
            1: [35.0, 20.0, 30.0],
            2: [10.0, 50.0, 30.0],
            3: [11.0, 20.0, 30.0],
        }
        eul = np.array([table[int(v)] for v in lab])
        return coords, lab, _mrp(eul), c, d

    def test_gate_keeps_a_differently_oriented_small_grain(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        coords, lab, mrp, c, d = self._case()
        blind = FragmentationAnalyzer.absorb_fragments(coords, lab, 10)
        assert len(np.unique(blind)) == 2  # legacy: both C and D vanish
        gated = FragmentationAnalyzer.absorb_fragments(
            coords, lab, 10, mrp=mrp, max_miso_deg=5.0
        )
        assert len(np.unique(gated)) == 3
        assert len(np.unique(gated[c])) == 1 and gated[c][0] not in gated[~c & ~d]
        assert gated[d][0] == gated[(lab == 0)][0]  # D joined A

    def test_gate_picks_the_closest_not_the_largest_contact(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        # fragment F (x == 5) touches A (x < 5, 2 deg away) on one face and
        # B (x > 5, 20 deg away) on one face; B is bigger so has more contact
        coords = _grid(14, 3, 3)
        lab = np.where(coords[:, 0] < 5, 0, np.where(coords[:, 0] == 5, 2, 1))
        table = {0: [12.0, 20.0, 30.0], 1: [30.0, 20.0, 30.0], 2: [10.0, 20.0, 30.0]}
        mrp = _mrp(np.array([table[int(v)] for v in lab]))
        out = FragmentationAnalyzer.absorb_fragments(
            coords, lab, 10, mrp=mrp, max_miso_deg=25.0
        )
        f = lab == 2
        assert out[f][0] == out[lab == 0][0]

    def test_segment_default_keeps_small_grain_legacy_absorbs_it(self, tmp_path):
        pytest.importorskip("neml2")
        pytest.importorskip("networkit")
        from graintrace.fragmentation import FragmentationAnalyzer

        coords, lab, mrp, c, _ = self._case()
        kw = dict(
            misori_tol_deg=5.0,
            graph_mode="grid",
            gamma=0.5,
            grain_threshold_final=10,
            remerge_miso_deg=3.0,
            verbose=False,
        )
        fa = FragmentationAnalyzer()
        default = fa.segment(coords, mrp, workdir=str(tmp_path / "a"), **kw)
        assert len(np.unique(default)) == 3
        assert len(np.unique(default[c])) == 1
        legacy = fa.segment(
            coords, mrp, percolation_max_frac=1.0, workdir=str(tmp_path / "b"), **kw
        )
        assert len(np.unique(legacy)) == 2  # orientation-blind absorb


class TestAgglomerativeRemerge:
    """The no-cap re-merge recomputes means, so it cannot chain."""

    @staticmethod
    def _chain(n_blocks=8, step=2.5):
        coords = _grid(n_blocks * 10, 10, 4)
        lab = (coords[:, 0] // 10).astype(np.int64)
        eul = np.zeros((len(coords), 3))
        eul[:, 0] = 20 + step * lab
        eul[:, 1] = 30
        eul += np.random.default_rng(0).normal(0, 0.2, eul.shape)
        return coords, lab, _mrp(eul)

    def test_chain_does_not_collapse(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        coords, lab, mrp = self._chain()
        out = FragmentationAnalyzer.adjacency_remerge(coords, mrp, lab, 3.0)
        assert len(np.unique(out)) >= 4
        # no merged cluster spans more than two neighbouring blocks (<= 2.5 deg)
        for c in np.unique(out):
            blocks = np.unique(lab[out == c])
            assert blocks.max() - blocks.min() <= 1
        # with the spread guard on (the segment default), also no chain
        guarded = FragmentationAnalyzer.adjacency_remerge(
            coords, mrp, lab, 3.0, spread_tol_deg=5.0
        )
        assert len(np.unique(guarded)) >= 4

    def test_legacy_union_find_still_chains(self):
        """The cap opt-in keeps the old fixed-mean union-find (legacy labels)."""
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        coords, lab, mrp = self._chain()
        out = FragmentationAnalyzer.adjacency_remerge(
            coords, mrp, lab, 3.0, percolation_max_frac=1.0
        )
        assert len(np.unique(out)) == 1

    def test_moment_merge_equals_the_union_mean_across_variants(self):
        pytest.importorskip("neml2")
        import torch

        from graintrace.fragmentation import (
            FragmentationAnalyzer,
            _cluster_moments,
            _left_quat_matrix,
            _matrix_to_quat_np,
            _moments_mean_matrix,
            _mrp_to_matrix_np,
            _sym_miso_deg,
            _symmetry_ops_np,
        )
        from graintrace.orientation_helper import matrix_to_mrp

        rng = np.random.default_rng(4)
        ops = _symmetry_ops_np("432")
        ra = _mrp_to_matrix_np(
            _mrp(np.array([10, 20, 30]) + rng.normal(0, 0.5, (40, 3)))
        )
        rb = _mrp_to_matrix_np(
            _mrp(np.array([11, 20, 30]) + rng.normal(0, 0.5, (30, 3)))
        )
        rb = np.matmul(ops[7], rb)  # same physical orientations, another variant
        inv = np.array([0] * 40 + [1] * 30)
        mom = _cluster_moments(np.concatenate([ra, rb]), inv, 2, ops)
        mean = _moments_mean_matrix(mom)
        cand = np.einsum("kij,jl->kil", ops, mean[1])
        s_op = int(np.argmax(np.einsum("ij,kij->k", mean[0], cand)))
        lq = _left_quat_matrix(_matrix_to_quat_np(ops)[s_op])
        merged = _moments_mean_matrix((mom[0] + lq @ mom[1] @ lq.T)[None])
        ref = FragmentationAnalyzer.mean_orientation_mrp(
            matrix_to_mrp(torch.as_tensor(np.concatenate([ra, rb]))).numpy(), "432"
        )
        assert _sym_miso_deg(merged, _mrp_to_matrix_np(ref), ops)[0] < 1e-4


class TestPickGammaDiagnostics:
    def test_every_row_is_annotated_in_every_mode(self):
        pytest.importorskip("neml2")
        from graintrace.fragmentation import FragmentationAnalyzer

        mrp = _mrp(
            np.vstack([np.tile([10, 20, 30], (20, 1)), np.tile([60, 30, 15], (20, 1))])
        )
        merged = np.zeros(40, dtype=np.int64)
        split = np.array([0] * 20 + [1] * 20, dtype=np.int64)

        def sweep():
            return [
                {"gamma": 0.5, "n_grains": 1, "max_frac": 1.0, "labels": merged},
                {"gamma": 4.0, "n_grains": 2, "max_frac": 0.5, "labels": split},
            ]

        for kw in (
            {"mode": "lowest"},
            {"percolation_max_frac": 0.6},
            {"mode": "spread"},
        ):
            sw = sweep()
            FragmentationAnalyzer.pick_gamma(sw, mrp=mrp, spread_tol_deg=5.0, **kw)
            assert [r["n_flagged"] for r in sw] == [1, 0]
            assert all("max_spread_deg" in r for r in sw)
        sw = sweep()
        assert (
            FragmentationAnalyzer.pick_gamma(sw, mrp=mrp, spread_tol_deg=5.0)["gamma"]
            == 0.5
        )


class TestGrainFragmentsGammaDefault:
    def test_default_gamma_is_the_documented_sub_grain_value(
        self, tmp_path, monkeypatch
    ):
        from graintrace.fragmentation import FragmentationAnalyzer

        rng = np.random.default_rng(0)
        coords = rng.normal(size=(30, 3))
        path = tmp_path / "s0.csv"
        pd.DataFrame(
            {
                "id": np.arange(30),
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "ori_rodrigues_x": 0.0,
                "ori_rodrigues_y": 0.0,
                "ori_rodrigues_z": 0.0,
                "block_id": 1,
            }
        ).to_csv(path, index=False)

        class _Res:
            def load_field_data(self, _):
                return pd.read_csv(path)

        seen = []

        def fake_segment(self, c, m, **kw):  # pylint: disable=unused-argument
            seen.append(kw)
            return np.zeros(len(c), dtype=np.int64)

        monkeypatch.setattr(FragmentationAnalyzer, "segment", fake_segment)
        fa = FragmentationAnalyzer()
        fa.grain_fragments(_Res(), 0, min_subgrain_miso_deg=0)
        fa.grain_fragments(
            _Res(), 0, min_subgrain_miso_deg=0, seg_kwargs={"gamma": 1.0}
        )
        fa.grain_fragments(
            _Res(), 0, min_subgrain_miso_deg=0, seg_kwargs={"gamma": None}
        )
        assert [k["gamma"] for k in seen] == [0.25, 1.0, None]
        assert all(k["absorb_max_miso_deg"] is None for k in seen)

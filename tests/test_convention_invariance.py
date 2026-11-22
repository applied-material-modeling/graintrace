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

"""The orientation-convention flip (passive ``g`` -> active ``R = g^T``) is invariant.

Layer A compares every flipped primitive against its verbatim 0544c83 copy
(``tests/_orientation_legacy.py``): matrices are transposed, Euler outputs and
misorientation angles are unchanged, MRP outputs are the inverse rotation.

Layer B re-runs the orientation-consuming pipelines on the inputs frozen in
``tests/data/orientation_convention_baseline.npz`` (recorded on 0544c83 by
``tests/data/make_orientation_baseline.py``) and requires the recorded results.
Euler-input pipelines get the same Euler input; MRP-input pipelines get ``-p``
(the legacy MRP was ``mrp(g) == -mrp(R)``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neml2")
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data"))

# pylint: disable=wrong-import-position,import-error
import _orientation_legacy as L  # noqa: E402
import make_orientation_baseline as MB  # noqa: E402

from graintrace import fragmentation as fr  # noqa: E402
from graintrace import orientation_helper as oh  # noqa: E402
from graintrace.frame_transform import FrameTransform  # noqa: E402
from graintrace.nf import metrics as nfm  # noqa: E402

SYMS = ("1", "432", "622")
BASELINE = Path(__file__).parent / "data" / "orientation_convention_baseline.npz"


def _T(m):
    return m.transpose(-2, -1)


def _random_euler(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(0, 360, n),
            np.degrees(np.arccos(rng.uniform(-1, 1, n))),
            rng.uniform(0, 360, n),
        ],
        axis=1,
    )


def _edge_euler():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [10.0, 180.0, 0.0],
            [359.999, 1e-7, 45.0],
            [0.0, 90.0, 0.0],
            [45.0, 54.7356, 45.0],
            [180.0, 180.0, 180.0],
            [12.0, 0.0, 27.0],
        ]
    )


def _near_pairs(n=200, seed=1):
    """Pairs 0.1-5 deg apart; 10% of the partners are reported in another variant."""
    rng = np.random.default_rng(seed)
    e1 = _random_euler(n, seed)
    sig = rng.uniform(0.1, 5.0)
    e2 = oh.perturb_orientation(e1, sig, rng=np.random.default_rng(seed + 7)).numpy()
    ops = oh.symmetry_operators("432")
    flip = rng.random(n) < 0.1
    pick = torch.as_tensor(rng.integers(1, len(ops), n))
    r2 = oh.euler_to_matrix(e2)
    var = oh.matrix_to_euler(r2 @ ops[pick]).numpy()
    e2[flip] = var[flip]
    return e1, e2


def _miso_deg(e1, e2, sym="1"):
    return oh.misorientation(e1, e2, "bunge", "degrees", sym).numpy()


# --------------------------------------------------------------------------- #
# Layer A: primitives vs the frozen legacy copies
# --------------------------------------------------------------------------- #
class TestPrimitives:
    """Every flipped primitive equals its legacy counterpart up to the documented side."""

    def test_matrix_is_legacy_transpose(self):
        e = np.vstack([_random_euler(), _edge_euler()])
        for conv in ("bunge", "kocks", "roe"):
            new = oh.euler_to_matrix(e, conv)
            old = L.legacy_euler_to_matrix(e, conv)
            assert torch.allclose(new, _T(old), atol=1e-12, rtol=0)

    def test_matrix_to_euler_unchanged(self):
        e = np.vstack([_random_euler(), _edge_euler()])
        for conv in ("bunge", "kocks"):
            for unit in ("degrees", "radians"):
                ei = e if unit == "degrees" else np.deg2rad(e)
                new = oh.matrix_to_euler(oh.euler_to_matrix(ei, conv, unit), conv, unit)
                old = L.legacy_matrix_to_euler(
                    L.legacy_euler_to_matrix(ei, conv, unit), conv, unit
                )
                assert torch.equal(new, old)

    def test_euler_to_mrp_is_inverse_rotation(self):
        e = _random_euler()
        new = oh.euler_to_mrp(e)
        old = L.legacy_euler_to_mrp(e)
        assert torch.allclose(
            oh.mrp_to_matrix(new), _T(oh.mrp_to_matrix(old)), atol=1e-12, rtol=0
        )
        assert torch.allclose(oh.legacy_mrp_to_mrp(old), new, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("sym", SYMS)
    def test_misorientation_equal(self, sym):
        e1, e2 = _near_pairs()
        r1, r2 = _random_euler(300, 3), _random_euler(300, 4)
        for a, b in ((e1, e2), (r1, r2), (_edge_euler(), _edge_euler()[::-1].copy())):
            new = oh.misorientation(a, b, "bunge", "degrees", sym)
            old = L.legacy_misorientation(a, b, "bunge", "degrees", sym)
            assert torch.allclose(new, old, atol=1e-5, rtol=0)
            new_nf = nfm.misorientation(a, b, "bunge", "degrees", sym)
            assert torch.allclose(new_nf, old, atol=1e-5, rtol=0)

    @pytest.mark.parametrize("sym", SYMS)
    def test_fundamental_zone_is_legacy_transpose(self, sym):
        e = np.vstack([_random_euler(), _edge_euler()])
        new = oh.move_to_fundamental_zone(oh.euler_to_matrix(e), sym)
        old = L.legacy_move_to_fundamental_zone(L.legacy_euler_to_matrix(e), sym)
        assert torch.allclose(new, _T(old), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("sym", SYMS)
    def test_nf_fold_is_legacy_transpose(self, sym):
        e1, e2 = _near_pairs()
        e = torch.as_tensor(np.stack([e1, e2], axis=1))  # (N, 2, 3) sets
        new = nfm.fold_to_reference_variant(oh.euler_to_matrix(e), sym)
        old = L.legacy_fold_to_reference_variant(L.legacy_euler_to_matrix(e), sym)
        assert torch.allclose(new, _T(old), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("sym", SYMS)
    def test_averages_equal_as_orientations(self, sym):
        e1, e2 = _near_pairs(60)
        rng = np.random.default_rng(5)
        w = rng.uniform(0.1, 1.0, 60)
        new = oh.average_orientation(np.vstack([e1[:30], e2[:30]]), None, symmetry=sym)
        old = L.legacy_average_orientation(
            np.vstack([e1[:30], e2[:30]]), None, symmetry=sym
        )
        assert _miso_deg(new[None], old[None]).max() < 1e-5
        new = oh.average_orientation(e2, w, symmetry=sym)
        old = L.legacy_average_orientation(e2, w, symmetry=sym)
        assert _miso_deg(new[None], old[None]).max() < 1e-5

        sets = torch.as_tensor(np.stack([e1, e2, e1], axis=1))  # (N, 3, 3) sets
        m_new, eu_new = nfm.average_rotations(sets, "bunge", "degrees", sym)
        m_old, eu_old = L.legacy_average_rotations(sets, "bunge", "degrees", sym)
        assert _miso_deg(eu_new.numpy(), eu_old.numpy()).max() < 1e-5
        assert torch.allclose(
            oh.mrp_to_matrix(m_new), _T(oh.mrp_to_matrix(m_old)), atol=1e-9, rtol=0
        )

    def test_seeded_perturbation_identical(self):
        e = _random_euler(200, 9)
        for sig in (0.5, 3.0, 20.0):
            new = oh.perturb_orientation(e, sig, rng=np.random.default_rng(3))
            old = L.legacy_perturb_orientation(e, sig, rng=np.random.default_rng(3))
            assert torch.allclose(new, old, atol=1e-9, rtol=0)


class TestFragmentationHelpers:
    """Module helpers fed the same orientation as MRP(-p) give the legacy result."""

    @staticmethod
    def _mats(sym="432"):
        e1, e2 = _near_pairs(300)
        e = np.vstack([e1, e2])
        p_old = L.legacy_euler_to_mrp(e).numpy()
        r_new = fr._mrp_to_matrix_np(-p_old)
        r_old = fr._mrp_to_matrix_np(p_old)
        return r_new, r_old, fr._symmetry_ops_np(sym), p_old

    @pytest.mark.parametrize("sym", SYMS)
    def test_fold_and_miso(self, sym):
        r_new, r_old, ops, _ = self._mats(sym)
        ref_new, ref_old = r_new[::-1].copy(), r_old[::-1].copy()
        new = fr._fold_to_reference(r_new, ref_new, ops)
        old = L.legacy_fold_to_reference(r_old, ref_old, ops)
        assert np.allclose(new, np.swapaxes(old, -1, -2), atol=1e-12)
        assert np.allclose(
            fr._sym_miso_deg(r_new, ref_new, ops),
            L.legacy_sym_miso_deg(r_old, ref_old, ops),
            atol=1e-5,
        )

    @pytest.mark.parametrize("sym", SYMS)
    def test_cluster_tangent(self, sym):
        _, _, ops, p_old = self._mats(sym)
        inv = np.arange(len(p_old)) % 7
        mean_new, vec_new = fr._cluster_tangent(-p_old, inv, 7, ops)
        mean_old, vec_old = L.legacy_cluster_tangent(p_old, inv, 7, ops)
        assert np.allclose(mean_new, np.swapaxes(mean_old, -1, -2), atol=1e-9)
        assert np.allclose(vec_new, vec_old, atol=1e-6)

    @pytest.mark.parametrize("sym", SYMS)
    def test_mean_orientation_mrp(self, sym):
        _, _, _, p_old = self._mats(sym)
        for sl in (slice(0, 300), slice(150, 450), slice(0, 1)):
            new = fr.FragmentationAnalyzer.mean_orientation_mrp(-p_old[sl], sym)
            old = L.legacy_mean_orientation_mrp(p_old[sl], sym)
            assert np.allclose(
                oh.mrp_to_matrix(new).numpy(),
                np.swapaxes(oh.mrp_to_matrix(old).numpy(), -1, -2),
                atol=1e-9,
            )

    def test_right_quat_fold_matches_matrix_fold(self):
        """``W(q_S)`` folds quaternion moments exactly like ``R -> R S``."""
        r_new, _, ops, _ = self._mats("432")
        q = fr._matrix_to_quat_np(r_new[:5])
        for k in range(1, 9):  # neml2 quaternions of the non-180-deg ops
            w = fr._right_quat_matrix(fr._matrix_to_quat_np(ops[k : k + 1])[0])
            got = fr._quat_to_matrix_np((w @ q.T).T)
            assert np.allclose(got, r_new[:5] @ ops[k], atol=1e-12)

    @pytest.mark.parametrize("sym", ["432", "622"])
    def test_op_quaternions_cover_180_degree_ops(self, sym):
        """``_ops_quat_np`` is exact for every op (neml2's route fails at 180 deg)."""
        r_new, _, _, _ = self._mats(sym)
        ops = fr._symmetry_ops_np(sym)
        q = fr._matrix_to_quat_np(r_new[:5])
        qs = fr._ops_quat_np(ops)
        assert np.allclose(fr._quat_to_matrix_np(qs), ops, atol=1e-12)
        for k in range(len(ops)):
            got = fr._quat_to_matrix_np((fr._right_quat_matrix(qs[k]) @ q.T).T)
            assert np.allclose(got, r_new[:5] @ ops[k], atol=1e-12)


class TestFrameTransformInvariance:
    """``R' = Q R`` is the legacy ``g' = g Q^T`` transposed."""

    def test_matrices_mrp_euler(self):
        ft = FrameTransform(rotation=[("z", 172.3), ("x", 180.0), ("y", -0.9)])
        e = _random_euler(200, 11)
        r = oh.euler_to_matrix(e)
        g = L.legacy_euler_to_matrix(e)
        assert torch.allclose(
            ft.orientation_matrices(r),
            _T(L.legacy_frame_orientation_matrices(ft, g)),
            atol=1e-12,
            rtol=0,
        )
        p_old = L.legacy_euler_to_mrp(e)
        new = ft.mrp(-p_old)
        old = L.legacy_frame_mrp(ft, p_old)
        assert torch.allclose(
            oh.mrp_to_matrix(new), _T(oh.mrp_to_matrix(old)), atol=1e-9, rtol=0
        )
        assert (new.norm(dim=-1) <= 1.0 + 1e-12).all()
        # Euler in, Euler out: identical to the legacy matrix path
        old_e = L.legacy_matrix_to_euler(L.legacy_frame_orientation_matrices(ft, g))
        assert _miso_deg(ft.euler(e), old_e.numpy()).max() < 1e-5


# --------------------------------------------------------------------------- #
# Layer B: pipelines vs the frozen 0544c83 baseline
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def base():
    if not BASELINE.exists():
        pytest.skip("orientation baseline npz missing")
    return dict(np.load(BASELINE))


def _tables_equal(new_df, old_arr):
    new = MB.table_array(new_df)
    assert new.shape == old_arr.shape
    assert np.allclose(new[:, :5], old_arr[:, :5], atol=1e-8)
    assert _miso_deg(new[:, 5:], old_arr[:, 5:]).max() < 1e-5


def _split_equal(res, base, tag):
    summ = MB.split_summary(res)
    for k, v in summ.items():
        ref = base[f"{tag}_split_{k}"]
        if v.dtype.kind in "US":
            assert v.tolist() == ref.tolist(), k
        else:
            assert np.allclose(v, ref, atol=1e-6), k


class TestPipelinesBaseline:
    """Pipelines re-run on the 0544c83 inputs reproduce the 0544c83 outputs."""

    def test_grid_mrp_is_negated_legacy(self, base):
        new = oh.euler_to_mrp(torch.as_tensor(base["grid_euler"])).numpy()
        assert np.allclose(
            oh.mrp_to_matrix(new).numpy(),
            np.swapaxes(oh.mrp_to_matrix(base["grid_mrp_legacy"]).numpy(), -1, -2),
            atol=1e-12,
        )

    def test_grid_segmentation_pipeline(self, base):
        an = fr.FragmentationAnalyzer(symmetry="432")
        c, over, truth = base["grid_coords"], base["grid_oversplit"], base["grid_truth"]
        mrp = -base["grid_mrp_legacy"]
        seg = an.segment(c, mrp, **MB.SEG_KW)
        assert np.array_equal(seg, base["grid_segment"])
        _tables_equal(an.seg_to_grain_table(seg, c, mrp=mrp), base["grid_table_mrp"])
        _tables_equal(
            an.seg_to_grain_table(
                seg,
                c,
                euler=base["grid_euler"],
                angle_convention="bunge",
                angle_type="degrees",
            ),
            base["grid_table_euler"],
        )
        _tables_equal(
            an.seg_to_grain_table(seg, c, mrp=mrp, trim_miso_deg=5.0),
            base["grid_table_trim"],
        )
        spread = an.cluster_orientation_spread(mrp, over, "432", max_points=0)
        assert np.allclose(
            np.array([[k, v] for k, v in sorted(spread.items())]),
            base["grid_spread"],
            atol=1e-6,
        )
        leak = an.cluster_leak_check(mrp, truth * 0 + (truth >= 2), 5.0, "432")
        got = np.array(
            [
                [k, v["flagged"], v["spread_deg"], v["leak_frac"], v["leak_sep_deg"]]
                for k, v in sorted(leak.items())
            ],
            dtype=float,
        )
        assert np.allclose(got, base["grid_leak"], atol=1e-6)
        assert np.array_equal(
            an.absorb_fragments(c, over, 30, mrp=mrp, max_miso_deg=5.0, symmetry="432"),
            base["grid_absorb"],
        )
        assert np.array_equal(
            an.adjacency_remerge(c, mrp, over, 3.5, "432"), base["grid_remerge_agg"]
        )
        assert np.array_equal(
            an.adjacency_remerge(c, mrp, over, 3.5, "432", spread_tol_deg=5.0),
            base["grid_remerge_spread"],
        )
        assert np.array_equal(
            an.adjacency_remerge(c, mrp, over, 3.5, "432", percolation_max_frac=0.5),
            base["grid_remerge_uf"],
        )
        assert np.array_equal(
            an.merge_fragments_by_misorientation(mrp, over, 3.5, "432"),
            base["grid_merge_frag"],
        )
        means = np.stack(
            [an.mean_orientation_mrp(mrp[truth == g], "432") for g in range(5)]
        )
        assert np.allclose(
            oh.mrp_to_matrix(means).numpy(),
            np.swapaxes(oh.mrp_to_matrix(base["grid_mean_mrp_legacy"]).numpy(), -1, -2),
            atol=1e-9,
        )

    def test_ff_split_detection(self, base):
        import glob
        import json

        an = fr.FragmentationAnalyzer(symmetry="432")
        ff = sorted(glob.glob(str(MB.MWE / "synthetic_split_ff" / "ff_step*.csv")))
        with open(
            MB.MWE / "synthetic_split_ff" / "ground_truth.json", encoding="utf-8"
        ) as fh:
            s = json.load(fh)["split_step"]
        res = an.detect_splits(
            ff[s - 1], ff[s], assignment="one_to_one", n_null=5, seed=0, **MB.SPLIT_KW
        )
        _split_equal(res, base, "ff")

    @pytest.mark.parametrize("tag", ["ebsd", "nf"])
    def test_voxel_split_detection(self, base, tag):
        an = fr.FragmentationAnalyzer(symmetry="432")
        tables = []
        for i, df in enumerate(MB.voxel_steps()[tag]):
            c = df[["x", "y", "z"]].to_numpy(float)
            m = oh.euler_to_mrp(
                torch.as_tensor(df[["Eul0", "Eul1", "Eul2"]].to_numpy(float))
            ).numpy()
            lab = an.segment(c, m, **MB.VOXEL_SEG_KW)
            assert np.array_equal(lab, base[f"{tag}_segment_{i}"])
            tab = an.seg_to_grain_table(lab, c, mrp=m)
            _tables_equal(tab, base[f"{tag}_table_{i}"])
            tables.append(tab)
        _split_equal(an.detect_splits(tables[0], tables[1], **MB.SPLIT_KW), base, tag)

    def test_grain_fragments_on_negated_cpfe_mrp(self, base, tmp_path):
        if "frag_table" not in base:
            pytest.skip("cpfe_ff_fragmentation was absent when the baseline was made")
        src = MB.FRAG_DIR  # frozen pre-regeneration inputs
        os.makedirs(tmp_path / "mesh_out")
        df = pd.read_csv(src / "mesh_out" / MB.FRAG_STEP_FILE)
        for ax in "xyz":
            df[f"ori_rodrigues_{ax}"] = -df[f"ori_rodrigues_{ax}"]
        df.to_csv(tmp_path / "mesh_out" / MB.FRAG_STEP_FILE, index=False)
        table, ids, codes = MB.run_grain_fragments(
            str(tmp_path / "mesh_out"), str(src / "out.csv")
        )
        assert np.array_equal(table, base["frag_table"])
        assert np.array_equal(ids, base["frag_ids"])
        assert codes.tolist() == base["frag_codes"].tolist()

    def test_graph_grain_matcher(self, base, tmp_path):
        pytest.importorskip("torch_geometric")
        if "match_a_to_b" not in base:
            pytest.skip("matcher baseline absent")
        frames = [
            base[f"match_in_{k}"] for k in ("xyz", "eul", "xyz_b", "eul_b", "perm")
        ]
        a2b, fa, fb = MB.run_matcher(*frames, str(tmp_path))
        assert np.array_equal(a2b, base["match_a_to_b"])
        assert np.allclose(fa, base["match_fa"], atol=1e-9)
        assert np.allclose(fb, base["match_fb"], atol=1e-9)

    def test_region_base_stitching(self, base, tmp_path):
        cols = ["X", "Y", "Z", "GrainRadius", "Eul0", "Eul1", "Eul2"]
        s0 = pd.DataFrame(base["stitch_s0"], columns=cols)
        s1 = pd.DataFrame(base["stitch_s1"], columns=cols)
        out = MB.run_stitch(s0, s1, str(tmp_path))
        ref = base["stitch_out"]
        assert out.shape == ref.shape
        assert np.allclose(out[:, :4], ref[:, :4], atol=1e-8)
        assert _miso_deg(out[:, 4:], ref[:, 4:]).max() < 1e-5

    def test_stitch_inputs_regenerate_bit_identically(self, base):
        """perturb_orientation keeps seeded draws across the flip."""
        s0, s1 = MB.stitch_inputs()
        assert np.allclose(s0.to_numpy(float), base["stitch_s0"], atol=1e-9)
        assert np.allclose(s1.to_numpy(float), base["stitch_s1"], atol=1e-9)

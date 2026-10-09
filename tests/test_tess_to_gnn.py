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

"""Tests for graintrace.tess_to_gnn (first tests for this module, issue #31).

Everything runs off one hand-written ``.tess`` fixture: two unit cubes sharing
the plane ``x = 1``, so 2 cells, 12 vertices, 20 edges, 11 faces, exactly one of
them internal. It is small enough to read and check by eye, and it needs no
NEPER -- ``NeperTessToGraphNN``'s geometry hooks are no-op placeholders, so
construction only parses the file.

The module imports ``torch_geometric`` at top level (the ``gnn`` extra, which CI
does not install), so every test gates on it; the orientation descriptors route
through ``neml2`` and gate on that too.

Three tests are ``xfail(strict=True)``. They pin known defects recorded in issue
#31 and are deliberately *not* fixed here -- this PR is tests only. A strict
xfail turns into a failure the moment the defect is fixed, which is the signal to
flip the test to a plain assertion.
"""
from __future__ import annotations

import pytest

# The *ori descriptor line, substituted into the template below.
_TESS_TEMPLATE = """\
***tess
 **format
   3.5
 **general
   3
   standard
 **cell
   2
  *id
   1 2
  *seed
   1  0.500000  0.500000  0.500000  1.000000
   2  1.500000  0.500000  0.500000  1.000000
{ori_block}\
 **vertex
   12
   1  0.0 0.0 0.0  0
   2  0.0 1.0 0.0  0
   3  0.0 1.0 1.0  0
   4  0.0 0.0 1.0  0
   5  1.0 0.0 0.0  0
   6  1.0 1.0 0.0  0
   7  1.0 1.0 1.0  0
   8  1.0 0.0 1.0  0
   9  2.0 0.0 0.0  0
   10 2.0 1.0 0.0  0
   11 2.0 1.0 1.0  0
   12 2.0 0.0 1.0  0
 **edge
   20
   1   1 2  0
   2   2 3  0
   3   3 4  0
   4   4 1  0
   5   5 6  0
   6   6 7  0
   7   7 8  0
   8   8 5  0
   9   9 10 0
   10  10 11 0
   11  11 12 0
   12  12 9  0
   13  1 5  0
   14  2 6  0
   15  3 7  0
   16  4 8  0
   17  5 9  0
   18  6 10 0
   19  7 11 0
   20  8 12 0
 **face
   11
   1 4   1 2 3 4
    4   1 2 3 4
    1.0 0.0 0.0 0.0
    0
   2 4   5 6 7 8
    4   5 6 7 8
    1.0 0.0 0.0 -1.0
    0
   3 4   9 10 11 12
    4   9 10 11 12
    1.0 0.0 0.0 -2.0
    0
   4 4   1 5 8 4
    4   13 -8 -16 4
    0.0 1.0 0.0 0.0
    0
   5 4   2 6 7 3
    4   14 6 -15 -2
    0.0 1.0 0.0 -1.0
    0
   6 4   1 2 6 5
    4   1 14 -5 -13
    0.0 0.0 1.0 0.0
    0
   7 4   4 8 7 3
    4   16 -7 -15 3
    0.0 0.0 1.0 -1.0
    0
   8 4   5 9 12 8
    4   17 -12 -20 8
    0.0 1.0 0.0 0.0
    0
   9 4   6 10 11 7
    4   18 10 -19 -6
    0.0 1.0 0.0 -1.0
    0
   10 4  5 6 10 9
    4   5 18 -9 -17
    0.0 0.0 1.0 0.0
    0
   11 4  8 12 11 7
    4   20 11 -19 -7
    0.0 0.0 1.0 -1.0
    0
 **polyhedron
   2
   1 6   -1 2 -4 5 -6 7
   2 6   -2 3 -8 9 -10 11
***end
"""

# Rodrigues (Gibbs) vectors -- what Neper writes by default.
_ORI_RODRIGUES = """\
  *ori
   rodrigues:active
   0.100000  0.000000  0.000000
   0.000000  0.200000  0.000000
"""

# Face 6's signed edge list and cell 1's signed face list, verbatim from the
# template; the sign tests rewrite exactly these two lines.
_FACE6_EDGES = "    4   1 14 -5 -13"
_CELL1_FACES = "   1 6   -1 2 -4 5 -6 7"


def _write_tess(tmp_path, ori_block: str = "", name: str = "two_cell.tess") -> str:
    """Write the two-cube fixture (optionally with an ``*ori`` block)."""
    path = tmp_path / name
    path.write_text(_TESS_TEMPLATE.format(ori_block=ori_block), encoding="utf-8")
    return str(path)


def _parser(tmp_path, ori_block: str = "", substitutions=(), name="two_cell.tess"):
    """Build a ``NeperTessToGraphNN`` over the fixture, with optional edits."""
    from graintrace.tess_to_gnn import NeperTessToGraphNN

    text = _TESS_TEMPLATE.format(ori_block=ori_block)
    for old, new in substitutions:
        assert old in text, f"fixture no longer contains {old!r}"
        text = text.replace(old, new)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return NeperTessToGraphNN(str(path))


def _to_mrp(parser, values, descriptor):
    """Call the private descriptor converter under test."""
    return parser._tess_ori_to_mrp(
        values, descriptor
    )  # pylint: disable=protected-access


class TestParseTess:
    """The ``.tess`` parser on a complete, well-formed file."""

    def test_counts_and_zero_based_indices(self, tmp_path):
        pytest.importorskip("torch_geometric")
        parser = _parser(tmp_path)

        assert parser.cell_seeds.shape == (2, 4)
        assert parser.cell_seeds[1].tolist() == [1.5, 0.5, 0.5, 1.0]
        assert parser.vertices.shape == (12, 3)
        assert parser.vertices[8].tolist() == [2.0, 0.0, 0.0]

        # Vertex ids are 1-based in the file, 0-based in memory.
        assert parser.edges.shape == (20, 2)
        assert parser.edges[0].tolist() == [0, 1]
        assert parser.edges[12].tolist() == [0, 4]

        assert len(parser.face_vertices) == 11
        assert len(parser.face_edges) == 11
        assert parser.face_vertices[0] == [0, 1, 2, 3]
        assert len(parser.cell_to_faces) == 2
        assert all(len(faces) == 6 for faces in parser.cell_to_faces)

    def test_negative_edge_and_face_indices_stay_negative(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from graintrace.tess_to_gnn import decode_signed_index

        parser = _parser(tmp_path)

        # Face 4's edge list is "13 -8 -16 4" -> the two reversed edges keep
        # their sign through the 1-based -> 0-based shift. Reversed entries are
        # stored as the bitwise complement of the 0-based index, so -8 encodes
        # "edge index 7, reversed" (see encode_signed_index).
        assert parser.face_edges[3] == [12, -8, -16, 3]
        assert [decode_signed_index(e) for e in parser.face_edges[3]] == [
            (12, False),
            (7, True),
            (15, True),
            (3, False),
        ]
        # Cell 2's face list is "-2 3 -8 9 -10 11".
        assert parser.cell_to_faces[1] == [-2, 2, -8, 8, -10, 10]
        assert [decode_signed_index(f) for f in parser.cell_to_faces[1]] == [
            (1, True),
            (2, False),
            (7, True),
            (8, False),
            (9, True),
            (10, False),
        ]

    def test_missing_file_raises(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from graintrace.tess_to_gnn import NeperTessToGraphNN

        with pytest.raises(FileNotFoundError, match="Tessellation file not found"):
            NeperTessToGraphNN(str(tmp_path / "absent.tess"))

    def test_ori_section_is_parsed_into_mrp(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        parser = _parser(tmp_path, ori_block=_ORI_RODRIGUES)

        assert parser.ori_type == "rodrigues"
        assert parser.orientations.shape == (2, 3)

    def test_absent_ori_section_leaves_an_empty_tensor(self, tmp_path):
        pytest.importorskip("torch_geometric")
        parser = _parser(tmp_path)

        assert parser.ori_type == "none"
        assert parser.orientations.shape == (0, 3)

    def test_sign_of_index_one_is_preserved(self, tmp_path):
        pytest.importorskip("torch_geometric")

        positive = _parser(
            tmp_path,
            substitutions=((_CELL1_FACES, "   1 6   1 2 -4 5 -6 7"),),
            name="plus.tess",
        )
        negative = _parser(tmp_path, name="minus.tess")  # template has "-1"
        assert positive.cell_to_faces[0][0] != negative.cell_to_faces[0][0]

        forward = _parser(tmp_path, name="fwd.tess")
        reversed_ = _parser(
            tmp_path,
            substitutions=((_FACE6_EDGES, "    4   -1 14 -5 -13"),),
            name="rev.tess",
        )
        assert forward.face_edges[5][0] != reversed_.face_edges[5][0]


class TestValidateTopology:
    """Face-to-cell connectivity checks."""

    def test_classifies_the_shared_face_as_internal(self, tmp_path):
        pytest.importorskip("torch_geometric")
        report = _parser(tmp_path).validate_topology(verbose=False)

        # Face 2 (index 1) is the only one both cubes carry.
        assert report["internal"] == [1]
        assert sorted(report["boundary"]) == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert report["nonmanifold"] == []

    def test_non_manifold_face_raises_when_verbose(self, tmp_path):
        pytest.importorskip("torch_geometric")
        # Cell 1 lists face 2 twice, so face index 1 is claimed three times.
        parser = _parser(
            tmp_path,
            substitutions=((_CELL1_FACES, "   1 7   -1 2 -4 5 -6 7 2"),),
        )
        assert parser.validate_topology(verbose=False)["nonmanifold"] == [1]
        with pytest.raises(ValueError, match="Faces not properly shared"):
            parser.validate_topology(verbose=True)

    @pytest.mark.xfail(
        strict=True,
        reason="issue #31 / survey 2.8: validate_topology only raises inside its "
        "`if verbose` branch, so a quiet call reports a broken tessellation by "
        "return value and lets the caller sail past it",
    )
    def test_non_manifold_face_raises_when_quiet(self, tmp_path):
        pytest.importorskip("torch_geometric")
        parser = _parser(
            tmp_path,
            substitutions=((_CELL1_FACES, "   1 7   -1 2 -4 5 -6 7 2"),),
        )
        with pytest.raises(ValueError, match="Faces not properly shared"):
            parser.validate_topology(verbose=False)

    def test_isolated_face_is_reported(self, tmp_path):
        pytest.importorskip("torch_geometric")
        # Cell 2 drops face 11, leaving face index 10 claimed by nobody.
        parser = _parser(
            tmp_path,
            substitutions=(("   2 6   -2 3 -8 9 -10 11", "   2 5   -2 3 -8 9 -10"),),
        )
        assert parser.validate_topology(verbose=False)["isolated"] == [10]


class TestBuildCellGraph:
    """``build_cell_graph`` turns shared faces into graph edges."""

    def test_shared_face_becomes_the_single_graph_edge(self, tmp_path):
        pytest.importorskip("torch_geometric")
        graph = _parser(tmp_path).build_cell_graph()

        assert graph.edge_index.shape == (2, 1)
        assert graph.edge_index.T.tolist() == [[0, 1]]
        # The default node feature is the 4-component seed (x, y, z, weight).
        assert graph.x.shape == (2, 4)
        assert graph.feature_names == ["seed_centroid"]
        assert graph.feature_slices == {"seed_centroid": (0, 4)}
        assert graph.edge_attr.shape == (1, 0)

    def test_dataframe_columns_become_node_features(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pd = pytest.importorskip("pandas")

        parser = _parser(tmp_path)
        parser.register_dataframe_features(
            pd.DataFrame({"Eul0": [10.0, 20.0], "flag": [True, False]}), verbose=False
        )
        graph = parser.build_cell_graph()

        assert graph.feature_slices == {
            "seed_centroid": (0, 4),
            "Eul0": (4, 5),
            "flag": (5, 6),
        }
        assert graph.x.shape == (2, 6)
        assert graph.x[:, 4].tolist() == [10.0, 20.0]
        assert graph.x[:, 5].tolist() == [1.0, 0.0]  # bool -> float

    def test_dataframe_length_must_match_the_cell_count(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pd = pytest.importorskip("pandas")

        parser = _parser(tmp_path)
        with pytest.raises(ValueError, match="DataFrame length mismatch"):
            parser.register_dataframe_features(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))

    def test_non_dataframe_input_is_rejected(self, tmp_path):
        pytest.importorskip("torch_geometric")
        parser = _parser(tmp_path)
        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            parser.register_dataframe_features({"a": [1.0, 2.0]})

    def test_broken_topology_stops_the_graph_build(self, tmp_path):
        pytest.importorskip("torch_geometric")
        parser = _parser(
            tmp_path,
            substitutions=((_CELL1_FACES, "   1 7   -1 2 -4 5 -6 7 2"),),
        )
        with pytest.raises(ValueError, match="Faces not properly shared"):
            parser.build_cell_graph()

    def test_no_shared_face_is_an_error(self, tmp_path):
        pytest.importorskip("torch_geometric")
        # Only cube 2 drops the dividing plane, so cube 1 still claims it and
        # every face has exactly one owner: nothing is shared, and -- unlike
        # dropping it from both cubes -- nothing is left isolated either, so
        # build_cell_graph reaches its own guard instead of tripping
        # validate_topology first.
        parser = _parser(
            tmp_path,
            substitutions=(
                ("   2 6   -2 3 -8 9 -10 11", "   2 5   3 -8 9 -10 11"),
            ),
        )
        report = parser.validate_topology(verbose=False)
        assert report["internal"] == []
        assert report["isolated"] == []
        assert len(report["boundary"]) == 11
        with pytest.raises(ValueError, match="No shared faces found"):
            parser.build_cell_graph()


class TestTessOriToMrp:
    """``_tess_ori_to_mrp`` must return the rotation the descriptor names.

    The assertions compare ``mrp_to_matrix(result)`` against the rotation matrix
    built independently from the same input. Checking only that the result *is*
    a rotation proves nothing: ``mrp_to_matrix`` returns an orthonormal matrix
    for any three numbers, so a corrupted MRP passes that check.
    """

    @staticmethod
    def _matrix(mrp):
        import torch

        from graintrace.orientation_helper import mrp_to_matrix

        return mrp_to_matrix(torch.as_tensor(mrp, dtype=torch.float64))

    def test_rotmat_descriptor_keeps_the_matrix(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        import torch

        values = [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # +90 deg about z
        ]
        parser = _parser(tmp_path)
        mrp = _to_mrp(parser, values, "rotmat")

        assert mrp.shape == (2, 3)
        assert torch.allclose(mrp[0], torch.zeros(3, dtype=torch.float64), atol=1e-12)
        expected = torch.tensor(values, dtype=torch.float64).reshape(-1, 3, 3)
        assert torch.allclose(self._matrix(mrp), expected, atol=1e-12)

    def test_quaternion_descriptor_is_read_scalar_first(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        import torch

        from graintrace.orientation_helper import quat_to_matrix

        # [w, x, y, z]: identity, then 120 deg about [1, 1, 1]/sqrt(3).
        values = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]
        parser = _parser(tmp_path)
        mrp = _to_mrp(parser, values, "quaternion")

        assert torch.allclose(mrp[0], torch.zeros(3, dtype=torch.float64), atol=1e-12)
        expected = quat_to_matrix(torch.tensor(values, dtype=torch.float64))
        assert torch.allclose(self._matrix(mrp), expected, atol=1e-12)

    def test_euler_descriptor_honours_the_named_convention(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        import torch

        from graintrace.orientation_helper import euler_to_matrix

        values = [[10.0, 20.0, 30.0], [200.0, 70.0, 250.0]]
        angles = torch.tensor(values, dtype=torch.float64)
        parser = _parser(tmp_path)

        for descriptor, convention in (
            ("euler-bunge", "bunge"),
            ("euler-kocks", "kocks"),
        ):
            mrp = _to_mrp(parser, values, descriptor)
            expected = euler_to_matrix(angles, convention, "degrees")
            assert torch.allclose(self._matrix(mrp), expected, atol=1e-12)

        # The two conventions must not collapse onto each other, or the
        # descriptor is being ignored.
        bunge = _to_mrp(parser, values, "euler-bunge")
        kocks = _to_mrp(parser, values, "euler-kocks")
        assert not torch.allclose(bunge, kocks)

    def test_unknown_descriptor_falls_back_to_the_first_three_values(self, tmp_path):
        pytest.importorskip("torch_geometric")

        parser = _parser(tmp_path)
        mrp = _to_mrp(parser, [[0.1, 0.2, 0.3, 9.9]], "something-neper-never-wrote")
        assert mrp.tolist() == [[0.1, 0.2, 0.3]]

    def test_rodrigues_descriptor_matches_the_normalised_quaternion(self, tmp_path):
        pytest.importorskip("torch_geometric")
        pytest.importorskip("neml2")
        import torch

        from graintrace.orientation_helper import quat_to_matrix

        values = [[0.3, -0.2, 0.5], [0.0, 0.0, 0.0]]
        parser = _parser(tmp_path)
        mrp = _to_mrp(parser, values, "rodrigues")

        gibbs = torch.tensor(values, dtype=torch.float64)
        quat = torch.cat(
            [torch.ones(len(values), 1, dtype=torch.float64), gibbs], dim=-1
        )
        quat = quat / quat.norm(dim=-1, keepdim=True)
        assert torch.allclose(self._matrix(mrp), quat_to_matrix(quat), atol=1e-12)

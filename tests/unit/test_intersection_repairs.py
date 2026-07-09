"""Unit tests for the intersection-repair helpers and public repair steps.

The pure geometric helpers in ``_intersection_repairs`` (plane extraction,
plane clipping, face-component collection, single-vertex face splitting) are
tested directly with small hand-built inputs. The two public repair steps are
then driven over a synthetic self-intersecting mesh so the iterative loops
execute end to end.
"""
from __future__ import annotations

import math

import pytest

from geometry_pipeline.core.ir import Face, Mesh, Vertex
from geometry_pipeline.repairs.mesh._intersection_repairs import (
    _boundary_chain,
    _build_edge_face_adjacency,
    _classify_multi_hit_face_collinear,
    _clip_face_loop_against_plane,
    _face_fid,
    _face_vids,
    _plane_from_face,
    _project_face_and_point_to_2d,
    _repair_single_endpoint_face_interior_touch,
    _repair_single_endpoint_face_interior_touch_by_triangulation,
    _reverse_face_vids,
    _set_face_vids,
    _signed_distance_to_plane,
    _split_face_at_single_interior_vertex,
    _visible_boundary_vertices_from_point,
    collect_face_component_from_seed_faces,
    orient_faces_consistently_by_adjacency,
    repair_multi_hit_face_collinear_chain,
    repair_plc_by_offset_iterative,
    repair_plc_single_splits_iterative,
)
from geometry_pipeline.repairs.mesh._common import _endpoint_vids_from_edge_t
from geometry_pipeline.repairs.mesh.repair_intersections import (
    RepairPlcSingleSplitsRepair,
    TrimSegmentFaceIntersectionsRepair,
)
from geometry_pipeline.validators.mesh.intersections import IntersectionsValidator

from conftest import make_mesh

# A unit square face in the z=0 plane, vertex ids 1..4.
SQUARE_PTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
SQUARE_IDS = [1, 2, 3, 4]


def _face(vids):
    return Face(vertex_indices=list(vids), group="default", material=None)


# --- Face accessors ----------------------------------------------------------

def test_face_vids_returns_copy():
    f = _face([1, 2, 3])
    out = _face_vids(f)
    assert out == [1, 2, 3]
    out.append(99)
    assert f.vertex_indices == [1, 2, 3]  # original untouched


def test_face_vids_rejects_non_face():
    with pytest.raises(TypeError):
        _face_vids(object())


def test_set_and_reverse_face_vids():
    f = _face([1, 2, 3])
    _set_face_vids(f, [4, 5, 6])
    assert f.vertex_indices == [4, 5, 6]
    _reverse_face_vids(f)
    assert f.vertex_indices == [6, 5, 4]


def test_face_fid_falls_back_to_index():
    # IR ``Face`` has no ``fid`` attribute, so the index is used.
    assert _face_fid(_face([1, 2, 3]), idx=7) == 7
    assert _face_fid(_face([1, 2, 3]), default=3) == 3


# --- Plane helpers -----------------------------------------------------------

def test_plane_from_square_face():
    plane_point, normal = _plane_from_face(_face(SQUARE_IDS), SQUARE_PTS)
    assert plane_point == (0.0, 0.0, 0.0)
    assert normal == pytest.approx((0.0, 0.0, 1.0))


def test_plane_from_degenerate_face_is_none():
    assert _plane_from_face(_face([1, 2]), SQUARE_PTS) is None
    collinear = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert _plane_from_face(_face([1, 2, 3]), collinear) is None


def test_signed_distance_to_plane():
    d = _signed_distance_to_plane((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert d == pytest.approx(2.0)
    d_below = _signed_distance_to_plane((0.0, 0.0, -1.5), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert d_below == pytest.approx(-1.5)


# --- Plane clipping ----------------------------------------------------------

def test_clip_square_against_plane_keeps_correct_half():
    points = list(SQUARE_PTS)  # helper appends intersection vertices
    loop = _clip_face_loop_against_plane(
        list(SQUARE_IDS),
        points,
        plane_point=(0.5, 0.0, 0.0),
        plane_normal=(1.0, 0.0, 0.0),
        keep_sign=1.0,  # keep x >= 0.5
    )
    assert len(loop) == 4
    kept = [points[v - 1] for v in loop]
    assert all(p[0] >= 0.5 - 1e-9 for p in kept)


def test_clip_degenerate_loop_returns_empty():
    assert _clip_face_loop_against_plane([1, 2], SQUARE_PTS, (0, 0, 0), (1, 0, 0), 1.0) == []


# --- Adjacency / components --------------------------------------------------

def test_build_edge_face_adjacency_shared_edge():
    faces = [_face([1, 2, 3]), _face([2, 3, 4])]  # share edge (2,3)
    adj = _build_edge_face_adjacency(faces)
    assert sorted(adj[(2, 3)]) == [0, 1]


def test_collect_component_spans_shared_edges_only():
    faces = [
        _face([1, 2, 3]),  # fid 0
        _face([2, 3, 4]),  # fid 1 - shares edge (2,3) with fid 0
        _face([5, 6, 7]),  # fid 2 - disconnected
    ]
    comp = collect_face_component_from_seed_faces(faces, [0])
    assert comp == [0, 1]


# --- Boundary chain / visibility --------------------------------------------

def test_boundary_chain_walks_forward_inclusive():
    assert _boundary_chain([10, 20, 30, 40], 0, 2) == [10, 20, 30]
    assert _boundary_chain([10, 20, 30, 40], 3, 1) == [40, 10, 20]


def test_all_corners_visible_from_center_of_square():
    poly2d = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert _visible_boundary_vertices_from_point(poly2d, (0.5, 0.5)) == [0, 1, 2, 3]


# --- Single interior-vertex split -------------------------------------------

def test_split_square_at_center_vertex():
    points = SQUARE_PTS + [(0.5, 0.5, 0.0)]  # vid 5 at the centre
    polys = _split_face_at_single_interior_vertex(_face(SQUARE_IDS), 5, points)
    assert polys is not None
    assert len(polys) == 2
    assert all(5 in poly for poly in polys)


def test_split_returns_none_when_vertex_already_in_face():
    assert _split_face_at_single_interior_vertex(_face(SQUARE_IDS), 1, SQUARE_PTS) is None


def test_project_face_and_point_drops_dominant_axis():
    poly2d, p2, dropped = _project_face_and_point_to_2d(SQUARE_IDS, 1, SQUARE_PTS)
    assert dropped == "z"
    assert poly2d == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert p2 == (0.0, 0.0)


# --- Public repair steps (end-to-end over the iterative loops) --------------

@pytest.fixture
def intersecting_mesh() -> Mesh:
    pytest.importorskip("shapely")
    # Triangle A in z=0; triangle B is vertical and its edge (4->5) pierces
    # A's interior at (0.5, 0.5, 0).
    return make_mesh(
        [
            (0.0, 0.0, 0.0),   # 1  A
            (2.0, 0.0, 0.0),   # 2  A
            (0.0, 2.0, 0.0),   # 3  A
            (0.5, 0.5, -1.0),  # 4  B below plane
            (0.5, 0.5, 1.0),   # 5  B above plane
            (0.5, 3.0, 1.0),   # 6  B
        ],
        [[1, 2, 3], [4, 5, 6]],
    )


def test_trim_repair_runs_over_intersection(ctx, intersecting_mesh):
    detector = IntersectionsValidator()
    assert len(detector.detect(intersecting_mesh, ctx)) >= 1  # precondition

    new_mesh, result = TrimSegmentFaceIntersectionsRepair(detector).apply(
        intersecting_mesh, detector.detect(intersecting_mesh, ctx), ctx
    )
    assert isinstance(new_mesh, Mesh)
    assert "remaining_intersections" in result.details
    assert "status" in result.details


def test_single_splits_repair_runs_over_intersection(ctx, intersecting_mesh):
    detector = IntersectionsValidator()
    new_mesh, result = RepairPlcSingleSplitsRepair(detector).apply(
        intersecting_mesh, detector.detect(intersecting_mesh, ctx), ctx
    )
    assert isinstance(new_mesh, Mesh)
    assert "remaining_intersections" in result.details
    assert "changed" in result.details


# --- Endpoint-touch PLC repairs (called directly with hand-built reports) ----

# A large triangle in z=0 plus one vertex (id 4) sitting inside its interior.
TRI_PTS = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.5, 0.5, 0.0)]
TRI_IDS = [1, 2, 3]


def _endpoint_report(facet_fid=0):
    # An edge (4->5) whose endpoint vertex 4 lands inside triangle face `fid`.
    return {
        "hit_type": "endpoint_face_interior_touch",
        "facet_fid": facet_fid,
        "edge": (4, 5),
        "t_param": 0.0,  # endpoint is vertex 4
        "point": [0.5, 0.5, 0.0],
    }


def test_endpoint_vids_from_edge_t():
    assert _endpoint_vids_from_edge_t((4, 5), 0.0) == (4, 5)
    assert _endpoint_vids_from_edge_t((4, 5), 1.0) == (5, 4)
    assert _endpoint_vids_from_edge_t((4, 5), 0.5) == (None, None)


def test_single_endpoint_split_repairs_touched_face():
    faces = [_face(TRI_IDS)]
    new_faces, changed, diag = _repair_single_endpoint_face_interior_touch(
        faces, _endpoint_report(), list(TRI_PTS)
    )
    assert changed is True
    assert diag["status"] == "ok"
    assert diag["inserted_vid"] == 4
    assert len(new_faces) >= 2  # face split into multiple


def test_single_endpoint_split_rejects_wrong_hit_type():
    report = {**_endpoint_report(), "hit_type": "segment_face_interior_intersection"}
    faces = [_face(TRI_IDS)]
    _, changed, diag = _repair_single_endpoint_face_interior_touch(faces, report, list(TRI_PTS))
    assert changed is False
    assert diag["status"] == "wrong_type"


def test_single_endpoint_split_reports_missing_face():
    report = _endpoint_report(facet_fid=99)  # no such face index
    faces = [_face(TRI_IDS)]
    _, changed, diag = _repair_single_endpoint_face_interior_touch(faces, report, list(TRI_PTS))
    assert changed is False
    assert diag["status"] == "missing_face"


def test_single_endpoint_split_by_triangulation():
    faces = [_face(TRI_IDS)]
    new_faces, changed, diag = _repair_single_endpoint_face_interior_touch_by_triangulation(
        faces, _endpoint_report(), list(TRI_PTS)
    )
    assert changed is True
    assert diag["status"] == "ok"
    assert diag["n_output_tris"] >= 1


def test_plc_single_splits_iterative_applies_and_converges():
    faces = [_face(TRI_IDS)]
    calls = {"n": 0}

    def supplier(_f, _p):
        calls["n"] += 1
        return [_endpoint_report()] if calls["n"] == 1 else []

    _faces, _points, changed, summary = repair_plc_single_splits_iterative(
        faces, list(TRI_PTS), (0.5, 0.5, 5.0), plc_report_supplier=supplier
    )
    assert changed is True
    assert summary["applied_repairs"] == 1
    assert summary["stopped_reason"] == "no_plc_hits"


def test_plc_offset_iterative_early_returns():
    faces = [_face(TRI_IDS)]
    # No hits at all.
    _, _, changed, summary = repair_plc_by_offset_iterative(
        faces, list(TRI_PTS), plc_report_supplier=lambda f, p: []
    )
    assert changed is False
    assert summary["stopped_reason"] == "no_plc_hits"

    # Hits present but none are endpoint-face touches.
    other = [{"hit_type": "segment_face_interior_intersection"}]
    _, _, changed2, summary2 = repair_plc_by_offset_iterative(
        faces, list(TRI_PTS), plc_report_supplier=lambda f, p: other
    )
    assert changed2 is False
    assert summary2["stopped_reason"] == "no_endpoint_face_interior_touch"


# --- Multi-hit collinear classification / repair ----------------------------

def test_classify_multi_hit_face_collinear_two_points_are_collinear():
    face = _face(SQUARE_IDS)
    reports = [{"point": [0.3, 0.5, 0.0]}, {"point": [0.6, 0.5, 0.0]}]
    cls = _classify_multi_hit_face_collinear(face, reports, SQUARE_PTS)
    assert cls["is_collinear"] is True


def test_classify_multi_hit_face_needs_at_least_two_points():
    face = _face(SQUARE_IDS)
    cls = _classify_multi_hit_face_collinear(face, [{"point": [0.3, 0.5, 0.0]}], SQUARE_PTS)
    assert cls["is_collinear"] is False
    assert cls["reason"] == "need_at_least_2_points"


def test_classify_multi_hit_face_non_collinear_three_points():
    face = _face(SQUARE_IDS)
    reports = [
        {"point": [0.2, 0.5, 0.0]},
        {"point": [0.6, 0.5, 0.0]},
        {"point": [0.5, 0.95, 0.0]},  # off the line
    ]
    cls = _classify_multi_hit_face_collinear(face, reports, SQUARE_PTS)
    assert cls["is_collinear"] is False


def test_repair_multi_hit_collinear_chain_splits_face():
    faces = [_face(SQUARE_IDS)]
    reports = [
        {"facet_fid": 0, "point": [0.3, 0.5, 0.0]},
        {"facet_fid": 0, "point": [0.6, 0.5, 0.0]},
    ]
    new_faces, _points, changed, diag = repair_multi_hit_face_collinear_chain(
        faces, reports, list(SQUARE_PTS)
    )
    assert changed is True
    assert diag["status"] == "ok"
    assert len(new_faces) == 2  # face replaced by two sub-faces


# --- Orientation by adjacency ------------------------------------------------

def test_orient_faces_consistently_runs_on_cube(unit_cube):
    faces = [_face(f.vertex_indices) for f in unit_cube.faces]
    diag = orient_faces_consistently_by_adjacency(faces)
    assert isinstance(diag, dict)

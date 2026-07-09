"""Unit tests for the ``geometry_pipeline.geometry_math`` primitives.

These are pure numeric functions with no I/O, so every case uses a small
hand-computed input with a known exact answer. Note the polygon helpers take
**1-based** vertex ids into the ``points`` list.
"""
from __future__ import annotations

import math

import pytest

from geometry_pipeline.geometry_math import geometry_math as gm
from geometry_pipeline.geometry_math.predicates import (
    classify_face_degeneracy,
    classify_face_planarity_m,
)


# A unit square in the z=0 plane, CCW. Vertex ids 1..4 index this list.
SQUARE = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
]
SQUARE_IDS = [1, 2, 3, 4]


# --- Vector primitives -------------------------------------------------------

def test_uedge_is_order_independent():
    assert gm.uedge(3, 1) == (1, 3)
    assert gm.uedge(1, 3) == (1, 3)


def test_sub_add_mul():
    assert gm.sub((3, 5, 7), (1, 2, 3)) == (2, 3, 4)
    assert gm.vadd((1, 2, 3), (4, 5, 6)) == (5, 7, 9)
    assert gm.vmul((1, 2, 3), 2) == (2, 4, 6)


def test_cross_and_dot():
    assert gm.cross((1, 0, 0), (0, 1, 0)) == (0, 0, 1)
    assert gm.dot((1, 2, 3), (4, 5, 6)) == 32


def test_norm_distance_unit():
    assert gm.norm((3, 4, 0)) == pytest.approx(5.0)
    assert gm.distance((0, 0, 0), (0, 3, 4)) == pytest.approx(5.0)
    assert gm.unit((3, 0, 0)) == pytest.approx((1.0, 0.0, 0.0))


def test_unit_of_zero_vector_is_zero():
    assert gm.unit((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


# --- Bounding boxes ----------------------------------------------------------

def test_aabb_of_tri_and_seg():
    assert gm.aabb_of_tri((0, 0, 0), (2, 1, 0), (1, 3, 5)) == (0, 0, 0, 2, 3, 5)
    assert gm.aabb_of_seg((0, 0, 0), (1, -2, 3)) == (0, -2, 0, 1, 0, 3)


def test_aabb_overlap_true_and_false():
    a = (0, 0, 0, 1, 1, 1)
    assert gm.aabb_overlap(a, (0.5, 0.5, 0.5, 2, 2, 2)) is True
    assert gm.aabb_overlap(a, (5, 5, 5, 6, 6, 6)) is False


def test_aabb_overlap_respects_padding():
    a = (0, 0, 0, 1, 1, 1)
    b = (1.05, 0, 0, 2, 1, 1)  # gap of 0.05 on x
    assert gm.aabb_overlap(a, b) is False
    assert gm.aabb_overlap(a, b, pad=0.1) is True


# --- Segment / triangle intersection ----------------------------------------

def _tri():
    return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


def test_segment_through_triangle_hits():
    a, b, c = _tri()
    hit, t, u, v = gm.segment_intersects_triangle(
        (0.25, 0.25, -1.0), (0.25, 0.25, 1.0), a, b, c
    )
    assert hit is True
    assert t == pytest.approx(0.5)


def test_segment_missing_triangle_does_not_hit():
    a, b, c = _tri()
    hit, *_ = gm.segment_intersects_triangle(
        (2.0, 2.0, -1.0), (2.0, 2.0, 1.0), a, b, c
    )
    assert hit is False


def test_segment_parallel_to_triangle_plane_does_not_hit():
    a, b, c = _tri()
    hit, *_ = gm.segment_intersects_triangle(
        (0.25, 0.25, 1.0), (0.75, 0.25, 1.0), a, b, c
    )
    assert hit is False


# --- Polygon area / normal ---------------------------------------------------

def test_polygon_area_3d_unit_square():
    assert gm.polygon_area_3d(SQUARE_IDS, SQUARE) == pytest.approx(1.0)


def test_polygon_area_3d_degenerate_loop_is_zero():
    assert gm.polygon_area_3d([1, 2], SQUARE) == 0.0


def test_newell_normal_points_up_for_ccw_square():
    assert gm.newell_normal_from_points(SQUARE_IDS, SQUARE) == pytest.approx((0.0, 0.0, 2.0))


def test_compute_face_unit_normal_square():
    assert gm.compute_face_unit_normal(SQUARE_IDS, SQUARE) == pytest.approx((0.0, 0.0, 1.0))


def test_polygon_centroid_square():
    assert gm.polygon_centroid(SQUARE_IDS, SQUARE) == pytest.approx((0.5, 0.5, 0.0))


def test_polygon_centroid_rejects_empty_loop():
    with pytest.raises(ValueError):
        gm.polygon_centroid([], SQUARE)


# --- 2D predicates -----------------------------------------------------------

def test_orient_sign():
    assert gm.orient((0, 0), (1, 0), (0, 1)) > 0  # CCW
    assert gm.orient((0, 0), (0, 1), (1, 0)) < 0  # CW
    assert gm.orient((0, 0), (1, 0), (2, 0)) == 0  # collinear


def test_area2_signed_for_ccw_square():
    poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert gm.area2(poly) == pytest.approx(2.0)  # 2 * signed area


def test_point_on_segment_2d():
    assert gm.point_on_segment_2d((0.5, 0.0), (0, 0), (1, 0)) is True
    assert gm.point_on_segment_2d((0.5, 0.5), (0, 0), (1, 0)) is False


def test_segments_intersect_2d_crossing_and_disjoint():
    assert gm.segments_intersect_2d((0, 0), (2, 2), (0, 2), (2, 0)) is True
    assert gm.segments_intersect_2d((0, 0), (1, 0), (0, 1), (1, 1)) is False


def test_segments_intersect_2d_touching_endpoint():
    # Endpoint of the second segment lies on the first.
    assert gm.segments_intersect_2d((0, 0), (2, 0), (1, 0), (1, 1)) is True


def test_point_in_polygon_2d_inside_outside_boundary():
    poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert gm.point_in_polygon_2d(poly, (0.5, 0.5)) == "inside"
    assert gm.point_in_polygon_2d(poly, (2.0, 2.0)) == "outside"
    assert gm.point_in_polygon_2d(poly, (0.0, 0.5)) == "boundary"


def test_point_segment_distance_2d_perpendicular_and_clamped():
    assert gm.point_segment_distance_2d((0, 1), (0, 0), (2, 0)) == pytest.approx(1.0)
    # Beyond segment end -> distance to nearest endpoint.
    assert gm.point_segment_distance_2d((3, 0), (0, 0), (2, 0)) == pytest.approx(1.0)


# --- Projection --------------------------------------------------------------

def test_project_face_to_2d_drops_dominant_axis():
    poly2d, dropped = gm.project_face_to_2d(SQUARE_IDS, SQUARE)
    assert dropped == "z"
    assert poly2d == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_offset_point_along_vector():
    assert gm.offset_point_along_vector((0, 0, 0), (0, 0, 1), 5.0) == (0.0, 0.0, 5.0)


# --- Face predicates ---------------------------------------------------------

def test_degeneracy_ok_for_real_triangle():
    tri = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    status, area2 = classify_face_degeneracy([1, 2, 3], tri)
    assert status == "ok"
    assert area2 == pytest.approx(1.0)  # |Newell normal|^2 for this triangle


def test_degeneracy_fatal_for_collinear_points():
    line = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    status, _ = classify_face_degeneracy([1, 2, 3], line)
    assert status == "fatal"


def test_degeneracy_fatal_for_fewer_than_three_unique_vertices():
    status, _ = classify_face_degeneracy([1, 1, 2], SQUARE)
    assert status == "fatal"


def test_planarity_skip_for_triangle():
    tri = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    status, _, _ = classify_face_planarity_m([1, 2, 3], tri)
    assert status == "skip"


def test_planarity_ok_for_flat_quad():
    status, max_abs, _ = classify_face_planarity_m(SQUARE_IDS, SQUARE)
    assert status == "ok"
    assert max_abs == pytest.approx(0.0)


def test_planarity_fatal_for_strongly_non_planar_quad():
    quad = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)]
    status, _, _ = classify_face_planarity_m([1, 2, 3, 4], quad)
    assert status == "fatal"


# --- Triangulation (requires shapely) ----------------------------------------

def test_triangulate_square_covers_all_vertices():
    shapely = pytest.importorskip("shapely")
    if not hasattr(shapely, "constrained_delaunay_triangles"):
        pytest.skip("shapely too old for constrained_delaunay_triangles")
    from geometry_pipeline.geometry_math.triangulation import triangulate_face_cdt_shapely

    tris = triangulate_face_cdt_shapely(SQUARE_IDS, SQUARE)
    assert len(tris) == 2
    assert all(len(t) == 3 for t in tris)
    assert {vid for tri in tris for vid in tri} == set(SQUARE_IDS)

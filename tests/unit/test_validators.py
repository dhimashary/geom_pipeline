"""Synthetic-fixture unit tests for each mesh validator.

Every positive case is the smallest mesh that exhibits exactly one defect
kind, so a failure points unambiguously at a single validator. Each validator
is also run against the clean ``unit_cube`` to guard against false positives.
Overlap/intersection tests need Shapely's constrained Delaunay triangulation
and skip gracefully when it is unavailable.
"""
from __future__ import annotations

import pytest

from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.validators.mesh.boundary_edges import BoundaryEdgesValidator
from geometry_pipeline.validators.mesh.collinear_faces import CollinearFacesValidator
from geometry_pipeline.validators.mesh.degenerate_faces import ZeroAreaFaceValidator
from geometry_pipeline.validators.mesh.duplicate_vertices import DuplicateVerticesValidator
from geometry_pipeline.validators.mesh.intersections import IntersectionsValidator
from geometry_pipeline.validators.mesh.non_planar_faces import NonPlanarFacesValidator
from geometry_pipeline.validators.mesh.overlapping_faces import OverlappingFacesValidator
from geometry_pipeline.validators.mesh.possible_holes import PossibleHolesValidator
from geometry_pipeline.validators.mesh.small_faces import SmallFacesValidator
from geometry_pipeline.validators.mesh.t_junctions import TJunctionsValidator

from conftest import make_mesh


def _kinds(issues) -> set[IssueKind]:
    return {i.kind for i in issues}


# --- duplicate_vertex --------------------------------------------------------

def test_duplicate_vertices_detected(ctx):
    mesh = make_mesh(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)],  # v1 and v4 coincide
        [[1, 2, 3]],
    )
    issues = DuplicateVerticesValidator().detect(mesh, ctx)
    assert len(issues) == 2  # one report per member of the duplicate group
    assert _kinds(issues) == {IssueKind.DUPLICATE_VERTEX}


def test_duplicate_vertices_none_on_clean_cube(ctx, unit_cube):
    assert DuplicateVerticesValidator().detect(unit_cube, ctx) == []


# --- zero_area_face ---------------------------------------------------------

def test_zero_area_face_detected_for_zero_area_triangle(ctx):
    mesh = make_mesh([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [[1, 2, 3]])
    issues = ZeroAreaFaceValidator().detect(mesh, ctx)
    assert len(issues) == 1
    assert issues[0].kind is IssueKind.ZERO_AREA_FACE


def test_zero_area_face_none_on_clean_cube(ctx, unit_cube):
    assert ZeroAreaFaceValidator().detect(unit_cube, ctx) == []


# --- non_planar_face ---------------------------------------------------------

def test_non_planar_face_detected(ctx):
    mesh = make_mesh(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)],  # 4th vertex lifted out of plane
        [[1, 2, 3, 4]],
    )
    issues = NonPlanarFacesValidator().detect(mesh, ctx)
    assert len(issues) == 1
    assert issues[0].kind is IssueKind.NON_PLANAR_FACE


def test_non_planar_face_none_on_clean_cube(ctx, unit_cube):
    assert NonPlanarFacesValidator().detect(unit_cube, ctx) == []


# --- boundary_edge -----------------------------------------------------------

def test_boundary_edges_detected_for_lone_triangle(ctx):
    mesh = make_mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[1, 2, 3]])
    issues = BoundaryEdgesValidator().detect(mesh, ctx)
    assert len(issues) == 3  # all three edges are used by only one face
    assert _kinds(issues) == {IssueKind.BOUNDARY_EDGE}


def test_boundary_edges_none_on_closed_cube(ctx, unit_cube):
    assert BoundaryEdgesValidator().detect(unit_cube, ctx) == []


# --- possible_hole -----------------------------------------------------------

def test_possible_hole_detected_for_open_cube(ctx, open_cube):
    issues = PossibleHolesValidator().detect(open_cube, ctx)
    assert len(issues) == 1  # the single empty top loop
    assert issues[0].kind is IssueKind.POSSIBLE_HOLE


def test_possible_hole_none_on_closed_cube(ctx, unit_cube):
    assert PossibleHolesValidator().detect(unit_cube, ctx) == []


# --- small_face --------------------------------------------------------------

def test_small_face_detected(ctx):
    mesh = make_mesh([(0, 0, 0), (0.05, 0, 0), (0, 0.05, 0)], [[1, 2, 3]])
    issues = SmallFacesValidator().detect(mesh, ctx)
    assert len(issues) == 1
    assert issues[0].kind is IssueKind.SMALL_FACE


def test_small_face_none_on_clean_cube(ctx, unit_cube):
    assert SmallFacesValidator().detect(unit_cube, ctx) == []


# --- collinear_face ----------------------------------------------------------

def test_collinear_face_detected(ctx):
    mesh = make_mesh([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [[1, 2, 3]])
    issues = CollinearFacesValidator().detect(mesh, ctx)
    assert len(issues) == 1
    assert issues[0].kind is IssueKind.COLLINEAR_FACE


def test_collinear_face_none_on_clean_cube(ctx, unit_cube):
    assert CollinearFacesValidator().detect(unit_cube, ctx) == []


# --- t_junction --------------------------------------------------------------

def test_t_junction_detected(ctx):
    # Vertex 4 sits at the midpoint of face F1's edge (1->2) but belongs only
    # to face F2, i.e. a classic T-junction.
    mesh = make_mesh(
        [
            (0.0, 0.0, 0.0),   # 1
            (2.0, 0.0, 0.0),   # 2
            (1.0, 1.0, 0.0),   # 3
            (1.0, 0.0, 0.0),   # 4  midpoint of edge (1,2)
            (0.0, -1.0, 0.0),  # 5
            (2.0, -1.0, 0.0),  # 6
        ],
        [[1, 2, 3], [4, 5, 6]],
    )
    issues = TJunctionsValidator().detect(mesh, ctx)
    assert len(issues) == 1
    assert issues[0].kind is IssueKind.T_JUNCTION


def test_t_junction_none_on_clean_cube(ctx, unit_cube):
    assert TJunctionsValidator().detect(unit_cube, ctx) == []


# --- intersection ------------------------------------------------------------

def test_intersection_detected(ctx):
    pytest.importorskip("shapely")
    # Triangle A lies in z=0; triangle B is vertical and its edge (4->5)
    # pierces A's interior at (0.5, 0.5, 0).
    mesh = make_mesh(
        [
            (0.0, 0.0, 0.0),   # 1  A
            (2.0, 0.0, 0.0),   # 2  A
            (0.0, 2.0, 0.0),   # 3  A
            (0.5, 0.5, -1.0),  # 4  B below plane
            (0.5, 0.5, 1.0),   # 5  B above plane
            (0.5, 3.0, 1.0),   # 6  B (second crossing lands outside A)
        ],
        [[1, 2, 3], [4, 5, 6]],
    )
    issues = IntersectionsValidator().detect(mesh, ctx)
    assert len(issues) >= 1
    assert _kinds(issues) == {IssueKind.INTERSECTION}


def test_intersection_none_on_clean_cube(ctx, unit_cube):
    pytest.importorskip("shapely")
    assert IntersectionsValidator().detect(unit_cube, ctx) == []


# --- overlapping_face --------------------------------------------------------

def test_overlapping_faces_detected(ctx):
    pytest.importorskip("shapely")
    # Two identical, coplanar triangles occupying the same region.
    mesh = make_mesh(
        [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),  # triangle A
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),  # triangle B
        ],
        [[1, 2, 3], [4, 5, 6]],
    )
    issues = OverlappingFacesValidator().detect(mesh, ctx)
    assert len(issues) == 1
    assert issues[0].kind is IssueKind.OVERLAPPING_FACE


def test_overlapping_faces_none_on_clean_cube(ctx, unit_cube):
    pytest.importorskip("shapely")
    assert OverlappingFacesValidator().detect(unit_cube, ctx) == []

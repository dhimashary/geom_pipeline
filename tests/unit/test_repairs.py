"""Synthetic-fixture unit tests for the mesh repair steps.

Each repair is driven by the smallest mesh exhibiting the defect it targets,
and the assertions check both the returned :class:`Mesh` and the
:class:`RepairResult` bookkeeping. Repairs that need a detector (T-junctions)
have one injected explicitly.
"""
from __future__ import annotations

import pytest

from geometry_pipeline.core.ir import BRep
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.repairs.mesh.compact_vertices import CompactVerticesRepair
from geometry_pipeline.repairs.mesh.deduplicate_vertices import DeduplicateVerticesRepair
from geometry_pipeline.repairs.mesh.fix_t_junctions import FixTJunctionsIterativeRepair
from geometry_pipeline.repairs.mesh.orient_outward import FlipFacesIfMajorityInwardRepair
from geometry_pipeline.repairs.mesh.remove_degenerate_faces import RemoveZeroAreaFaceRepair
from geometry_pipeline.repairs.mesh.sort_vertices import SortVerticesDeterministicallyRepair
from geometry_pipeline.validators.mesh.t_junctions import TJunctionsValidator

from conftest import make_mesh


# --- deduplicate_vertices ----------------------------------------------------

def test_deduplicate_merges_coincident_vertices(ctx):
    mesh = make_mesh(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)],  # v1 and v4 coincide
        [[1, 2, 3]],
    )
    new_mesh, result = DeduplicateVerticesRepair().apply(mesh, [], ctx)
    assert len(new_mesh.vertices) == 3
    assert result.before_count == 4
    assert result.after_count == 3
    assert result.details["merged_vertex_count"] == 1


def test_deduplicate_handles_declared_issues(ctx):
    repair = DeduplicateVerticesRepair()
    assert IssueKind.DUPLICATE_VERTEX in repair.handles


# --- remove_zero_area_faces -------------------------------------------------

def test_remove_zero_area_drops_only_the_zero_area_face(ctx):
    mesh = make_mesh(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0)],
        [[1, 2, 3], [1, 2, 4]],  # second face is collinear (zero area)
    )
    new_mesh, result = RemoveZeroAreaFaceRepair().apply(mesh, [], ctx)
    assert len(new_mesh.faces) == 1
    assert result.before_count == 2
    assert result.after_count == 1
    assert result.details["fatal_removed"] == 1


# --- compact_vertices --------------------------------------------------------

def test_compact_drops_unreferenced_vertices(ctx):
    mesh = make_mesh(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 5, 5)],  # v4 unused
        [[1, 2, 3]],
    )
    new_mesh, result = CompactVerticesRepair().apply(mesh, [], ctx)
    assert len(new_mesh.vertices) == 3
    assert result.before_count == 4
    assert result.after_count == 3


# --- sort_vertices -----------------------------------------------------------

def test_sort_vertices_orders_by_coordinate_and_preserves_geometry(ctx):
    mesh = make_mesh([(1, 0, 0), (0, 0, 0), (0, 1, 0)], [[1, 2, 3]])
    new_mesh, _ = SortVerticesDeterministicallyRepair().apply(mesh, [], ctx)

    coords = [(v.x, v.y, v.z) for v in new_mesh.vertices]
    assert coords == sorted(coords)  # ascending coordinate order
    # The face still references the same three physical points.
    face = new_mesh.faces[0]
    referenced = {coords[i - 1] for i in face.vertex_indices}
    assert referenced == {(1, 0, 0), (0, 0, 0), (0, 1, 0)}


def test_sort_vertices_is_deterministic(ctx):
    def _run():
        mesh = make_mesh([(1, 0, 0), (0, 0, 0), (0, 1, 0)], [[1, 2, 3]])
        m, _ = SortVerticesDeterministicallyRepair().apply(mesh, [], ctx)
        return [(v.x, v.y, v.z) for v in m.vertices]

    assert _run() == _run()


# --- orient_outward ----------------------------------------------------------

def test_orient_outward_leaves_outward_cube_untouched(ctx, unit_cube):
    _, result = FlipFacesIfMajorityInwardRepair().apply(unit_cube, [], ctx)
    assert result.details["flipped_all"] is False


def test_orient_outward_flips_inward_cube(ctx, unit_cube):
    inward = make_mesh(
        [(v.x, v.y, v.z) for v in unit_cube.vertices],
        [list(reversed(f.vertex_indices)) for f in unit_cube.faces],
    )
    _, result = FlipFacesIfMajorityInwardRepair().apply(inward, [], ctx)
    assert result.details["flipped_all"] is True


# --- fix_t_junctions ---------------------------------------------------------

def test_fix_t_junctions_resolves_the_junction(ctx):
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
    detector = TJunctionsValidator()
    assert len(detector.detect(mesh, ctx)) == 1  # precondition

    new_mesh, result = FixTJunctionsIterativeRepair(detector).apply(
        mesh, detector.detect(mesh, ctx), ctx
    )
    assert result.details["changed"] is True
    assert result.details["remaining_t_junctions"] == 0
    assert detector.detect(new_mesh, ctx) == []


# --- accept-guard ------------------------------------------------------------

def test_repair_rejects_unsupported_geometry(ctx):
    with pytest.raises(TypeError):
        DeduplicateVerticesRepair().apply(BRep(), [], ctx)

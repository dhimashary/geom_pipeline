"""Unit tests for the ``geometry_pipeline.core`` domain layer.

These cover the pure value objects and helpers that every other layer depends
on: issue identity/serialisation, the geometry IR, tolerances, snapshot
diffing, and the pipeline result accessors.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from geometry_pipeline.core.diff import diff_snapshots
from geometry_pipeline.core.ir import BRep, Face, Mesh, PointCloud, Vertex
from geometry_pipeline.core.issues import (
    DetectionStage,
    Issue,
    IssueKind,
)
from geometry_pipeline.core.jsonable import to_jsonable
from geometry_pipeline.core.report import (
    PipelineResult,
    RepairReport,
    RepairResult,
    ValidationSnapshot,
)
from geometry_pipeline.core.tolerances import Tolerances

# --- Issue -------------------------------------------------------------------


def _issue(kind: IssueKind = IssueKind.T_JUNCTION, payload: dict | None = None) -> Issue:
    return Issue.create(kind, DetectionStage.PRE, payload=payload or {})


def test_issue_create_populates_fields():
    issue = Issue.create(
        IssueKind.INTERSECTION,
        DetectionStage.POST_STAGE,
        stage_name="repair_intersections",
        payload={"a": 1},
    )
    assert issue.kind is IssueKind.INTERSECTION
    assert issue.stage is DetectionStage.POST_STAGE
    assert issue.stage_name == "repair_intersections"
    assert issue.payload == {"a": 1}
    assert isinstance(issue.id, str) and len(issue.id) == 12


def test_issue_id_is_deterministic_for_same_kind_and_payload():
    a = _issue(IssueKind.SMALL_FACE, {"fid": 7, "area": 0.5})
    b = _issue(IssueKind.SMALL_FACE, {"area": 0.5, "fid": 7})  # key order differs
    assert a.id == b.id


def test_issue_id_changes_with_kind_or_payload():
    base = _issue(IssueKind.SMALL_FACE, {"fid": 7})
    assert base.id != _issue(IssueKind.ZERO_AREA_FACE, {"fid": 7}).id
    assert base.id != _issue(IssueKind.SMALL_FACE, {"fid": 8}).id


def test_issue_id_ignores_severity_and_stage():
    a = Issue.create(IssueKind.SMALL_FACE, DetectionStage.PRE, payload={"fid": 1})
    b = Issue.create(IssueKind.SMALL_FACE, DetectionStage.FINAL, payload={"fid": 1})
    assert a.id == b.id


def test_issue_is_frozen():
    issue = _issue()
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.id = "tampered"  # type: ignore[misc]


def test_issue_payload_is_coerced_to_json_native():
    issue = _issue(IssueKind.INTERSECTION, {"fid": np.int64(3), "pt": np.array([1.0, 2.0])})
    assert type(issue.payload["fid"]) is int
    assert issue.payload["pt"] == [1.0, 2.0]


def test_issuekind_values_are_stable_strings():
    # The frontend/report layers key off these string values, so a rename here
    # is a breaking change that must be caught.
    assert {k.value for k in IssueKind} == {
        "duplicate_vertex",
        "zero_area_face",
        "non_planar_face",
        "t_junction",
        "intersection",
        "boundary_edge",
        "possible_hole",
        "small_face",
        "overlapping_face",
        "collinear_face",
    }


# --- IR ----------------------------------------------------------------------


def test_mesh_defaults_are_empty_and_independent():
    a, b = Mesh(), Mesh()
    assert a.vertices == [] and a.faces == [] and a.materials == {} and a.metadata == {}
    a.vertices.append(Vertex(0, 0, 0))
    assert b.vertices == [], "default_factory must not share state between instances"


def test_geometry_kind_discriminators():
    assert Mesh.kind == "mesh"
    assert BRep.kind == "brep"
    assert PointCloud.kind == "pointcloud"


def test_vertex_is_frozen_and_hashable():
    v = Vertex(1.0, 2.0, 3.0)
    assert (v.x, v.y, v.z) == (1.0, 2.0, 3.0)
    assert hash(v) == hash(Vertex(1.0, 2.0, 3.0))


def test_face_holds_indices_group_and_material():
    f = Face(vertex_indices=[0, 1, 2], group="wall", material="brick")
    assert f.vertex_indices == [0, 1, 2]
    assert f.group == "wall"
    assert f.material == "brick"


# --- Tolerances --------------------------------------------------------------


def test_tolerances_have_documented_defaults():
    t = Tolerances()
    assert t.vertex_merge == pytest.approx(1e-2)
    assert t.degenerate_area == pytest.approx(1e-12)
    assert t.max_reports == 200


def test_tolerances_is_frozen():
    t = Tolerances()
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.vertex_merge = 0.5  # type: ignore[misc]


def test_tolerances_overridable_at_construction():
    assert Tolerances(vertex_merge=0.25).vertex_merge == 0.25


# --- Snapshot diffing --------------------------------------------------------


def _snapshot(*issues: Issue) -> ValidationSnapshot:
    return ValidationSnapshot(when=DetectionStage.PRE, issues=list(issues))


def test_diff_reports_fixed_introduced_and_remaining():
    shared = _issue(IssueKind.T_JUNCTION, {"id": "keep"})
    only_before = _issue(IssueKind.SMALL_FACE, {"id": "gone"})
    only_after = _issue(IssueKind.INTERSECTION, {"id": "new"})

    diff = diff_snapshots(_snapshot(shared, only_before), _snapshot(shared, only_after))

    assert [i.id for i in diff.fixed] == [only_before.id]
    assert [i.id for i in diff.introduced] == [only_after.id]
    assert [i.id for i in diff.remaining] == [shared.id]


def test_diff_identical_snapshots_has_no_changes():
    snap = _snapshot(_issue(), _issue(IssueKind.SMALL_FACE, {"fid": 1}))
    diff = diff_snapshots(snap, snap)
    assert diff.fixed == [] and diff.introduced == []
    assert {i.id for i in diff.remaining} == {i.id for i in snap.issues}


# --- PipelineResult ----------------------------------------------------------


def test_pipeline_result_initial_and_final_snapshots():
    first = ValidationSnapshot(when=DetectionStage.PRE, issues=[_issue()])
    last = ValidationSnapshot(when=DetectionStage.FINAL, issues=[])
    result = PipelineResult(geometry=Mesh(), snapshots=[first, last])
    assert result.initial is first
    assert result.final is last


def test_pipeline_result_accessors_handle_no_snapshots():
    result = PipelineResult(geometry=Mesh())
    assert result.initial is None
    assert result.final is None
    assert result.composite_issues == []


def test_composite_issues_take_latest_detection_per_kind():
    tj_early = _issue(IssueKind.T_JUNCTION, {"stage": "pre"})
    tj_late = _issue(IssueKind.T_JUNCTION, {"stage": "post"})
    small = _issue(IssueKind.SMALL_FACE, {"fid": 1})

    pre = ValidationSnapshot(when=DetectionStage.PRE, issues=[tj_early, small])
    post = ValidationSnapshot(when=DetectionStage.POST_STAGE, issues=[tj_late])
    result = PipelineResult(geometry=Mesh(), snapshots=[pre, post])

    composite_ids = {i.id for i in result.composite_issues}
    # T_JUNCTION comes from the later snapshot; SMALL_FACE (only in PRE) survives.
    assert tj_late.id in composite_ids
    assert tj_early.id not in composite_ids
    assert small.id in composite_ids


def test_repair_report_defaults():
    rr = RepairResult(step_name="s", stage_name="t")
    assert rr.affected_ids == [] and rr.iterations == 1 and rr.details == {}
    assert RepairReport().results == []


# --- to_jsonable edge cases (complements test_jsonable.py) -------------------


def test_to_jsonable_passes_through_native_scalars():
    assert to_jsonable(None) is None
    assert to_jsonable(True) is True
    assert to_jsonable("x") == "x"


def test_to_jsonable_converts_sets_and_tuples_to_lists():
    assert to_jsonable((1, 2)) == [1, 2]
    assert sorted(to_jsonable({1, 2, 3})) == [1, 2, 3]


def test_to_jsonable_stringifies_dict_keys():
    out = to_jsonable({1: "a"})
    assert out == {"1": "a"}

"""Tests for the generic frontend report translator (tech-debt #10)."""
from __future__ import annotations

from geometry_pipeline.core.issues import (
    DetectionStage,
    Issue,
    IssueKind,
    Severity,
)
from geometry_pipeline.reporting.frontend_schema import (
    KIND_TO_LEGACY_KEY,
    kind_dict,
)
from geometry_pipeline.validators.mesh.intersections import IntersectionsValidator
from geometry_pipeline.validators.mesh.t_junctions import TJunctionsValidator


def _issue(kind: IssueKind, payload: dict | None = None) -> Issue:
    return Issue.create(kind, Severity.WARN, DetectionStage.PRE, payload=payload or {})


def test_every_issue_kind_is_mapped_so_none_is_silently_dropped():
    # Regression for the "translator drift" smell: a kind missing from the
    # mapping would vanish from the report. All kinds must be present.
    assert set(KIND_TO_LEGACY_KEY) == set(IssueKind)


def test_kind_dict_is_generic_and_routes_each_kind_to_its_key():
    issues = [_issue(k, {"elements": [{"type": "vertex", "points": [[0, 0, 0]]}]}) for k in IssueKind]
    report = kind_dict(issues)

    for k in IssueKind:
        key = KIND_TO_LEGACY_KEY[k]
        assert len(report[key]) == 1, f"{k} not routed to {key}"
        assert report[key][0]["elements"] == [{"type": "vertex", "points": [[0, 0, 0]]}]


def test_summary_issues_are_excluded():
    # Summary marker Issues emitted by cap_and_summarize are not part of the
    # frontend shape.
    report = kind_dict([_issue(IssueKind.T_JUNCTION, {"summary": True})])
    assert report["T-junctions"] == []



def test_inverted_normal_is_no_longer_dropped():
    report = kind_dict([_issue(IssueKind.INVERTED_NORMAL, {"elements": []})])
    assert "inverted_normals" in report
    assert len(report["inverted_normals"]) == 1


def test_t_junction_validator_builds_elements_in_payload():
    raw = {
        "edge_coordinates": [[0, 0, 0], [1, 0, 0]],
        "split_vertex_coordinates": [0.5, 0, 0],
    }
    payload = TJunctionsValidator().payload_of(raw)
    assert payload["elements"] == [
        {"type": "edge", "points": [[0, 0, 0], [1, 0, 0]]},
        {"type": "vertex", "points": [0.5, 0, 0]},
    ]
    # The translator then surfaces those elements with no per-kind logic.
    issue = _issue(IssueKind.T_JUNCTION, payload)
    assert kind_dict([issue])["T-junctions"][0]["elements"] == payload["elements"]


def test_intersection_validator_builds_elements_in_payload():
    raw = {
        "hit_type": "interior",
        "edge_coordinates": [[0, 0, 0], [1, 0, 0]],
        "facet_fid_coordinates": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "point": [0.5, 0.5, 0],
    }
    payload = IntersectionsValidator().payload_of(raw)
    assert payload["elements"] == [
        {"type": "edge", "points": [[0, 0, 0], [1, 0, 0]]},
        {"type": "face", "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
        {"type": "vertex", "points": [[0.5, 0.5, 0]]},
    ]
    assert payload["sub_kind"] == "interior"

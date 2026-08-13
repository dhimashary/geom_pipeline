"""End-to-end smoke test on the real, all-defects room geometry.

Unlike the synthetic per-validator/per-repair unit tests, this drives the full
public pipeline (import -> validate -> repair -> export) over
``tests/models/public/01_Apartment_Room/Apartment_Room.obj`` and asserts
the run completes, detects defects, and writes output artifacts. Native cavity
detection is disabled so the test does not depend on the compiled C++ kernel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geometry_pipeline.api import GeometryResult, process_geometry, repair_geometry
from geometry_pipeline.core.issues import IssueKind


@pytest.fixture
def result(real_room_obj, tmp_path) -> GeometryResult:
    pytest.importorskip("shapely")
    return repair_geometry(real_room_obj, tmp_path, detect_cavities=False)


def test_pipeline_runs_and_detects_defects(result):
    assert isinstance(result, GeometryResult)
    # The real room is known to contain multiple kinds of defects.
    assert result.issue_count > 0


def test_issue_report_is_keyed_by_every_issue_kind(result):
    # ``kind_dict`` always emits a slot for every kind so none is silently
    # dropped, regardless of whether that kind was present.
    assert set(result.issue_report) == {k.value for k in IssueKind}


def test_pipeline_writes_output_artifacts(result, tmp_path):
    # Exporters write next to the output base; at least the repaired OBJ/GEO
    # should exist after a successful run.
    produced = list(tmp_path.rglob("*"))
    suffixes = {p.suffix for p in produced if p.is_file()}
    assert ".obj" in suffixes
    assert ".geo" in suffixes


def test_report_is_serialisable(result):
    import json

    # The report/issue payloads must be JSON-native (no numpy leaking through).
    json.dumps(result.issue_report)
    json.dumps(result.report)


# --- merged single-pass pipeline (process_geometry) --------------------------


@pytest.fixture
def merged_run(real_room_obj, tmp_path):
    """Run the merged pipeline once and capture the checkpoint payload(s)."""
    pytest.importorskip("shapely")
    seen: list[dict] = []
    result = process_geometry(
        real_room_obj,
        tmp_path,
        detect_cavities=False,
        on_checkpoint=seen.append,
    )
    return result, seen, tmp_path


def test_merged_fires_checkpoint_once_with_initial_issues(merged_run):
    _, seen, _ = merged_run
    # The inspect checkpoint fires exactly once, before the repair finishes.
    assert len(seen) == 1
    payload = seen[0]
    assert payload["stage"] == "t_junctions"
    assert payload["issue_count"] > 0
    assert set(payload["issue_report"]) == {k.value for k in IssueKind}


def test_merged_emits_checkpoint_and_repaired_artifacts(merged_run):
    _, _, tmp_path = merged_run
    names = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    stem = "Apartment_Room"
    # inspect checkpoint (initial 3dm/zip is produced by the backend
    # map_to_3dm_and_geo flow, not the pipeline)
    assert f"{stem}.geo" in names
    assert f"{stem}_inspect_issue.json" in names
    # repaired bundle + remaining report
    assert f"{stem}_repaired.geo" in names
    assert f"{stem}_repaired.zip" in names
    assert f"{stem}_remaining_issue.json" in names


def test_merged_result_describes_remaining_state(merged_run):
    result, _, _ = merged_run
    assert isinstance(result, GeometryResult)
    assert set(result.issue_report) == {k.value for k in IssueKind}
    assert "post" in result.report

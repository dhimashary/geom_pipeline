"""End-to-end smoke test on the real, all-defects room geometry.

Unlike the synthetic per-validator/per-repair unit tests, this drives the full
public pipeline (import -> validate -> repair -> export) over
``tests/models/vert2.0.6.obj`` and asserts the run completes, detects defects,
and writes output artifacts. Native cavity detection is disabled so the test
does not depend on the compiled C++ kernel.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from geometry_pipeline.api import GeometryResult, repair_geometry
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

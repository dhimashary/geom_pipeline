"""Unit tests for the public facade metadata helpers in ``api``.

``list_issue_kinds`` is pure (no I/O), so it lives with the unit tests. The
heavier ``repair_geometry`` / ``process_geometry`` entry points are exercised
by the integration smoke test.
"""
from __future__ import annotations

from geometry_pipeline.api import list_issue_kinds
from geometry_pipeline.core.issues import IssueKind


def test_list_issue_kinds_covers_every_kind():
    infos = list_issue_kinds()
    assert {info.kind for info in infos} == {k.value for k in IssueKind}


def test_list_issue_kinds_marks_repairable_kinds():
    by_kind = {info.kind: info for info in list_issue_kinds()}
    # These kinds have a repair step wired in the pipeline.
    for kind in ("duplicate_vertex", "degenerate_face", "t_junction", "intersection"):
        assert by_kind[kind].repairable is True
    # Detection-only kinds have no repair.
    assert by_kind["small_face"].repairable is False
    assert by_kind["collinear_face"].repairable is False


def test_list_issue_kinds_have_descriptions():
    assert all(info.description for info in list_issue_kinds())

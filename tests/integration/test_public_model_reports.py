"""Data-driven regression test over committed public room models.

Each ``tests/models/public/NN_<name>/`` folder is a self-contained case: it
holds one ``<name>.obj`` plus the expected ``<name>_inspect_issue.json`` and
``<name>_remaining_issue.json`` reports. This test runs the full merged
pipeline over the OBJ and asserts the freshly produced issue reports match the
committed references exactly. Dropping in a new ``02_<name>/`` folder is picked
up automatically — no code changes required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from geometry_pipeline.api import process_geometry

PUBLIC_DIR = Path(__file__).parent.parent / "models" / "public"
CASE_DIR_RE = re.compile(r"^\d+_")
INSPECT_SUFFIX = "_inspect_issue.json"
REMAINING_SUFFIX = "_remaining_issue.json"


def _discover_cases() -> list[Path]:
    """Return every ``NN_<name>`` folder holding an inspect-issue reference."""
    if not PUBLIC_DIR.is_dir():
        return []
    return sorted(
        d
        for d in PUBLIC_DIR.iterdir()
        if d.is_dir() and CASE_DIR_RE.match(d.name) and list(d.glob(f"*{INSPECT_SUFFIX}"))
    )


CASES = _discover_cases()


def _counts(report: dict) -> dict[str, int]:
    """Per-kind issue counts from a produced issue report."""
    return {kind: len(items) for kind, items in report.items()}


@pytest.mark.parametrize("case_dir", CASES, ids=[c.name for c in CASES])
def test_public_model_reports_match_reference(
    case_dir: Path, tmp_path: Path, record_public_model_coverage
) -> None:
    pytest.importorskip("shapely")

    # The source model stem is derived from the inspect-issue reference so
    # stray artifacts (e.g. ``<stem>_repaired.obj``) don't confuse discovery.
    inspects = list(case_dir.glob(f"*{INSPECT_SUFFIX}"))
    assert len(inspects) == 1, f"expected one {INSPECT_SUFFIX} in {case_dir}, found {len(inspects)}"
    stem = inspects[0].name[: -len(INSPECT_SUFFIX)]
    obj = case_dir / f"{stem}.obj"
    assert obj.exists(), f"missing source model: {obj}"

    process_geometry(obj, tmp_path, detect_cavities=False)

    produced_reports: dict[str, dict] = {}
    matches: dict[str, bool] = {}
    for suffix in (INSPECT_SUFFIX, REMAINING_SUFFIX):
        reference = case_dir / f"{stem}{suffix}"
        assert reference.exists(), f"missing reference report: {reference}"
        produced = tmp_path / f"{stem}{suffix}"
        assert produced.exists(), f"pipeline did not produce: {produced.name}"

        expected = json.loads(reference.read_text())
        actual = json.loads(produced.read_text())
        produced_reports[suffix] = actual
        matches[suffix] = actual == expected

    # Record coverage from the freshly executed run (not the reference) so the
    # terminal summary can show detected issues per model.
    record_public_model_coverage(
        model=case_dir.name,
        initial=_counts(produced_reports[INSPECT_SUFFIX]),
        remaining=_counts(produced_reports[REMAINING_SUFFIX]),
        match=matches[INSPECT_SUFFIX] and matches[REMAINING_SUFFIX],
    )

    assert matches[INSPECT_SUFFIX], f"{stem}{INSPECT_SUFFIX} does not match reference"
    assert matches[REMAINING_SUFFIX], f"{stem}{REMAINING_SUFFIX} does not match reference"

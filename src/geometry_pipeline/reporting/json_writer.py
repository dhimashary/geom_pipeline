from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from geometry_pipeline.core.jsonable import to_jsonable
from geometry_pipeline.core.report import PipelineResult

# Module logger
logger = logging.getLogger(__name__)


class JsonReportWriter:
    """Writes JSON artifacts derived from the pipeline result:

    - `<stem>_issue.json`: flat frontend-shaped issue dict (kind counts)
    - `<stem>_report.json`: PRE/POST snapshots + optional repairs
      (only when ``write_report=True``)

    ``issue_source`` selects which snapshot feeds `<stem>_issue.json`:
      - ``"initial"``   — the PRE snapshot (raw input issues)
      - ``"final"``     — the FINAL snapshot (post-repair, from the
        profile's ``final_validators``)
      - ``"composite"`` — per-kind "last detection wins" across all stages

    Unlike the geometry exporters in `io.exporters`, this writer ignores the
    `geom` argument and instead serializes the full `PipelineResult`, which it
    receives via `set_pipeline_result`.
    """

    def __init__(
        self,
        *,
        issue_suffix: str = "_issue.json",
        report_suffix: str = "_report.json",
        issue_source: str = "initial",
        write_report: bool = True,
    ) -> None:
        if issue_source not in ("initial", "final", "composite"):
            raise ValueError(
                f"issue_source must be 'initial', 'final' or 'composite', got {issue_source!r}"
            )
        self.issue_suffix = issue_suffix
        self.report_suffix = report_suffix
        self.issue_source = issue_source
        self.write_report = write_report
        self._result: PipelineResult | None = None

    def set_pipeline_result(self, result: PipelineResult) -> None:
        self._result = result

    def path_for(self, base: Path) -> Path:
        # receive base path (folder/stem) as the canonical base; exporter will
        # append its own suffixes when writing files.
        return Path(base)

    def write(self, geom: Any, path: Path) -> None:
        base = Path(path)
        stem = base.name
        parent = base.parent
        if self._result is None:
            logger.debug("No pipeline result to write for %s", base)
            return

        # Import translators lazily to avoid circular imports at module import time
        try:
            from .frontend_schema import kind_dict, snapshot_report
        except Exception:
            # fallback: build minimal shapes
            def kind_dict(x):  # type: ignore[misc]
                return {}

            def snapshot_report(x):  # type: ignore[misc]
                return {"pre": {}, "post": {}, "repairs": []}

        if self.issue_source == "composite":
            issues = self._result.composite_issues
        elif self.issue_source == "final":
            snap = getattr(self._result, "final", None)
            issues = snap.issues if snap else None
        else:  # "initial"
            snap = getattr(self._result, "initial", None)
            issues = snap.issues if snap else None
        issue_report = kind_dict(issues)

        issue_path = parent / f"{stem}{self.issue_suffix}"
        issue_path.parent.mkdir(parents=True, exist_ok=True)
        issue_path.write_text(json.dumps(to_jsonable(issue_report), indent=2, default=str))

        if self.write_report:
            report = snapshot_report(self._result)
            report_path = parent / f"{stem}{self.report_suffix}"
            report_path.write_text(json.dumps(to_jsonable(report), indent=2, default=str))


__all__ = ["JsonReportWriter"]

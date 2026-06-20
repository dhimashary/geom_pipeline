from __future__ import annotations
from pathlib import Path
import json
from typing import Any
import logging

# Module logger
logger = logging.getLogger(__name__)

from geometry_pipeline.core.report import PipelineResult
from geometry_pipeline.core.jsonable import to_jsonable


class JsonReportWriter:
    """Writes two JSON artifacts derived from the pipeline result:

    - `<stem>_issue.json`: flat frontend-shaped issue dict (kind counts)
    - `<stem>_report.json`: PRE/POST snapshots + optional repairs

    Unlike the geometry exporters in `io.exporters`, this writer ignores the
    `geom` argument and instead serializes the full `PipelineResult`, which it
    receives via `set_pipeline_result`.
    """

    def __init__(self, *, issue_suffix: str = "_issue.json", report_suffix: str = "_report.json", use_composite: bool = False) -> None:
        self.issue_suffix = issue_suffix
        self.report_suffix = report_suffix
        self.use_composite = use_composite
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
            def kind_dict(x):
                return {}

            def snapshot_report(x):
                return {"pre": {}, "post": {}, "repairs": []}
        
        if self.use_composite:
            issues = self._result.composite_issues
        elif getattr(self._result, "initial", None):
            issues = self._result.initial.issues
        else:
            issues = None
        issue_report = kind_dict(issues)
        report = snapshot_report(self._result)

        issue_path = parent / f"{stem}{self.issue_suffix}"
        report_path = parent / f"{stem}{self.report_suffix}"

        issue_path.parent.mkdir(parents=True, exist_ok=True)
        issue_path.write_text(json.dumps(to_jsonable(issue_report), indent=2, default=str))
        report_path.write_text(json.dumps(to_jsonable(report), indent=2, default=str))


__all__ = ["JsonReportWriter"]

"""Helpers shared by the seven legacy-detector wrappers.

Each detector in `app.services.geometry_inspection_service` returns a list
of dicts. These helpers build an `Issue`, and apply the global `max_reports`
cap with a summary issue.
"""

from __future__ import annotations

from typing import Callable

from geometry_pipeline.core.issues import DetectionStage, Issue, IssueKind


def cap_and_summarize(
    raw: list[dict],
    *,
    kind: IssueKind,
    stage: DetectionStage,
    stage_name: str,
    max_reports: int,
    payload_of: Callable[[dict], dict] | None = None,
) -> list[Issue]:
    """Convert raw detector dicts to Issues, applying the report cap.

    Parameters
    ----------
    raw : list[dict]
        Detector output.
    payload_of : callable(dict) -> dict
        Override for payload extraction..

    Cap behaviour: if len(raw) >= max_reports we keep the first max_reports
    entries and append one `summary` Issue with payload["summary"] = True.
    Each capped entry's payload also carries `capped=True`.
    """
    if payload_of is None:

        def payload_of(d: dict) -> dict:
            return d

    capped = len(raw) >= max_reports
    kept = raw[:max_reports] if capped else raw

    issues: list[Issue] = []
    for d in kept:
        payload = payload_of(d)
        if capped:
            payload = {**payload, "capped": True}
        issues.append(
            Issue.create(
                kind=kind,
                stage=stage,
                stage_name=stage_name,
                payload=payload,
            )
        )

    if capped:
        issues.append(
            Issue.create(
                kind=kind,
                stage=stage,
                stage_name=stage_name,
                payload={
                    "summary": True,
                    "capped": True,
                    "actual_count_lower_bound": len(raw),
                    "max_reports": max_reports,
                },
            )
        )

    return issues


__all__ = ["cap_and_summarize"]

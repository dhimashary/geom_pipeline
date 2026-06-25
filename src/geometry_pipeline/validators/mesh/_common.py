"""Helpers shared by the seven legacy-detector wrappers.

Each detector in `app.services.geometry_inspection_service` returns a list
of dicts with a "severity" string ("medium" / "high"). These helpers map
that to `Severity`, build an `Issue`, and apply the global `max_reports`
cap with a summary issue per migration-plan decision #4.
"""
from __future__ import annotations

from geometry_pipeline.core.issues import DetectionStage, Issue, IssueKind, Severity


def severity_from_legacy(s: str) -> Severity:
    """Map legacy severity strings to the new enum.

    "high" → FATAL, "medium" → WARN. Anything unknown → WARN (conservative).
    """
    return Severity.FATAL if s == "high" else Severity.WARN


def cap_and_summarize(
    raw: list[dict],
    *,
    kind: IssueKind,
    stage: DetectionStage,
    stage_name: str,
    max_reports: int,
    severity_of: callable = None,
    payload_of: callable = None,
) -> list[Issue]:
    """Convert raw detector dicts to Issues, applying the report cap.

    Parameters
    ----------
    raw : list[dict]
        Detector output, in the legacy "elements/severity/details" shape.
    severity_of : callable(dict) -> Severity
        Override for severity extraction. Defaults to looking at d["severity"]
        and applying `severity_from_legacy`.
    payload_of : callable(dict) -> dict
        Override for payload extraction. Defaults to dropping the
        "severity" key (it is encoded by the Issue.severity field) and
        keeping everything else.

    Cap behaviour: if len(raw) >= max_reports we keep the first max_reports
    entries and append one `summary` Issue with payload["summary"] = True.
    Each capped entry's payload also carries `capped=True`.
    """
    if severity_of is None:
        def severity_of(d: dict) -> Severity:
            return severity_from_legacy(d.get("severity", "medium"))
    if payload_of is None:
        def payload_of(d: dict) -> dict:
            return {k: v for k, v in d.items() if k != "severity"}

    capped = len(raw) >= max_reports
    kept = raw[:max_reports] if capped else raw

    issues: list[Issue] = []
    for d in kept:
        payload = payload_of(d)
        if capped:
            payload = {**payload, "capped": True}
        issues.append(Issue.create(
            kind=kind,
            severity=severity_of(d),
            stage=stage,
            stage_name=stage_name,
            payload=payload,
        ))

    if capped:
        issues.append(Issue.create(
            kind=kind,
            severity=Severity.WARN,
            stage=stage,
            stage_name=stage_name,
            payload={
                "summary": True,
                "capped": True,
                "actual_count_lower_bound": len(raw),
                "max_reports": max_reports,
            },
        ))

    return issues


__all__ = ["severity_from_legacy", "cap_and_summarize"]

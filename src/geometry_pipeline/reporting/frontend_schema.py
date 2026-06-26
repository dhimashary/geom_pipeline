from __future__ import annotations

from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.report import PipelineResult, ValidationSnapshot


def _legacy_severity(i: Issue) -> str:
    return {"fatal": "high", "warn": "medium"}.get(i.severity.value, "low")


def _normalise_elements(i: Issue) -> list[dict]:
    """Read the validator-supplied `elements` shape — no per-kind logic.

    Validators populate `payload["elements"]` with the frontend element shape
    (`[{"type": ..., "points": ...}, ...]`). Accept either a single dict or a
    list; anything else yields an empty list.
    """
    elements = i.payload.get("elements")
    if isinstance(elements, dict):
        return [elements]
    if isinstance(elements, list):
        return elements
    return []


def _entry(i: Issue) -> dict:
    return {
        "elements": _normalise_elements(i),
        "severity": _legacy_severity(i),
        "id": i.id,
    }


def kind_dict(issues: list[Issue]) -> dict:
    """Flat dict — one list per IssueKind, keyed by the kind's enum value.

    Enumerates every `IssueKind`, so each kind always gets a (possibly empty)
    list and no kind is ever silently dropped.

    Summary marker Issues emitted by `validators.mesh._common.cap_and_summarize`
    when a list is capped are excluded from the frontend shape.
    """
    out: dict[str, list[dict]] = {kind.value: [] for kind in IssueKind}
    for i in issues or []:
        if i.payload.get("summary"):
            continue
        out[i.kind.value].append(_entry(i))
    return out


def issue_detection_report_from_snapshot(snapshot: ValidationSnapshot) -> dict:
    return kind_dict(snapshot.issues)


def revalidation_report_from_snapshot(snapshot: ValidationSnapshot) -> dict:
    return kind_dict(snapshot.issues)


def repair_report_from_pipeline(result: PipelineResult) -> list[dict]:
    out: list[dict] = []
    for r in result.repairs.results:
        out.append({
            "repair_type": r.step_name,
            "affected_count": len(r.affected_ids),
            "before": r.before_count,
            "after": r.after_count,
            "details": dict(r.details),
            "stage": r.stage_name,
            "iterations": r.iterations,
            "affected_issue_ids": list(r.affected_ids),
        })
    return out


def snapshot_report(pipeline_result: PipelineResult) -> dict:
    pre = {}
    post = {}
    repairs = []

    if pipeline_result is None:
        return {"pre": pre, "post": post, "repairs": repairs}

    if hasattr(pipeline_result, "initial") and hasattr(pipeline_result, "final"):
        pre = kind_dict(getattr(pipeline_result.initial, "issues", []))
        post = kind_dict(getattr(pipeline_result.final, "issues", []))

    repairs = repair_report_from_pipeline(pipeline_result)

    return {"pre": pre, "post": post, "repairs": repairs}

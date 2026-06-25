from __future__ import annotations

from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.report import PipelineResult, ValidationSnapshot


# The single source of truth for the frontend's exact JSON keys. This is the
# *only* place that knows the frontend's key strings. Element-shaping lives in
# each validator's payload (`payload["elements"]`), so this translator stays
# generic: adding a new IssueKind means adding one entry here and (optionally)
# building `elements` inside that validator — no per-kind branches below.
KIND_TO_LEGACY_KEY: dict[IssueKind, str] = {
    IssueKind.DUPLICATE_VERTEX: "duplicate_vertices",
    IssueKind.NON_PLANAR_FACE:  "non_coplanar_faces",
    IssueKind.T_JUNCTION:       "T-junctions",
    IssueKind.POSSIBLE_HOLE:    "possible_holes",
    IssueKind.BOUNDARY_EDGE:    "boundary_edges",
    IssueKind.DEGENERATE_FACE:  "degenerate_faces",
    IssueKind.INTERSECTION:     "intersections",
    IssueKind.OVERLAPPING_FACE: "overlapping_faces",
    IssueKind.SMALL_FACE:       "small_faces",
    IssueKind.INVERTED_NORMAL:  "inverted_normals",
}


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
    """Frontend-shaped flat dict — one list per IssueKind, built generically.

    Iterates a single mapping rather than enumerating kinds, so a new
    `IssueKind` is never silently dropped: every mapped kind always gets a
    (possibly empty) list, and any unmapped kind would surface as a missing
    key rather than vanishing from a hand-written branch.

    Summary marker Issues emitted by `validators.mesh._common.cap_and_summarize`
    when a list is capped are excluded from the frontend shape.
    """
    out: dict[str, list[dict]] = {key: [] for key in KIND_TO_LEGACY_KEY.values()}
    for i in issues or []:
        if i.payload.get("summary"):
            continue
        key = KIND_TO_LEGACY_KEY.get(i.kind)
        if key is None:
            continue
        out[key].append(_entry(i))
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

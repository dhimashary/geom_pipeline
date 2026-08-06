from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .core.context import Context
from .core.issues import IssueKind
from .core.tolerances import Tolerances

# Internal imports — all private to the package; the facade is the public surface.
from .io.registry import ImporterRegistry
from .pipeline.runner import run_pipeline
from .profiles.wave_based import wave_based_profile
from .reporting.frontend_schema import kind_dict, snapshot_report

SUPPORTED_INPUTS = (".obj", ".3dm", ".dxf")

_log = logging.getLogger("choras_geometry")


class GeometryError(RuntimeError):
    """Single public error type for facade consumers."""


@dataclass(frozen=True)
class GeometryResult:
    outputs: dict[str, str]
    issue_report: dict
    report: dict
    issue_count: int


def _collect_outputs(result: Any) -> dict[str, str]:
    # Best-effort: try common attribute names added by the pipeline result
    if result is None:
        return {}
    if hasattr(result, "outputs") and isinstance(result.outputs, dict):
        return result.outputs
    if hasattr(result, "exported_paths") and isinstance(result.exported_paths, dict):
        return result.exported_paths
    # try to gather from exporters if present
    out = {}
    exporters = getattr(result, "exporters", None) or getattr(result, "exporter_targets", None)
    if exporters:
        try:
            for name, target in exporters.items():
                out[name] = str(target)
        except Exception:
            pass
    return out


def _count_issues(issue_report: dict) -> int:
    # Count issues in the report. `kind_dict` shapes the report as a mapping of
    # issue-kind -> list of issue entries, so list lengths must be counted.
    def _sum(v):
        if isinstance(v, bool):
            return 0
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            return sum(_sum(x) for x in v.values())
        if isinstance(v, list):
            return len(v)
        return 0

    return _sum(issue_report or {})


def repair_geometry(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    volume_name: str = "RoomVolume",
    detect_cavities: bool = True,
) -> GeometryResult:
    in_path = Path(input_path)
    try:
        geom = ImporterRegistry.for_extension(in_path.suffix).load(in_path)
        profile = wave_based_profile(detect_cavities=detect_cavities, volume_name=volume_name)
        ctx = Context(
            tolerances=Tolerances(), logger=_log, profile_name=getattr(profile, "name", None) or ""
        )
        base = Path(output_dir) / in_path.stem
        result = run_pipeline(geom, profile, base, ctx)
    except Exception as exc:
        raise GeometryError(str(exc)) from exc

    issue_report = kind_dict(result.initial.issues) if result.initial else {}
    report = snapshot_report(result)
    return GeometryResult(
        outputs=_collect_outputs(result),
        issue_report=issue_report,
        report=report,
        issue_count=_count_issues(issue_report),
    )


def process_geometry(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    volume_name: str = "RoomVolume",
    detect_cavities: bool = True,
    on_checkpoint: Optional[Callable[[dict], None]] = None,
) -> GeometryResult:
    """Run the merged wave-based pipeline in a single pass.

    One run emits all geometry artifacts:
      * the initial bundle ``<stem>.3dm`` + ``<stem>.zip`` (raw upload),
      * the inspect checkpoint ``<stem>.geo`` + ``<stem>_inspect_issue.json``
        (tjunc-fixed, pre-intersection-repair mesh), and
      * the repaired bundle ``<stem>_repaired.{obj,3dm,geo,zip}`` +
        ``<stem>_remaining_issue.json`` + ``<stem>_report.json``.

    ``on_checkpoint``, if given, is invoked once when the inspect checkpoint
    fires — *before* the repair finishes — with a dict
    ``{"stage", "issue_report", "issue_count"}`` describing the initial
    (AfterUpload) issues, so callers can persist them and report progress early.

    The returned :class:`GeometryResult` describes the *repaired* (remaining)
    state: ``report["post"]`` / ``issue_count`` reflect the final validation.
    """
    in_path = Path(input_path)

    def _raw_cb(stage_name: str, interim: Any) -> None:
        if on_checkpoint is None:
            return
        checkpoint_report = kind_dict(interim.composite_issues)
        on_checkpoint(
            {
                "stage": stage_name,
                "issue_report": checkpoint_report,
                "issue_count": _count_issues(checkpoint_report),
            }
        )

    try:
        geom = ImporterRegistry.for_extension(in_path.suffix).load(in_path)
        profile = wave_based_profile(detect_cavities=detect_cavities, volume_name=volume_name)
        ctx = Context(
            tolerances=Tolerances(),
            logger=_log,
            profile_name=getattr(profile, "name", None) or "",
            extras={"on_checkpoint": _raw_cb} if on_checkpoint is not None else {},
        )
        base = Path(output_dir) / in_path.stem
        result = run_pipeline(geom, profile, base, ctx)
    except Exception as exc:
        raise GeometryError(str(exc)) from exc

    final_snap = getattr(result, "final", None)
    issue_report = kind_dict(final_snap.issues if final_snap else [])
    report = snapshot_report(result)
    return GeometryResult(
        outputs=_collect_outputs(result),
        issue_report=issue_report,
        report=report,
        issue_count=_count_issues(issue_report),
    )


# Short, human-readable description for every IssueKind the validators can emit.
_ISSUE_DESCRIPTIONS: dict[IssueKind, str] = {
    IssueKind.DUPLICATE_VERTEX: "Two or more vertices share the same position.",
    IssueKind.ZERO_AREA_FACE: "A face has zero (or near-zero) area.",
    IssueKind.NON_PLANAR_FACE: "A polygon face's vertices are not coplanar.",
    IssueKind.T_JUNCTION: "A vertex lies on another face's edge without being shared (T-junction).",
    IssueKind.INTERSECTION: "An edge passes through a face (segment-facet intersection).",
    IssueKind.BOUNDARY_EDGE: "An edge is used by only one face (open boundary).",
    IssueKind.POSSIBLE_HOLE: "A loop of boundary edges suggests a missing face (hole).",
    IssueKind.SMALL_FACE: "A face is smaller than the minimum-area tolerance.",
    IssueKind.OVERLAPPING_FACE: "Two faces occupy the same region of space.",
    IssueKind.COLLINEAR_FACE: "A face's vertices are collinear or nearly collinear (collapsed to a line).",
}


@dataclass(frozen=True)
class IssueInfo:
    """Metadata about one kind of geometry issue this pipeline can detect."""

    kind: str
    description: str
    repairable: bool


def _repairable_issue_kinds() -> set[IssueKind]:
    """Union of the `handles` sets declared by every exported repair step.

    Computed dynamically so it never drifts from the actual repair classes.
    """
    from . import repairs as _repairs_pkg

    kinds: set[IssueKind] = set()
    for name in getattr(_repairs_pkg, "__all__", []):
        handles = getattr(getattr(_repairs_pkg, name, None), "handles", None)
        if handles:
            kinds |= set(handles)
    return kinds


def list_issue_kinds() -> list[IssueInfo]:
    """List every issue kind the pipeline can detect.

    Each entry carries a short description and whether a repair step in this
    pipeline can fix it (``repairable``). Kinds with no repair are
    detection-only and must be resolved upstream (e.g. in the CAD tool).
    """
    repairable = _repairable_issue_kinds()
    return [
        IssueInfo(
            kind=kind.value,
            description=_ISSUE_DESCRIPTIONS[kind],
            repairable=kind in repairable,
        )
        for kind in IssueKind
    ]

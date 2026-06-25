from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import tempfile
from typing import Any

# Internal imports — all private to the package; the facade is the public surface.
from .io.registry import ImporterRegistry
from .core.context import Context
from .core.tolerances import Tolerances
from .core.issues import IssueKind
from .pipeline.runner import run_pipeline
from .profiles.wave_based import wave_based_profile, wave_based_inspect_profile
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
    # Sum numeric values in the issue_report; if nested, sum recursively
    def _sum(v):
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            return sum(_sum(x) for x in v.values())
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
        ctx = Context(tolerances=Tolerances(), logger=_log, profile_name=getattr(profile, "name", None))
        base = Path(output_dir) / in_path.stem
        result = run_pipeline(geom, profile, base, ctx)
    except Exception as exc:
        raise GeometryError(str(exc)) from exc

    issue_report = kind_dict(result.initial.issues)
    report = snapshot_report(result)
    return GeometryResult(
        outputs=_collect_outputs(result),
        issue_report=issue_report,
        report=report,
        issue_count=_count_issues(issue_report),
    )


def inspect_geometry(
    input_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> GeometryResult:
    in_path = Path(input_path)
    try:
        geom = ImporterRegistry.for_extension(in_path.suffix).load(in_path)
        profile = wave_based_inspect_profile()
        ctx = Context(tolerances=Tolerances(), logger=_log, profile_name=getattr(profile, "name", None))
        base = Path(output_dir) / in_path.stem if output_dir is not None else Path(tempfile.gettempdir()) / in_path.stem
        result = run_pipeline(geom, profile, base, ctx)
    except Exception as exc:
        raise GeometryError(str(exc)) from exc

    issue_report = kind_dict(result.initial.issues)
    report = snapshot_report(result)
    return GeometryResult(
        outputs={},
        issue_report=issue_report,
        report=report,
        issue_count=_count_issues(issue_report),
    )


# Short, human-readable description for every IssueKind the validators can emit.
_ISSUE_DESCRIPTIONS: dict[IssueKind, str] = {
    IssueKind.DUPLICATE_VERTEX: "Two or more vertices share the same position.",
    IssueKind.DEGENERATE_FACE: "A face has zero (or near-zero) area.",
    IssueKind.NON_PLANAR_FACE: "A polygon face's vertices are not coplanar.",
    IssueKind.T_JUNCTION: "A vertex lies on another face's edge without being shared (T-junction).",
    IssueKind.INTERSECTION: "An edge passes through a face (segment-facet intersection).",
    IssueKind.BOUNDARY_EDGE: "An edge is used by only one face (open boundary).",
    IssueKind.POSSIBLE_HOLE: "A loop of boundary edges suggests a missing face (hole).",
    IssueKind.INVERTED_NORMAL: "A face is wound so its normal points the wrong way.",
    IssueKind.SMALL_FACE: "A face is smaller than the minimum-area tolerance.",
    IssueKind.OVERLAPPING_FACE: "Two faces occupy the same region of space.",
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

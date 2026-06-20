"""Reporting value objects produced by the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

from geometry_pipeline.core.ir import Geometry
from geometry_pipeline.core.issues import DetectionStage, Issue, IssueKind


@dataclass
class ValidationSnapshot:
    """Issues observed at one point in the pipeline."""
    when: DetectionStage
    stage_name: str = ""
    issues: list[Issue] = field(default_factory=list)


@dataclass
class RepairResult:
    step_name: str
    stage_name: str
    affected_ids: list[str] = field(default_factory=list)
    before_count: int = 0
    after_count: int = 0
    iterations: int = 1
    details: dict = field(default_factory=dict)


@dataclass
class RepairReport:
    results: list[RepairResult] = field(default_factory=list)


@dataclass
class PipelineResult:
    geometry: Geometry
    snapshots: list[ValidationSnapshot] = field(default_factory=list)
    repairs: RepairReport = field(default_factory=RepairReport)
    output_path: str | None = None

    @property
    def initial(self) -> ValidationSnapshot | None:
        return self.snapshots[0] if self.snapshots else None

    @property
    def final(self) -> ValidationSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def composite_issues(self) -> list[Issue]:
        """Per-kind "last detection wins" across all snapshots.

        Each ``IssueKind`` is reported from the *latest* snapshot that
        detected it, so kinds measured at a specific stage (e.g. T-junctions
        before their fix, intersections after the T-junction fix) come from
        that stage rather than the raw PRE pass or a fully-processed FINAL
        pass. Kinds only ever seen in PRE keep their PRE issues.
        """
        latest: dict[IssueKind, list[Issue]] = {}
        for snap in self.snapshots:
            by_kind: dict[IssueKind, list[Issue]] = {}
            for issue in snap.issues:
                by_kind.setdefault(issue.kind, []).append(issue)
            latest.update(by_kind)
        return [issue for issues in latest.values() for issue in issues]

"""Repair protocol + shared base class."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Protocol

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Geometry
from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.report import RepairResult


class RepairStep(Protocol):
    name: ClassVar[str]
    accepts: ClassVar[set[str]]  # IR kinds it can run on
    handles: ClassVar[set[IssueKind]]  # IssueKinds this step addresses

    def apply(
        self,
        geom: Geometry,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Geometry, RepairResult]: ...


class BaseRepair(ABC):
    """Shared plumbing for repair steps."""

    name: ClassVar[str]
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = set()

    def ensure_accepts(self, geom: Geometry) -> None:
        if geom.kind not in self.accepts:
            raise TypeError(
                f"{self.__class__.__name__} accepts {sorted(self.accepts)!r}, got {geom.kind!r}"
            )

    def affected_ids(self, issues: list[Issue]) -> list[str]:
        if not self.handles:
            return []
        return [i.id for i in issues if i.kind in self.handles]

    def make_result(
        self,
        *,
        stage_name: str = "",
        before_count: int,
        after_count: int,
        details: dict,
        issues: list[Issue],
        iterations: int | None = None,
        affected_ids: list[str] | None = None,
    ) -> RepairResult:
        return RepairResult(
            step_name=self.name,
            stage_name=stage_name,
            affected_ids=self.affected_ids(issues) if affected_ids is None else affected_ids,
            before_count=before_count,
            after_count=after_count,
            iterations=iterations,
            details=details,
        )

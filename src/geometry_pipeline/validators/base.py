"""Validator protocol + shared base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Geometry
from geometry_pipeline.core.issues import DetectionStage, Issue, IssueKind, Severity
from geometry_pipeline.validators.mesh._common import cap_and_summarize, severity_from_legacy


class Validator(Protocol):
    name: ClassVar[str]
    accepts: ClassVar[set[str]]      # IR kinds, e.g. {"mesh"}
    kind: ClassVar[IssueKind]

    def detect(self, geom: Geometry, ctx: Context) -> list[Issue]: ...


class BaseValidator(ABC):
    """Shared plumbing for validators.

    Concrete validators only implement `detect_raw` and, when needed,
    override `severity_of` / `payload_of`.
    """

    name: ClassVar[str]
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind]
    when: ClassVar[DetectionStage] = DetectionStage.PRE
    stage_name: ClassVar[str] = ""

    def detect(self, geom: Geometry, ctx: Context) -> list[Issue]:
        self.ensure_accepts(geom)
        raw = self.detect_raw(geom, ctx)
        return cap_and_summarize(
            raw,
            kind=self.kind,
            stage=self.when,
            stage_name=self.stage_name,
            max_reports=ctx.tolerances.max_reports,
            severity_of=self.severity_of,
            payload_of=self.payload_of,
        )

    def ensure_accepts(self, geom: Geometry) -> None:
        if geom.kind not in self.accepts:
            raise TypeError(
                f"{self.__class__.__name__} accepts {sorted(self.accepts)!r}, got {geom.kind!r}"
            )

    def severity_of(self, payload: dict) -> Severity:
        return severity_from_legacy(payload.get("severity", "medium"))

    def payload_of(self, payload: dict) -> dict:
        return {k: v for k, v in payload.items() if k != "severity"}

    @abstractmethod
    def detect_raw(self, geom: Geometry, ctx: Context) -> list[dict]:
        """Return legacy detector dictionaries before Issue conversion."""

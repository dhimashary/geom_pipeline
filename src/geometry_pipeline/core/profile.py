"""SimulationProfile bundles all per-solver policy in one object."""
from __future__ import annotations

from dataclasses import dataclass, field
from geometry_pipeline.core.ir import Exporter, Geometry
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.repairs.base import RepairStep
from geometry_pipeline.core.tolerances import Tolerances
from geometry_pipeline.validators.base import Validator


@dataclass
class Stage:
    """One ordered repair-then-detect stage of the pipeline.

    ``exporters`` turns the stage into an export point: after its repairs and
    post-validators run, the pipeline writes these exporters against the
    current (intermediate) mesh and the snapshots gathered so far. This lets a
    single run emit intermediate artifacts (e.g. the raw or inspect bundle)
    without a second pass. Empty by default, so ordinary stages are unaffected.

    ``checkpoint`` additionally fires the ``on_checkpoint`` callback (from
    ``ctx.extras``) after the exporters run, so callers can react mid-pipeline
    (persist rows, report progress). An export point is not necessarily a
    notify point: e.g. the raw stage exports the initial bundle but does not
    notify, while the inspect checkpoint does both.
    """
    name: str
    repairs: list[RepairStep] = field(default_factory=list)
    post_validators: list[Validator] = field(default_factory=list)
    fail_fast_on: set[IssueKind] = field(default_factory=set)
    exporters: list[Exporter] = field(default_factory=list)
    checkpoint: bool = False


@dataclass
class SimulationProfile:
    name: str
    target_ir: type[Geometry]
    pre_validators: list[Validator]
    stages: list[Stage]
    final_validators: list[Validator]
    exporters: list[Exporter]
    tolerances: Tolerances

    def __post_init__(self) -> None:
        ir_kind = self.target_ir.kind
        bad: list[str] = []

        def _check(component, role: str) -> None:
            accepts = getattr(component, "accepts", None)
            name = getattr(component, "name", type(component).__name__)
            if accepts is None or ir_kind not in accepts:
                bad.append(f"{role} {name!r} accepts={accepts!r} but profile target_ir.kind={ir_kind!r}")

        for v in self.pre_validators:
            _check(v, "pre_validator")
        for v in self.final_validators:
            _check(v, "final_validator")
        for stage in self.stages:
            for r in stage.repairs:
                _check(r, f"stage[{stage.name}].repair")
            for v in stage.post_validators:
                _check(v, f"stage[{stage.name}].post_validator")

        if bad:
            joined = "\n  - ".join(bad)
            raise ValueError(
                f"SimulationProfile {self.name!r} has IR-kind mismatches:\n  - {joined}"
            )

"""Top-level orchestration: importer -> (convert) -> pre-validate -> stages -> final-validate -> export."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Geometry
from geometry_pipeline.core.issues import DetectionStage, Issue
from geometry_pipeline.core.profile import SimulationProfile, Stage
from geometry_pipeline.core.report import (
    PipelineResult,
    RepairReport,
    ValidationSnapshot,
)
from geometry_pipeline.validators.base import Validator


def run_validators(
    geom: Geometry,
    validators: list[Validator],
    ctx: Context,
    when: DetectionStage,
    stage_name: str = "",
) -> ValidationSnapshot:
    """Run every validator whose `accepts` includes `geom.kind`."""
    issues = []
    for v in validators:
        if geom.kind not in v.accepts:
            ctx.logger.warning(
                "[validators] skipping %s: accepts=%r but geom.kind=%r",
                v.name, v.accepts, geom.kind,
            )
            continue
        out = v.detect(geom, ctx)
        for i in out:
            if i.stage is when and i.stage_name == stage_name:
                issues.append(i)
            else:
                issues.append(Issue.create(
                    kind=i.kind,
                    stage=when,
                    stage_name=stage_name,
                    payload=i.payload,
                ))
    return ValidationSnapshot(
        when=when,
        stage_name=stage_name,
        issues=issues,
    )


def run_stage(
    geom: Geometry,
    stage: Stage,
    ctx: Context,
    repair_report: RepairReport,
    pre_issues: list,
) -> tuple[Geometry, ValidationSnapshot]:
    """Apply one repair stage, run its post-validators, return new snapshot."""
    for repair in stage.repairs:
        if geom.kind not in repair.accepts:
            ctx.logger.warning(
                "[stage %s] skipping repair %s: accepts=%r but geom.kind=%r",
                stage.name, repair.name, repair.accepts, geom.kind,
            )
            continue
        relevant = [i for i in pre_issues if i.kind in repair.handles] if repair.handles else pre_issues
        geom, result = repair.apply(geom, relevant, ctx, stage_name=stage.name)
        repair_report.results.append(result)

    snapshot = run_validators(
        geom, stage.post_validators, ctx,
        when=DetectionStage.POST_STAGE, stage_name=stage.name,
    )

    if stage.fail_fast_on:
        offending = [i for i in snapshot.issues if i.kind in stage.fail_fast_on]
        if offending:
            ctx.logger.error(
                "[stage %s] fail_fast_on triggered (%d issues of kinds %s)",
                stage.name, len(offending),
                {i.kind.value for i in offending},
            )
            raise PipelineFailFastError(stage_name=stage.name, issues=offending)

    return geom, snapshot


class PipelineFailFastError(RuntimeError):
    def __init__(self, stage_name: str, issues: list) -> None:
        super().__init__(f"Pipeline aborted at stage {stage_name!r}: {len(issues)} blocking issue(s)")
        self.stage_name = stage_name
        self.issues = issues


def run_checkpoint_exporters(
    geom: Geometry,
    stage: Stage,
    snapshots: list,
    repairs: RepairReport,
    output_path: Path,
    ctx: Context,
) -> None:
    """Fire a stage's exporters against the intermediate mesh.

    The interim ``PipelineResult`` carries only the snapshots gathered so far,
    so its ``composite_issues`` reflects the mesh *at this point* (no later or
    FINAL snapshots leak in). When ``stage.checkpoint`` is set, an optional
    ``ctx.extras['on_checkpoint']`` callback is invoked afterwards as
    ``cb(stage_name, interim_result)`` so callers (e.g. the backend) can react
    mid-pipeline (write DB rows, bump progress). Export points that are not
    checkpoints (e.g. the raw stage) write their files without notifying.
    """
    interim = PipelineResult(
        geometry=geom,
        snapshots=list(snapshots),
        repairs=repairs,
        output_path=str(output_path),
    )
    for exporter in stage.exporters:
        if hasattr(exporter, "set_pipeline_result"):
            try:
                exporter.set_pipeline_result(interim)
            except Exception:
                ctx.logger.exception("[pipeline] checkpoint exporter.set_pipeline_result failed")
        target = getattr(exporter, "path_for", lambda p: p)(output_path)
        exporter.write(geom, target)
        ctx.logger.info("[pipeline] export(%s) wrote %s", stage.name, target)

    if not stage.checkpoint:
        return
    cb = ctx.extras.get("on_checkpoint") if getattr(ctx, "extras", None) else None
    if cb is not None:
        try:
            cb(stage.name, interim)
        except Exception:
            ctx.logger.exception("[pipeline] on_checkpoint callback failed for stage %s", stage.name)


def run_pipeline(
    geom: Geometry,
    profile: SimulationProfile,
    output_path: Path,
    ctx: Context,
) -> PipelineResult:
    """Run the full pipeline for `geom` under `profile`."""
    logger = ctx.logger

    logger.warning("Running pipeline %r on geometry kind %r", profile.name, geom.kind)
    if geom.kind != profile.target_ir.kind:
        raise ValueError(
            f"Profile {profile.name!r} expects IR kind {profile.target_ir.kind!r}, "
            f"got {geom.kind!r}"
        )

    snapshots: list[ValidationSnapshot] = []
    repairs = RepairReport()
    logger.warning("Pipeline %r: starting PRE-validation", profile.name)
    pre = run_validators(
        geom, profile.pre_validators, ctx,
        when=DetectionStage.PRE, stage_name="",
    )
    snapshots.append(pre)
    accumulated_issues = list(pre.issues)
    logger.warning("Pipeline %r: completed PRE-validation with %d issue(s)", profile.name, len(pre.issues))

    # Ensure the output directory exists before any (possibly mid-pipeline)
    # checkpoint export writes to it.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for stage in profile.stages:
        geom, snap = run_stage(geom, stage, ctx, repairs, accumulated_issues)
        snapshots.append(snap)
        accumulated_issues = list(snap.issues)
        if stage.exporters:
            run_checkpoint_exporters(geom, stage, snapshots, repairs, output_path, ctx)

    logger.warning("Pipeline %r: completed all stages, starting FINAL validation", profile.name)
    final = run_validators(
        geom, profile.final_validators, ctx,
        when=DetectionStage.FINAL, stage_name="",
    )
    snapshots.append(final)
    logger.warning("Pipeline %r: completed FINAL validation with %d issue(s)", profile.name, len(final.issues))

    logger.warning("Pipeline %r: starting export to %s", profile.name, output_path)

    pipeline_result = PipelineResult(
        geometry=geom,
        snapshots=snapshots,
        repairs=repairs,
        output_path=str(output_path),
    )

    for exporter in profile.exporters:
        # Allow exporters to receive the full pipeline result when they support it.
        if hasattr(exporter, "set_pipeline_result"):
            try:
                exporter.set_pipeline_result(pipeline_result)
            except Exception:
                ctx.logger.exception("[pipeline] exporter.set_pipeline_result failed")

        target = getattr(exporter, "path_for", lambda p: p)(output_path)
        exporter.write(geom, target)
        ctx.logger.info(
            "[pipeline] wrote %s at %s",
            target, datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    logger.warning("Pipeline %r: completed export to %s", profile.name, output_path)
    return pipeline_result

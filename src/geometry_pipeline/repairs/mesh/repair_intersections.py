"""Three intersection repairs sharing the same injected detector.

* ``TrimSegmentFaceIntersectionsRepair`` — for "edge cuts through face
  interior" cases. Trims one component at a time against the offending
  face's plane until no `segment_face_interior_intersection` remains.
* ``RepairPlcSingleSplitsRepair`` — for `endpoint_face_interior_touch`
  cases where the touching endpoint lies inside a face: split the face
  at that endpoint via triangulation.
* ``RepairPlcByOffsetRepair`` — last-resort fallback: nudge a touching
  endpoint along the touched face's normal by ``tolerances.plc_offset``.
"""
from __future__ import annotations

from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.ir import Vertex
from geometry_pipeline.core.report import RepairResult
from geometry_pipeline.repairs.base import BaseRepair
from geometry_pipeline.repairs.mesh._common import room_center_from_mesh
from geometry_pipeline.validators.base import Validator
from geometry_pipeline.repairs.mesh._intersection_repairs import (
    repair_plc_by_offset_iterative,
    repair_plc_single_splits_iterative,
    trim_segment_face_intersections_iterative,
)


def _affected(issues: list[Issue]) -> list[str]:
    return [i.id for i in issues if i.kind == IssueKind.INTERSECTION]


class TrimSegmentFaceIntersectionsRepair(BaseRepair):
    name: ClassVar[str] = "trim_segment_face_intersections_iterative"
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = {IssueKind.INTERSECTION}

    def __init__(self, detector: Validator, max_iters: int | None = None) -> None:
        self.detector = detector
        self.max_iters = max_iters

    def apply(
        self,
        geom: Mesh,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Mesh, RepairResult]:
        self.ensure_accepts(geom)
        max_iters = self.max_iters or ctx.tolerances.max_plc_iters
        faces = list(geom.faces)
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        before_faces = len(faces)
        before_points = len(points)

        new_faces, new_points, changed_any, diag = trim_segment_face_intersections_iterative(
            faces, points,
            room_center_from_mesh(geom),
            max_iters=max_iters,
            tol=ctx.tolerances.clipping,
            logger=ctx.logger,
            plc_report_supplier=lambda f, p: [
                i.payload
                for i in self.detector.detect(
                    Mesh(
                        vertices=[Vertex(x=pt[0], y=pt[1], z=pt[2]) for pt in p],
                        faces=list(f),
                        materials=dict(geom.materials),
                        metadata=dict(geom.metadata),
                    ),
                    ctx,
                )
            ],
        )
        new_mesh = Mesh(
            vertices=[Vertex(x=p[0], y=p[1], z=p[2]) for p in new_points],
            faces=list(new_faces),
            materials=dict(geom.materials),
            metadata=dict(geom.metadata),
        )
        remaining = self.detector.detect(new_mesh, ctx)

        result = self.make_result(
            stage_name=stage_name,
            before_count=before_faces,
            after_count=len(new_mesh.faces),
            iterations=int(diag.get("iterations", max_iters if changed_any else 0)),
            details={
                **diag,
                "changed": bool(changed_any),
                "vertices_before": before_points,
                "vertices_after": len(new_points),
                "remaining_intersections": len(remaining),
            },
            issues=issues,
            affected_ids=_affected(issues),
        )
        return new_mesh, result


class RepairPlcSingleSplitsRepair(BaseRepair):
    name: ClassVar[str] = "repair_plc_single_splits_iterative"
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = {IssueKind.INTERSECTION}

    def __init__(self, detector: Validator, max_iters: int | None = None) -> None:
        self.detector = detector
        self.max_iters = max_iters

    def apply(
        self,
        geom: Mesh,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Mesh, RepairResult]:
        self.ensure_accepts(geom)
        max_iters = self.max_iters or ctx.tolerances.max_plc_iters
        faces = list(geom.faces)
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        before = len(faces)

        result_tuple = repair_plc_single_splits_iterative(
            faces, points,
            room_center_from_mesh(geom),
            logger=ctx.logger,
            max_iters=max_iters,
            planarity_tol_m=ctx.tolerances.planarity_split,
            plc_report_supplier=lambda f, p: [
                i.payload
                for i in self.detector.detect(
                    Mesh(
                        vertices=[Vertex(x=pt[0], y=pt[1], z=pt[2]) for pt in p],
                        faces=list(f),
                        materials=dict(geom.materials),
                        metadata=dict(geom.metadata),
                    ),
                    ctx,
                )
            ],
        )
        # Defensive unpack: legacy helpers sometimes return shorter tuples.
        new_faces, new_points, changed_any, diag = faces, points, False, {}
        if isinstance(result_tuple, tuple):
            if len(result_tuple) >= 4:
                new_faces, new_points, changed_any, diag = result_tuple[:4]
            elif len(result_tuple) == 3:
                new_faces, new_points, changed_any = result_tuple[:3]
            elif len(result_tuple) == 2:
                new_faces, new_points = result_tuple[:2]

        new_mesh = Mesh(
            vertices=[Vertex(x=p[0], y=p[1], z=p[2]) for p in new_points],
            faces=list(new_faces),
            materials=dict(geom.materials),
            metadata=dict(geom.metadata),
        )
        remaining = self.detector.detect(new_mesh, ctx)

        return new_mesh, self.make_result(
            stage_name=stage_name,
            before_count=before,
            after_count=len(new_mesh.faces),
            details={
                **diag,
                "changed": bool(changed_any),
                "remaining_intersections": len(remaining),
                "planarity_tol_m": ctx.tolerances.planarity_split,
            },
            iterations=int(diag.get("iterations", max_iters if changed_any else 0)),
            issues=issues,
            affected_ids=_affected(issues),
        )


class RepairPlcByOffsetRepair(BaseRepair):
    name: ClassVar[str] = "repair_plc_by_offset_iterative"
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = {IssueKind.INTERSECTION}

    def __init__(self, detector: Validator, max_iters: int | None = None) -> None:
        self.detector = detector
        self.max_iters = max_iters

    def apply(
        self,
        geom: Mesh,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Mesh, RepairResult]:
        self.ensure_accepts(geom)
        max_iters = self.max_iters or ctx.tolerances.max_plc_iters
        faces = list(geom.faces)
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        before = len(faces)

        result_tuple = repair_plc_by_offset_iterative(
            faces, points,
            logger=ctx.logger,
            max_iters=max_iters,
            offset_m=ctx.tolerances.plc_offset,
            plc_report_supplier=lambda f, p: [
                i.payload
                for i in self.detector.detect(
                    Mesh(
                        vertices=[Vertex(x=pt[0], y=pt[1], z=pt[2]) for pt in p],
                        faces=list(f),
                        materials=dict(geom.materials),
                        metadata=dict(geom.metadata),
                    ),
                    ctx,
                )
            ],
        )
        # Defensive unpack: legacy helpers sometimes return shorter tuples.
        new_faces, new_points, changed_any, diag = faces, points, False, {}
        if isinstance(result_tuple, tuple):
            if len(result_tuple) >= 4:
                new_faces, new_points, changed_any, diag = result_tuple[:4]
            elif len(result_tuple) == 3:
                new_faces, new_points, changed_any = result_tuple[:3]
            elif len(result_tuple) == 2:
                new_faces, new_points = result_tuple[:2]

        new_mesh = Mesh(
            vertices=[Vertex(x=p[0], y=p[1], z=p[2]) for p in new_points],
            faces=list(new_faces),
            materials=dict(geom.materials),
            metadata=dict(geom.metadata),
        )
        remaining = self.detector.detect(new_mesh, ctx)

        return new_mesh, self.make_result(
            stage_name=stage_name,
            before_count=before,
            after_count=len(new_mesh.faces),
            details={
                **diag,
                "changed": bool(changed_any),
                "remaining_intersections": len(remaining),
                "offset_m": ctx.tolerances.plc_offset,
            },
            iterations=int(diag.get("iterations", max_iters if changed_any else 0)),
            issues=issues,
            affected_ids=_affected(issues),
        )


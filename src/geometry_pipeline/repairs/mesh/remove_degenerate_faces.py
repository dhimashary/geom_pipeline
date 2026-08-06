"""Drop faces classified as fatally zero-area (effectively zero area)."""

from __future__ import annotations

import logging
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Face, Mesh
from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.report import RepairResult
from geometry_pipeline.geometry_math.predicates import classify_face_degeneracy
from geometry_pipeline.repairs.base import BaseRepair


def remove_zero_area_faces_mesh(
    mesh: Mesh,
    *,
    fatal_area_tol: float = 1e-12,
    min_altitude_tol: float = 0.0,
    logger: logging.Logger = None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
    points = [(v.x, v.y, v.z) for v in mesh.vertices]
    kept_faces: list[Face] = []
    fatal_removed = 0

    for face in mesh.faces:
        vids = [int(i) for i in face.vertex_indices]
        status, _area2 = classify_face_degeneracy(
            vids,
            points,
            fatal_area_tol=fatal_area_tol,
            min_altitude_tol=min_altitude_tol,
        )
        if status == "fatal":
            fatal_removed += 1
            continue

        kept_faces.append(
            Face(
                vertex_indices=list(vids),
                group=getattr(face, "group", "default") or "default",
                material=getattr(face, "material", None),
            )
        )

    new_mesh = Mesh(
        vertices=list(mesh.vertices),
        faces=kept_faces,
        materials=dict(mesh.materials),
        metadata=dict(mesh.metadata),
    )

    return new_mesh, fatal_removed


class RemoveZeroAreaFaceRepair(BaseRepair):
    name: ClassVar[str] = "remove_zero_area_faces"
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = {IssueKind.ZERO_AREA_FACE}

    def apply(
        self,
        geom: Mesh,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Mesh, RepairResult]:
        self.ensure_accepts(geom)
        before = len(geom.faces)
        new_mesh, fatal_removed = remove_zero_area_faces_mesh(
            geom,
            fatal_area_tol=ctx.tolerances.degenerate_area,
            min_altitude_tol=ctx.tolerances.degenerate_min_altitude_m,
            logger=ctx.logger,
        )

        result = self.make_result(
            stage_name=stage_name,
            before_count=before,
            after_count=len(new_mesh.faces),
            details={
                "fatal_removed": fatal_removed,
                "fatal_area_tol": ctx.tolerances.degenerate_area,
                "min_altitude_tol": ctx.tolerances.degenerate_min_altitude_m,
            },
            issues=issues,
        )
        return new_mesh, result

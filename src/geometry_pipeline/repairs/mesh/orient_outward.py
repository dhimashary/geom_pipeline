"""Global outward-normal repair.

If more faces point toward the room centre than away, flip every face.
Cheap and catches the common "OBJ exporter inverted everything" case.
For local edge-by-edge consistency see ``orient_faces_consistently_by_adjacency``
in ``_intersection_repairs.py``.
"""

from __future__ import annotations

from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.report import RepairResult
from geometry_pipeline.repairs.base import BaseRepair
from geometry_pipeline.repairs.mesh._common import (
    flip_all_faces_if_majority_inward,
    room_center_from_mesh,
)


class FlipFacesIfMajorityInwardRepair(BaseRepair):
    name: ClassVar[str] = "flip_all_faces_if_majority_inward"
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = set()

    def apply(
        self,
        geom: Mesh,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Mesh, RepairResult]:
        self.ensure_accepts(geom)
        faces = list(geom.faces)
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        room_center = room_center_from_mesh(geom)
        flipped = flip_all_faces_if_majority_inward(
            faces,
            points,
            room_center,
            logger=ctx.logger,
        )
        new_mesh = Mesh(
            vertices=list(geom.vertices),
            faces=faces,
            materials=dict(geom.materials),
            metadata=dict(geom.metadata),
        )
        result = self.make_result(
            stage_name=stage_name,
            before_count=len(faces),
            after_count=len(faces),
            details={"flipped_all": bool(flipped), "room_center": list(room_center)},
            issues=issues,
        )
        return new_mesh, result

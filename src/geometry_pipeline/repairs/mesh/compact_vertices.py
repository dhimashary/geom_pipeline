"""Compact vertices: drop any not referenced by a face and remap indices."""
from __future__ import annotations

from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh, Face, Vertex
from geometry_pipeline.core.issues import IssueKind, Issue
from geometry_pipeline.core.report import RepairResult
from geometry_pipeline.repairs.base import BaseRepair
from geometry_pipeline.repairs.mesh._common import compact_vertices_and_remove_unused


class CompactVerticesRepair(BaseRepair):
    name: ClassVar[str] = "compact_vertices_and_remove_unused"
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
        # operate directly on Mesh: build points list and face vid lists
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        faces = [Face(vertex_indices=list(f.vertex_indices), group=getattr(f, "group", "default"), material=getattr(f, "material", None)) for f in geom.faces]

        new_faces, new_points, _changed, diag = compact_vertices_and_remove_unused(faces, points)

        # build new Mesh preserving materials/metadata
        new_mesh = Mesh(
            vertices=[Vertex(x=p[0], y=p[1], z=p[2]) for p in new_points],
            faces=[Face(vertex_indices=list(f.vertex_indices), group=getattr(f, "group", "default"), material=getattr(f, "material", None)) for f in new_faces],
            materials=dict(geom.materials),
            metadata=dict(geom.metadata),
        )

        result = self.make_result(
            stage_name=stage_name,
            before_count=len(points),
            after_count=len(new_points),
            details=dict(diag),
            issues=issues,
            affected_ids=[],
        )
        return new_mesh, result

"""Re-index vertices in a stable order to make output reproducible.

Sorting is keyed on rounded coordinates (8 decimals) so that runs on the
same input produce byte-identical Gmsh `.geo` output even when the
original OBJ file ordering changes.
"""

from __future__ import annotations

from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh, Vertex
from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.report import RepairResult
from geometry_pipeline.repairs.base import BaseRepair


def sort_vertices_deterministically(unique_vertices, faces):
    unique_vertices_sorted = sorted(
        enumerate(unique_vertices, start=1),
        key=lambda kv: (
            round(kv[1][0], 8),
            round(kv[1][1], 8),
            round(kv[1][2], 8),
        ),
    )
    index_map = {old: new for new, (old, _) in enumerate(unique_vertices_sorted, start=1)}
    unique_vertices = [v for _, v in unique_vertices_sorted]
    for face in faces:
        face.vertex_indices = [index_map[i] for i in face.vertex_indices]

    return unique_vertices, faces


class SortVerticesDeterministicallyRepair(BaseRepair):
    name: ClassVar[str] = "sort_vertices_deterministically"
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
        faces = geom.faces
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        new_points, new_faces = sort_vertices_deterministically(points, faces)
        # Build a new Mesh directly (Mesh-only path). `sort_vertices_deterministically`
        # updates `face.vertex_indices` in-place so we can reuse `new_faces`.
        new_mesh = Mesh(
            vertices=[Vertex(x=p[0], y=p[1], z=p[2]) for p in new_points],
            faces=list(new_faces),
            materials=dict(geom.materials),
            metadata=dict(geom.metadata),
        )
        result = self.make_result(
            stage_name=stage_name,
            before_count=len(points),
            after_count=len(new_points),
            details={},
            issues=issues,
            affected_ids=[],
        )
        return new_mesh, result

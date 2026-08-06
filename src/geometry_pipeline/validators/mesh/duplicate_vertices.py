"""Validator: detects vertices that coincide within `tolerances.vertex_merge`."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.validators.base import BaseValidator


def detect_duplicate_vertices(
    vertices: List[Tuple[float, float, float]],
    tol: float = 1e-2,
) -> List[Dict[str, Any]]:
    unique_vertices: List[Tuple[float, float, float]] = []
    orig_to_unique: Dict[int, int] = {}

    for i, v in enumerate(vertices, start=1):
        found = None

        for j, uv in enumerate(unique_vertices, start=1):
            if abs(uv[0] - v[0]) < tol and abs(uv[1] - v[1]) < tol and abs(uv[2] - v[2]) < tol:
                found = j
                break
        if found is None:
            unique_vertices.append(v)
            orig_to_unique[i] = len(unique_vertices)
        else:
            orig_to_unique[i] = found

    unique_to_originals: Dict[int, List[int]] = {}
    for orig, uniq in orig_to_unique.items():
        unique_to_originals.setdefault(uniq, []).append(orig)

    duplicate_reports: List[Dict[str, Any]] = []
    for _uniq_idx, origs in unique_to_originals.items():
        if len(origs) > 1:
            for orig in sorted(origs):
                coord = vertices[orig - 1]
                duplicate_reports.append(
                    {
                        "elements": {
                            "type": "vertex",
                            "points": [[coord[0], coord[1], coord[2]]],
                        },
                    }
                )

    return duplicate_reports


class DuplicateVerticesValidator(BaseValidator):
    name: ClassVar[str] = "duplicate_vertices"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.DUPLICATE_VERTEX

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        return detect_duplicate_vertices(points, tol=ctx.tolerances.vertex_merge)

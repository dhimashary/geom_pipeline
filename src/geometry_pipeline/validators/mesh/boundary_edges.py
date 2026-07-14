"""Validator: detects open boundary edges (edges adjacent to one face only)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.geometry_math.geometry_math import uedge
from geometry_pipeline.core.ir import Face
from geometry_pipeline.validators.base import BaseValidator


def detect_boundary_edges(
    faces: List[Face],
    unique_vertices: List[Tuple[float, float, float]],
) -> List[Dict[str, Any]]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for fid, f in enumerate(faces):
        vids = list(getattr(f, "vertex_indices", []))
        n = len(vids)
        if n < 2:
            continue

        for i in range(n):
            a = vids[i]
            b = vids[(i + 1) % n]
            edge_to_faces[uedge(a, b)].append(fid)

    boundary_edges: List[Dict[str, Any]] = []
    for edge, face_fids in edge_to_faces.items():
        if len(face_fids) == 1:
            a, b = edge
            coord_a = unique_vertices[a - 1]
            coord_b = unique_vertices[b - 1]
            boundary_edges.append({
                "elements": {
                    "type": "edge",
                    "points": [[coord_a[0], coord_a[1], coord_a[2]], [coord_b[0], coord_b[1], coord_b[2]]],
                },
            })

    return boundary_edges


class BoundaryEdgesValidator(BaseValidator):
    name: ClassVar[str] = "boundary_edges"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.BOUNDARY_EDGE

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        return detect_boundary_edges(list(geom.faces), points)

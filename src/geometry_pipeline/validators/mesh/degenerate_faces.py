"""Validator: detects faces with effectively zero area."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.geometry_math.predicates import classify_face_degeneracy
from geometry_pipeline.validators.base import BaseValidator


def detect_degenerate_faces_mesh(
    mesh: Mesh,
    *,
    fatal_area2_tol: float = 1e-16,
    min_altitude_tol: float = 0.0,
) -> List[Dict[str, Any]]:
    """Detect degenerate faces directly on the Mesh IR (zero-area faces).

    Returns legacy-style detector dicts so downstream plumbing is unchanged.
    """
    degenerate_faces: List[Dict[str, Any]] = []
    points = [(v.x, v.y, v.z) for v in mesh.vertices]

    for fi, face in enumerate(mesh.faces):
        vids = [int(i) for i in face.vertex_indices]
        status, _area2 = classify_face_degeneracy(
            vids,
            points,
            fatal_area_tol=fatal_area2_tol,
            min_altitude_tol=min_altitude_tol,
        )

        if status == "fatal":
            coordinates = [points[vid - 1] for vid in vids]
            degenerate_faces.append({
                "elements": {
                    "type": "face",
                    "points": [[coord[0], coord[1], coord[2]] for coord in coordinates],
                },
            })

    return degenerate_faces


def detect_degenerate_faces(
    faces,
    points: List[Tuple[float, float, float]],
    *,
    fatal_area2_tol: float = 1e-16,
) -> List[Dict[str, Any]]:
    """Compatibility wrapper for legacy callers that provide FaceRecord-style
    `faces` (with `.verts` and `.fid`) and a `points` list.
    """
    degenerate_faces: List[Dict[str, Any]] = []

    for f in faces:
        vids = [int(i) for i in getattr(f, "verts", getattr(f, "vertex_indices", []))]
        status, _area2 = classify_face_degeneracy(
            vids,
            points,
            fatal_area_tol=fatal_area2_tol,
        )

        if status == "fatal":
            coordinates = [points[vid - 1] for vid in vids]
            degenerate_faces.append({
                "elements": {
                    "type": "face",
                    "points": [[coord[0], coord[1], coord[2]] for coord in coordinates],
                },
            })

    return degenerate_faces


class DegenerateFacesValidator(BaseValidator):
    name: ClassVar[str] = "degenerate_faces"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.DEGENERATE_FACE

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        return detect_degenerate_faces_mesh(
            geom,
            fatal_area2_tol=ctx.tolerances.degenerate_area,
            min_altitude_tol=ctx.tolerances.degenerate_min_altitude_m,
        )

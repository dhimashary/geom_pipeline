"""Validator: detects faces whose vertices deviate from a best-fit plane."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Face, Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.geometry_math.predicates import (
    classify_face_degeneracy,
    classify_face_planarity_m,
)
from geometry_pipeline.validators.base import BaseValidator


def inspect_face_planarity_issues(
    faces: List[Face],
    unique_vertices: List[Tuple[float, float, float]],
    *,
    warn_planar_tol_m: float = 1e-4,
    fatal_planar_tol_m: float = 1e-3,
) -> List[Dict[str, Any]]:
    problematic_faces: List[Dict[str, Any]] = []

    for face in faces:
        degeneracy_status, _ = classify_face_degeneracy(face.vertex_indices, unique_vertices)
        if degeneracy_status == "fatal":
            continue

        status, max_dist_m, rms_dist_m = classify_face_planarity_m(
            face.vertex_indices,
            unique_vertices,
            warn_planar_tol_m=warn_planar_tol_m,
            fatal_planar_tol_m=fatal_planar_tol_m,
        )

        if status in ("warning", "fatal"):
            coordinates = [unique_vertices[vid - 1] for vid in face.vertex_indices]
            problematic_faces.append(
                {
                    "elements": {
                        "type": "face",
                        "points": [[coord[0], coord[1], coord[2]] for coord in coordinates],
                    },
                    "details": {
                        "worst_vertex_deviation": max_dist_m,
                        "overall_spread_deviation": rms_dist_m,
                    },
                }
            )

    return problematic_faces


def inspect_face_planarity_issues_mesh(
    mesh: Mesh,
    *,
    warn_planar_tol_m: float = 1e-4,
    fatal_planar_tol_m: float = 1e-3,
) -> List[Dict[str, Any]]:
    points = [(v.x, v.y, v.z) for v in mesh.vertices]
    problematic_faces: List[Dict[str, Any]] = []

    for face in mesh.faces:
        vids = [int(i) for i in face.vertex_indices]
        degeneracy_status, _ = classify_face_degeneracy(vids, points)
        if degeneracy_status == "fatal":
            continue

        status, max_dist_m, rms_dist_m = classify_face_planarity_m(
            vids,
            points,
            warn_planar_tol_m=warn_planar_tol_m,
            fatal_planar_tol_m=fatal_planar_tol_m,
        )

        if status in ("warning", "fatal"):
            coordinates = [points[vid - 1] for vid in vids]
            problematic_faces.append(
                {
                    "elements": {
                        "type": "face",
                        "points": [[coord[0], coord[1], coord[2]] for coord in coordinates],
                    },
                    "details": {
                        "worst_vertex_deviation": max_dist_m,
                        "overall_spread_deviation": rms_dist_m,
                    },
                }
            )

    return problematic_faces


class NonPlanarFacesValidator(BaseValidator):
    name: ClassVar[str] = "non_planar_faces"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.NON_PLANAR_FACE

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:  # type: ignore[override]
        return inspect_face_planarity_issues_mesh(
            geom,
            warn_planar_tol_m=ctx.tolerances.planarity_warn_m,
            fatal_planar_tol_m=ctx.tolerances.planarity_fatal_m,
        )

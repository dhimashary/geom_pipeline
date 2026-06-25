"""Validator: flags faces whose bounding-box max-dimension is below threshold.

Acoustic wave-based solvers degrade when the mesh contains faces much
smaller than the wavelength of interest; faces below ~10 cm are almost
always either modelling artefacts or unintended slivers. Default threshold
comes from `Tolerances.small_face_max_dim` (0.10 m).

NOTE: not wired into `wave_based_profile` yet — see tech-debt #11.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.geometry_math.geometry_math import polygon_area_3d
from geometry_pipeline.core.ir import Mesh, Face
from geometry_pipeline.core.issues import IssueKind, Severity
from geometry_pipeline.validators.base import BaseValidator


def detect_faces_with_area_below_threshold(
    faces: List[Face],
    vertices: List[Tuple[float, float, float]],
    *,
    area_threshold_m2: float = 0.001,
) -> List[Dict[str, Any]]:
    small_faces: List[Dict[str, Any]] = []

    for fid, f in enumerate(faces):
        vids = list(getattr(f, "vertex_indices", []))
        area = polygon_area_3d(vids, vertices)
        if area < area_threshold_m2:
            small_faces.append({
                "fid": getattr(f, "fid", fid),
                "verts": vids[:],
                "area_m2": area,
                "threshold_m2": area_threshold_m2,
            })

    return small_faces


class SmallFacesValidator(BaseValidator):
    name: ClassVar[str] = "small_faces"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.SMALL_FACE

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        threshold = ctx.tolerances.small_face_max_dim

        raw: list[dict] = []
        for f in geom.faces:
            vids = list(getattr(f, "vertex_indices", []))
            pts = [points[i - 1] for i in vids]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            max_dim = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            if max_dim < threshold:
                raw.append({
                    "fid": f.fid,
                    "max_dim": max_dim,
                    "threshold": threshold,
                    "elements": {
                        "type": "face",
                        "points": pts,
                    },
                })

        return raw

    def severity_of(self, payload: dict) -> Severity:
        return Severity.WARN

    def payload_of(self, payload: dict) -> dict:
        return dict(payload)

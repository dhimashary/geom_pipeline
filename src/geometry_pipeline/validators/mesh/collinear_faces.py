"""Validator: flags collinear / nearly-collinear faces.

A face is *collinear* when all its vertices lie on (or very close to) a single
straight line: the polygon has effectively collapsed to a line segment. We
measure this with the maximum perpendicular deviation of any vertex from the
line spanned by the two farthest-apart vertices of the face. When that
deviation is below ``Tolerances.collinear_face_max_deviation_m`` the face is
reported.

Unlike the sliver check (a dimensionless aspect ratio), this uses an absolute
distance tolerance in metres, so it flags faces that are geometrically a line
regardless of their length.

Detection-only (WARN); no repair is wired.
"""
from __future__ import annotations

from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind, Severity
from geometry_pipeline.geometry_math.geometry_math import cross, distance, norm, sub, unit
from geometry_pipeline.validators.base import BaseValidator


class CollinearFacesValidator(BaseValidator):
    name: ClassVar[str] = "collinear_faces"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.COLLINEAR_FACE

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        max_deviation = ctx.tolerances.collinear_face_max_deviation_m

        raw: list[dict] = []
        for fid, f in enumerate(geom.faces):
            vids = list(getattr(f, "vertex_indices", []))
            if len(vids) < 3:
                continue

            pts = [points[i - 1] for i in vids]

            # Longest chord = the two farthest-apart vertices; defines the line.
            span = 0.0
            i0 = j0 = 0
            n = len(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    d = distance(pts[i], pts[j])
                    if d > span:
                        span, i0, j0 = d, i, j

            if span <= 0.0:
                # All vertices coincident: a duplicate-vertex / degenerate case.
                continue

            base = pts[i0]
            axis = unit(sub(pts[j0], base))

            # Max perpendicular distance of any vertex from the chord's line.
            max_perp = 0.0
            for p in pts:
                perp = norm(cross(axis, sub(p, base)))
                if perp > max_perp:
                    max_perp = perp

            if max_perp < max_deviation:
                raw.append({
                    "fid": getattr(f, "fid", fid),
                    "max_deviation_m": max_perp,
                    "threshold_m": max_deviation,
                    "span_m": span,
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

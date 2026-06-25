"""Validator: detects open holes (leaks) by voxel flood-fill.

Unlike :mod:`possible_holes`, which traces boundary-edge loops on the surface
(fragile on folds, T-junctions and unwelded meshes), this validator asks the
volumetric question "can outside air reach the room interior?". It reports one
marker per distinct opening, located at the point where air squeezes through
the surface. See :mod:`geometry_pipeline.cavity_detection.leak_detector`.
"""
from __future__ import annotations

from typing import ClassVar

from geometry_pipeline.cavity_detection.leak_detector import detect_leaks_flood_fill
from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.validators.base import BaseValidator


class FloodFillHolesValidator(BaseValidator):
    name: ClassVar[str] = "flood_fill_holes"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.POSSIBLE_HOLE

    def __init__(self, *, pitch: float = 0.05, closing_iterations: int = 0) -> None:
        self._pitch = pitch
        self._closing_iterations = closing_iterations

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        leaks = detect_leaks_flood_fill(
            list(geom.faces),
            points,
            pitch=self._pitch,
            closing_iterations=self._closing_iterations,
        )
        out: list[dict] = []
        for leak in leaks:
            out.append(
                {
                    "elements": [
                        {"type": "vertex", "points": [leak["world"]]},
                        {"type": "vertex", "points": [leak["nearest_vertex_xyz"]]},
                    ],
                    "nearest_vertex_id": leak["nearest_vertex_id"],
                    "leak_world": leak["world"],
                    "severity": "high",
                }
            )
        return out

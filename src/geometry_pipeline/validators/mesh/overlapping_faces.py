"""Validator: detects pairs of coplanar faces whose surfaces overlap.

Two faces "overlap" when they lie in the same plane (parallel normals and a
zero plane-to-plane offset) and their projected 2D footprints share a region
of positive area. This is distinct from `IntersectionsValidator` (an edge of
one face piercing the *interior* of another) and from `DuplicateVerticesValidator`:
overlapping faces typically come from doubled-up surfaces, re-imported
geometry, or modelling mistakes, and they break volumetric meshing because the
solver cannot decide which face bounds the volume.

Detection strategy
------------------
1. Triangulate every face (CDT) so overlap area can be measured robustly even
   for concave / non-convex polygons.
2. Compute each face's unit normal, an axis-aligned bounding box, and a
   representative point on its plane.
3. For every face pair sharing an AABB overlap, test coplanarity
   (normals near-parallel *and* the offset between the two planes below
   `overlap_coplanar_dist_m`).
4. Project both faces onto the dominant axis plane and measure the area of the
   2D intersection of their triangulations. If it exceeds
   `overlap_min_area_m2` *and* survives erosion by half `overlap_sliver_width_m`
   (so long, micron-thin slivers from edge-adjacent faces are discarded) the
   pair is reported.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.geometry_math.geometry_math import (
    aabb_of_tri,
    aabb_overlap,
    cross,
    dot,
    newell_normal_from_points,
    norm,
    sub,
    unit,
)
from geometry_pipeline.geometry_math.triangulation import triangulate_face_cdt_shapely
from geometry_pipeline.validators.base import BaseValidator

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
    from shapely.ops import unary_union as _shapely_unary_union
except Exception:  # pragma: no cover - shapely is a hard dependency in practice
    _ShapelyPolygon = None  # type: ignore
    _shapely_unary_union = None  # type: ignore


Point3 = Tuple[float, float, float]


def _dropped_axis_from_normal(n: Point3) -> str:
    ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
    if az >= ax and az >= ay:
        return "z"
    if ay >= ax and ay >= az:
        return "y"
    return "x"


def _project(p: Point3, dropped_axis: str) -> Tuple[float, float]:
    if dropped_axis == "z":
        return (p[0], p[1])
    if dropped_axis == "y":
        return (p[0], p[2])
    return (p[1], p[2])


def _face_triangles_3d(
    vids: List[int],
    points: List[Point3],
) -> List[Tuple[Point3, Point3, Point3]]:
    """Return the face's triangulation as triples of 3D points."""
    if len(vids) < 3:
        return []
    if len(vids) == 3:
        tris = [vids[:]]
    else:
        tris = triangulate_face_cdt_shapely(vids, points)
    out: List[Tuple[Point3, Point3, Point3]] = []
    for tri in tris:
        if len(tri) != 3:
            continue
        a, b, c = tri
        out.append((points[a - 1], points[b - 1], points[c - 1]))
    return out


def _tris_to_polygon(
    tris: List[Tuple[Point3, Point3, Point3]],
    dropped_axis: str,
):
    """Build a single (possibly multi-) 2D polygon from a triangulation."""
    polys = []
    for A, B, C in tris:
        ring = [
            _project(A, dropped_axis),
            _project(B, dropped_axis),
            _project(C, dropped_axis),
        ]
        poly = _ShapelyPolygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if (not poly.is_empty) and poly.area > 0.0:
            polys.append(poly)
    if not polys:
        return None
    merged = _shapely_unary_union(polys)
    if merged.is_empty:
        return None
    return merged


def detect_overlapping_faces_mesh(
    mesh: Mesh,
    *,
    coplanar_dist_m: float = 1e-4,
    normal_cos_eps: float = 1e-6,
    min_overlap_area_m2: float = 1e-9,
    sliver_width_m: float = 1e-3,
    bbox_pad: float = 1e-9,
    max_reports: int = 2000,
) -> List[Dict[str, Any]]:
    """Detect coplanar overlapping face pairs on the Mesh IR.

    Returns legacy-style detector dicts so the shared cap/summary plumbing in
    `BaseValidator` can convert them to `Issue`s unchanged.
    """
    if _ShapelyPolygon is None or _shapely_unary_union is None:
        raise ImportError("Shapely is required for overlapping-face detection")

    points: List[Point3] = [(v.x, v.y, v.z) for v in mesh.vertices]

    # Pre-compute per-face geometry once.
    face_infos: List[Dict[str, Any]] = []
    for fi, face in enumerate(mesh.faces):
        vids = [int(i) for i in face.vertex_indices]
        if len(vids) < 3:
            continue
        nrm = newell_normal_from_points(vids, points)
        if norm(nrm) <= 0.0:
            continue  # degenerate face, handled by DegenerateFacesValidator
        unit_n = unit(nrm)
        tris = _face_triangles_3d(vids, points)
        if not tris:
            continue
        # AABB across all triangles of the face.
        bb = None
        for A, B, C in tris:
            tbb = aabb_of_tri(A, B, C)
            if bb is None:
                bb = tbb
            else:
                bb = (
                    min(bb[0], tbb[0]), min(bb[1], tbb[1]), min(bb[2], tbb[2]),
                    max(bb[3], tbb[3]), max(bb[4], tbb[4]), max(bb[5], tbb[5]),
                )
        face_infos.append({
            "fid": getattr(face, "fid", fi),
            "vids": vids,
            "normal": unit_n,
            "point": points[vids[0] - 1],
            "aabb": bb,
            "tris": tris,
        })

    reports: List[Dict[str, Any]] = []
    n_faces = len(face_infos)
    for i in range(n_faces):
        fa = face_infos[i]
        for j in range(i + 1, n_faces):
            fb = face_infos[j]

            if not aabb_overlap(fa["aabb"], fb["aabb"], pad=bbox_pad):
                continue

            # Coplanarity: normals parallel (cross magnitude ~ 0) ...
            n1 = fa["normal"]
            n2 = fb["normal"]
            if norm(cross(n1, n2)) > normal_cos_eps:
                continue
            # ... and both planes coincide (offset along the shared normal ~ 0).
            offset = abs(dot(sub(fb["point"], fa["point"]), n1))
            if offset > coplanar_dist_m:
                continue

            dropped_axis = _dropped_axis_from_normal(n1)
            poly_a = _tris_to_polygon(fa["tris"], dropped_axis)
            poly_b = _tris_to_polygon(fb["tris"], dropped_axis)
            if poly_a is None or poly_b is None:
                continue

            inter = poly_a.intersection(poly_b)
            overlap_area = float(getattr(inter, "area", 0.0))
            if overlap_area <= min_overlap_area_m2:
                continue

            # Reject thin slivers: two coplanar faces that share an edge can
            # produce a long, micron-thin intersection when their common
            # vertices differ by coordinate noise. Eroding the intersection by
            # half the sliver width collapses such slivers to nothing while a
            # genuine area overlap survives.
            if sliver_width_m > 0.0 and not inter.is_empty:
                core = inter.buffer(-0.5 * sliver_width_m)
                if core.is_empty or float(getattr(core, "area", 0.0)) <= min_overlap_area_m2:
                    continue

            coords_a = [points[vid - 1] for vid in fa["vids"]]
            coords_b = [points[vid - 1] for vid in fb["vids"]]
            reports.append({
                "elements": [
                    {
                        "type": "face",
                        "points": [[c[0], c[1], c[2]] for c in coords_a],
                    },
                    {
                        "type": "face",
                        "points": [[c[0], c[1], c[2]] for c in coords_b],
                    },
                ],
                "details": {
                    "face_a_fid": fa["fid"],
                    "face_b_fid": fb["fid"],
                    "overlap_area_m2": overlap_area,
                    "plane_offset_m": offset,
                },
            })
            if len(reports) >= max_reports:
                return reports

    return reports


class OverlappingFacesValidator(BaseValidator):
    name: ClassVar[str] = "overlapping_faces"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.OVERLAPPING_FACE

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        return detect_overlapping_faces_mesh(
            geom,
            coplanar_dist_m=ctx.tolerances.overlap_coplanar_dist_m,
            normal_cos_eps=ctx.tolerances.overlap_normal_cos_eps,
            min_overlap_area_m2=ctx.tolerances.overlap_min_area_m2,
            sliver_width_m=ctx.tolerances.overlap_sliver_width_m,
            bbox_pad=ctx.tolerances.bbox_pad,
            max_reports=ctx.tolerances.max_reports,
        )

    def payload_of(self, payload: dict) -> dict:
        return payload

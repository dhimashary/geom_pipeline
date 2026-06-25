"""Validator: detects segment-facet intersections (CDT-based)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.ir import Mesh, Face
from geometry_pipeline.core.issues import IssueKind, Severity
from geometry_pipeline.geometry_math.geometry_math import (
    aabb_of_seg,
    aabb_of_tri,
    aabb_overlap,
    segment_intersects_triangle,
    sub,
    vadd,
    vmul,
)
from geometry_pipeline.geometry_math.predicates import classify_face_planarity_m
from geometry_pipeline.geometry_math.triangulation import triangulate_face_cdt_shapely
from geometry_pipeline.validators.base import BaseValidator


def _classify_segment_triangle_hit(t, u, v, *, t_eps=1e-9, bary_eps=1e-9):
    w = 1.0 - u - v

    at_start = abs(t) <= t_eps
    at_end = abs(t - 1.0) <= t_eps
    at_endpoint = at_start or at_end

    vals = [u, v, w]
    near_zero = [abs(x) <= bary_eps for x in vals]
    near_one = [abs(x - 1.0) <= bary_eps for x in vals]

    n_zero = sum(near_zero)
    n_one = sum(near_one)

    if n_one == 1 and n_zero >= 2:
        tri_loc = "vertex"
    elif n_zero == 1 and n_one == 0:
        tri_loc = "edge"
    elif all((bary_eps < x < 1.0 - bary_eps) for x in vals):
        tri_loc = "interior"
    else:
        tri_loc = "unknown"

    if at_endpoint:
        if tri_loc == "vertex":
            return "endpoint_vertex_touch"
        if tri_loc == "edge":
            return "endpoint_edge_touch"
        if tri_loc == "interior":
            return "endpoint_face_interior_touch"
        return "unknown"

    if tri_loc == "vertex":
        return "segment_vertex_touch"
    if tri_loc == "edge":
        return "segment_edge_intersection"
    if tri_loc == "interior":
        return "segment_face_interior_intersection"
    return "unknown"


def detect_segment_facet_intersections_cdt(
    faces: List[Face],
    points: List[Tuple[float, float, float]],
    *,
    warn_planar_tol_m=1e-4,
    fatal_planar_tol_m=1e-3,
    eps=1e-10,
    bbox_pad=1e-9,
    max_reports=200,
    skip_warped_faces=True,
) -> List[Dict[str, Any]]:
    tri_list = []

    for f in faces:
        poly = f.vertex_indices
        if len(poly) < 3:
            continue

        planar_flag = None
        if len(poly) > 3:
            pstat, _pmax_m, _prms_m = classify_face_planarity_m(
                poly,
                points,
                warn_planar_tol_m=warn_planar_tol_m,
                fatal_planar_tol_m=fatal_planar_tol_m,
            )
            planar_flag = pstat
            if skip_warped_faces and pstat == "fatal":
                continue

        if len(poly) == 3:
            tris = [poly[:]]
        else:
            tris = triangulate_face_cdt_shapely(poly, points)

        if not tris:
            continue

        for tri in tris:
            if len(tri) != 3:
                continue
            a, b, c = tri
            A, B, C = points[a - 1], points[b - 1], points[c - 1]
            tri_list.append({
                "fid": f.fid,
                "tri": (a, b, c),
                "aabb": aabb_of_tri(A, B, C),
                "planar_flag": planar_flag,
            })

    edge_to_faces = defaultdict(set)
    edge_set = set()
    for f in faces:
        poly = f.vertex_indices
        n = len(poly)
        if n < 2:
            continue
        for i in range(n):
            u = poly[i]
            v = poly[(i + 1) % n]
            e = (u, v) if u < v else (v, u)
            edge_set.add(e)
            edge_to_faces[e].add(f.fid)

    tri_vset = [set(t["tri"]) for t in tri_list]

    reports = []
    for (u, v) in edge_set:
        P0 = points[u - 1]
        P1 = points[v - 1]
        seg_bb = aabb_of_seg(P0, P1)

        for ti, tinfo in enumerate(tri_list):
            if not aabb_overlap(seg_bb, tinfo["aabb"], pad=bbox_pad):
                continue

            a, b, c = tinfo["tri"]

            if u in tri_vset[ti] or v in tri_vset[ti]:
                continue

            A, B, C = points[a - 1], points[b - 1], points[c - 1]
            hit, t, uu, vv = segment_intersects_triangle(P0, P1, A, B, C, eps=eps)
            if not hit:
                continue

            hit_type = _classify_segment_triangle_hit(t, uu, vv, t_eps=1e-9, bary_eps=1e-9)
            I = vadd(P0, vmul(sub(P1, P0), t))

            original_face = next(f for f in faces if f.fid == tinfo["fid"])
            facet_fid_coordinates = [points[vid - 1] for vid in original_face.vertex_indices]

            reports.append({
                "edge": (u, v),
                "edge_coordinates": [P0, P1],
                "edge_fids": sorted(edge_to_faces[(u, v) if u < v else (v, u)]),
                "facet_fid": tinfo["fid"],
                "facet_fid_coordinates": facet_fid_coordinates,
                "facet_tri": (a, b, c),
                "point": I,
                "t_param": float(t),
                "bary_u": float(uu),
                "bary_v": float(vv),
                "bary_w": float(1.0 - uu - vv),
                "hit_type": hit_type,
                "facet_planarity_flag": tinfo["planar_flag"],
            })
            if len(reports) >= max_reports:
                return reports

    return reports


def detect_segment_facet_intersections_cdt_mesh(
    mesh: Mesh,
    *,
    warn_planar_tol_m=1e-4,
    fatal_planar_tol_m=1e-3,
    eps=1e-10,
    bbox_pad=1e-9,
    max_reports=200,
    skip_warped_faces=True,
) -> List[Dict[str, Any]]:
    points = [(v.x, v.y, v.z) for v in mesh.vertices]
    tri_list = []

    for fi, face in enumerate(mesh.faces):
        vids = [int(i) for i in face.vertex_indices]
        poly = vids
        if len(poly) < 3:
            continue

        planar_flag = None
        if len(poly) > 3:
            pstat, _pmax_m, _prms_m = classify_face_planarity_m(
                poly,
                points,
                warn_planar_tol_m=warn_planar_tol_m,
                fatal_planar_tol_m=fatal_planar_tol_m,
            )
            planar_flag = pstat
            if skip_warped_faces and pstat == "fatal":
                continue

        if len(poly) == 3:
            tris = [poly[:]]
        else:
            tris = triangulate_face_cdt_shapely(poly, points)

        if not tris:
            continue

        for tri in tris:
            if len(tri) != 3:
                continue
            a, b, c = tri
            A, B, C = points[a - 1], points[b - 1], points[c - 1]
            tri_list.append({
                "fid": getattr(face, "fid", fi),
                "tri": (a, b, c),
                "aabb": aabb_of_tri(A, B, C),
                "planar_flag": planar_flag,
            })

    edge_to_faces = defaultdict(set)
    edge_set = set()
    for fi, face in enumerate(mesh.faces):
        poly = [int(i) for i in face.vertex_indices]
        n = len(poly)
        if n < 2:
            continue
        for i in range(n):
            u = poly[i]
            v = poly[(i + 1) % n]
            e = (u, v) if u < v else (v, u)
            edge_set.add(e)
            edge_to_faces[e].add(getattr(face, "fid", fi))

    tri_vset = [set(t["tri"]) for t in tri_list]

    reports = []
    for (u, v) in edge_set:
        P0 = points[u - 1]
        P1 = points[v - 1]
        seg_bb = aabb_of_seg(P0, P1)

        for ti, tinfo in enumerate(tri_list):
            if not aabb_overlap(seg_bb, tinfo["aabb"], pad=bbox_pad):
                continue

            a, b, c = tinfo["tri"]

            if u in tri_vset[ti] or v in tri_vset[ti]:
                continue

            A, B, C = points[a - 1], points[b - 1], points[c - 1]
            hit, t, uu, vv = segment_intersects_triangle(P0, P1, A, B, C, eps=eps)
            if not hit:
                continue

            hit_type = _classify_segment_triangle_hit(t, uu, vv, t_eps=1e-9, bary_eps=1e-9)
            I = vadd(P0, vmul(sub(P1, P0), t))

            original_face = next((f for f in mesh.faces if getattr(f, "fid", None) == tinfo["fid"]), None)
            facet_fid_coordinates = [points[vid - 1] for vid in (original_face.vertex_indices if original_face is not None else [])]

            reports.append({
                "edge": (u, v),
                "edge_coordinates": [P0, P1],
                "edge_fids": sorted(edge_to_faces[(u, v) if u < v else (v, u)]),
                "facet_fid": tinfo["fid"],
                "facet_fid_coordinates": facet_fid_coordinates,
                "facet_tri": (a, b, c),
                "point": I,
                "t_param": float(t),
                "bary_u": float(uu),
                "bary_v": float(vv),
                "bary_w": float(1.0 - uu - vv),
                "hit_type": hit_type,
                "facet_planarity_flag": tinfo["planar_flag"],
            })
            if len(reports) >= max_reports:
                return reports

    return reports


class IntersectionsValidator(BaseValidator):
    name: ClassVar[str] = "intersections"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.INTERSECTION

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        return detect_segment_facet_intersections_cdt_mesh(
            geom,
            warn_planar_tol_m=ctx.tolerances.planarity_warn_m,
            fatal_planar_tol_m=ctx.tolerances.planarity_fatal_m,
            eps=ctx.tolerances.intersection_eps,
            bbox_pad=ctx.tolerances.bbox_pad,
            max_reports=ctx.tolerances.max_reports,
        )

    def severity_of(self, payload: dict) -> Severity:
        return Severity.FATAL

    def payload_of(self, payload: dict) -> dict:
        # Detector returns a `hit_type` discriminator; surface it as
        # payload["sub_kind"] so downstream code can route on it without
        # widening IssueKind.
        out = dict(payload)
        out["sub_kind"] = payload.get("hit_type", "interior")
        # Build the frontend `elements` shape here so the report translator
        # stays generic (no per-kind branches in reporting/frontend_schema).
        out["elements"] = [
            {"type": "edge",   "points": out.get("edge_coordinates", [])},
            {"type": "face",   "points": out.get("facet_fid_coordinates", [])},
            {"type": "vertex", "points": [out.get("point", [0, 0, 0])]},
        ]
        return out

"""Validator: detects segment-facet intersections (CDT-based)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, ClassVar, Dict, List

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.geometry_math.geometry_math import (
    aabb_of_seg,
    aabb_of_tri,
    aabb_overlap,
    dot,
    newell_normal_from_points,
    point_in_polygon_2d,
    project_point_by_dropped_axis,
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


def _edge_is_face_side(u, v, face_ids):
    """True if (u, v) is a consecutive boundary side of the face loop."""
    n = len(face_ids)
    for i in range(n):
        if {face_ids[i], face_ids[(i + 1) % n]} == {u, v}:
            return True
    return False


def _coplanar_segment_face_overlap(
    p0,
    p1,
    face_ids,
    points,
    *,
    coplanar_dist_tol,
    min_overlap_len,
    tol_2d=1e-9,
):
    """Detect an edge lying in a face's plane and overlapping its interior.

    Moller-Trumbore rejects segments parallel to the triangle plane (det ~ 0),
    so coplanar edge-on-face overlaps are invisible to it. This routine covers
    that gap: it gates on coplanarity, projects the face polygon and the
    segment to 2D by dropping the dominant normal axis, and measures the length
    of the segment lying strictly inside the polygon.

    Returns None when there is no interior overlap, otherwise a dict describing
    the parametric enter/exit along ``p0 -> p1`` and the matching 3D points.
    """
    nrm = newell_normal_from_points(face_ids, points)
    nn = math.sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2])
    if nn <= 1e-18:
        return None
    normal = (nrm[0] / nn, nrm[1] / nn, nrm[2] / nn)
    plane_point = points[face_ids[0] - 1]

    # Coplanarity gate: both endpoints must lie in the face plane.
    d0 = dot(sub(p0, plane_point), normal)
    d1 = dot(sub(p1, plane_point), normal)
    if abs(d0) > coplanar_dist_tol or abs(d1) > coplanar_dist_tol:
        return None

    # Drop the dominant normal axis for a stable 2D projection.
    ax, ay, az = abs(normal[0]), abs(normal[1]), abs(normal[2])
    if az >= ax and az >= ay:
        dropped_axis = "z"
    elif ay >= ax and ay >= az:
        dropped_axis = "y"
    else:
        dropped_axis = "x"

    poly2d = [project_point_by_dropped_axis(points[pid - 1], dropped_axis) for pid in face_ids]
    s0 = project_point_by_dropped_axis(p0, dropped_axis)
    s1 = project_point_by_dropped_axis(p1, dropped_axis)

    dx = s1[0] - s0[0]
    dy = s1[1] - s0[1]
    seg_len2d = math.sqrt(dx * dx + dy * dy)
    if seg_len2d <= tol_2d:
        return None

    # Parametric cut points along the segment: endpoints plus every crossing
    # with a polygon edge.
    cuts = [0.0, 1.0]
    n = len(poly2d)
    for i in range(n):
        a2 = poly2d[i]
        b2 = poly2d[(i + 1) % n]
        ex = b2[0] - a2[0]
        ey = b2[1] - a2[1]
        den = dx * ey - dy * ex
        if abs(den) <= tol_2d:
            continue  # segment parallel to this polygon edge
        apx = a2[0] - s0[0]
        apy = a2[1] - s0[1]
        s = (apx * ey - apy * ex) / den  # param along the segment
        t = (apx * dy - apy * dx) / den  # param along the polygon edge
        if -tol_2d <= s <= 1.0 + tol_2d and -tol_2d <= t <= 1.0 + tol_2d:
            cuts.append(min(1.0, max(0.0, s)))

    cuts = sorted({round(c, 12) for c in cuts})

    # Sum the sub-intervals whose midpoint falls strictly inside the polygon.
    # Boundary-coincident intervals classify as "boundary" and are excluded, so
    # an edge merely running along a polygon boundary is not reported here.
    inside_len = 0.0
    t_enter = None
    t_exit = None
    for i in range(len(cuts) - 1):
        sa = cuts[i]
        sb = cuts[i + 1]
        if sb - sa <= tol_2d:
            continue
        sm = 0.5 * (sa + sb)
        mid = (s0[0] + sm * dx, s0[1] + sm * dy)
        if point_in_polygon_2d(poly2d, mid, tol=tol_2d) != "inside":
            continue
        inside_len += (sb - sa) * seg_len2d
        if t_enter is None:
            t_enter = sa
        t_exit = sb

    if t_enter is None or inside_len < min_overlap_len:
        return None

    seg3 = sub(p1, p0)
    return {
        "t_enter": float(t_enter),
        "t_exit": float(t_exit),
        "overlap_length": float(inside_len),
        "enter": vadd(p0, vmul(seg3, t_enter)),
        "exit": vadd(p0, vmul(seg3, t_exit)),
        "mid": vadd(p0, vmul(seg3, 0.5 * (t_enter + t_exit))),
    }


def detect_segment_facet_intersections_cdt_mesh(
    mesh: Mesh,
    *,
    warn_planar_tol_m=1e-4,
    fatal_planar_tol_m=1e-3,
    eps=1e-10,
    bbox_pad=1e-9,
    max_reports=200,
    skip_warped_faces=True,
    coplanar_dist_tol=1e-4,
    min_coplanar_overlap_len=1e-6,
) -> List[Dict[str, Any]]:
    points = [(v.x, v.y, v.z) for v in mesh.vertices]
    tri_list: list[dict[str, Any]] = []
    face_polys: list[dict[str, Any]] = []

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

        pts3 = [points[pid - 1] for pid in poly]
        face_polys.append(
            {
                "fid": getattr(face, "fid", fi),
                "vids": poly,
                "vset": set(poly),
                "aabb": (
                    min(p[0] for p in pts3),
                    min(p[1] for p in pts3),
                    min(p[2] for p in pts3),
                    max(p[0] for p in pts3),
                    max(p[1] for p in pts3),
                    max(p[2] for p in pts3),
                ),
                "planar_flag": planar_flag,
            }
        )

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
            tri_list.append(
                {
                    "fid": getattr(face, "fid", fi),
                    "face_vids": poly,
                    "tri": (a, b, c),
                    "aabb": aabb_of_tri(A, B, C),
                    "planar_flag": planar_flag,
                }
            )

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
    for u, v in edge_set:
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

            facet_fid_coordinates = [points[vid - 1] for vid in tinfo["face_vids"]]

            reports.append(
                {
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
                }
            )
            if len(reports) >= max_reports:
                return reports

    # Second pass: coplanar edge-on-face overlaps. Moller-Trumbore rejects
    # segments parallel to a face plane (det ~ 0), so an edge lying flat on a
    # face never appears above. Detect those interior overlaps here at polygon
    # granularity (one report per overlapping edge/face pair).
    for u, v in edge_set:
        P0 = points[u - 1]
        P1 = points[v - 1]
        seg_bb = aabb_of_seg(P0, P1)

        for finfo in face_polys:
            if _edge_is_face_side(u, v, finfo["vids"]):
                continue  # the edge is one of this face's own boundary sides
            if not aabb_overlap(seg_bb, finfo["aabb"], pad=bbox_pad):
                continue

            overlap = _coplanar_segment_face_overlap(
                P0,
                P1,
                finfo["vids"],
                points,
                coplanar_dist_tol=coplanar_dist_tol,
                min_overlap_len=min_coplanar_overlap_len,
                tol_2d=bbox_pad,
            )
            if overlap is None:
                continue

            reports.append(
                {
                    "edge": (u, v),
                    "edge_coordinates": [P0, P1],
                    "edge_fids": sorted(edge_to_faces[(u, v) if u < v else (v, u)]),
                    "facet_fid": finfo["fid"],
                    "facet_fid_coordinates": [points[vid - 1] for vid in finfo["vids"]],
                    "point": overlap["mid"],
                    "overlap_coordinates": [overlap["enter"], overlap["exit"]],
                    "overlap_length": overlap["overlap_length"],
                    "t_param": overlap["t_enter"],
                    "hit_type": "coplanar_segment_face_overlap",
                    "facet_planarity_flag": finfo["planar_flag"],
                }
            )
            if len(reports) >= max_reports:
                return reports

    return reports


class IntersectionsValidator(BaseValidator):
    name: ClassVar[str] = "intersections"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.INTERSECTION

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:  # type: ignore[override]
        return detect_segment_facet_intersections_cdt_mesh(
            geom,
            warn_planar_tol_m=ctx.tolerances.planarity_warn_m,
            fatal_planar_tol_m=ctx.tolerances.planarity_fatal_m,
            eps=ctx.tolerances.intersection_eps,
            bbox_pad=ctx.tolerances.bbox_pad,
            max_reports=ctx.tolerances.max_reports,
            coplanar_dist_tol=ctx.tolerances.overlap_coplanar_dist_m,
            min_coplanar_overlap_len=ctx.tolerances.planarity_split,
        )

    def payload_of(self, payload: dict) -> dict:
        # Detector returns a `hit_type` discriminator; surface it as
        # payload["sub_kind"] so downstream code can route on it without
        # widening IssueKind.
        out = dict(payload)
        out["sub_kind"] = payload.get("hit_type", "interior")
        # Build the frontend `elements` shape here so the report translator
        # stays generic (no per-kind branches in reporting/frontend_schema).
        out["elements"] = [
            {"type": "edge", "points": out.get("edge_coordinates", [])},
            {"type": "face", "points": out.get("facet_fid_coordinates", [])},
            {"type": "vertex", "points": [out.get("point", [0, 0, 0])]},
        ]
        return out

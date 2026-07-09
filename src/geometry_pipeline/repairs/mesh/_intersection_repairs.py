"""Legacy repair implementations moved from src.services.geometry_repair_service.

This module contains a small, focused subset needed by the high-level
repair classes in `app.geometry.repairs` so the repairs package can be
independent from `app.services` during migration.
"""
from __future__ import annotations

from collections import defaultdict, deque
import logging
import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from geometry_pipeline.core.ir import Face
from geometry_pipeline.geometry_math.geometry_math import (
    area2,
    cross,
    distance,
    dot,
    newell_normal_from_points,
    point_on_segment_2d,
    segments_intersect_2d,
    sub,
    tri_area2,
    unit,
    uedge,
    project_point_by_dropped_axis,
    project_face_to_2d,
    project_vid_to_2d,
    point_in_polygon_2d,
    fit_chain_direction_2d,
    line_segment_intersection_signed,
    polygon_area2_newell,
)
from geometry_pipeline.geometry_math.triangulation import triangulate_face_cdt_shapely
from geometry_pipeline.geometry_math.predicates import (
    classify_face_degeneracy,
    classify_face_planarity_m,
    planarity_deviation_m,
)
from geometry_pipeline.io.importers.obj import clean_face_loop

logger = logging.getLogger(__name__)

from geometry_pipeline.repairs.mesh._common import (
    compact_vertices_and_remove_unused,
    get_or_create_vertex,
    flip_all_faces_if_majority_inward as _flip_all_faces_if_majority_inward,
)


# Mesh-only helpers: require `Face` from `app.geometry.ir` with
# `vertex_indices`. Legacy `FaceRecord` is not accepted here.
def _face_vids(face) -> List[int]:
    if not hasattr(face, "vertex_indices"):
        raise TypeError(
            "_face_vids: legacy FaceRecord not accepted; migrate callers to use app.geometry.ir.Face"
        )
    return list(face.vertex_indices)


def _set_face_vids(face, vids: List[int]):
    if not hasattr(face, "vertex_indices"):
        raise TypeError(
            "_set_face_vids: legacy FaceRecord not accepted; migrate callers to use app.geometry.ir.Face"
        )
    face.vertex_indices = vids


def _face_fid(face, default=None, idx=None):
    # Mesh `Face` has no `fid`; fall back to provided `default` or `idx`.
    if hasattr(face, "fid"):
        return getattr(face, "fid")
    return default if default is not None else idx


def _reverse_face_vids(face):
    if not hasattr(face, "vertex_indices"):
        raise TypeError(
            "_reverse_face_vids: legacy FaceRecord not accepted; migrate callers to use app.geometry.ir.Face"
        )
    face.vertex_indices.reverse()

# Wrappers for larger PLC repair routines (delegate to legacy service for now)
def trim_segment_face_intersections_iterative(
    faces: List[Face],
    points: List[Tuple[float, float, float]],
    room_center: Tuple[float, float, float],
    *,
    plc_report_supplier,
    max_iters: int = 20,
    tol: float = 1e-9,
    logger=None,
) -> Tuple[List[Face], List[Tuple[float, float, float]], bool, Dict[str, Any]]:
    changed_any = False
    actions = []

    for it in range(1, max_iters + 1):
        plc_hits = plc_report_supplier(faces, points)

        seg_face_hits = [
            r
            for r in plc_hits
            if r.get("hit_type") in ("segment_face_interior_intersection", "segment_edge_intersection")
        ]
        if not seg_face_hits:
            diag = {
                "status": "ok" if changed_any else "no_supported_hits",
                "iterations": it - 1,
                "applied_repairs": len(actions),
                "actions": actions,
            }
            return faces, points, changed_any, diag

        target = seg_face_hits[0]
        faces2, points2, changed, trim_diag = trim_component_from_segment_face_intersection_report(
            faces,
            points,
            target,
            room_center,
            tol=tol,
            logger=logger,
        )
        if not changed:
            diag = {
                "status": "stalled",
                "iterations": it,
                "applied_repairs": len(actions),
                "last_trim": trim_diag,
                "actions": actions,
            }
            return faces, points, changed_any, diag

        faces2, points2, compact_changed, compact_diag = compact_vertices_and_remove_unused(faces2, points2)
        action = {
            "iteration": it,
            "target_hit": {
                "edge": target.get("edge"),
                "edge_fids": target.get("edge_fids"),
                "facet_fid": target.get("facet_fid"),
                "point": target.get("point"),
            },
            "trim_diag": trim_diag,
            "compact_diag": compact_diag,
        }
        actions.append(action)
        faces, points = faces2, points2
        changed_any = True

        if logger is not None:
            logger.info(
                "[PLC TRIM LOOP] iter=%d trim_status=%s removed_unused_vertices=%d",
                it,
                trim_diag.get("status"),
                compact_diag.get("removed_unused_vertices", 0),
            )

    diag = {
        "status": "max_iters_reached",
        "iterations": max_iters,
        "applied_repairs": len(actions),
        "actions": actions,
    }
    return faces, points, changed_any, diag


# ------ Trimming Helper

def _clip_face_loop_against_plane(
    face_loop: List[int],
    points: List[Tuple[float, float, float]],
    plane_point: Tuple[float, float, float],
    plane_normal: Tuple[float, float, float],
    keep_sign: float,
    *,
    tol: float = 1e-9,
) -> List[int]:
    if len(face_loop) < 3:
        return []

    def is_inside(sd: float) -> bool:
        return sd * keep_sign >= -tol

    out: List[int] = []
    n = len(face_loop)

    for i in range(n):
        a_vid = face_loop[i]
        b_vid = face_loop[(i + 1) % n]
        A = points[a_vid - 1]
        B = points[b_vid - 1]

        da = _signed_distance_to_plane(A, plane_point, plane_normal)
        db = _signed_distance_to_plane(B, plane_point, plane_normal)
        a_inside = is_inside(da)
        b_inside = is_inside(db)

        if a_inside and b_inside:
            if not out or out[-1] != b_vid:
                out.append(b_vid)
        elif a_inside and not b_inside:
            denom = da - db
            if abs(denom) > tol:
                t = da / denom
                X = (
                    A[0] + t * (B[0] - A[0]),
                    A[1] + t * (B[1] - A[1]),
                    A[2] + t * (B[2] - A[2]),
                )
                x_vid = get_or_create_vertex(points, X, tol)
                if not out or out[-1] != x_vid:
                    out.append(x_vid)
        elif (not a_inside) and b_inside:
            denom = da - db
            if abs(denom) > tol:
                t = da / denom
                X = (
                    A[0] + t * (B[0] - A[0]),
                    A[1] + t * (B[1] - A[1]),
                    A[2] + t * (B[2] - A[2]),
                )
                x_vid = get_or_create_vertex(points, X, tol)
                if not out or out[-1] != x_vid:
                    out.append(x_vid)
            if not out or out[-1] != b_vid:
                out.append(b_vid)

    out = clean_face_loop(out)
    if len(out) >= 2 and out[0] == out[-1]:
        out = out[:-1]
    return out


def _plane_from_face(face: Face, points: List[Tuple[float, float, float]]):
    vids = _face_vids(face)
    if len(vids) < 3:
        return None

    nrm = newell_normal_from_points(vids, points)
    nn = math.sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2])
    if nn <= 1e-18:
        return None

    plane_point = points[vids[0] - 1]
    plane_normal = (nrm[0] / nn, nrm[1] / nn, nrm[2] / nn)
    return plane_point, plane_normal


def _signed_distance_to_plane(
    p: Tuple[float, float, float],
    plane_point: Tuple[float, float, float],
    plane_normal: Tuple[float, float, float],
) -> float:
    return (
        (p[0] - plane_point[0]) * plane_normal[0]
        + (p[1] - plane_point[1]) * plane_normal[1]
        + (p[2] - plane_point[2]) * plane_normal[2]
    )


def collect_face_component_from_seed_faces(
    faces: List[Face],
    seed_face_fids: List[int],
    *,
    excluded_face_fids: List[int] | None = None,
) -> List[int]:
    excluded = set(excluded_face_fids or [])
    valid_fids = { _face_fid(f, idx=i) for i,f in enumerate(faces) }
    seeds = [fid for fid in seed_face_fids if fid in valid_fids and fid not in excluded]
    if not seeds:
        return []

    edge_to_fids = _build_edge_face_adjacency(faces)
    face_adj: Dict[int, set] = defaultdict(set)
    for fids in edge_to_fids.values():
        for a in fids:
            for b in fids:
                if a != b:
                    face_adj[a].add(b)

    component = set()
    q = deque(seeds)
    while q:
        fid = q.popleft()
        if fid in component or fid in excluded:
            continue
        component.add(fid)
        for nbr in face_adj.get(fid, []):
            if nbr not in component and nbr not in excluded:
                q.append(nbr)

    return sorted(component)


def _build_edge_face_adjacency(faces: List[Face]) -> Dict[Tuple[int, int], List[int]]:
    edge_to_fids: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, face in enumerate(faces):
        vids = _face_vids(face)
        n = len(vids)
        fid = _face_fid(face, idx=i)
        for j in range(n):
            u = vids[j]
            v = vids[(j + 1) % n]
            edge_to_fids[uedge(u, v)].append(fid)
    return edge_to_fids


# use shared `get_or_create_vertex` from repairs._common

# ------ End of Trimming Helper


# ------ Repair Segment-Face Intersection 


def flip_all_faces_if_majority_inward(
    faces: List[Face],
    unique_vertices: List[Tuple[float, float, float]],
    room_center: Tuple[float, float, float],
    logger=None,
) -> bool:
    # Re-exported from the shared helper so existing call sites in this module
    # keep working. Single source of truth lives in ``repairs._common``.
    return _flip_all_faces_if_majority_inward(
        faces, unique_vertices, room_center, logger=logger
    )


def repair_plc_single_splits_iterative(
    faces: List[Face],
    points: List[Tuple[float, float, float]],
    room_center: Tuple[float, float, float],
    *,
    plc_report_supplier,
    logger=None,
    max_iters: int = 20,
    planarity_tol_m: float = 1e-6,
):
    summary = {
        "iterations": 0,
        "applied_repairs": 0,
        "stopped_reason": "unknown",
        "remaining_plc_hits": 0,
        "remaining_endpoint_face_hits": 0,
        "remaining_single_hit_candidates": 0,
        "remaining_multi_hit_faces": 0,
    }

    changed_any = False

    for it in range(1, max_iters + 1):
        summary["iterations"] = it

        plc_hits = plc_report_supplier(faces, points)

        summary["remaining_plc_hits"] = len(plc_hits)

        if not plc_hits:
            summary["remaining_plc_hits"] = 0
            summary["remaining_endpoint_face_hits"] = 0
            summary["remaining_single_hit_candidates"] = 0
            summary["remaining_multi_hit_faces"] = 0
            summary["stopped_reason"] = "no_plc_hits"
            if logger:
                logger.info("[PLC REPAIR] stable after %d iterations: no PLC hits", it - 1)
            return faces, points, changed_any, summary
        
        endpoint_face_hits = [
            r
            for r in plc_hits
            if r.get("hit_type") in ("endpoint_face_interior_touch", "endpoint_edge_touch")
        ]
        summary["remaining_endpoint_face_hits"] = len(endpoint_face_hits)

        if not endpoint_face_hits:
            summary["remaining_endpoint_face_hits"] = 0
            summary["remaining_single_hit_candidates"] = 0
            summary["remaining_multi_hit_faces"] = 0
            summary["stopped_reason"] = "no_endpoint_face_hits"
            if logger:
                logger.info("[PLC REPAIR] stop: PLC hits remain, but none are endpoint_face_interior_touch")
            return faces, points, changed_any, summary

        hits_by_face = defaultdict(list)
        for r in endpoint_face_hits:
            hits_by_face[r["facet_fid"]].append(r)

        multi_hit_faces = [fid for fid, rs in hits_by_face.items() if len(rs) > 1]
        single_hit_candidates = [rs[0] for fid, rs in hits_by_face.items() if len(rs) == 1]

        summary["remaining_multi_hit_faces"] = len(multi_hit_faces)
        summary["remaining_single_hit_candidates"] = len(single_hit_candidates)

        if logger:
            logger.info(
                "[PLC REPAIR] iter=%d plc_hits=%d endpoint_face_hits=%d multi_hit_faces=%d single_hit_candidates=%d",
                it,
                len(plc_hits),
                len(endpoint_face_hits),
                len(multi_hit_faces),
                len(single_hit_candidates),
            )

        changed = False
        diag = None

        # ---------------------------------------------------------
        # Priority 1: multi-hit same-face repair
        # ---------------------------------------------------------
        if multi_hit_faces:

            chosen_fid = max(multi_hit_faces, key=lambda fid: len(hits_by_face[fid]))

            chosen_reports = hits_by_face[chosen_fid]

            chosen_face = _find_face_by_fid(faces, chosen_fid)

            cls = _classify_multi_hit_face_collinear(

                chosen_face,

                chosen_reports,

                points,

                tol_m=0.01,

            )

            if cls["is_collinear"]:

                if logger:

                    logger.info(

                        "[PLC REPAIR] multi-hit face=%d classified as COLLINEAR (max_dev=%.6g)",

                        chosen_fid,

                        cls["max_dev"],

                    )

                faces, points, changed, diag = repair_multi_hit_face_collinear_chain(

                    faces,

                    chosen_reports,

                    points,

                    logger=logger,

                )

            else:

                if logger:

                    logger.info(

                        "CURRENTLY ONLY COLLINEAR multi-hit repair is implemented; face=%d classified as NONCOLLINEAR (max_dev=%.6g); skipping for now",
                        chosen_fid,
                        cls["max_dev"],
                    )
        # ---------------------------------------------------------
        # Priority 2: single-hit repair
        # ---------------------------------------------------------
        elif single_hit_candidates:
            chosen_report = single_hit_candidates[0]

            if logger:
                logger.info(
                    "[PLC REPAIR] chosen single-hit iter=%d facet_fid=%d edge=%s point=(%.6f,%.6f,%.6f)",
                    it,
                    chosen_report["facet_fid"],
                    chosen_report["edge"],
                    chosen_report["point"][0],
                    chosen_report["point"][1],
                    chosen_report["point"][2],
                )

            faces, changed, diag = _repair_single_endpoint_face_interior_touch_by_triangulation(
                faces,
                chosen_report,
                points,
                logger=logger,
                planarity_tol_m=planarity_tol_m,
            )

        else:
            summary["stopped_reason"] = "no_candidates"
            if logger:
                logger.info("[PLC REPAIR] stop: no repair candidates")
            return faces, points, changed_any, summary

        if not changed:
            summary["stopped_reason"] = "selected_candidate_not_changed"
            if logger:
                logger.info("[PLC REPAIR] stop: selected candidate produced no topology change; diag=%s", diag)
            return faces, points, changed_any, summary

        changed_any = True
        summary["applied_repairs"] += 1

        if logger:
            logger.info("[PLC REPAIR] applied iter=%d diag=%s", it, diag)

        # re-orient after topology change
        diag_orient = orient_faces_consistently_by_adjacency(faces, logger=logger)
        if logger:
            logger.info("[ORIENT AFTER PLC REPAIR] iter=%d diag=%s", it, diag_orient)

        flip_all_faces_if_majority_inward(
            faces,
            points,
            room_center,
            logger=logger,
        )

    summary["stopped_reason"] = "max_iters_reached"
    if logger:
        logger.warning("[PLC REPAIR] reached max_iters=%d", max_iters)

    return faces, points, changed_any, summary


def _repair_multi_endpoint_face_touch_same_face_by_triangulation(
    faces: List[Face],
    plc_reports: List[Dict[str, Any]],
    points: List[Tuple[float, float, float]],
    *,
    logger=None,
    planarity_tol_m: float = 1e-6,
):
    diag = {
        "status": "noop",
        "facet_fid": None,
        "n_hits": 0,
        "n_inserted_points": 0,
        "created_boundary_points": 0,
        "n_output_tris": 0,
    }

    if not plc_reports:
        diag["status"] = "no_reports"
        return faces, points, False, diag

    facet_ids = {r["facet_fid"] for r in plc_reports}
    if len(facet_ids) != 1:
        diag["status"] = "reports_not_same_face"
        return faces, points, False, diag

    facet_fid = next(iter(facet_ids))
    diag["facet_fid"] = facet_fid
    diag["n_hits"] = len(plc_reports)

    touched_face = _find_face_by_fid(faces, facet_fid)
    if touched_face is None:
        diag["status"] = "missing_face"
        return faces, points, False, diag
    vids = _face_vids(touched_face)
    pstat, pmax_m, prms_m = classify_face_planarity_m(vids, points)
    if pstat == "fatal":
        diag["status"] = "face_nonplanar"
        return faces, points, False, diag

    # --------------------------------------------------
    # 1) Extract unique inserted endpoint vids
    # --------------------------------------------------
    inserted_vids = []
    seen = set()

    for r in plc_reports:
        if r.get("hit_type") != "endpoint_face_interior_touch":
            continue

        vid, _other_vid = _endpoint_vids_from_edge_t(
            r["edge"],
            r["t_param"],
            t_eps=1e-9,
        )
        if vid is None:
            continue

        if vid not in seen:
            seen.add(vid)
            inserted_vids.append(vid)

    if len(inserted_vids) < 2:
        diag["status"] = "need_at_least_two_points"
        return faces, points, False, diag

    diag["n_inserted_points"] = len(inserted_vids)

    # --------------------------------------------------
    # 2) Project touched face + inserted points to 2D
    # --------------------------------------------------
    poly_ids = clean_face_loop(_face_vids(touched_face))
    poly2d, dropped_axis = project_face_to_2d(poly_ids, points)

    if area2(poly2d) < 0:
        poly_ids.reverse()
        poly2d.reverse()

    chain_pts_2d = [project_vid_to_2d(vid, points, dropped_axis) for vid in inserted_vids]

    # --------------------------------------------------
    # 3) Fit one dominant chain direction and sort points
    # --------------------------------------------------
    c2, d2 = fit_chain_direction_2d(chain_pts_2d)

    proj_vals = []
    for vid in inserted_vids:
        p2 = project_vid_to_2d(vid, points, dropped_axis)
        s = (p2[0] - c2[0]) * d2[0] + (p2[1] - c2[1]) * d2[1]
        proj_vals.append((s, vid))

    proj_vals.sort(key=lambda x: x[0])
    inserted_vids_sorted = [vid for _, vid in proj_vals]

    # --------------------------------------------------
    # 4) Extend fitted line to both boundary sides
    # --------------------------------------------------
    best_neg = None
    best_pos = None
    edge_neg = None
    edge_pos = None

    n = len(poly2d)
    for i in range(n):
        a2 = poly2d[i]
        b2 = poly2d[(i + 1) % n]

        hit = line_segment_intersection_signed(c2, d2, a2, b2, tol=1e-12)
        if hit is None:
            continue

        s, tseg = hit
        if s < 0.0:
            if best_neg is None or s > best_neg[0]:
                best_neg = (s, tseg)
                edge_neg = i
        elif s > 0.0:
            if best_pos is None or s < best_pos[0]:
                best_pos = (s, tseg)
                edge_pos = i

    if best_neg is None or best_pos is None:
        diag["status"] = "failed_boundary_extension"
        return faces, points, False, diag

    # --------------------------------------------------
    # 5) Create/reuse two boundary points
    # --------------------------------------------------
    bneg_vid, points, used_neg_existing = create_or_reuse_boundary_point_on_edge(
        poly_ids, poly2d, edge_neg, best_neg[1], points
    )
    bpos_vid, points, used_pos_existing = create_or_reuse_boundary_point_on_edge(
        poly_ids, poly2d, edge_pos, best_pos[1], points
    )

    diag["created_boundary_points"] = int(not used_neg_existing) + int(not used_pos_existing)

    # refresh polygon references after possible point insertion
    poly_ids = clean_face_loop(_face_vids(touched_face))
    poly2d, dropped_axis = project_face_to_2d(poly_ids, points)
    if area2(poly2d) < 0:
        poly_ids.reverse()
        poly2d.reverse()

    # insert boundary point A if new
    if bneg_vid not in poly_ids:
        poly_ids = _insert_vertex_on_polygon_edge(poly_ids, edge_neg, bneg_vid)

    # insert boundary point B if new
    if bpos_vid not in poly_ids:
        poly2d, dropped_axis = project_face_to_2d(poly_ids, points)
        if area2(poly2d) < 0:
            poly_ids.reverse()
            poly2d.reverse()

        p_bpos_2d = project_vid_to_2d(bpos_vid, points, dropped_axis)
        found_edge = None
        for i in range(len(poly_ids)):
            a2 = poly2d[i]
            b2 = poly2d[(i + 1) % len(poly_ids)]
            if point_on_segment_2d(p_bpos_2d, a2, b2, tol=1e-8):
                found_edge = i
                break

        if found_edge is None:
            diag["status"] = "cannot_reinsert_second_boundary_point"
            return faces, points, False, diag

        poly_ids = _insert_vertex_on_polygon_edge(poly_ids, found_edge, bpos_vid)

    if bneg_vid not in poly_ids or bpos_vid not in poly_ids:
        diag["status"] = "boundary_points_not_in_loop"
        return faces, points, False, diag

    # --------------------------------------------------
    # 6) Build chain split and 2 polygons
    # --------------------------------------------------
    chain_vids = [bneg_vid] + inserted_vids_sorted + [bpos_vid]

    idx_neg = poly_ids.index(bneg_vid)
    idx_pos = poly_ids.index(bpos_vid)

    boundary_a = _boundary_chain(poly_ids, idx_neg, idx_pos)
    boundary_b = _boundary_chain(poly_ids, idx_pos, idx_neg)

    # remove duplicated boundary endpoints before concatenation
    poly_a = clean_face_loop(boundary_a + list(reversed(chain_vids[1:-1])))
    poly_b = clean_face_loop(boundary_b + chain_vids[1:-1])

    split_polys = []
    for poly in (poly_a, poly_b):
        if len(poly) < 3:
            continue
        if polygon_area2_newell(poly, points) <= 1e-22:
            continue
        split_polys.append(poly)

    if len(split_polys) != 2:
        diag["status"] = "invalid_split_polys"
        return faces, points, False, diag

    # --------------------------------------------------
    # 7) Triangulate both polygons
    # --------------------------------------------------
    out_tris = []
    for poly in split_polys:
        if len(poly) == 3:
            tris = [poly]
        else:
            tris = triangulate_face_cdt_shapely(poly, points)

        if not tris:
            diag["status"] = "triangulation_failed"
            return faces, points, False, diag

        for tri in tris:
            if len(set(tri)) < 3:
                continue
            if tri_area2(tri[0], tri[1], tri[2], points) <= 1e-22:
                continue
            out_tris.append(tri)

    if not out_tris:
        diag["status"] = "no_output_tris"
        return faces, points, False, diag

    diag["n_output_tris"] = len(out_tris)

    # --------------------------------------------------
    # 8) Replace touched face with new triangles
    # --------------------------------------------------
    new_faces = []
    for i, f in enumerate(faces):
        if _face_fid(f, idx=i) != facet_fid:
            new_faces.append(f)
            continue

        for tri in out_tris:
            new_faces.append(Face(vertex_indices=tri, group=getattr(f, "group", "default"), material=getattr(f, "material", None)))

    diag["status"] = "ok"

    if logger:
        logger.info(
            "[PLC MULTI TRI REPAIR] facet_fid=%d hits=%d inserted=%d created_boundary_points=%d output_tris=%d",
            facet_fid,
            len(plc_reports),
            len(inserted_vids_sorted),
            diag["created_boundary_points"],
            diag["n_output_tris"],
        )

    return new_faces, points, True, diag


# Segment Facet Intersection helper math now imported from geometry_math


def _boundary_chain(poly_ids, i, j):
    n = len(poly_ids)
    out = [poly_ids[i]]
    k = i
    while k != j:
        k = (k + 1) % n
        out.append(poly_ids[k])
    return out


def _visible_boundary_vertices_from_point(poly2d, p2, tol=1e-12):
    n = len(poly2d)
    visible = []

    for i in range(n):
        vi = poly2d[i]
        ok = True

        for k in range(n):
            a = poly2d[k]
            b = poly2d[(k + 1) % n]

            if k == i or (k + 1) % n == i:
                continue

            if segments_intersect_2d(p2, vi, a, b, tol):
                ok = False
                break

        if ok:
            visible.append(i)

    return visible


def _split_face_at_single_interior_vertex(
    face: Face,
    inserted_vid: int,
    points: List[Tuple[float, float, float]],
    *,
    planarity_tol_m: float = 1e-6,
    boundary_tol_2d: float = 1e-10,
):
    vids = _face_vids(face)
    if inserted_vid in vids:
        return None

    max_abs, rms, nu, c = planarity_deviation_m(vids, points)
    if not math.isfinite(max_abs) or max_abs > planarity_tol_m:
        return None
    poly_ids = clean_face_loop(vids)
    poly2d, dropped_axis = project_face_to_2d(poly_ids, points)

    if area2(poly2d) < 0:
        poly_ids.reverse()
        poly2d.reverse()

    p2 = project_point_by_dropped_axis(points[inserted_vid - 1], dropped_axis)

    pos = point_in_polygon_2d(poly2d, p2, tol=boundary_tol_2d)
    if pos != "inside":
        return None

    visible = _visible_boundary_vertices_from_point(poly2d, p2, tol=boundary_tol_2d)
    if len(visible) < 2:
        return None

    best_pair = None
    best_score = -1.0
    n = len(poly_ids)

    for a in visible:
        for b in visible:
            if a >= b:
                continue
            if (a + 1) % n == b or (b + 1) % n == a:
                continue

            pa = poly2d[a]
            pb = poly2d[b]
            score = (pa[0] - pb[0])**2 + (pa[1] - pb[1])**2
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is not None:
        i, j = best_pair

        chain_ij = _boundary_chain(poly_ids, i, j)
        chain_ji = _boundary_chain(poly_ids, j, i)

        poly_a = clean_face_loop(chain_ij + [inserted_vid])
        poly_b = clean_face_loop(chain_ji + [inserted_vid])

        out = []
        for poly in (poly_a, poly_b):
            if len(poly) < 3:
                continue
            if polygon_area2_newell(poly, points) <= 1e-22:
                continue
            out.append(poly)

        if len(out) == 2:
            return out

    out = []
    for i in range(len(poly_ids)):
        a = poly_ids[i]
        b = poly_ids[(i + 1) % len(poly_ids)]
        tri = [a, b, inserted_vid]

        if len(set(tri)) < 3:
            continue
        if tri_area2(tri[0], tri[1], tri[2], points) <= 1e-22:
            continue

        out.append(tri)

    return out if out else None


# delegated to app.geometry.geometry_math.geometry_math.polygon_area2_newell


def _repair_single_endpoint_face_interior_touch(
    faces: List[Face],
    plc_report: Dict[str, Any],
    points: List[Tuple[float, float, float]],
    *,
    logger=None,
    planarity_tol_m: float = 1e-6,
):
    diag = {
        "status": "noop",
        "touched_fid": None,
        "inserted_vid": None,
        "n_new_faces": 0,
    }

    if plc_report.get("hit_type") != "endpoint_face_interior_touch":
        diag["status"] = "wrong_type"
        return faces, False, diag

    facet_fid = plc_report["facet_fid"]
    inserted_vid, _other_vid = _endpoint_vids_from_edge_t(
        plc_report["edge"],
        plc_report["t_param"],
        t_eps=1e-9,
    )
    if inserted_vid is None:
        diag["status"] = "bad_t_param"
        return faces, False, diag

    diag["touched_fid"] = facet_fid
    diag["inserted_vid"] = inserted_vid

    touched_face = _find_face_by_fid(faces, facet_fid)
    if touched_face is None:
        diag["status"] = "missing_face"
        return faces, False, diag

    split_polys = _split_face_at_single_interior_vertex(
        touched_face,
        inserted_vid,
        points,
        planarity_tol_m=planarity_tol_m,
    )

    if not split_polys:
        diag["status"] = "split_failed"
        if logger:
            logger.warning(
                "[SINGLE SPLIT] failed facet_fid=%d inserted_vid=%d vids=%s",
                facet_fid, inserted_vid, _face_vids(touched_face)
            )
        return faces, False, diag

    new_faces = []
    for i, f in enumerate(faces):
        if _face_fid(f, idx=i) != facet_fid:
            new_faces.append(f)
            continue

        for poly in split_polys:
            new_faces.append(Face(vertex_indices=poly, group=getattr(f, "group", "default"), material=getattr(f, "material", None)))

    diag["status"] = "ok"
    diag["n_new_faces"] = len(split_polys)

    if logger:
        logger.info(
            "[SINGLE SPLIT] repaired facet_fid=%d inserted_vid=%d -> %d new faces",
            facet_fid, inserted_vid, len(split_polys)
        )

    return new_faces, True, diag


# point_segment_distance_2d moved to geometry_math

def _project_face_and_point_to_2d(face_ids, point_vid, points):
    poly2d, dropped_axis = project_face_to_2d(face_ids, points)
    p2 = project_point_by_dropped_axis(points[point_vid - 1], dropped_axis)
    return poly2d, p2, dropped_axis

def _repair_single_endpoint_face_interior_touch_by_triangulation(
    faces: List[Face],
    plc_report: Dict[str, Any],
    points: List[Tuple[float, float, float]],
    *,
    logger=None,
    planarity_tol_m: float = 1e-6,
):
    diag = {
        "status": "noop",
        "touched_fid": None,
        "inserted_vid": None,
        "n_split_polys": 0,
        "n_output_tris": 0,
    }

    if plc_report.get("hit_type") != "endpoint_face_interior_touch":
        diag["status"] = "wrong_type"
        return faces, False, diag

    facet_fid = plc_report["facet_fid"]
    inserted_vid, _other_vid = _endpoint_vids_from_edge_t(
        plc_report["edge"],
        plc_report["t_param"],
        t_eps=1e-9,
    )
    if inserted_vid is None:
        diag["status"] = "bad_t_param"
        return faces, False, diag

    diag["touched_fid"] = facet_fid
    diag["inserted_vid"] = inserted_vid

    touched_face = _find_face_by_fid(faces, facet_fid)
    if touched_face is None:
        diag["status"] = "missing_face"
        return faces, False, diag

    split_polys = _split_face_at_single_interior_vertex(
        touched_face,
        inserted_vid,
        points,
        planarity_tol_m=planarity_tol_m,
    )

    if not split_polys:
        diag["status"] = "split_failed"
        return faces, False, diag

    diag["n_split_polys"] = len(split_polys)

    out_tris = []
    for poly in split_polys:
        if len(poly) < 3:
            continue

        if len(poly) == 3:
            tris = [poly]
        else:
            tris = triangulate_face_cdt_shapely(poly, points)

        if not tris:
            diag["status"] = "triangulation_failed"
            if logger:
                logger.warning(
                    "[PLC TRI REPAIR] triangulation failed facet_fid=%d poly=%s",
                    facet_fid, poly
                )
            return faces, False, diag

        for tri in tris:
            if len(set(tri)) < 3:
                continue
            if tri_area2(tri[0], tri[1], tri[2], points) <= 1e-22:
                continue
            out_tris.append(tri)

    if not out_tris:
        diag["status"] = "no_output_tris"
        return faces, False, diag

    diag["n_output_tris"] = len(out_tris)

    new_faces = []
    for i, f in enumerate(faces):
        if _face_fid(f, idx=i) != facet_fid:
            new_faces.append(f)
            continue

        for tri in out_tris:
            new_faces.append(Face(vertex_indices=tri, group=getattr(f, "group", "default"), material=getattr(f, "material", None)))

    diag["status"] = "ok"

    if logger:
        logger.info(
            "[PLC TRI REPAIR] repaired facet_fid=%d inserted_vid=%d split_polys=%d output_tris=%d",
            facet_fid,
            inserted_vid,
            diag["n_split_polys"],
            diag["n_output_tris"],
        )

    return new_faces, True, diag


def _insert_vertex_on_polygon_edge(poly_ids, edge_index, new_vid):
    n = len(poly_ids)

    if n < 2:
        return poly_ids

    return (
        poly_ids[: edge_index + 1]
        + [new_vid]
        + poly_ids[edge_index + 1 :]
    )


# project_vid_to_2d moved to geometry_math


# fit_chain_direction_2d moved to geometry_math


# line_segment_intersection_signed moved to geometry_math


from geometry_pipeline.repairs.mesh._common import (
    create_or_reuse_boundary_point_on_edge,
    move_touching_endpoint_off_face,
    _endpoint_vids_from_edge_t,
    compute_face_unit_normal,
    offset_point_along_vector,
    _find_face_by_fid,
)


# helpers moved to repairs._common


def repair_plc_by_offset_iterative(
    faces,
    points,
    *,
    plc_report_supplier,
    logger=None,
    max_iters=20,
    offset_m=0.1,
):
    summary = {
        "iterations": 0,
        "applied_repairs": 0,
        "stopped_reason": "unknown",
        "remaining_plc_hits": 0,
        "remaining_endpoint_face_hits": 0,
    }

    changed_any = False

    for it in range(1, max_iters + 1):
        summary["iterations"] = it

        plc_hits = plc_report_supplier(faces, points)

        summary["remaining_plc_hits"] = len(plc_hits)

        if not plc_hits:
            summary["remaining_endpoint_face_hits"] = 0
            summary["stopped_reason"] = "no_plc_hits"
            if logger:
                logger.info("[PLC OFFSET] stable after %d iterations: no PLC hits", it - 1)
            return faces, points, changed_any, summary

        endpoint_face_hits = [
            r for r in plc_hits
            if r.get("hit_type") == "endpoint_face_interior_touch"
        ]
        summary["remaining_endpoint_face_hits"] = len(endpoint_face_hits)

        if not endpoint_face_hits:
            summary["stopped_reason"] = "no_endpoint_face_interior_touch"
            if logger:
                logger.info("[PLC OFFSET] stop: PLC hits remain, but none are endpoint_face_interior_touch")
            return faces, points, changed_any, summary

        target = endpoint_face_hits[0]

        points, changed, diag = move_touching_endpoint_off_face(
            faces,
            points,
            target,
            offset_m=offset_m,
            logger=logger,
        )

        if not changed:
            summary["stopped_reason"] = "selected_offset_not_applied"
            if logger:
                logger.info("[PLC OFFSET] stop: selected report was not changed; diag=%s", diag)
            return faces, points, changed_any, summary

        changed_any = True
        summary["applied_repairs"] += 1

        if logger:
            logger.info("[PLC OFFSET] applied iter=%d diag=%s", it, diag)

    summary["stopped_reason"] = "max_iters_reached"
    if logger:
        logger.warning("[PLC OFFSET] reached max_iters=%d", max_iters)

    return faces, points, changed_any, summary


# -------- REPAIR MULTI HIT POINT-FACE INTERSECTION BY SPLITTING WITH NEW VERTEX --------
def _find_face_by_fid(faces: List[Face], fid: int):
    for i, f in enumerate(faces):
        if _face_fid(f, idx=i) == fid:
            return f
    return None


def _face_plane_basis(face, points):
    vids = _face_vids(face)
    n = newell_normal_from_points(vids, points)
    n = unit(n)
    c = polygon_centroid(vids, points)
    ref = (1.0, 0.0, 0.0)
    if abs(dot(ref, n)) > 0.9:
        ref = (0.0, 1.0, 0.0)
    u = unit(cross(ref, n))
    v = unit(cross(n, u))
    return c, u, v, n


def _project_to_face_2d(p, c, u, v):
    d = sub(p, c)
    return (dot(d, u), dot(d, v))


from geometry_pipeline.repairs.mesh._common import polygon_centroid, get_or_create_vertex


def _classify_multi_hit_face_collinear(
    face: Face,
    reports: List[Dict[str, Any]],
    points: List[Tuple[float, float, float]],
    *,
    tol_m: float = 1e-4,
):
    if len(reports) < 2:
        return {
            "is_collinear": False,
            "reason": "need_at_least_2_points",
            "max_dev": 0.0,
        }
    c, u, v, n = _face_plane_basis(face, points)
    pts3 = [r["point"] for r in reports]
    pts2 = [_project_to_face_2d(p, c, u, v) for p in pts3]
    best_i = 0
    best_j = 1
    best_d = -1.0
    for i in range(len(pts2)):
        for j in range(i + 1, len(pts2)):
            d = math.dist(pts2[i], pts2[j])
            if d > best_d:
                best_d = d
                best_i, best_j = i, j
    a = pts2[best_i]
    b = pts2[best_j]
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    L = math.sqrt(dx*dx + dy*dy)
    if L <= 1e-12:
        return {
            "is_collinear": False,
            "reason": "degenerate_points",
            "max_dev": 0.0,
        }
    max_dev = 0.0
    params = []
    for p in pts2:
        px = p[0] - a[0]
        py = p[1] - a[1]
        t = (px*dx + py*dy) / (L*L)
        params.append(t)
        perp = abs(px*dy - py*dx) / L
        max_dev = max(max_dev, perp)
    ordered = sorted(zip(params, pts2, pts3), key=lambda x: x[0])
    return {
        "is_collinear": max_dev <= 0.01,
        "max_dev": max_dev,
        "ordered_points_2d": [x[1] for x in ordered],
        "ordered_points_3d": [x[2] for x in ordered],
    }


def repair_multi_hit_face_collinear_chain(
    faces: List[Face],
    reports: List[Dict[str, Any]],
    points: List[Tuple[float, float, float]],
    *,
    logger=None,
):
    facet_fid = reports[0]["facet_fid"]
    face = _find_face_by_fid(faces, facet_fid)
    if face is None:
        return faces, points, False, {"status": "face_not_found"}
    cls = _classify_multi_hit_face_collinear(face, reports, points)
    if not cls["is_collinear"]:
        return faces, points, False, {"status": "not_collinear"}
    chain_vids = []
    for p in cls["ordered_points_3d"]:
        vid = get_or_create_vertex(points, p)
        chain_vids.append(vid)
    vids = _face_vids(face)
    # Chain direction in 3D, from the first to the last (ordered) touch point.
    a3 = points[chain_vids[0] - 1]
    b3 = points[chain_vids[-1] - 1]
    d3 = (b3[0] - a3[0], b3[1] - a3[1], b3[2] - a3[2])
    dlen = math.sqrt(d3[0] * d3[0] + d3[1] * d3[1] + d3[2] * d3[2])

    def _chain_param(vid: int) -> float:
        """Signed position of a vertex along the chain direction."""
        if dlen <= 1e-12:
            return 0.0
        p = points[vid - 1]
        return (
            (p[0] - a3[0]) * d3[0]
            + (p[1] - a3[1]) * d3[1]
            + (p[2] - a3[2]) * d3[2]
        ) / dlen

    # The two chain endpoints sit *inside* the face. Snapping each end to the
    # globally nearest corner collapses both onto the same vertex when the
    # chain hugs one side (e.g. 134 and 138 are both nearest to corner 123),
    # which produces a self-touching "bridged" loop. Instead, anchor the chain
    # *start* to the nearest boundary vertex behind it (param <= start) and the
    # chain *end* to the nearest boundary vertex ahead of it (param >= end), so
    # the two anchors land on opposite sides and the split is a clean cut.
    tol_param = 1e-9
    start_param = _chain_param(chain_vids[0])
    end_param = _chain_param(chain_vids[-1])

    behind = [v for v in vids if _chain_param(v) <= start_param + tol_param]
    ahead = [v for v in vids if _chain_param(v) >= end_param - tol_param]

    start_vid = min(
        behind or vids,
        key=lambda vid: distance(points[vid - 1], points[chain_vids[0] - 1]),
    )
    end_vid = min(
        ahead or vids,
        key=lambda vid: distance(points[vid - 1], points[chain_vids[-1] - 1]),
    )
    split_chain = [start_vid] + chain_vids + [end_vid]
    verts = vids
    i0 = verts.index(start_vid)
    i1 = verts.index(end_vid)
    if i0 <= i1:
        path1 = verts[i0:i1 + 1]
        path2 = verts[i1:] + verts[:i0 + 1]
    else:
        path1 = verts[i0:] + verts[:i1 + 1]
        path2 = verts[i1:i0 + 1]
    new_loop1 = clean_face_loop(path1 + list(reversed(split_chain[1:-1])))
    new_loop2 = clean_face_loop(path2 + split_chain[1:-1])
    if len(new_loop1) < 3 or len(new_loop2) < 3:
        return faces, points, False, {"status": "bad_split"}
    new_faces = []
    for i, f in enumerate(faces):
        if _face_fid(f, idx=i) != facet_fid:
            new_faces.append(f)
        else:
            new_faces.append(Face(vertex_indices=new_loop1, group=getattr(f, "group", "default"), material=getattr(f, "material", None)))
            new_faces.append(Face(vertex_indices=new_loop2, group=getattr(f, "group", "default"), material=getattr(f, "material", None)))
    diag = {
        "status": "ok",
        "repair_type": "collinear_chain_split",
        "facet_fid": facet_fid,
        "n_chain_points": len(chain_vids),
    }
    if logger:
        logger.info("[PLC REPAIR] structured collinear split face=%d", facet_fid)
    return new_faces, points, True, diag


def orient_faces_consistently_by_adjacency(
    faces: List[Face],
    logger=None,
) -> Dict[str, Any]:
    edge_to_uses: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]] = defaultdict(list)

    for fi, f in enumerate(faces):
        vs = _face_vids(f)
        n = len(vs)
        if n < 2:
            continue
        for i in range(n):
            a = vs[i]
            b = vs[(i + 1) % n]
            edge_to_uses[uedge(a, b)].append((fi, (a, b)))

    boundary_edges = 0
    nonmanifold_edges = 0
    for e, uses in edge_to_uses.items():
        if len(uses) == 1:
            boundary_edges += 1
        elif len(uses) > 2:
            nonmanifold_edges += 1

    if logger:
        logger.info(
            "[ORIENT] edges=%d boundary=%d nonmanifold=%d",
            len(edge_to_uses), boundary_edges, nonmanifold_edges
        )

    face_adj: Dict[int, List[Tuple[int, Tuple[int,int], Tuple[int,int]]]] = defaultdict(list)

    for e, uses in edge_to_uses.items():
        if len(uses) != 2:
            continue
        (f0, dir0), (f1, dir1) = uses
        face_adj[f0].append((f1, e, dir0, dir1))
        face_adj[f1].append((f0, e, dir1, dir0))

    visited = [False] * len(faces)
    flipped = [False] * len(faces)
    flipped_count = 0
    components = 0

    def flip_face(fi: int):
        nonlocal flipped_count
        _reverse_face_vids(faces[fi])
        flipped[fi] = not flipped[fi]
        flipped_count += 1

    for seed in range(len(faces)):
        if visited[seed]:
            continue

        components += 1
        visited[seed] = True
        q = deque([seed])

        while q:
            fa = q.popleft()
            for (fb, e, dir_a, dir_b) in face_adj.get(fa, []):
                if not visited[fb]:
                    def current_edge_dir(face_verts: List[int], undirected_edge: Tuple[int,int]) -> Optional[Tuple[int,int]]:
                        u, v = undirected_edge
                        n = len(face_verts)
                        for i in range(n):
                            a = face_verts[i]
                            b = face_verts[(i + 1) % n]
                            if (a == u and b == v) or (a == v and b == u):
                                return (a, b)
                        return None

                    da = current_edge_dir(_face_vids(faces[fa]), e)
                    db = current_edge_dir(_face_vids(faces[fb]), e)

                    if da is None or db is None:
                        if logger:
                            logger.warning("[ORIENT] missing edge %s in fa=%d or fb=%d", e, fa, fb)
                        visited[fb] = True
                        q.append(fb)
                        continue

                    if da == db:
                        flip_face(fb)
                        if logger:
                            logger.debug("[ORIENT] flipped face fid=%d to fix shared edge %s", _face_fid(faces[fb], idx=fb), e)

                    visited[fb] = True
                    q.append(fb)
                else:
                    pass

    return {
        "components": components,
        "flipped_faces": flipped_count,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
    }


def trim_component_against_facet_plane(
    faces: List[Face],
    points: List[Tuple[float, float, float]],
    *,
    clipping_facet_fid: int,
    seed_face_fids: List[int],
    room_center: Tuple[float, float, float],
    tol: float = 1e-9,
    max_component_fraction: float = 0.5,
    logger=None,
) -> Tuple[List[Face], List[Tuple[float, float, float]], bool, Dict[str, Any]]:
    
    """
    Trim a connected face component against the plane of a clipping facet.

    Parameters
    ----------
    faces : list[Face]
        Current face list.
    points : list[(x, y, z)]
        Global mutable point list. New intersection vertices may be appended.
    clipping_facet_fid : int
        Face id whose plane defines the clipping boundary.
    seed_face_fids : list[int]
        One or more faces belonging to the protruding component that should be clipped.
        For PLC reports, ``edge_fids`` is usually a good seed.
    room_center : tuple[float, float, float]
        A point known to lie on the side of the clipping plane that should be kept.
        In CHORAS this is usually the room center.
    tol : float
        Numerical tolerance for clipping and vertex reuse.
    max_component_fraction : float
        Safety cap in ``[0, 1]``. If the flood-filled component exceeds this
        fraction of the total face count, the trim is skipped (treated as a
        connectivity leak rather than an isolated protrusion). Defaults to 0.5.
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    tuple
        (updated_faces, updated_points, changed, diagnostics)

        diagnostics contains:
        - status
        - clipping_facet_fid
        - seed_face_fids
        - component_face_fids
        - keep_sign
        - faces_removed
        - faces_clipped
        - new_vertices_added

    Purpose
    -------
    This helper performs a best-effort half-space clip of a selected connected
    component. It is useful when a protruding object crosses a room boundary face and
    the desired behavior is to keep only the part on the room side of that face plane.

    Limitation
    ----------
    The clipping facet itself is not split or reconstructed. Therefore, this helper is
    best used as a controlled trimming utility for clearly unwanted outside geometry.
    """
    diag: Dict[str, Any] = {
        "status": "noop",
        "clipping_facet_fid": clipping_facet_fid,
        "seed_face_fids": list(seed_face_fids),
        "component_face_fids": [],
        "keep_sign": None,
        "faces_removed": 0,
        "faces_clipped": 0,
        "new_vertices_added": 0,
    }

    clipping_face = _find_face_by_fid(faces, clipping_facet_fid)
    if clipping_face is None:
        diag["status"] = "clipping_face_not_found"
        return faces, points, False, diag

    plane = _plane_from_face(clipping_face, points)
    if plane is None:
        diag["status"] = "invalid_clipping_plane"
        return faces, points, False, diag

    plane_point, plane_normal = plane
    room_sd = _signed_distance_to_plane(room_center, plane_point, plane_normal)
    keep_sign = 1.0 if room_sd >= 0.0 else -1.0
    diag["keep_sign"] = keep_sign

    component_face_fids = collect_face_component_from_seed_faces(
        faces,
        seed_face_fids,
        excluded_face_fids=[clipping_facet_fid],
    )
    diag["component_face_fids"] = component_face_fids

    if not component_face_fids:
        diag["status"] = "empty_component"
        return faces, points, False, diag

    # Component-size cap: a legitimate trim target is a small protruding
    # component (e.g. an object poking through a wall). On watertight rooms
    # with internal partitions the flood fill can leak across shared edges and
    # swallow most of the mesh; clipping such a component deletes important
    # faces. If the component is too large relative to the whole mesh, treat it
    # as a leak and skip the trim instead of gutting the geometry.
    component_fraction = len(component_face_fids) / max(1, len(faces))
    diag["component_fraction"] = component_fraction
    if component_fraction > max_component_fraction:
        diag["status"] = "component_too_large_skipped"
        diag["max_component_fraction"] = max_component_fraction
        if logger is not None:
            logger.warning(
                "[TRIM] facet_fid=%d component_faces=%d/%d (%.1f%%) exceeds cap %.1f%% — skipping trim",
                clipping_facet_fid,
                len(component_face_fids),
                len(faces),
                component_fraction * 100.0,
                max_component_fraction * 100.0,
            )
        return faces, points, False, diag

    start_n_points = len(points)
    changed = False
    updated_faces: List[Face] = []

    for i, face in enumerate(faces):
        if _face_fid(face, idx=i) not in component_face_fids:
            updated_faces.append(face)
            continue

        clipped_loop = _clip_face_loop_against_plane(
            _face_vids(face),
            points,
            plane_point,
            plane_normal,
            keep_sign,
            tol=tol,
        )

        if len(clipped_loop) < 3:
            diag["faces_removed"] += 1
            changed = True
            continue

        if clipped_loop != clean_face_loop(_face_vids(face)):
            diag["faces_clipped"] += 1
            changed = True

        status, _area2 = classify_face_degeneracy(
            clipped_loop,
            points,
            fatal_area_tol=1e-18,
        )
        if status == "fatal":
            diag["faces_removed"] += 1
            changed = True
            continue

        updated_faces.append(
            Face(vertex_indices=clipped_loop, group=getattr(face, "group", "default"), material=getattr(face, "material", "unknown"))
        )

    diag["new_vertices_added"] = len(points) - start_n_points
    diag["status"] = "ok" if changed else "no_effect"

    if logger is not None:
        logger.info(
            "[TRIM] facet_fid=%d component_faces=%d clipped=%d removed=%d new_vertices=%d status=%s",
            clipping_facet_fid,
            len(component_face_fids),
            diag["faces_clipped"],
            diag["faces_removed"],
            diag["new_vertices_added"],
            diag["status"],
        )

    return updated_faces, points, changed, diag


def trim_component_from_segment_face_intersection_report(
    faces: List[Face],
    points: List[Tuple[float, float, float]],
    plc_report: Dict[str, Any],
    room_center: Tuple[float, float, float],
    *,
    tol: float = 1e-9,
    logger=None,
) -> Tuple[List[Face], List[Tuple[float, float, float]], bool, Dict[str, Any]]:
    """
    Convenience wrapper that trims a protruding component using one PLC report.

    Parameters
    ----------
    faces : list[Face]
        Current face list.
    points : list[(x, y, z)]
        Global mutable point list.
    plc_report : dict
        One PLC report generated by ``detect_segment_facet_intersections_cdt``.
        The report must contain ``facet_fid`` and ``edge_fids``.
    room_center : tuple[float, float, float]
        Reference point used to choose the kept half-space.
    tol : float
        Numerical tolerance for clipping.
    logger : logging.Logger | None
        Optional logger.

    Returns
    -------
    tuple
        (updated_faces, updated_points, changed, diagnostics)

    Purpose
    -------
    This wrapper is especially useful for PLC reports of type
    ``segment_face_interior_intersection`` where the edge's incident faces belong to a
    protruding component that should be clipped back to the clipping facet plane.
    """
    clipping_facet_fid = plc_report.get("facet_fid")
    seed_face_fids = list(plc_report.get("edge_fids") or [])

    if clipping_facet_fid is None or not seed_face_fids:
        diag = {
            "status": "invalid_plc_report",
            "clipping_facet_fid": clipping_facet_fid,
            "seed_face_fids": seed_face_fids,
        }
        return faces, points, False, diag

    return trim_component_against_facet_plane(
        faces,
        points,
        clipping_facet_fid=clipping_facet_fid,
        seed_face_fids=seed_face_fids,
        room_center=room_center,
        tol=tol,
        logger=logger,
    )

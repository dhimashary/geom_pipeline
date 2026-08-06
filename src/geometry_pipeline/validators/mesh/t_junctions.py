"""Validator: detects PLC-level T-junctions (vertex on another face's edge)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Dict, List

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.geometry_math.geometry_math import cross, dot, sub, uedge
from geometry_pipeline.validators.base import BaseValidator


def detect_t_junctions_mesh_global_plc(
    mesh: Mesh,
    *,
    tol: float = 1e-8,
    max_reports: int = 2000,
) -> List[Dict[str, Any]]:
    points = [(v.x, v.y, v.z) for v in mesh.vertices]

    def point_on_segment_scale_correct(P, A, B, tol_):
        AB = sub(B, A)
        AP = sub(P, A)
        ab2 = dot(AB, AB)
        if ab2 <= 0.0:
            return (False, 0.0)

        cr = cross(AB, AP)
        if dot(cr, cr) > (tol_ * tol_) * ab2:
            return (False, 0.0)

        t = dot(AP, AB) / ab2
        if not (-tol_ <= t <= 1.0 + tol_):
            return (False, t)

        return (True, t)

    edge_to_face_idxs = defaultdict(list)
    vert_to_face_idxs = defaultdict(list)
    edge_set = set()

    for fi, face in enumerate(mesh.faces):
        vids = [int(i) for i in face.vertex_indices]
        n = len(vids)
        for v in vids:
            vert_to_face_idxs[v].append(fi)
        for i in range(n):
            a = vids[i]
            b = vids[(i + 1) % n]
            e = uedge(a, b)
            edge_set.add(e)
            edge_to_face_idxs[e].append(fi)

    all_verts = list(range(1, len(points) + 1))

    reports = []
    for u, v in edge_set:
        A = points[u - 1]
        B = points[v - 1]
        face_idxs_using_edge = edge_to_face_idxs[uedge(u, v)]
        if not face_idxs_using_edge:
            continue

        for w in all_verts:
            if w == u or w == v:
                continue

            P = points[w - 1]
            ok, t = point_on_segment_scale_correct(P, A, B, tol)
            if not ok or not (tol < t < 1.0 - tol):
                continue

            culprit_fid = None
            for fi in face_idxs_using_edge:
                face_vids = [int(i) for i in mesh.faces[fi].vertex_indices]
                if w not in face_vids:
                    culprit_fid = getattr(mesh.faces[fi], "fid", fi)
                    break

            if culprit_fid is None:
                continue

            edge_face_fids = [getattr(mesh.faces[fi], "fid", fi) for fi in face_idxs_using_edge]
            v_face_fids = [
                getattr(mesh.faces[fi], "fid", fi) for fi in vert_to_face_idxs.get(w, [])
            ]

            if len(v_face_fids) > 0:
                reports.append(
                    {
                        "edge": (u, v),
                        "edge_coordinates": [[A[0], A[1], A[2]], [B[0], B[1], B[2]]],
                        "split_vertex": w,
                        "split_vertex_coordinates": [P[0], P[1], P[2]],
                        "t_param": t,
                        "edge_face_fids": edge_face_fids,
                        "culprit_face_fid": culprit_fid,
                        "v_face_fids": v_face_fids,
                    }
                )

            if len(reports) >= max_reports:
                return reports

    return reports


class TJunctionsValidator(BaseValidator):
    name: ClassVar[str] = "t_junctions"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.T_JUNCTION

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        return detect_t_junctions_mesh_global_plc(
            geom,
            tol=ctx.tolerances.t_junction,
            max_reports=ctx.tolerances.max_reports,
        )

    def payload_of(self, payload: dict) -> dict:
        # Build the frontend `elements` shape here so the report translator
        # stays generic (no per-kind branches in reporting/frontend_schema).
        p = dict(payload)
        p["elements"] = [
            {"type": "edge", "points": p.get("edge_coordinates", [])},
            {"type": "vertex", "points": p.get("split_vertex_coordinates", [])},
        ]
        return p

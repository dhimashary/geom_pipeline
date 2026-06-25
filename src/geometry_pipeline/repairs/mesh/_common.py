"""Shared repair helpers (C3 target).

Start with `room_center_from_mesh` (was `mesh_helpers.py`) so callers
can import a stable `_common` module. Additional shared helpers will be
moved here as the refactor proceeds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geometry_pipeline.core.ir import Mesh
from typing import List, Tuple, Dict, Any

import math

from geometry_pipeline.core.ir import Face
from geometry_pipeline.geometry_math.geometry_math import (
    distance,
    polygon_centroid,
    compute_face_unit_normal,
    offset_point_along_vector,
    newell_normal_from_points,
)


def _vertex_mean(vertices: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    n = len(vertices)
    return (
        sum(p[0] for p in vertices) / n,
        sum(p[1] for p in vertices) / n,
        sum(p[2] for p in vertices) / n,
    )


def _enclosed_volume_centroid(
    vertices: List[Tuple[float, float, float]],
    faces,
) -> Tuple[float, float, float] | None:
    """Centre of mass of the solid enclosed by a closed, consistently
    oriented surface.

    Uses the divergence-theorem decomposition into tetrahedra
    ``(origin, a, b, c)`` (polygon faces are fan-triangulated). A global
    orientation flip cancels in the ``moment / volume`` ratio, so only the
    *relative* consistency of the faces matters, not whether they all point
    in or out. Returns ``None`` when the mesh has no enclosed volume
    (open / planar / zero-volume), in which case the caller should fall back
    to a simpler estimate.
    """
    total_v = 0.0
    cx = cy = cz = 0.0
    for f in faces:
        vids = f.vertex_indices
        if len(vids) < 3:
            continue
        a = vertices[vids[0] - 1]
        for k in range(1, len(vids) - 1):
            b = vertices[vids[k] - 1]
            c = vertices[vids[k + 1] - 1]
            # 6 x signed volume of tetrahedron (origin, a, b, c)
            vol6 = (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            total_v += vol6
            # tetra centroid is (origin + a + b + c) / 4
            cx += vol6 * (a[0] + b[0] + c[0])
            cy += vol6 * (a[1] + b[1] + c[1])
            cz += vol6 * (a[2] + b[2] + c[2])

    if abs(total_v) < 1e-15:
        return None
    # the constant 1/6 (volume) and 1/4 (centroid) factors cancel except for
    # the 1/4 on the moment, which survives as (.../4) / (total_v) -> /4.
    inv = 1.0 / (4.0 * total_v)
    return (cx * inv, cy * inv, cz * inv)


def room_center_from_mesh(mesh: "Mesh") -> tuple[float, float, float]:
    """A reference point inside the room.

    Several repairs ('which side of this plane is the room?', 'do the normals
    point outward?') need an interior reference point. The naive mean of the
    vertices is biased by vertex density and, for non-convex rooms, is easily
    pulled outside the solid.

    We therefore prefer the **centre of mass of the enclosed volume**
    (divergence-theorem centroid), which ignores vertex-density bias and lies
    inside the solid for any star-convex room. Two safety nets:

    * if the mesh encloses no volume (open / planar surface) we fall back to
      the vertex mean;
    * if the computed centroid falls outside the vertex bounding box -- the
      tell-tale sign of an *inconsistently* oriented surface, for which the
      signed-volume integral is meaningless -- we also fall back to the
      vertex mean.

    Note: for strongly non-convex rooms (e.g. an L- or U-shaped plan) no
    single centroid is guaranteed to sit inside the material; a guaranteed
    interior point would require a heavier method (voxel sampling / pole of
    inaccessibility). The volume centroid is a strict improvement over the
    vertex mean and is sufficient for the orientation/PLC heuristics here.
    """
    if not mesh.vertices:
        return (0.0, 0.0, 0.0)

    vertices = [(v.x, v.y, v.z) for v in mesh.vertices]
    centroid = _enclosed_volume_centroid(vertices, mesh.faces)

    if centroid is not None:
        min_x = min(p[0] for p in vertices)
        min_y = min(p[1] for p in vertices)
        min_z = min(p[2] for p in vertices)
        max_x = max(p[0] for p in vertices)
        max_y = max(p[1] for p in vertices)
        max_z = max(p[2] for p in vertices)
        if (
            min_x <= centroid[0] <= max_x
            and min_y <= centroid[1] <= max_y
            and min_z <= centroid[2] <= max_z
        ):
            return centroid

    return _vertex_mean(vertices)


def flip_all_faces_if_majority_inward(
    faces,
    unique_vertices,
    room_center,
    logger=None,
) -> bool:
    """Flip every face when the majority currently point inward.

    For each face, the Newell normal is compared against the vector from the
    face centroid to ``room_center``. An outward normal points away from the
    room, so that dot product is negative -> 'outward'; otherwise 'inward'.
    If inward strictly outnumbers outward, all faces are reversed.

    A tie (``inward == outward``) is deliberately resolved as *keep*: with no
    majority signal there is no reason to prefer a global flip, and keeping
    the input orientation is the least-surprising, idempotent choice. Returns
    ``True`` iff the faces were flipped.
    """
    inward = 0
    outward = 0

    for f in faces:
        vids = f.vertex_indices
        pts = [unique_vertices[i - 1] for i in vids]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)

        nx, ny, nz = newell_normal_from_points(vids, unique_vertices)
        to_center = (room_center[0] - cx, room_center[1] - cy, room_center[2] - cz)
        dotp = nx * to_center[0] + ny * to_center[1] + nz * to_center[2]

        if dotp < 0:
            outward += 1
        else:
            inward += 1

    if inward > outward:
        for f in faces:
            f.vertex_indices.reverse()
        if logger:
            logger.info(
                "[ORIENT] flipped ALL faces (majority inward: inward=%d outward=%d)",
                inward,
                outward,
            )
        return True

    if logger:
        logger.info(
            "[ORIENT] kept orientation (majority outward/tie: inward=%d outward=%d)",
            inward,
            outward,
        )
    return False


__all__ = ["room_center_from_mesh", "flip_all_faces_if_majority_inward"]




def get_or_create_vertex(points: list[tuple[float, float, float]], p: tuple[float, float, float], tol: float = 1e-9) -> int:
    for i, q in enumerate(points, start=1):
        if distance(p, q) <= tol:
            return i
    points.append(p)
    return len(points)


def compact_vertices_and_remove_unused(
    faces: List["Face"],
    points: List[Tuple[float, float, float]],
) -> Tuple[List["Face"], List[Tuple[float, float, float]], bool, Dict[str, Any]]:
    # Mesh-only: require `Face` objects from `app.geometry.ir` with
    # a `vertex_indices` attribute. Legacy `FaceRecord` is no longer
    # accepted here — callers must be migrated to the Mesh IR.
    def _face_vids(face) -> List[int]:
        if not hasattr(face, "vertex_indices"):
            raise TypeError(
                "compact_vertices_and_remove_unused: legacy FaceRecord not accepted; "
                "migrate callers to use app.geometry.ir.Face with `vertex_indices`."
            )
        return list(face.vertex_indices)

    used = sorted({vid for f in faces for vid in _face_vids(f)})
    mapping = {old_vid: new_vid for new_vid, old_vid in enumerate(used, start=1)}
    new_points = [points[old_vid - 1] for old_vid in used]
    changed = len(used) != len(points) or any(old_vid != mapping[old_vid] for old_vid in used)
    new_faces: List["Face"] = []
    for i, f in enumerate(faces):
        vids = [mapping[vid] for vid in _face_vids(f)]
        # Always construct Mesh `Face` instances.
        new_faces.append(
            Face(vertex_indices=vids, group=getattr(f, "group", "default"), material=getattr(f, "material", None))
        )

    diag = {
        "status": "ok",
        "vertices_before": len(points),
        "vertices_after": len(new_points),
        "removed_unused_vertices": len(points) - len(new_points),
    }
    return new_faces, new_points, changed, diag


def _find_face_by_fid(faces: List["Face"], fid: int) -> "Any | None":
    for i, face in enumerate(faces):
        if getattr(face, "fid", None) == fid:
            return face
    return None


def _endpoint_vids_from_edge_t(edge, t, *, t_eps=1e-9):
    u, v = edge

    if abs(t) <= t_eps:
        return u, v
    if abs(t - 1.0) <= t_eps:
        return v, u
    return None, None


# pure-math helpers `polygon_centroid`, `compute_face_unit_normal`,
# and `offset_point_along_vector` are provided by
# app.geometry.geometry_math.geometry_math


def create_or_reuse_boundary_point_on_edge(
    poly_ids,
    poly2d,
    edge_index,
    tseg,
    points,
    *,
    merge_tol_2d=1e-8,
):
    n = len(poly_ids)

    a_vid = poly_ids[edge_index]
    b_vid = poly_ids[(edge_index + 1) % n]

    a2 = poly2d[edge_index]
    b2 = poly2d[(edge_index + 1) % n]

    hit2 = (
        a2[0] + tseg * (b2[0] - a2[0]),
        a2[1] + tseg * (b2[1] - a2[1]),
    )

    da = math.sqrt((hit2[0] - a2[0]) ** 2 + (hit2[1] - a2[1]) ** 2)
    db = math.sqrt((hit2[0] - b2[0]) ** 2 + (hit2[1] - b2[1]) ** 2)

    if da <= merge_tol_2d:
        return a_vid, points, True
    if db <= merge_tol_2d:
        return b_vid, points, True

    a3 = points[a_vid - 1]
    b3 = points[b_vid - 1]
    new_p3 = (
        a3[0] + tseg * (b3[0] - a3[0]),
        a3[1] + tseg * (b3[1] - a3[1]),
        a3[2] + tseg * (b3[2] - a3[2]),
    )

    points = points + [new_p3]
    return len(points), points, False


def move_touching_endpoint_off_face(
    faces,
    points,
    plc_report,
    *,
    offset_m=0.01,
    logger=None,
    t_eps=1e-9,
):
    diag = {
        "status": "noop",
        "touched_vid": None,
        "other_vid": None,
        "facet_fid": plc_report.get("facet_fid"),
        "old_point": None,
        "new_point": None,
        "normal": None,
        "offset_m": offset_m,
        "len_plus": None,
        "len_minus": None,
        "chosen_direction": None,
    }

    if plc_report.get("hit_type") != "endpoint_face_interior_touch":
        diag["status"] = "wrong_hit_type"
        return points, False, diag

    touched_vid, other_vid = _endpoint_vids_from_edge_t(
        plc_report["edge"],
        plc_report["t_param"],
        t_eps=t_eps,
    )

    if touched_vid is None or other_vid is None:
        diag["status"] = "not_an_endpoint_hit"
        return points, False, diag

    diag["touched_vid"] = touched_vid
    diag["other_vid"] = other_vid

    facet_fid = plc_report["facet_fid"]
    touched_face = _find_face_by_fid(faces, facet_fid)

    if touched_face is None:
        diag["status"] = "missing_facet_face"
        return points, False, diag

    # Mesh-only: require `vertex_indices` on the touched face.
    if not hasattr(touched_face, "vertex_indices"):
        raise TypeError(
            "move_touching_endpoint_off_face: legacy FaceRecord not accepted; "
            "migrate callers to use app.geometry.ir.Face with `vertex_indices`."
        )
    tf_verts = list(touched_face.vertex_indices)
    normal = compute_face_unit_normal(tf_verts, points)
    if normal is None:
        diag["status"] = "invalid_face_normal"
        return points, False, diag

    old_point = points[touched_vid - 1]
    other_point = points[other_vid - 1]

    p_plus = offset_point_along_vector(old_point, normal, +offset_m)
    p_minus = offset_point_along_vector(old_point, normal, -offset_m)

    len_plus = distance(p_plus, other_point)
    len_minus = distance(p_minus, other_point)

    diag["old_point"] = old_point
    diag["normal"] = normal
    diag["len_plus"] = len_plus
    diag["len_minus"] = len_minus

    if len_plus < len_minus:
        new_point = p_plus
        chosen = "+normal"
    else:
        new_point = p_minus
        chosen = "-normal"

    points[touched_vid - 1] = new_point

    diag["status"] = "ok"
    diag["new_point"] = new_point
    diag["chosen_direction"] = chosen

    if logger:
        logger.info(
            "[PLC OFFSET] moved vid=%d away from facet_fid=%d by %.6f m "
            "using shorter-edge rule other_vid=%d "
            "len(+n)=%.6f len(-n)=%.6f chosen=%s "
            "old=(%.6f,%.6f,%.6f) new=(%.6f,%.6f,%.6f) normal=(%.6f,%.6f,%.6f)",
            diag["touched_vid"],
            facet_fid,
            offset_m,
            diag["other_vid"],
            diag["len_plus"],
            diag["len_minus"],
            diag["chosen_direction"],
            diag["old_point"][0], diag["old_point"][1], diag["old_point"][2],
            diag["new_point"][0], diag["new_point"][1], diag["new_point"][2],
            diag["normal"][0], diag["normal"][1], diag["normal"][2],
        )

    return points, True, diag


__all__ = ["room_center_from_mesh", "polygon_centroid", "get_or_create_vertex", "compact_vertices_and_remove_unused"]

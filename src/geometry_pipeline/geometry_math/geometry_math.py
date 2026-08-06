"""Vector / polygon math primitives implemented inside geometry_math."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def uedge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(dot(v, v))


def unit(v: Tuple[float, float, float], eps: float = 1e-30) -> Tuple[float, float, float]:
    n = norm(v)
    if n <= eps:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return norm(sub(a, b))


def vadd(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vmul(a: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def aabb_of_tri(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    return (
        min(a[0], b[0], c[0]),
        min(a[1], b[1], c[1]),
        min(a[2], b[2], c[2]),
        max(a[0], b[0], c[0]),
        max(a[1], b[1], c[1]),
        max(a[2], b[2], c[2]),
    )


def aabb_of_seg(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        min(a[2], b[2]),
        max(a[0], b[0]),
        max(a[1], b[1]),
        max(a[2], b[2]),
    )


def aabb_overlap(
    bb1: Tuple[float, float, float, float, float, float],
    bb2: Tuple[float, float, float, float, float, float],
    pad: float = 0.0,
) -> bool:
    ax0, ay0, az0, ax1, ay1, az1 = bb1
    bx0, by0, bz0, bx1, by1, bz1 = bb2
    return not (
        ax1 + pad < bx0
        or bx1 + pad < ax0
        or ay1 + pad < by0
        or by1 + pad < ay0
        or az1 + pad < bz0
        or bz1 + pad < az0
    )


def segment_intersects_triangle(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    c: Tuple[float, float, float],
    eps: float = 1e-12,
):
    """Moller-Trumbore segment-triangle test.

    Returns (hit, t, u, v) where t in [0, 1] parameterizes p0->p1.
    """
    d = sub(p1, p0)
    e1 = sub(b, a)
    e2 = sub(c, a)

    h = cross(d, e2)
    det = dot(e1, h)

    if abs(det) < eps:
        return (False, None, None, None)

    inv_det = 1.0 / det
    s = sub(p0, a)
    u = inv_det * dot(s, h)
    if u < -eps or u > 1.0 + eps:
        return (False, None, None, None)

    q = cross(s, e1)
    v = inv_det * dot(d, q)
    if v < -eps or (u + v) > 1.0 + eps:
        return (False, None, None, None)

    t = inv_det * dot(e2, q)
    if t < -eps or t > 1.0 + eps:
        return (False, None, None, None)

    return (True, t, u, v)


def polygon_area_3d(
    loop_vids: List[int],
    vertices: List[Tuple[float, float, float]],
) -> float:
    """Compute polygon area in 3D using Newell's method.

    `loop_vids` are 1-based vertex ids into `vertices`.
    """
    if len(loop_vids) < 3:
        return 0.0

    nx = ny = nz = 0.0
    n = len(loop_vids)

    for i in range(n):
        p = vertices[loop_vids[i] - 1]
        q = vertices[loop_vids[(i + 1) % n] - 1]

        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])

    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def newell_normal_from_points(face_ids: Iterable[int], points: List[Tuple[float, float, float]]):
    nx = ny = nz = 0.0
    face_ids_list = list(face_ids)
    n = len(face_ids_list)
    for i in range(n):
        p = points[face_ids_list[i] - 1]
        q = points[face_ids_list[(i + 1) % n] - 1]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    return (nx, ny, nz)


def orient(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def area2(poly2d: List[Tuple[float, float]]) -> float:
    s = 0.0
    m = len(poly2d)
    for i in range(m):
        x1, y1 = poly2d[i]
        x2, y2 = poly2d[(i + 1) % m]
        s += x1 * y2 - x2 * y1
    return s


def point_on_segment_2d(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    tol: float = 1e-12,
) -> bool:
    if abs(orient(a, b, p)) > tol:
        return False

    return (
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
    )


def segments_intersect_2d(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
    tol: float = 1e-12,
) -> bool:
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and (
        (o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)
    ):
        return True

    if abs(o1) <= tol and point_on_segment_2d(c, a, b, tol):
        return True
    if abs(o2) <= tol and point_on_segment_2d(d, a, b, tol):
        return True
    if abs(o3) <= tol and point_on_segment_2d(a, c, d, tol):
        return True
    if abs(o4) <= tol and point_on_segment_2d(b, c, d, tol):
        return True

    return False


def tri_area2(
    i: int,
    j: int,
    k: int,
    points: List[Tuple[float, float, float]],
) -> float:
    a = points[i - 1]
    b = points[j - 1]
    c = points[k - 1]
    ab = sub(b, a)
    ac = sub(c, a)
    cr = cross(ab, ac)
    return dot(cr, cr)


def project_point_by_dropped_axis(p3, dropped_axis):
    if dropped_axis == "z":
        return (p3[0], p3[1])
    elif dropped_axis == "y":
        return (p3[0], p3[2])
    else:
        return (p3[1], p3[2])


def _dominant_dropped_axis(face_ids, points):
    nrm = newell_normal_from_points(face_ids, points)
    ax, ay, az = abs(nrm[0]), abs(nrm[1]), abs(nrm[2])

    if az >= ax and az >= ay:
        return "z"
    elif ay >= ax and ay >= az:
        return "y"
    return "x"


def project_face_to_2d(face_ids, points):
    dropped_axis = _dominant_dropped_axis(face_ids, points)
    poly2d = [project_point_by_dropped_axis(points[pid - 1], dropped_axis) for pid in face_ids]
    return poly2d, dropped_axis


def project_vid_to_2d(vid, points, dropped_axis):
    x, y, z = points[vid - 1]
    if dropped_axis == "z":
        return (x, y)
    elif dropped_axis == "y":
        return (x, z)
    else:
        return (y, z)


def project_face_and_point_to_2d(face_ids, point_vid, points):
    """Project a face and a single point to 2D using dominant-axis projection.

    Returns (poly2d, p2, dropped_axis)
    """
    poly2d, dropped_axis = project_face_to_2d(face_ids, points)
    p2 = project_vid_to_2d(point_vid, points, dropped_axis)
    return poly2d, p2, dropped_axis


def polygon_area2_newell(face_ids, points):
    """Squared norm of Newell normal (area proxy)."""
    nrm = newell_normal_from_points(face_ids, points)
    return dot(nrm, nrm)


def polygon_centroid(
    loop_vids: List[int],
    points: List[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    if not loop_vids:
        raise ValueError("polygon_centroid(): empty vertex loop")
    sx = sy = sz = 0.0
    n = len(loop_vids)
    for vid in loop_vids:
        p = points[vid - 1]
        sx += p[0]
        sy += p[1]
        sz += p[2]
    return (sx / n, sy / n, sz / n)


def compute_face_unit_normal(face_verts, points, *, eps=1e-30):
    nx, ny, nz = newell_normal_from_points(face_verts, points)
    nlen = math.sqrt(nx * nx + ny * ny + nz * nz)

    if nlen <= eps:
        return None

    return (nx / nlen, ny / nlen, nz / nlen)


def offset_point_along_vector(point, direction, distance_val):
    return (
        point[0] + direction[0] * distance_val,
        point[1] + direction[1] * distance_val,
        point[2] + direction[2] * distance_val,
    )


def point_in_polygon_2d(poly, p, tol=1e-12):
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if point_on_segment_2d(p, a, b, tol):
            return "boundary"

    x, y = p
    inside = False

    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        crosses = (y1 > y) != (y2 > y)
        if crosses:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xinters:
                inside = not inside

    return "inside" if inside else "outside"


def point_segment_distance_2d(p, a, b):
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    apx = p[0] - a[0]
    apy = p[1] - a[1]

    ab2 = abx * abx + aby * aby
    if ab2 <= 0.0:
        dx = p[0] - a[0]
        dy = p[1] - a[1]
        return math.sqrt(dx * dx + dy * dy)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))

    qx = a[0] + t * abx
    qy = a[1] + t * aby

    dx = p[0] - qx
    dy = p[1] - qy
    return math.sqrt(dx * dx + dy * dy)


__all__ = [
    "aabb_of_seg",
    "aabb_of_tri",
    "aabb_overlap",
    "area2",
    "cross",
    "dot",
    "distance",
    "newell_normal_from_points",
    "norm",
    "orient",
    "polygon_area_3d",
    "point_on_segment_2d",
    "segment_intersects_triangle",
    "segments_intersect_2d",
    "sub",
    "tri_area2",
    "project_point_by_dropped_axis",
    "_dominant_dropped_axis",
    "project_face_to_2d",
    "project_vid_to_2d",
    "point_in_polygon_2d",
    "point_segment_distance_2d",
    "uedge",
    "unit",
    "vadd",
    "vmul",
    "polygon_centroid",
    "compute_face_unit_normal",
    "offset_point_along_vector",
    "project_face_and_point_to_2d",
    "polygon_area2_newell",
]

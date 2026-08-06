"""Face classification helpers (degeneracy, planarity)."""

from __future__ import annotations

import math
from typing import List, Tuple


def classify_face_degeneracy(
    verts_ids: List[int],
    points: List[Tuple[float, float, float]],
    *,
    fatal_area_tol: float = 1e-12,
    min_altitude_tol: float = 0.0,
) -> Tuple[str, float]:
    """Classify polygon degeneracy using Newell area.

    A face is "fatal" when either:
      * its squared Newell area is <= ``fatal_area_tol`` (collapsed / zero
        area), or
      * ``min_altitude_tol > 0`` and its thinness (2*area / longest_edge,
        i.e. the smallest altitude) falls below ``min_altitude_tol``. This
        catches *slivers*: near-collinear faces that still have a small but
        non-zero area and so slip past the area-only test.
    """
    if len(verts_ids) < 3:
        return "fatal", 0.0

    unique = []
    seen = set()
    for vid in verts_ids:
        if vid not in seen:
            unique.append(vid)
            seen.add(vid)

    if len(unique) < 3:
        return "fatal", 0.0

    nx = ny = nz = 0.0
    n = len(unique)

    for i in range(n):
        p = points[unique[i] - 1]
        q = points[unique[(i + 1) % n] - 1]

        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])

    area2 = nx * nx + ny * ny + nz * nz

    if area2 <= fatal_area_tol:
        return "fatal", area2

    if min_altitude_tol > 0.0:
        # Smallest altitude = 2*area / longest_edge = |Newell normal| / Lmax.
        twice_area = math.sqrt(area2)
        max_edge_len = 0.0
        for i in range(n):
            p = points[unique[i] - 1]
            q = points[unique[(i + 1) % n] - 1]
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            dz = p[2] - q[2]
            edge_len = math.sqrt(dx * dx + dy * dy + dz * dz)
            if edge_len > max_edge_len:
                max_edge_len = edge_len

        if max_edge_len > 0.0:
            min_altitude = twice_area / max_edge_len
            if min_altitude < min_altitude_tol:
                return "fatal", area2

    return "ok", area2


def planarity_deviation_m(face_ids, points, *, eps=1e-30):
    """Returns: (max_abs_dist_m, rms_dist_m, normal_unit, plane_point_centroid)."""
    n = len(face_ids)
    if n < 3:
        return (float("inf"), float("inf"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    nx = ny = nz = 0.0
    for i in range(n):
        p = points[face_ids[i] - 1]
        q = points[face_ids[(i + 1) % n] - 1]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])

    nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nlen <= eps:
        return (float("inf"), float("inf"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    nu = (nx / nlen, ny / nlen, nz / nlen)

    cx = cy = cz = 0.0
    for vid in face_ids:
        x, y, z = points[vid - 1]
        cx += x
        cy += y
        cz += z
    inv = 1.0 / n
    c = (cx * inv, cy * inv, cz * inv)

    max_abs = 0.0
    s2 = 0.0
    for vid in face_ids:
        x, y, z = points[vid - 1]
        dx = x - c[0]
        dy = y - c[1]
        dz = z - c[2]
        d = nu[0] * dx + nu[1] * dy + nu[2] * dz
        ad = abs(d)
        if ad > max_abs:
            max_abs = ad
        s2 += d * d

    rms = math.sqrt(s2 / n)
    return (max_abs, rms, nu, c)


def classify_face_planarity_m(
    face_ids,
    points,
    *,
    warn_planar_tol_m=1e-4,
    fatal_planar_tol_m=1e-3,
):
    n = len(face_ids)
    if n < 3:
        return ("fatal", float("inf"), float("inf"))
    if n == 3:
        return ("skip", 0.0, 0.0)

    max_abs, rms, _, _ = planarity_deviation_m(face_ids, points)
    if not math.isfinite(max_abs):
        return ("fatal", max_abs, rms)
    if max_abs > fatal_planar_tol_m:
        return ("fatal", max_abs, rms)
    if max_abs > warn_planar_tol_m:
        return ("warning", max_abs, rms)
    return ("ok", max_abs, rms)


__all__ = [
    "classify_face_degeneracy",
    "classify_face_planarity_m",
    "planarity_deviation_m",
]

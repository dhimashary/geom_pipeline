"""Constrained Delaunay triangulation helper implemented in geometry_math."""

from __future__ import annotations

from typing import List, Tuple

try:
    from shapely import constrained_delaunay_triangles  # type: ignore[import-untyped]
    from shapely.geometry import Polygon  # type: ignore[import-untyped]
except Exception:
    Polygon = None  # type: ignore
    constrained_delaunay_triangles = None  # type: ignore

from geometry_pipeline.geometry_math.geometry_math import area2, newell_normal_from_points


def triangulate_face_cdt_shapely(
    face: List[int],
    points: List[Tuple[float, float, float]],
    *,
    tol: float = 1e-12,
    round_ndigits: int = 12,
) -> List[List[int]]:
    """Constrained Delaunay triangulation of a polygon face using Shapely 2.1+."""
    if constrained_delaunay_triangles is None or Polygon is None:
        raise ImportError("Shapely constrained Delaunay triangulation not available")

    if len(face) < 3:
        return []
    if len(face) == 3:
        return [face[:]]

    nrm = newell_normal_from_points(face, points)
    ax, ay, az = abs(nrm[0]), abs(nrm[1]), abs(nrm[2])

    if az >= ax and az >= ay:

        def proj(pid):
            return (points[pid - 1][0], points[pid - 1][1])

    elif ay >= ax and ay >= az:

        def proj(pid):
            return (points[pid - 1][0], points[pid - 1][2])

    else:

        def proj(pid):
            return (points[pid - 1][1], points[pid - 1][2])

    face_ids = face[:]
    poly2d = [proj(pid) for pid in face_ids]

    if area2(poly2d) < 0:
        face_ids.reverse()
        poly2d.reverse()

    cleaned_ids = [face_ids[0]]
    cleaned_2d = [poly2d[0]]
    for pid, p2 in zip(face_ids[1:], poly2d[1:]):
        if abs(p2[0] - cleaned_2d[-1][0]) <= tol and abs(p2[1] - cleaned_2d[-1][1]) <= tol:
            continue
        cleaned_ids.append(pid)
        cleaned_2d.append(p2)

    if len(cleaned_2d) >= 2:
        if (
            abs(cleaned_2d[0][0] - cleaned_2d[-1][0]) <= tol
            and abs(cleaned_2d[0][1] - cleaned_2d[-1][1]) <= tol
        ):
            cleaned_ids.pop()
            cleaned_2d.pop()

    if len(cleaned_ids) < 3:
        return []
    if len(cleaned_ids) == 3:
        return [cleaned_ids]

    def key(x, y):
        return (round(x, round_ndigits), round(y, round_ndigits))

    coord_to_vid = {}
    for vid, (x, y) in zip(cleaned_ids, cleaned_2d):
        coord_to_vid[key(x, y)] = vid

    poly = Polygon(cleaned_2d)

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty or (not poly.is_valid):
        return []

    tris_geom = constrained_delaunay_triangles(poly)
    geoms = getattr(tris_geom, "geoms", [])
    if not geoms:
        return []

    out: List[List[int]] = []
    for tri in geoms:
        coords = list(tri.exterior.coords)
        if len(coords) < 4:
            continue
        coords = coords[:-1]
        if len(coords) != 3:
            continue

        vids = []
        ok = True
        for x, y in coords:
            vid = coord_to_vid.get(key(x, y))  # type: ignore[assignment]
            if vid is None:
                ok = False
                break
            vids.append(vid)

        if not ok or len(set(vids)) < 3:
            continue
        out.append(vids)

    return out


__all__ = ["triangulate_face_cdt_shapely"]

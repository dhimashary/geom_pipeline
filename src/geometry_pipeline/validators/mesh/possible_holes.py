"""Validator: detects closed boundary loops (candidate holes in the surface)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh, Face
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.core.tolerances import Tolerances
from geometry_pipeline.geometry_math.geometry_math import (
    compute_face_unit_normal,
    newell_normal_from_points,
    point_in_polygon_2d,
    project_point_by_dropped_axis,
    uedge,
)
from geometry_pipeline.validators.base import BaseValidator

# Default plane tolerance (metres) for deciding whether a face is coplanar with
# a candidate loop when testing if the loop interior is filled by surface.
_COPLANAR_TOL_M = Tolerances().hole_fill_coplanar_m

# When separating the room shell from interior objects, each mesh component is
# tested for containment only against this many of the largest components. A
# room fragmented into a few big panels is still covered, while a badly
# fragmented mesh stays O(C) instead of O(C^2).
_MAX_CONTAINER_CANDIDATES = 16


def _loop_dropped_axis(
    loop_vids: List[int],
    unique_vertices: List[Tuple[float, float, float]],
) -> str:
    nx, ny, nz = newell_normal_from_points(loop_vids, unique_vertices)
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if az >= ax and az >= ay:
        return "z"
    if ay >= ax and ay >= az:
        return "y"
    return "x"


def _interior_point_2d(
    loop2d: List[Tuple[float, float]],
) -> Tuple[float, float] | None:
    """A point guaranteed to lie inside the (simple) 2D loop.

    The centroid works for convex loops; for non-convex ones it may fall
    outside, so we fall back to fan-triangle centroids until one lands inside.
    """
    n = len(loop2d)
    if n < 3:
        return None

    cx = sum(p[0] for p in loop2d) / n
    cy = sum(p[1] for p in loop2d) / n
    if point_in_polygon_2d(loop2d, (cx, cy)) != "outside":
        return (cx, cy)

    a = loop2d[0]
    for i in range(1, n - 1):
        b = loop2d[i]
        c = loop2d[i + 1]
        tc = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)
        if point_in_polygon_2d(loop2d, tc) != "outside":
            return tc
    return (cx, cy)


def _loop_interior_is_filled(
    loop_vids: List[int],
    faces: List[Face],
    unique_vertices: List[Tuple[float, float, float]],
    *,
    coplanar_tol_m: float = _COPLANAR_TOL_M,
) -> bool:
    """True when the region enclosed by the loop is covered by coplanar surface.

    This is the real distinction between a *hole* and the *outer rim of an open
    surface* (e.g. a table top): a hole's interior is empty, whereas an open
    surface fills its own perimeter. We test the enclosed *region* directly, so
    the result does not depend on how the boundary edges happen to be split
    among the surrounding faces — a single face spanning several loop edges, or
    a corner-capping triangle, cannot skew it.
    """
    if len(loop_vids) < 3:
        return False

    dropped = _loop_dropped_axis(loop_vids, unique_vertices)
    unit_n = compute_face_unit_normal(loop_vids, unique_vertices)
    if unit_n is None:
        return False

    loop2d = [project_point_by_dropped_axis(unique_vertices[v - 1], dropped) for v in loop_vids]
    interior = _interior_point_2d(loop2d)
    if interior is None:
        return False

    p0 = unique_vertices[loop_vids[0] - 1]

    for f in faces:
        vids = list(getattr(f, "vertex_indices", []))
        if len(vids) < 3:
            continue
        # The face must lie (near) in the loop's plane to count as a fill;
        # otherwise a parallel surface elsewhere in 3D could overlap in 2D.
        if any(
            abs(
                (unique_vertices[v - 1][0] - p0[0]) * unit_n[0]
                + (unique_vertices[v - 1][1] - p0[1]) * unit_n[1]
                + (unique_vertices[v - 1][2] - p0[2]) * unit_n[2]
            )
            > coplanar_tol_m
            for v in vids
        ):
            continue
        face2d = [project_point_by_dropped_axis(unique_vertices[v - 1], dropped) for v in vids]
        if point_in_polygon_2d(face2d, interior) != "outside":
            return True

    return False


def _face_components(
    faces: List[Face],
    edge_to_faces: Dict[Tuple[int, int], List[int]],
) -> List[int]:
    """Label every face with the id of the connected component it belongs to.

    Two faces are connected when they share an edge. The result is a list
    ``comp`` such that ``comp[fid]`` is the component id of face ``fid``. A
    room shell and a free-standing object inside it (furniture) end up as
    separate components because they share no edges.
    """
    parent = list(range(len(faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for adj_faces in edge_to_faces.values():
        first = adj_faces[0]
        for other in adj_faces[1:]:
            union(first, other)

    return [find(fid) for fid in range(len(faces))]


def _component_aabb(
    face_ids: Set[int],
    faces: List[Face],
    unique_vertices: List[Tuple[float, float, float]],
) -> Tuple[float, float, float, float, float, float] | None:
    """Axis-aligned bounding box (minx, miny, minz, maxx, maxy, maxz)."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    seen = False
    for fid in face_ids:
        for v in getattr(faces[fid], "vertex_indices", []):
            p = unique_vertices[v - 1]
            seen = True
            for i in range(3):
                if p[i] < lo[i]:
                    lo[i] = p[i]
                if p[i] > hi[i]:
                    hi[i] = p[i]
    if not seen:
        return None
    return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])


def _aabb_contains(
    outer: Tuple[float, float, float, float, float, float],
    inner: Tuple[float, float, float, float, float, float],
    *,
    tol: float,
) -> bool:
    """True when ``inner`` lies within ``outer`` (padded by ``tol``)."""
    return all(outer[i] - tol <= inner[i] for i in range(3)) and all(
        inner[i + 3] <= outer[i + 3] + tol for i in range(3)
    )


def _room_face_ids(
    faces: List[Face],
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    unique_vertices: List[Tuple[float, float, float]],
    *,
    containment_tol_m: float,
) -> Set[int]:
    """Return the faces that belong to the room shell rather than to objects.

    The room is the enclosing structure (walls, ceiling, floor); furniture and
    other props are smaller meshes that sit *inside* it. We separate faces into
    connected components and treat a component as an *interior object* when its
    bounding box is contained within a strictly larger component's box. Every
    component that is not contained in another (the room shell, and any other
    top-level structure) is kept. Boundary loops on interior objects are then
    ignored, so a hole in a chair or table is never reported — only holes in the
    room itself are.

    A badly fragmented mesh can produce a huge number of components. To stay
    linear we never compare every component against every other (which would be
    O(C^2)); instead we sort components by size once and test each one for
    containment only against the few largest components. A fragmented room may
    be split into several large panels, so testing against the top-K largest
    (rather than just the single biggest) keeps the room-vs-object split correct
    while remaining O(C * K) ~= O(C).
    """
    comp_of = _face_components(faces, edge_to_faces)

    members: Dict[int, Set[int]] = defaultdict(set)
    for fid, cid in enumerate(comp_of):
        members[cid].add(fid)

    # A single connected mesh (typical clean room) is entirely the room.
    if len(members) <= 1:
        return set(range(len(faces)))

    boxes: Dict[int, Tuple[float, float, float, float, float, float]] = {}
    diag2: Dict[int, float] = {}
    for cid, fids in members.items():
        box = _component_aabb(fids, faces, unique_vertices)
        if box is None:
            continue
        boxes[cid] = box
        diag2[cid] = sum((box[i + 3] - box[i]) ** 2 for i in range(3))

    # Largest-first; only a strictly larger component can enclose another, so
    # each component is tested for containment against the handful of biggest
    # candidates rather than against all others.
    order = sorted(boxes, key=lambda c: diag2[c], reverse=True)
    candidates = order[:_MAX_CONTAINER_CANDIDATES]

    interior: Set[int] = set()
    for cid in order:
        box = boxes[cid]
        for other_cid in candidates:
            if other_cid == cid:
                continue
            # Strictly larger only, so two equal-size shells are never treated
            # as inside one another.
            if diag2[other_cid] <= diag2[cid]:
                continue
            if _aabb_contains(boxes[other_cid], box, tol=containment_tol_m):
                interior.add(cid)
                break

    room_face_ids: Set[int] = set()
    for fid, cid in enumerate(comp_of):
        if cid not in interior:
            room_face_ids.add(fid)
    return room_face_ids


def detect_possible_holes_from_faces(
    faces: List[Face],
    unique_vertices: List[Tuple[float, float, float]],
    *,
    coplanar_tol_m: float = _COPLANAR_TOL_M,
    containment_tol_m: float = 0.0,
) -> List[Dict[str, Any]]:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for fid, f in enumerate(faces):
        vids = list(getattr(f, "vertex_indices", []))
        n = len(vids)
        if n < 2:
            continue
        for i in range(n):
            a = vids[i]
            b = vids[(i + 1) % n]
            edge_to_faces[uedge(a, b)].append(fid)

    boundary_edges = {e for e, adj in edge_to_faces.items() if len(adj) == 1}
    if not boundary_edges:
        return []

    # Faces that belong to the room shell. Boundary loops on interior objects
    # (furniture) are not holes we care about, so we drop them up front.
    room_face_ids = _room_face_ids(
        faces,
        edge_to_faces,
        unique_vertices,
        containment_tol_m=containment_tol_m,
    )
    boundary_edges = {
        e for e in boundary_edges if edge_to_faces[e][0] in room_face_ids
    }
    if not boundary_edges:
        return []

    adj: Dict[int, Set[int]] = defaultdict(set)
    for a, b in boundary_edges:
        adj[a].add(b)
        adj[b].add(a)

    visited: Set[int] = set()
    components: List[Set[int]] = []
    for v in adj:
        if v in visited:
            continue
        component: Set[int] = set()
        stack = [v]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                component.add(curr)
                stack.extend(adj[curr] - visited)
        components.append(component)

    loops: List[Dict[str, Any]] = []
    for comp in components:
        if not all(len(adj[v]) == 2 for v in comp):
            continue

        comp_boundary_edges = [e for e in boundary_edges if e[0] in comp and e[1] in comp]
        if not comp_boundary_edges:
            continue

        start_v = min(comp)
        edge_loop = []
        loop_vids = [start_v]
        prev_v = start_v
        cur_v = next(iter(adj[start_v]))
        while cur_v != start_v:
            edge_loop.append((unique_vertices[prev_v - 1], unique_vertices[cur_v - 1]))
            loop_vids.append(cur_v)
            neighbors = adj[cur_v]
            next_v = next(n for n in neighbors if n != prev_v)
            prev_v = cur_v
            cur_v = next_v
        edge_loop.append((unique_vertices[prev_v - 1], unique_vertices[cur_v - 1]))

        # Skip outer rims of open/floating surfaces: if the region enclosed by
        # the loop is filled by coplanar surface it is the perimeter of a sheet
        # (e.g. a table top), not a hole.
        if _loop_interior_is_filled(loop_vids, faces, unique_vertices, coplanar_tol_m=coplanar_tol_m):
            continue

        elements = [{"type": "edge", "points": [list(edge[0]), list(edge[1])]} for edge in edge_loop]
        loops.append({"elements": elements})

    return loops


class PossibleHolesValidator(BaseValidator):
    name: ClassVar[str] = "possible_holes"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.POSSIBLE_HOLE

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        return detect_possible_holes_from_faces(
            list(geom.faces),
            points,
            coplanar_tol_m=ctx.tolerances.hole_fill_coplanar_m,
            containment_tol_m=ctx.tolerances.hole_object_containment_m,
        )

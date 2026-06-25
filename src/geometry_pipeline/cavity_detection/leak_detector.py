"""Voxel flood-fill leak (open-hole) detector.

A surface mesh that should enclose a room "leaks" when the interior air is
connected to the exterior air through a gap in the surface (a missing face, an
unwelded seam, a penetrating wall, ...). Loop-based hole detection is fragile
on folded sheets, T-junctions and unwelded meshes; this detector instead asks
a purely volumetric question:

    Can outside air reach the room interior?

Approach
--------
1. Triangulate the polygon faces and voxelize the surface (reusing the same
   machinery as the cavity detector) into an occupancy grid.
2. Flood-fill the *empty* voxels. After padding the grid by one voxel, the
   label touching the padded corner is "outside".
3. Pick an interior seed (the mesh centroid, nudged to the nearest empty
   voxel). If its label equals the outside label, outside air reached the
   interior — the surface leaks.
4. Localise: BFS the empty-voxel graph from the interior seed to the grid
   border and report the *bottleneck* of that path (the voxel most surrounded
   by wall), which is where the air squeezes through. Snap it to the nearest
   mesh vertex.
5. Multi-hole: plug a small ball at the bottleneck and re-flood; repeat until
   the interior seals or ``max_leaks`` is reached, so every distinct opening is
   reported once.

Accuracy is bound by ``pitch`` — walls thinner than ``pitch`` may be voxelised
shut (false negative) and sub-pitch gaps may be bridged. Use a ``pitch``
smaller than the smallest opening you care about.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging

import numpy as np

from geometry_pipeline.cavity_detection.cavity_detector import (
    _triangulate_face_with_fallback,
)

logger = logging.getLogger(__name__)

_NEIGHBORS_6 = (
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
)


def _build_occupancy(
    faces: Sequence,
    unique_vertices: Sequence[Tuple[float, float, float]],
    *,
    pitch: float,
    closing_iterations: int,
):
    """Triangulate + voxelize. Returns (occ, origin, shape) or None."""
    import trimesh
    from scipy import ndimage

    pts_arr = np.asarray(unique_vertices, dtype=float)
    tris: List[Tuple[int, int, int]] = []
    for f in faces:
        for tri in _triangulate_face_with_fallback(list(f.vertex_indices), unique_vertices):
            tris.append((tri[0] - 1, tri[1] - 1, tri[2] - 1))
    if not tris:
        return None

    mesh = trimesh.Trimesh(
        vertices=pts_arr, faces=np.asarray(tris, dtype=np.int64), process=False
    )
    vox = mesh.voxelized(pitch=pitch)
    occ = np.asarray(vox.matrix, dtype=bool)
    if closing_iterations > 0:
        occ = ndimage.binary_closing(occ, iterations=closing_iterations)

    transform = np.asarray(vox.transform, dtype=float)
    origin = transform[:3, 3]
    return occ, origin, np.asarray(occ.shape)


def _interior_seed_voxel(
    occ: np.ndarray,
    empty: np.ndarray,
    shape: np.ndarray,
    iv: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Return iv if empty, else the nearest empty voxel (expanding box search)."""
    if empty[iv]:
        return iv
    for r in range(1, max(shape) + 1):
        x0, x1 = max(0, iv[0] - r), min(shape[0], iv[0] + r + 1)
        y0, y1 = max(0, iv[1] - r), min(shape[1], iv[1] + r + 1)
        z0, z1 = max(0, iv[2] - r), min(shape[2], iv[2] + r + 1)
        sub = empty[x0:x1, y0:y1, z0:z1]
        if sub.any():
            local = np.argwhere(sub)[0]
            return (int(x0 + local[0]), int(y0 + local[1]), int(z0 + local[2]))
    return iv


def _wall_neighbor_count(occ: np.ndarray, shape: np.ndarray, c: Tuple[int, int, int]) -> int:
    cnt = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nc = (c[0] + dx, c[1] + dy, c[2] + dz)
                if 0 <= nc[0] < shape[0] and 0 <= nc[1] < shape[1] and 0 <= nc[2] < shape[2]:
                    if occ[nc]:
                        cnt += 1
    return cnt


def _trace_leak_bottleneck(
    occ: np.ndarray,
    empty: np.ndarray,
    shape: np.ndarray,
    start: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int]]:
    """BFS from interior seed to any border empty voxel; return the path voxel
    most surrounded by wall (the squeeze-through point)."""
    sx, sy, sz = int(shape[0]), int(shape[1]), int(shape[2])
    visited = np.zeros((sx, sy, sz), dtype=bool)
    prev: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    dq = deque([start])
    visited[start] = True
    target: Optional[Tuple[int, int, int]] = None

    while dq:
        c = dq.popleft()
        if (c[0] == 0 or c[1] == 0 or c[2] == 0
                or c[0] == sx - 1 or c[1] == sy - 1 or c[2] == sz - 1):
            target = c
            break
        for dx, dy, dz in _NEIGHBORS_6:
            nc = (c[0] + dx, c[1] + dy, c[2] + dz)
            if 0 <= nc[0] < sx and 0 <= nc[1] < sy and 0 <= nc[2] < sz:
                if not visited[nc] and empty[nc]:
                    visited[nc] = True
                    prev[nc] = c
                    dq.append(nc)

    if target is None:
        return None

    path = [target]
    while path[-1] != start:
        path.append(prev[path[-1]])

    return max(path, key=lambda c: _wall_neighbor_count(occ, shape, c))


def detect_leaks_flood_fill(
    faces: Sequence,
    unique_vertices: Sequence[Tuple[float, float, float]],
    *,
    pitch: float = 0.05,
    closing_iterations: int = 0,
    interior_seed: Optional[Sequence[float]] = None,
    max_leaks: int = 25,
    merge_distance_m: float = 0.6,
) -> List[Dict[str, Any]]:
    """Detect open holes (leaks) by voxel flood-fill.

    Returns a list of leak dicts, one per distinct opening::

        {
            "world": [x, y, z],            # leak centre (voxel-snapped)
            "nearest_vertex_id": int,      # 1-based id of closest mesh vertex
            "nearest_vertex_xyz": [x,y,z],
            "wall_neighbors": int,         # how pinched the throat is (0..26)
        }

    An empty list means the surface is watertight at this ``pitch`` (no outside
    air reaches the interior).
    """
    from scipy import ndimage

    if not faces or not unique_vertices:
        return []

    built = _build_occupancy(
        faces, unique_vertices, pitch=pitch, closing_iterations=closing_iterations
    )
    if built is None:
        return []
    occ, origin, shape = built
    pts_arr = np.asarray(unique_vertices, dtype=float)

    if interior_seed is None:
        interior_seed = pts_arr.mean(axis=0)
    seed_idx = np.floor((np.asarray(interior_seed, float) - origin) / pitch).astype(int)
    seed_idx = tuple(int(v) for v in np.clip(seed_idx, 0, shape - 1))

    def voxel_to_world(idx) -> np.ndarray:
        return origin + (np.asarray(idx) + 0.5) * pitch

    plug_r = max(1, int(round(0.10 / pitch)))  # ~10 cm plug per iteration
    occ = occ.copy()

    # ---- Pass 1: enumerate every squeeze-through point ----------------------
    # A single wide opening is plugged ~10 cm at a time, so it yields a chain of
    # bottlenecks; we cluster them below into one report per opening.
    raw: List[Tuple[np.ndarray, int]] = []  # (world, wall_neighbors)
    for _ in range(max_leaks):
        empty = ~occ
        empty_padded = np.pad(empty, 1, constant_values=True)
        labeled_padded, _n = ndimage.label(empty_padded)
        outside_label = int(labeled_padded[0, 0, 0])
        labeled = labeled_padded[1:-1, 1:-1, 1:-1]

        iv = _interior_seed_voxel(occ, empty, shape, seed_idx)
        if int(labeled[iv]) != outside_label:
            break  # interior sealed -> no (more) leaks

        bottleneck = _trace_leak_bottleneck(occ, empty, shape, iv)
        if bottleneck is None:
            break

        raw.append((voxel_to_world(bottleneck), _wall_neighbor_count(occ, shape, bottleneck)))

        # Plug a ball around the bottleneck so the next pass advances rather than
        # rerouting through the same spot.
        bx, by, bz = bottleneck
        x0, x1 = max(0, bx - plug_r), min(shape[0], bx + plug_r + 1)
        y0, y1 = max(0, by - plug_r), min(shape[1], by + plug_r + 1)
        z0, z1 = max(0, bz - plug_r), min(shape[2], bz + plug_r + 1)
        occ[x0:x1, y0:y1, z0:z1] = True

    if not raw:
        return []

    # ---- Pass 2: single-linkage cluster the points into openings -----------
    n = len(raw)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(raw[i][0] - raw[j][0])) <= merge_distance_m:
                parent[find(i)] = find(j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    leaks: List[Dict[str, Any]] = []
    for members in clusters.values():
        # Representative = the most pinched point of the opening.
        rep = max(members, key=lambda i: raw[i][1])
        world = raw[rep][0]
        d2 = np.sum((pts_arr - world) ** 2, axis=1)
        nearest_vid = int(np.argmin(d2)) + 1
        leaks.append(
            {
                "world": [float(x) for x in world],
                "nearest_vertex_id": nearest_vid,
                "nearest_vertex_xyz": [float(x) for x in pts_arr[nearest_vid - 1]],
                "wall_neighbors": int(raw[rep][1]),
            }
        )

    leaks.sort(key=lambda l: -l["wall_neighbors"])
    return leaks

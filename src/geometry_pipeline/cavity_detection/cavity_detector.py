"""Voxel-based enclosed-cavity detector.

Given a list of polygon faces (FaceRecord-compatible) and unique vertices,
detect enclosed empty regions (cavities) inside the geometry — robust to
non-manifold input — and return one `Cavity` per detected pocket with
oriented face references suitable for the GEO exporter.

Approach
--------
1. Triangulate input polygons (fan triangulation; assumes planar / convex).
2. Voxelize the surface (rasterize triangles into a regular grid).
3. Pad the grid by 1 voxel; flood-fill / label empty voxels.
4. The label at any padded-corner voxel is "outside"; every other label is
   an enclosed cavity.
5. For each original face, probe `centroid ± normal * offset` to determine
   which cavity sits on which side, and emit `(face_index, sign)` with
   sign convention: +1 if the face normal points *out of* the cavity
   (Gmsh's outward-loop convention), -1 otherwise.

Caveats: see /memories notes — accuracy is bound by `pitch`; thin walls may
leak; orientation probe is heuristic. Good enough as a first pass for
acoustic enclosure detection.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

from geometry_pipeline.core.ir import Cavity

try:
    # Prefer constrained Delaunay triangulation implemented in the geometry
    # kernel; import lazily to avoid requiring Shapely unless detection runs.
    from geometry_pipeline.geometry_math.triangulation import triangulate_face_cdt_shapely
except Exception:
    triangulate_face_cdt_shapely = None  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def _polygon_centroid_and_normal(
    verts_1based: Sequence[int],
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (centroid, unit-normal) for a polygon. Newell's method for normal."""
    pts = np.array([points[i - 1] for i in verts_1based], dtype=float)
    centroid = pts.mean(axis=0)
    n = np.zeros(3)
    m = len(pts)
    for i in range(m):
        a = pts[i]
        b = pts[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return centroid, np.array([0.0, 0.0, 1.0])
    return centroid, n / norm


def _fan_triangulate(verts: Sequence[int]) -> List[Tuple[int, int, int]]:
    if len(verts) < 3:
        return []
    return [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]


def _triangulate_face_with_fallback(
    verts: Sequence[int], points: Sequence[Tuple[float, float, float]]
):
    """Triangulate a polygon face using CDT if available, otherwise fall back
    to a simple fan triangulation. Returns a list of triangles as 1-based
    vertex id triples.
    """
    if triangulate_face_cdt_shapely is not None:
        try:
            tris = triangulate_face_cdt_shapely(list(verts), list(points))
            if tris:
                return tris
        except Exception:
            # Fall through to fan triangulation on any error
            logger.debug(
                "CDT triangulation failed for a face; falling back to fan triangulation",
                exc_info=True,
            )
    return _fan_triangulate(verts)


def detect_cavities(
    faces: Sequence,
    unique_vertices: Sequence[Tuple[float, float, float]],
    *,
    pitch: float = 0.05,
    closing_iterations: int = 0,
    name_largest_as_room: bool = True,
    min_cavity_voxels: int = 1,
) -> List[Cavity]:
    """Detect enclosed cavities. Returns a list of `Cavity` sorted by
    descending volume.

    Parameters
    ----------
    faces : list of FaceRecord-like objects with `.verts` (1-based vertex ids)
    unique_vertices : list of (x, y, z)
    pitch : voxel size in model units. Must be smaller than the smallest
        feature/wall thickness you care about.
    closing_iterations : optional binary closing iterations to bridge sub-pitch
        gaps (snaps gaps ≤ ~pitch * iters). Use 0 to disable.
    name_largest_as_room : name the largest cavity "Room"; others "Cavity_2"...
    min_cavity_voxels : ignore cavities below this voxel count (noise filter).
    """
    import trimesh
    from scipy import ndimage  # type: ignore[import-untyped]

    if not faces or not unique_vertices:
        return []

    logger.warning(
        "Cavity detection is an experimental feature; results may be inaccurate and should be verified visually. Use `cavity_pitch` to adjust voxel size and `cavity_closing_iterations` to bridge small gaps."
    )
    # ---- 1. Triangulate (fan) and build a trimesh ---------------------------
    pts_arr = np.asarray(unique_vertices, dtype=float)
    tris: List[Tuple[int, int, int]] = []
    tri_to_face: List[int] = []  # parallel: tri index -> original face index
    for fi, f in enumerate(faces):
        # Use CDT where possible to respect polygon shape and avoid
        # creating skinny triangles from arbitrary polygons.
        tris_for_face = _triangulate_face_with_fallback(list(f.vertex_indices), unique_vertices)
        for tri in tris_for_face:
            tris.append((tri[0] - 1, tri[1] - 1, tri[2] - 1))  # 0-based
            tri_to_face.append(fi)

    if not tris:
        return []

    tris_arr = np.asarray(tris, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=pts_arr, faces=tris_arr, process=False)
    logger.warning(
        "Triangulated mesh for cavity detection: %d faces -> %d triangles", len(faces), len(tris)
    )

    # ---- 2. Voxelize surface ------------------------------------------------
    vox = mesh.voxelized(pitch=pitch)
    occ = np.asarray(vox.matrix, dtype=bool)

    if closing_iterations > 0:
        occ = ndimage.binary_closing(occ, iterations=closing_iterations)

    empty = ~occ

    # ---- 3. Pad + label -----------------------------------------------------
    empty_padded = np.pad(empty, 1, constant_values=True)
    labeled_padded, n_labels = ndimage.label(empty_padded)
    outside_label = int(labeled_padded[0, 0, 0])
    labeled = labeled_padded[1:-1, 1:-1, 1:-1]

    cavity_labels = [
        lbl
        for lbl in range(1, n_labels + 1)
        if lbl != outside_label and int(np.sum(labeled == lbl)) >= min_cavity_voxels
    ]
    if not cavity_labels:
        logger.info("No enclosed cavities detected at pitch=%s", pitch)
        return []

    # ---- 4. World ↔ voxel mapping ------------------------------------------
    # trimesh VoxelGrid: indices_to_points(idx) returns world-space centers.
    # We invert: world -> idx via origin + scale.
    logger.warning(
        "Probing %d faces against %d cavity labels (non-cavity space is label %d)",
        len(faces),
        len(cavity_labels),
        outside_label,
    )
    transform = np.asarray(vox.transform, dtype=float)
    origin = transform[:3, 3]
    # axis-aligned scale = pitch on the diagonal (assume isotropic)
    shape = np.asarray(labeled.shape)

    def world_to_voxel(pts: np.ndarray) -> np.ndarray:
        # voxel center at origin + (idx + 0.5) * pitch  =>  idx = (pt-origin)/pitch - 0.5
        idx = np.floor((pts - origin) / pitch).astype(int)
        return np.clip(idx, 0, shape - 1)

    # ---- 5. Probe each original face on both sides --------------------------
    offset = pitch * 1.5
    centroids = np.empty((len(faces), 3))
    normals = np.empty((len(faces), 3))
    for fi, f in enumerate(faces):
        c, n = _polygon_centroid_and_normal(f.vertex_indices, unique_vertices)
        centroids[fi] = c
        normals[fi] = n

    probe_pos = centroids + normals * offset
    probe_neg = centroids - normals * offset

    idx_pos = world_to_voxel(probe_pos)
    idx_neg = world_to_voxel(probe_neg)

    lbl_pos = labeled[idx_pos[:, 0], idx_pos[:, 1], idx_pos[:, 2]]
    lbl_neg = labeled[idx_neg[:, 0], idx_neg[:, 1], idx_neg[:, 2]]

    # ---- 6. Build cavities --------------------------------------------------
    # cavity_label -> {face_index: sign}
    cavity_faces: dict[int, dict[int, int]] = {lbl: {} for lbl in cavity_labels}

    for fi in range(len(faces)):
        lp = int(lbl_pos[fi])
        ln = int(lbl_neg[fi])
        # +normal probe lands in cavity X => normal points INTO X => sign -1 for X
        if lp in cavity_faces:
            cavity_faces[lp].setdefault(fi, -1)
        # -normal probe lands in cavity Y => normal points OUT of Y => sign +1 for Y
        if ln in cavity_faces:
            # if face already added (rare same-cavity both sides), keep first
            cavity_faces[ln].setdefault(fi, +1)

    # Compute volumes and sort desc
    voxel_vol = pitch**3
    cavity_volumes = {lbl: float(np.sum(labeled == lbl)) * voxel_vol for lbl in cavity_labels}
    sorted_labels = sorted(cavity_labels, key=lambda l: -cavity_volumes[l])

    cavities: List[Cavity] = []
    for rank, lbl in enumerate(sorted_labels, start=1):
        faces_for_cavity = cavity_faces[lbl]
        if not faces_for_cavity:
            logger.warning("Cavity %s has no assigned faces (probe missed); skipping", lbl)
            continue
        oriented = sorted(faces_for_cavity.items())  # [(face_idx, sign), ...]
        if rank == 1 and name_largest_as_room:
            name = "Room"
        else:
            name = f"Cavity_{rank}"
        cavities.append(
            Cavity(
                id=rank,
                name=name,
                volume=cavity_volumes[lbl],
                oriented_faces=oriented,
            )
        )

    logger.info(
        "Detected %d cavity/cavities at pitch=%s: %s",
        len(cavities),
        pitch,
        [(c.name, round(c.volume, 4), len(c.oriented_faces)) for c in cavities],
    )
    return cavities

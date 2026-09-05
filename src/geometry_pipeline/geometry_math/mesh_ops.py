"""Pure vertex/face-loop operations shared by the importer and repairs.

These helpers are IR-agnostic: they operate on plain coordinate tuples and
index loops, so both the I/O adapters (e.g. the OBJ importer) and the repair
layer can depend on them without depending on each other.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def deduplicate_vertices(
    vertices: List[Tuple[float, float, float]],
    tol: float = 1e-2,
) -> Tuple[List[Tuple[float, float, float]], Dict[int, int]]:
    unique_vertices: List[Tuple[float, float, float]] = []
    orig_to_unique: Dict[int, int] = {}

    for i, v in enumerate(vertices, start=1):
        found = None

        for j, uv in enumerate(unique_vertices, start=1):
            if abs(uv[0] - v[0]) < tol and abs(uv[1] - v[1]) < tol and abs(uv[2] - v[2]) < tol:
                found = j
                break
        if found is None:
            unique_vertices.append(v)
            orig_to_unique[i] = len(unique_vertices)
        else:
            orig_to_unique[i] = found
    return unique_vertices, orig_to_unique


def clean_face_loop(verts: List[int]) -> List[int]:
    if not verts:
        return verts

    cleaned = [verts[0]]

    for v in verts[1:]:
        if v != cleaned[-1]:
            cleaned.append(v)

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    return cleaned

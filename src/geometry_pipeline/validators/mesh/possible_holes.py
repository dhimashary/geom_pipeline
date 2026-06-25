"""Validator: detects closed boundary loops (candidate holes in the surface)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh, Face
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.geometry_math.geometry_math import uedge
from geometry_pipeline.validators.base import BaseValidator


def detect_possible_holes_from_faces(
    faces: List[Face],
    unique_vertices: List[Tuple[float, float, float]],
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
        prev_v = start_v
        cur_v = next(iter(adj[start_v]))
        while cur_v != start_v:
            edge_loop.append((unique_vertices[prev_v - 1], unique_vertices[cur_v - 1]))
            neighbors = adj[cur_v]
            next_v = next(n for n in neighbors if n != prev_v)
            prev_v = cur_v
            cur_v = next_v
        edge_loop.append((unique_vertices[prev_v - 1], unique_vertices[cur_v - 1]))

        elements = [{"type": "edge", "points": [list(edge[0]), list(edge[1])]} for edge in edge_loop]
        loops.append({"elements": elements, "severity": "high"})

    return loops


class PossibleHolesValidator(BaseValidator):
    name: ClassVar[str] = "possible_holes"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.POSSIBLE_HOLE

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        return detect_possible_holes_from_faces(list(geom.faces), points)

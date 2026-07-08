"""Make polygon winding globally consistent across shared edges.

For every manifold shared edge (used by exactly two faces), ensure the
edge direction is opposite in the two faces. This is required for a
valid PLC topology and for correct outward-normal computation downstream.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from typing import ClassVar

from geometry_pipeline.core.ir import Face

from geometry_pipeline.core.context import Context
from geometry_pipeline.geometry_math.geometry_math import uedge
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.report import RepairResult
from geometry_pipeline.repairs.base import BaseRepair


def orient_faces_consistently_by_adjacency(faces: list[Face], logger=None) -> Dict[str, Any]:
    edge_to_uses: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]] = defaultdict(list)

    for fi, f in enumerate(faces):
        vs = f.vertex_indices
        n = len(vs)
        if n < 2:
            continue
        for i in range(n):
            a = vs[i]
            b = vs[(i + 1) % n]
            edge_to_uses[uedge(a, b)].append((fi, (a, b)))

    boundary_edges = 0
    nonmanifold_edges = 0
    for _e, uses in edge_to_uses.items():
        if len(uses) == 1:
            boundary_edges += 1
        elif len(uses) > 2:
            nonmanifold_edges += 1

    face_adj: Dict[int, List[Tuple[int, Tuple[int, int], Tuple[int, int]]]] = defaultdict(list)
    for e, uses in edge_to_uses.items():
        if len(uses) != 2:
            continue
        (f0, dir0), (f1, dir1) = uses
        face_adj[f0].append((f1, e, dir0, dir1))
        face_adj[f1].append((f0, e, dir1, dir0))

    visited = [False] * len(faces)
    flipped_count = 0
    components = 0

    def flip_face(fi: int):
        nonlocal flipped_count
        faces[fi].vertex_indices.reverse()
        flipped_count += 1

    def current_edge_dir(face_verts: List[int], undirected_edge: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        u, v = undirected_edge
        n = len(face_verts)
        for i in range(n):
            a = face_verts[i]
            b = face_verts[(i + 1) % n]
            if (a == u and b == v) or (a == v and b == u):
                return (a, b)
        return None

    for seed in range(len(faces)):
        if visited[seed]:
            continue

        components += 1
        visited[seed] = True
        q = deque([seed])

        while q:
            fa = q.popleft()
            for (fb, e, _dir_a, _dir_b) in face_adj.get(fa, []):
                if not visited[fb]:
                    da = current_edge_dir(faces[fa].vertex_indices, e)
                    db = current_edge_dir(faces[fb].vertex_indices, e)

                    if da is None or db is None:
                        visited[fb] = True
                        q.append(fb)
                        continue

                    if da == db:
                        flip_face(fb)

                    visited[fb] = True
                    q.append(fb)

    if logger:
        logger.info(
            "[ORIENT] components=%d flipped=%d boundary=%d nonmanifold=%d",
            components,
            flipped_count,
            boundary_edges,
            nonmanifold_edges,
        )

    return {
        "components": components,
        "flipped_faces": flipped_count,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
    }


class OrientFacesConsistentlyByAdjacencyRepair(BaseRepair):
    name: ClassVar[str] = "orient_faces_consistently_by_adjacency"
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = set()

    def apply(
        self,
        geom: Mesh,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Mesh, RepairResult]:
        self.ensure_accepts(geom)
        faces = list(geom.faces)
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        diag = orient_faces_consistently_by_adjacency(faces, logger=ctx.logger)
        new_mesh = Mesh(
            vertices=list(geom.vertices),
            faces=faces,
            materials=dict(geom.materials),
            metadata=dict(geom.metadata),
        )
        result = self.make_result(
            stage_name=stage_name,
            before_count=len(faces),
            after_count=len(faces),
            details=dict(diag),
            issues=issues,
        )
        return new_mesh, result

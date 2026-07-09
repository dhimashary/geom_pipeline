"""Iteratively split polygon edges until no PLC T-junctions remain.

The detector is injected (decision #6) so the loop can be re-run with a
faster detector without changing this class. Each iteration:
    1. detector.detect(mesh, ctx) → list of T-junction Issues
    2. if empty → done
    3. otherwise call `fix_t_junctions_iterative_mesh`, which itself
       contains a fast inner loop that splits all reported edges in one
       pass; we exit this outer loop after that single call.

The outer loop is kept in case a future detector reports a single batch
that the inner repair only partially fixes — today's detector is global
so a single inner call is sufficient.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.geometry_math.geometry_math import uedge
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.validators.mesh.t_junctions import detect_t_junctions_mesh_global_plc
from geometry_pipeline.core.report import RepairResult
from geometry_pipeline.repairs.base import BaseRepair
from geometry_pipeline.validators.base import Validator
from geometry_pipeline.core.ir import Face


def _insert_vertex_on_edge_in_poly(poly, u, v, w):
    if w in poly:
        return poly, False

    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if a == u and b == v:
            return poly[: i + 1] + [w] + poly[i + 1 :], True

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if a == v and b == u:
            return poly[: i + 1] + [w] + poly[i + 1 :], True

    return poly, False


def fix_t_junctions_by_edge_splitting_mesh(
    mesh: Mesh,
    tjunc_reports,
    points,
    *,
    tol: float = 1e-8,
    max_passes: int = 10,
    logger: Optional[Any] = None,
):
    fid_to_face = {getattr(f, "fid", i): f for i, f in enumerate(mesh.faces)}

    def edge_param_t(u, v, w):
        Ax, Ay, Az = points[u - 1]
        Bx, By, Bz = points[v - 1]
        Px, Py, Pz = points[w - 1]
        ABx, ABy, ABz = (Bx - Ax, By - Ay, Bz - Az)
        APx, APy, APz = (Px - Ax, Py - Ay, Pz - Az)
        ab2 = ABx * ABx + ABy * ABy + ABz * ABz
        if ab2 <= 0.0:
            return 0.0
        return (APx * ABx + APy * ABy + APz * ABz) / ab2

    changed_any = False

    for pass_i in range(max_passes):
        ops = defaultdict(list)

        for r in tjunc_reports:
            u, v = r["edge"]
            w = r["split_vertex"]

            if "culprit_face_fid" in r and r["culprit_face_fid"] is not None:
                target_fids = [r["culprit_face_fid"]]
            else:
                target_fids = r.get("edge_face_fids") or r.get("edge_fids") or []

            for fid in target_fids:
                if fid not in fid_to_face:
                    continue
                t = edge_param_t(u, v, w)
                if not (tol < t < 1.0 - tol):
                    continue
                ops[(fid, uedge(u, v))].append((u, v, w, t))

        if not ops:
            if logger:
                logger.info("[TJUNC FIX] pass=%d: no ops; done", pass_i)
            break

        pass_changed = 0

        for (fid, _e), items in ops.items():
            face = fid_to_face[fid]
            poly = list(face.vertex_indices)
            items_sorted = sorted(items, key=lambda x: x[3])

            for (u, v, w, _t) in items_sorted:
                new_poly, did = _insert_vertex_on_edge_in_poly(poly, u, v, w)
                if did:
                    poly = new_poly
                    pass_changed += 1

            face.vertex_indices = poly

        if logger:
            logger.info("[TJUNC FIX] pass=%d: faces_changed=%d", pass_i, pass_changed)

        if pass_changed == 0:
            break

        changed_any = True
        break

    return mesh, changed_any


def fix_t_junctions_iterative_mesh(
    mesh: Mesh,
    *,
    tol: float = 1e-8,
    max_iters: int = 100,
    max_reports: int = 5000,
    logger: Optional[Any] = None,
):
    changed_any = False

    for it in range(max_iters):
        tjs = detect_t_junctions_mesh_global_plc(mesh, tol=tol, max_reports=max_reports)
        if not tjs:
            if logger:
                logger.info("[TJUNC FIX] stable after %d iterations", it)
            return mesh, changed_any

        mesh, changed = fix_t_junctions_by_edge_splitting_mesh(
            mesh,
            tjs,
            [(v.x, v.y, v.z) for v in mesh.vertices],
            tol=tol,
            max_passes=1,
            logger=logger,
        )
        if not changed:
            if logger:
                logger.warning("[TJUNC FIX] no changes applied but TJUNC still detected; stopping")
            return mesh, changed_any

        changed_any = True

    if logger:
        logger.warning("[TJUNC FIX] reached max_iters=%d; may still have TJUNCs", max_iters)
    return mesh, changed_any


class FixTJunctionsIterativeRepair(BaseRepair):
    name: ClassVar[str] = "fix_t_junctions_iterative"
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = {IssueKind.T_JUNCTION}

    def __init__(self, detector: Validator, max_iters: int | None = None) -> None:
        self.detector = detector
        self.max_iters = max_iters

    def apply(
        self,
        geom: Mesh,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Mesh, RepairResult]:
        self.ensure_accepts(geom)
        max_iters = self.max_iters or ctx.tolerances.max_t_junction_iters
        before = len(geom.faces)

        # The legacy `fix_t_junctions_iterative` already uses the legacy
        # detector internally. We pre-compute Issues via the injected
        # detector only for `affected_ids` book-keeping; the actual fix
        # is delegated.
        affected = self.affected_ids(issues)

        new_mesh, changed = fix_t_junctions_iterative_mesh(
            geom,
            tol=ctx.tolerances.t_junction,
            max_iters=max_iters,
            max_reports=ctx.tolerances.max_reports,
            logger=ctx.logger,
        )

        # Count remaining T-junctions to record convergence.
        remaining = self.detector.detect(new_mesh, ctx)
        result = self.make_result(
            stage_name=stage_name,
            before_count=before,
            after_count=len(new_mesh.faces),
            iterations=max_iters if changed else 0,
            details={
                "changed": bool(changed),
                "remaining_t_junctions": len(remaining),
                "tolerance": ctx.tolerances.t_junction,
            },
            issues=issues,
            affected_ids=affected,
        )
        return new_mesh, result

"""Tessellate a BRep into a Mesh."""

from __future__ import annotations

from geometry_pipeline.core.ir import BRep, Mesh


def brep_to_mesh(brep: BRep) -> Mesh:
    raise NotImplementedError

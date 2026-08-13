"""Typed geometry issues produced by validators."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from geometry_pipeline.core.jsonable import to_jsonable


class IssueKind(str, Enum):
    DUPLICATE_VERTEX = "duplicate_vertex"
    ZERO_AREA_FACE = "zero_area_face"
    NON_PLANAR_FACE = "non_planar_face"
    T_JUNCTION = "t_junction"
    INTERSECTION = "intersection"
    BOUNDARY_EDGE = "boundary_edge"
    POSSIBLE_HOLE = "possible_hole"
    SMALL_FACE = "small_face"
    OVERLAPPING_FACE = "overlapping_face"
    COLLINEAR_FACE = "collinear_face"


class DetectionStage(str, Enum):
    """When in the pipeline an issue was observed."""

    PRE = "pre"
    POST_STAGE = "post_stage"
    FINAL = "final"


@dataclass(frozen=True)
class Issue:
    """A defect detected by a validator.

    `id` is a stable content hash of (kind, geometry) — derived from the
    issue's `elements` when present — so the *same* physical defect produces
    the *same* id across snapshots.
    """

    id: str
    kind: IssueKind
    stage: DetectionStage
    stage_name: str = ""
    payload: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        kind: IssueKind,
        stage: DetectionStage,
        stage_name: str = "",
        payload: dict | None = None,
    ) -> "Issue":
        # Coerce numpy scalars/arrays to JSON-native builtins so the stored
        # payload (and the id hash derived from it) are stable and serialise
        # cleanly regardless of the numpy version.
        data: dict = to_jsonable(payload or {})
        # Derive the id from the intrinsic geometry (`elements`) when present so
        # the same physical defect keeps a stable id across snapshots, even when
        # repair shifts face indices or nudges derived metrics (max_dim,
        # overlap_area, ...). Fall back to the full payload for issues that
        # carry no elements (e.g. capped-list summary markers).
        id_source = data.get("elements") or data
        key = json.dumps([kind.value, id_source], sort_keys=True, default=str)
        issue_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return cls(
            id=issue_id,
            kind=kind,
            stage=stage,
            stage_name=stage_name,
            payload=data,
        )

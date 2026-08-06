"""Public exports for the repair-step wrappers.

Concrete repair steps live under `mesh/` (they all operate on mesh geometry).
Kind-agnostic infrastructure (`base`, `_common`) stays at this top level.
"""

from geometry_pipeline.repairs.mesh.compact_vertices import CompactVerticesRepair
from geometry_pipeline.repairs.mesh.deduplicate_vertices import DeduplicateVerticesRepair
from geometry_pipeline.repairs.mesh.fix_t_junctions import FixTJunctionsIterativeRepair
from geometry_pipeline.repairs.mesh.orient_outward import FlipFacesIfMajorityInwardRepair
from geometry_pipeline.repairs.mesh.remove_degenerate_faces import RemoveZeroAreaFaceRepair
from geometry_pipeline.repairs.mesh.repair_intersections import (
    RepairPlcByOffsetRepair,
    RepairPlcSingleSplitsRepair,
    TrimSegmentFaceIntersectionsRepair,
)
from geometry_pipeline.repairs.mesh.sort_vertices import SortVerticesDeterministicallyRepair

__all__ = [
    "CompactVerticesRepair",
    "DeduplicateVerticesRepair",
    "FixTJunctionsIterativeRepair",
    "FlipFacesIfMajorityInwardRepair",
    "RemoveZeroAreaFaceRepair",
    "RepairPlcByOffsetRepair",
    "RepairPlcSingleSplitsRepair",
    "SortVerticesDeterministicallyRepair",
    "TrimSegmentFaceIntersectionsRepair",
]

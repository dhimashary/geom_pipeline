"""Public exports for the seven detection validators.

Concrete validators live under `mesh/` (they all operate on mesh geometry).
Kind-agnostic infrastructure (`base`, `_common`) stays at this top level.
"""
from geometry_pipeline.validators.mesh.boundary_edges import BoundaryEdgesValidator
from geometry_pipeline.validators.mesh.degenerate_faces import ZeroAreaFaceValidator
from geometry_pipeline.validators.mesh.duplicate_vertices import DuplicateVerticesValidator
from geometry_pipeline.validators.mesh.intersections import IntersectionsValidator
from geometry_pipeline.validators.mesh.non_planar_faces import NonPlanarFacesValidator
from geometry_pipeline.validators.mesh.overlapping_faces import OverlappingFacesValidator
from geometry_pipeline.validators.mesh.possible_holes import PossibleHolesValidator
from geometry_pipeline.validators.mesh.t_junctions import TJunctionsValidator

__all__ = [
    "BoundaryEdgesValidator",
    "ZeroAreaFaceValidator",
    "DuplicateVerticesValidator",
    "IntersectionsValidator",
    "NonPlanarFacesValidator",
    "OverlappingFacesValidator",
    "PossibleHolesValidator",
    "TJunctionsValidator",
]

"""Pure-geometry math and predicates shared by validators and repairs."""

from geometry_pipeline.geometry_math.geometry_math import (
    area2,
    cross,
    dot,
    newell_normal_from_points,
    orient,
    sub,
    uedge,
)
from geometry_pipeline.geometry_math.mesh_ops import (
    clean_face_loop,
    deduplicate_vertices,
)
from geometry_pipeline.geometry_math.predicates import (
    classify_face_degeneracy,
    classify_face_planarity_m,
    planarity_deviation_m,
)
from geometry_pipeline.geometry_math.triangulation import triangulate_face_cdt_shapely

__all__ = [
    "area2",
    "cross",
    "dot",
    "newell_normal_from_points",
    "orient",
    "sub",
    "uedge",
    "triangulate_face_cdt_shapely",
    "classify_face_degeneracy",
    "classify_face_planarity_m",
    "planarity_deviation_m",
    "clean_face_loop",
    "deduplicate_vertices",
]

"""Canonical tolerance bundle attached to a SimulationProfile.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerances:
    # ---- Length tolerances (metres)
    vertex_merge: float = 1e-2
    planarity_warn_m: float = 1e-4
    planarity_fatal_m: float = 1e-3
    planarity_split: float = 1e-6
    t_junction: float = 1e-2
    clipping: float = 1e-6
    intersection_eps: float = 1e-10
    bbox_pad: float = 1e-9
    plc_offset: float = 0.01
    small_face_max_dim: float = 0.10
    # ---- Collinear-face tolerance (metres)
    # A face is (nearly) collinear when the maximum perpendicular deviation of
    # any vertex from the line through its two farthest-apart vertices is below
    # this value: the polygon has effectively collapsed to a line segment.
    collinear_face_max_deviation_m: float = 1e-3

    # ---- Possible-hole tolerances
    # Max out-of-plane distance (metres) for a face to count as coplanar with a
    # candidate boundary loop when deciding whether the loop interior is filled
    # by surface (an open-surface rim) rather than an empty hole.
    hole_fill_coplanar_m: float = 1e-2
    # Padding (metres) applied to a component's bounding box when deciding
    # whether one mesh component sits inside another. Used to separate the room
    # shell from free-standing objects (furniture) so that holes inside objects
    # are not reported as room holes.
    hole_object_containment_m: float = 1e-3

    # ---- Overlapping-face tolerances
    overlap_coplanar_dist_m: float = 1e-4
    overlap_normal_cos_eps: float = 1e-6
    # Minimum width (metres) of an overlap region. The 2D intersection is eroded
    # by half this value; anything that does not survive is a thin sliver caused
    # by coordinate noise at a shared edge (edge-adjacent faces) rather than a
    # genuine area overlap, and is not reported.
    overlap_sliver_width_m: float = 1e-3

    # ---- Area tolerances (m²)
    degenerate_area: float = 1e-12
    overlap_min_area_m2: float = 1e-9

    # ---- Sliver tolerance (metres): smallest face altitude below which a
    # near-collinear face is treated as degenerate.
    degenerate_min_altitude_m: float = 1e-4

    # ---- Iteration caps
    max_t_junction_iters: int = 100
    max_plc_iters: int = 20
    max_edge_split_passes: int = 10

    # ---- Reporting caps
    max_reports: int = 200

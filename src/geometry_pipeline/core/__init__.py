"""Core value objects and orchestration data models for geometry."""
from geometry_pipeline.core.context import Context
from geometry_pipeline.core.diff import SnapshotDiff, diff_snapshots
from geometry_pipeline.core.ir import (
    BRep,
    Cavity,
    Curve,
    Face,
    Geometry,
    LayerInfo,
    MaterialInfo,
    Mesh,
    PointCloud,
    Surface,
    Vertex,
)
from geometry_pipeline.core.issues import DetectionStage, Issue, IssueKind, Severity
from geometry_pipeline.core.profile import SimulationProfile, Stage
from geometry_pipeline.core.report import PipelineResult, RepairReport, RepairResult, ValidationSnapshot
from geometry_pipeline.core.tolerances import Tolerances

__all__ = [
    "BRep",
    "Cavity",
    "Context",
    "Curve",
    "DetectionStage",
    "Face",
    "Geometry",
    "Issue",
    "IssueKind",
    "LayerInfo",
    "MaterialInfo",
    "Mesh",
    "PipelineResult",
    "PointCloud",
    "RepairReport",
    "RepairResult",
    "Severity",
    "SimulationProfile",
    "SnapshotDiff",
    "Stage",
    "Surface",
    "Tolerances",
    "ValidationSnapshot",
    "Vertex",
    "diff_snapshots",
]

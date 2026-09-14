"""Core value objects and orchestration data models for geometry."""

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.diff import SnapshotDiff, diff_snapshots
from geometry_pipeline.core.ir import (
    Cavity,
    Face,
    Geometry,
    MaterialInfo,
    Mesh,
    Vertex,
)
from geometry_pipeline.core.issues import DetectionStage, Issue, IssueKind
from geometry_pipeline.core.ports import Exporter, ReportWriter
from geometry_pipeline.core.profile import SimulationProfile, Stage
from geometry_pipeline.core.report import (
    PipelineResult,
    RepairReport,
    RepairResult,
    ValidationSnapshot,
)
from geometry_pipeline.core.tolerances import Tolerances

__all__ = [
    "Cavity",
    "Context",
    "DetectionStage",
    "Exporter",
    "Face",
    "Geometry",
    "Issue",
    "IssueKind",
    "MaterialInfo",
    "Mesh",
    "PipelineResult",
    "RepairReport",
    "RepairResult",
    "ReportWriter",
    "SimulationProfile",
    "SnapshotDiff",
    "Stage",
    "Tolerances",
    "ValidationSnapshot",
    "Vertex",
    "diff_snapshots",
]

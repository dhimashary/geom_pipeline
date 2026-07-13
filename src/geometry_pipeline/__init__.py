"""geometry_pipeline — detached geometry processing package for CHORAS.

The public surface is the facade in :mod:`geometry_pipeline.api`. Everything else
is an implementation detail and may change without notice.
"""

from .api import (
    repair_geometry,
    process_geometry,
    list_issue_kinds,
    GeometryResult,
    IssueInfo,
    GeometryError,
    SUPPORTED_INPUTS,
)

__all__ = [
    "repair_geometry",
    "process_geometry",
    "list_issue_kinds",
    "GeometryResult",
    "IssueInfo",
    "GeometryError",
    "SUPPORTED_INPUTS",
]

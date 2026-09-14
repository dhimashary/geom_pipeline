"""Boundary port protocols for the geometry pipeline.

These are the ports that connect the domain core to external systems: file
sources (``Importer``), file sinks (``Exporter``), and result sinks
(``ReportWriter``). They live in the core so that ``core`` (e.g. ``profile``)
can reference them without importing the ``io``/``reporting`` adapter packages.
Concrete adapters under ``io/`` and ``reporting/`` implement these protocols
structurally, without inheriting from them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol

from geometry_pipeline.core.ir import Geometry

if TYPE_CHECKING:
    from geometry_pipeline.core.report import PipelineResult


class Importer(Protocol):
    """Port for geometry sources: read ``path`` into the geometry IR."""

    extensions: ClassVar[tuple[str, ...]]

    def load(self, path: Path) -> Geometry: ...


class Exporter(Protocol):
    """Port for geometry sinks: write the given geometry to ``path``.

    ``path_for`` lets a sink derive its own filename from a base path.
    """

    def path_for(self, base: Path) -> Path: ...
    def write(self, geom: Geometry, path: Path) -> None: ...


class ReportWriter(Protocol):
    """Port for result sinks: serialize a ``PipelineResult`` to ``path``.

    Unlike ``Exporter`` this consumes the run's result (issues + snapshots)
    rather than the geometry, so it is a distinct port.
    """

    def path_for(self, base: Path) -> Path: ...
    def write(self, result: PipelineResult, path: Path) -> None: ...

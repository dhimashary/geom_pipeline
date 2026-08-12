"""Intermediate Representation (IR) for geometry.

The IR is a tagged union: every variant implements `Geometry` and carries
a `kind` discriminator so validators / repairs / exporters can declare which
variants they accept. `Mesh` is currently the only variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from geometry_pipeline.core.report import PipelineResult

# ---- Primitive value objects ------------------------------------------------


@dataclass(frozen=True)
class Vertex:
    x: float
    y: float
    z: float


@dataclass
class Face:
    vertex_indices: list[int]
    group: str
    material: str | None


@dataclass(frozen=True)
class MaterialInfo:
    name: str
    properties: dict[str, float] = field(default_factory=dict)


# ---- Tagged-union geometry types --------------------------------------------


@runtime_checkable
class Geometry(Protocol):
    kind: ClassVar[str]


class Exporter(Protocol):
    """Port for exporters: write the given geometry to ``path``.

    Defined here in the core domain so that ``core`` (e.g. ``profile``) and
    ``io`` can both depend on it without ``core`` importing ``io``. Concrete
    exporters live under ``io/exporters`` and implement this protocol.
    """

    def write(self, geom: Geometry, path: Path) -> None: ...


@runtime_checkable
class SupportsPathFor(Protocol):
    """Capability: an exporter that resolves its own output path from a base."""

    def path_for(self, base: Path) -> Path: ...


@runtime_checkable
class SupportsPipelineResult(Protocol):
    """Capability: an exporter that needs the pipeline result before ``write``."""

    def set_pipeline_result(self, result: PipelineResult) -> None: ...


@dataclass
class Mesh:
    kind: ClassVar[str] = "mesh"
    vertices: list[Vertex] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    materials: dict[str, MaterialInfo] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class Cavity:
    """Represents a detected enclosed cavity.

    oriented_faces: list of (face_index, sign) where sign is +1 if the
    face's normal points out of the cavity, -1 if it points into the cavity.
    face_index refers to the index in the `faces` list passed to the exporter.
    """

    id: int
    name: str
    volume: float
    oriented_faces: List[Tuple[int, int]]
    is_manifold: bool = False

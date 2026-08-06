"""Intermediate Representation (IR) for geometry.

The IR is a tagged union: every variant implements `Geometry` and carries
a `kind` discriminator so validators / repairs / exporters can declare which
variants they accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Protocol, Tuple, runtime_checkable

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


@dataclass
class Curve:
    """A 2D/3D curve segment from a B-Rep input (line, polyline, arc, spline)."""

    points: list[Vertex]
    layer: str
    closed: bool


@dataclass
class Surface:
    """A trimmed surface from a B-Rep input."""

    boundary: list[Curve]
    layer: str


@dataclass(frozen=True)
class MaterialInfo:
    name: str
    properties: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerInfo:
    name: str
    color: str | None = None


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


@dataclass
class Mesh:
    kind: ClassVar[str] = "mesh"
    vertices: list[Vertex] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    materials: dict[str, MaterialInfo] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class BRep:
    kind: ClassVar[str] = "brep"
    curves: list[Curve] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)
    layers: dict[str, LayerInfo] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class PointCloud:
    kind: ClassVar[str] = "pointcloud"
    points: list[Vertex] = field(default_factory=list)
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

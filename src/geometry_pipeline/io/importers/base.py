"""Importer Protocol: file -> Geometry IR."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol

from geometry_pipeline.core.ir import Geometry


class Importer(Protocol):
    extensions: ClassVar[tuple[str, ...]]

    def load(self, path: Path) -> Geometry: ...

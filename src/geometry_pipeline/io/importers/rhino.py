"""Rhino .3dm importer -> Mesh (with material extraction)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from geometry_pipeline.core.ir import Mesh


class Rhino3dmImporter:
    extensions: ClassVar[tuple[str, ...]] = (".3dm",)

    def load(self, path: Path) -> Mesh:
        raise NotImplementedError

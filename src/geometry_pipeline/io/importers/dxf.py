"""DXF importer -> BRep."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from geometry_pipeline.core.ir import BRep


class DxfImporter:
    extensions: ClassVar[tuple[str, ...]] = (".dxf",)

    def load(self, path: Path) -> BRep:
        raise NotImplementedError

"""Registry for IR <-> IR converters (e.g. BRep -> Mesh tessellation)."""
from __future__ import annotations

from typing import Callable

from geometry_pipeline.core.ir import Geometry


Converter = Callable[[Geometry], Geometry]


class ConverterRegistry:
    @classmethod
    def register(
        cls,
        src: type[Geometry],
        dst: type[Geometry],
        fn: Converter,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def convert(cls, geom: Geometry, target: type[Geometry]) -> Geometry:
        raise NotImplementedError

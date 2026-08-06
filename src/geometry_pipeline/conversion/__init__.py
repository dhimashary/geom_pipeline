"""IR conversion package (e.g. BRep -> Mesh)."""

from geometry_pipeline.conversion.brep_to_mesh import brep_to_mesh
from geometry_pipeline.conversion.registry import Converter, ConverterRegistry

__all__ = ["Converter", "ConverterRegistry", "brep_to_mesh"]

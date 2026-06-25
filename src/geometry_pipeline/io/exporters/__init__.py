"""Public API for built-in geometry exporters.

Exporters are named by the IR *kind* they consume and the *format* they
emit, e.g. `MeshObjExporter` consumes a `Mesh` IR and writes an OBJ. This
leaves room for kind-specific variants (e.g. a future `BRepObjExporter`)
without overloading a single class with `if kind == ...` branches.
"""

from geometry_pipeline.core.ir import Exporter
from geometry_pipeline.io.exporters.mesh_geo import GeoExporter, GmshGeoExporter
from geometry_pipeline.io.exporters.mesh_obj import MeshObjExporter
from geometry_pipeline.io.exporters.mesh_three_dm import MeshThreeDMExporter

# Backward-compatibility aliases for the previous, less specific names.
ObjExporter = MeshObjExporter
ThreeDMExporter = MeshThreeDMExporter
ThreeDmExporter = MeshThreeDMExporter

__all__ = [
	"Exporter",
	"GeoExporter",
	"GmshGeoExporter",
	"MeshObjExporter",
	"MeshThreeDMExporter",
	# Backward-compatibility aliases
	"ObjExporter",
	"ThreeDMExporter",
	"ThreeDmExporter",
]

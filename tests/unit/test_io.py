"""Unit tests for the importers, exporters, and their registries.

The OBJ path is exercised with real geometry (a round-trip through the
exporter and back through the importer) plus the committed all-defects file.
Exporters are checked for the files/markers they produce. Heavy cavity
detection is disabled for the ``.geo`` exporter so the test stays a pure I/O
check.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.io.exporters.mesh_geo import GmshGeoExporter
from geometry_pipeline.io.exporters.mesh_obj import MeshObjExporter
from geometry_pipeline.io.exporters.mesh_three_dm import MeshThreeDMExporter
from geometry_pipeline.io.importers.obj import ObjImporter
from geometry_pipeline.io.registry import ExporterRegistry, ImporterRegistry


def _coord_set(mesh: Mesh) -> set[tuple[float, float, float]]:
    return {(round(v.x, 6), round(v.y, 6), round(v.z, 6)) for v in mesh.vertices}


# --- OBJ import --------------------------------------------------------------

def test_obj_import_of_real_room(real_room_obj):
    mesh = ObjImporter().load(real_room_obj)
    assert isinstance(mesh, Mesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.metadata.get("source_path") is not None


# --- OBJ round-trip ----------------------------------------------------------

def test_obj_export_reimport_round_trip(tmp_path, unit_cube):
    out = tmp_path / "cube.obj"
    MeshObjExporter().write(unit_cube, out)
    assert out.exists()

    reimported = ObjImporter().load(out)
    assert len(reimported.vertices) == len(unit_cube.vertices) == 8
    assert len(reimported.faces) == len(unit_cube.faces) == 6
    # The importer/exporter coordinate flips must cancel exactly.
    assert _coord_set(reimported) == _coord_set(unit_cube)


def test_obj_exporter_path_for_preserves_dotted_stem():
    # ``path_for`` appends `_repaired.obj` without truncating dotted names.
    base = Path("out") / "Vertigo_2.06_model"
    result = MeshObjExporter().path_for(base)
    assert result.name == "Vertigo_2.06_model_repaired.obj"


# --- GEO export --------------------------------------------------------------

def test_geo_export_writes_expected_markers(tmp_path, unit_cube):
    out = tmp_path / "cube.geo"
    GmshGeoExporter(detect_cavities=False).write(unit_cube, out)
    assert out.exists()

    text = out.read_text()
    assert "Point(1)" in text
    assert "Plane Surface(1)" in text


# --- 3DM export --------------------------------------------------------------

def test_three_dm_export_creates_non_empty_file(tmp_path, unit_cube):
    out = tmp_path / "cube.3dm"
    MeshThreeDMExporter().write(unit_cube, out)
    assert out.exists()
    assert out.stat().st_size > 0


# --- Registries --------------------------------------------------------------

def test_importer_registry_resolves_obj():
    imp = ImporterRegistry.for_extension(".obj")
    assert isinstance(imp, ObjImporter)
    # Extension normalization: no leading dot works too.
    assert isinstance(ImporterRegistry.for_extension("obj"), ObjImporter)


def test_importer_registry_rejects_unknown_extension():
    with pytest.raises(ValueError):
        ImporterRegistry.for_extension(".nope")


def test_exporter_registry_resolves_obj_mesh():
    exp = ExporterRegistry.get("obj", "mesh")
    assert isinstance(exp, MeshObjExporter)
    assert isinstance(ExporterRegistry.get(".OBJ", "mesh"), MeshObjExporter)


def test_exporter_registry_rejects_unknown_format():
    with pytest.raises(ValueError):
        ExporterRegistry.get("nope", "mesh")

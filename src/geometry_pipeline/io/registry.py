"""Registries for importers and exporters keyed by file extension / format."""
from __future__ import annotations

from geometry_pipeline.core.ir import Exporter
from geometry_pipeline.io.importers.base import Importer


class ImporterRegistry:
    _by_ext: dict[str, type[Importer] | Importer] = {}

    @classmethod
    def register(cls, importer: type[Importer] | Importer) -> None:
        """Register an importer class or instance for all its declared extensions.

        Extensions may include a leading dot (".obj") or not; they are
        normalized to lowercase without a dot.
        """
        exts = getattr(importer, "extensions", None)
        if not exts:
            return
        for e in exts:
            key = e.lower().lstrip(".")
            cls._by_ext[key] = importer

    @classmethod
    def for_extension(cls, ext: str) -> Importer:
        """Return an importer instance for the given extension (e.g. '.obj' or 'obj')."""
        key = ext.lower().lstrip(".")
        imp = cls._by_ext.get(key)
        if imp is None:
            # Try to lazily register built-in importers once and retry
            cls._register_builtins()
            imp = cls._by_ext.get(key)
            if imp is None:
                raise ValueError(f"No importer registered for extension: {ext}")
        return imp() if isinstance(imp, type) else imp

    @classmethod
    def _register_builtins(cls) -> None:
        """Register the known builtin importers (idempotent).

        This keeps startup simple: importing the registry will load the
        common importer modules and register their classes.
        """
        if cls._by_ext:
            return
        try:
            from geometry_pipeline.io.importers.obj import ObjImporter
        except Exception:
            ObjImporter = None
        try:
            from geometry_pipeline.io.importers.dxf import DxfImporter
        except Exception:
            DxfImporter = None
        try:
            from geometry_pipeline.io.importers.rhino import Rhino3dmImporter
        except Exception:
            Rhino3dmImporter = None

        for imp in (ObjImporter, DxfImporter, Rhino3dmImporter):
            if imp is not None:
                cls.register(imp)


class ExporterRegistry:
    """Resolve an exporter from a (format, IR kind) pair.

    A single output format (e.g. ``"obj"``) can have more than one
    implementation depending on the input IR kind — a ``Mesh`` is written
    differently than a ``BRep``. Keying on both lets the runner pick the
    right algorithm without the exporter branching internally on geometry
    type. ``kind`` defaults to ``"mesh"`` since that is the only IR the
    built-in exporters currently consume.
    """

    _by_key: dict[tuple[str, str], type[Exporter] | Exporter] = {}

    @staticmethod
    def _key(fmt: str, kind: str) -> tuple[str, str]:
        return (fmt.lower().lstrip("."), kind.lower())

    @classmethod
    def register(
        cls,
        fmt: str,
        kind: str,
        exporter: type[Exporter] | Exporter,
    ) -> None:
        """Register an exporter class or instance for a (format, kind) pair.

        Format names are normalized to lowercase without a leading dot, so
        ``"OBJ"``, ``".obj"`` and ``"obj"`` all map to the same entry.
        """
        cls._by_key[cls._key(fmt, kind)] = exporter

    @classmethod
    def get(cls, fmt: str, kind: str = "mesh") -> Exporter:
        """Return an exporter instance for the given format and IR kind."""
        key = cls._key(fmt, kind)
        exp = cls._by_key.get(key)
        if exp is None:
            # Try to lazily register built-in exporters once and retry
            cls._register_builtins()
            exp = cls._by_key.get(key)
            if exp is None:
                raise ValueError(
                    f"No exporter registered for format={fmt!r}, kind={kind!r}"
                )
        return exp() if isinstance(exp, type) else exp

    @classmethod
    def _register_builtins(cls) -> None:
        """Register the known builtin exporters (idempotent)."""
        if cls._by_key:
            return
        try:
            from geometry_pipeline.io.exporters.mesh_obj import MeshObjExporter
        except Exception:
            MeshObjExporter = None
        try:
            from geometry_pipeline.io.exporters.mesh_geo import GmshGeoExporter
        except Exception:
            GmshGeoExporter = None
        try:
            from geometry_pipeline.io.exporters.mesh_three_dm import MeshThreeDMExporter
        except Exception:
            MeshThreeDMExporter = None

        for fmt, kind, exp in (
            ("obj", "mesh", MeshObjExporter),
            ("geo", "mesh", GmshGeoExporter),
            ("3dm", "mesh", MeshThreeDMExporter),
        ):
            if exp is not None:
                cls.register(fmt, kind, exp)

"""OBJ exporter — writes a `Mesh` IR via the legacy export helper (inlined)."""

from __future__ import annotations

import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List

from geometry_pipeline.core.ir import Mesh

logger = logging.getLogger(__name__)


class MeshObjExporter:
    def __init__(self, *, repaired: bool = True) -> None:
        """``repaired=True`` writes ``<stem>_repaired.obj``; ``False`` writes the
        plain ``<stem>.obj`` (raw/initial bundle)."""
        self.repaired = repaired

    def path_for(self, base_path: Path) -> Path:
        """Write ``<stem>_repaired.obj`` (repaired) or ``<stem>.obj`` (raw)."""
        base_path = Path(base_path)
        # ``base_path`` is an extension-less base; use ``.name`` (not ``.stem``)
        # so filenames containing dots (e.g. ``Vertigo_2.06_...``) are not
        # truncated by Path treating ``.06_...`` as a suffix.
        suffix = "_repaired.obj" if self.repaired else ".obj"
        return base_path.with_name(base_path.name + suffix)

    def write(self, geom: Mesh, path: Path) -> None:
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        faces = list(geom.faces)
        self._export_processed_topology_to_obj(str(path), points, faces)

    def _export_processed_topology_to_obj(
        self, obj_output_path: str, unique_vertices: List, faces: List
    ) -> bool:
        """Export the processed topology to an OBJ file."""
        try:

            def _face_verts(face):
                if hasattr(face, "vertex_indices"):
                    return list(face.vertex_indices)
                if hasattr(face, "verts"):
                    return list(face.verts)
                raise TypeError("Face must provide `vertex_indices` or legacy `verts`")

            faces_by_group_and_material: dict = defaultdict(lambda: defaultdict(list))
            for face in faces:
                # Coerce missing/None group and material to "default" so the keys
                # are always strings. This prevents emitting a literal
                # ``usemtl None`` and avoids a TypeError when ``sorted()`` mixes
                # None with str keys below.
                group = getattr(face, "group", None) or "default"
                group_material = (
                    getattr(face, "group_material", None)
                    or getattr(face, "material", None)
                    or "default"
                )
                faces_by_group_and_material[group][group_material].append(face)

            # Write to a temporary file then atomically replace
            target_path = Path(obj_output_path)
            tmp_fd = None
            tmp_path = None
            try:
                tmp_dir = target_path.parent if target_path.parent.exists() else None
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    delete=False,
                    dir=tmp_dir,
                    prefix=target_path.stem + "_",
                    suffix=".obj",
                ) as tf:
                    tmp_fd = tf.fileno()
                    tmp_path = Path(tf.name)

                    tf.write("# Processed topology from geometry conversion\n\n")

                    # Vertices (note: preserve legacy coordinate ordering used by OBJ export)
                    for x, y, z in unique_vertices:
                        tf.write(f"v {x} {z} {-y}\n")

                    tf.write("\n")

                    # Faces grouped by group, then by group_material
                    for group in sorted(faces_by_group_and_material.keys()):
                        tf.write(f"\ng {group}\n")
                        for group_material in sorted(faces_by_group_and_material[group].keys()):
                            tf.write(f"usemtl {group_material}\n")
                            for face in faces_by_group_and_material[group][group_material]:
                                verts = _face_verts(face)
                                tf.write("f " + " ".join(f"{v}//1" for v in verts) + "\n")

                    tf.flush()
                    os.fsync(tmp_fd)

                os.replace(str(tmp_path), str(target_path))
                return True

            except Exception as ex:
                # Clean up temp file on failure
                if tmp_path and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                logger.error("Failed to export processed topology to OBJ: %s", ex)
                return False
        except Exception as ex:
            logger.error("Failed to export processed topology to OBJ: %s", ex)
            return False

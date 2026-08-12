"""Gmsh `.geo` exporter — writes a `Mesh` IR via the legacy export helper."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rhino3dm

from geometry_pipeline.core.ir import Cavity, Mesh
from geometry_pipeline.geometry_math.geometry_math import uedge

logger = logging.getLogger(__name__)


class GmshGeoExporter:
    def __init__(
        self,
        volume_name: str = "RoomVolume",
        *,
        repaired: bool = False,
        detect_cavities: bool = True,
        detection_mode: str = "native",
        cavity_pitch: float = 0.05,
        cavity_closing_iterations: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        volume_name : Name used for the single Physical Volume in legacy
            output, or as fallback when detection fails / yields nothing.
        repaired : If True, write ``<stem>_repaired.geo`` (and read the matching
            ``<stem>_repaired.3dm`` for material mapping); otherwise write the
            plain ``<stem>.geo`` and read ``<stem>.3dm``.
        detect_cavities : If True, run cavity detection before writing and
            emit one `Surface Loop` + `Volume` per detected cavity (Gmsh
            requires every enclosed space to be its own volume).
                detection_mode : One of:
                        - "native" (default): use the C++ CGAL grid+visibility detector
                            (``bin/volume_detector``); robust for multi-scale scenes
                            (big rooms + small furniture cavities). This path is the
                            production default and does not fall back to the voxel detector.
                        - "voxel": use the pure-Python voxel detector (kept for testing).
        cavity_pitch : Voxel size (model units) for the *voxel* detector. Must
            be smaller than the smallest wall thickness you care about.
        cavity_closing_iterations : Optional morphological closing iterations
            (voxel detector only) to bridge sub-pitch gaps before labeling.
        """
        self.volume_name = volume_name
        self.repaired = repaired
        self.detect_cavities = detect_cavities
        self.detection_mode = detection_mode
        self.cavity_pitch = cavity_pitch
        self.cavity_closing_iterations = cavity_closing_iterations

    def write(
        self,
        geom: Mesh,
        path: Path,
        cavities: Optional[List[Cavity]] = None,
    ) -> None:
        """Write GEO file.

        Cavity sources, in priority order:
          1. Explicit `cavities` argument (caller-provided).
          2. Auto-detected cavities (when `self.detect_cavities=True`).
          3. None -> legacy single-volume output.
        Detection failures fall back to legacy output with a warning.
        """
        points = [(v.x, v.y, v.z) for v in geom.vertices]
        faces = list(geom.faces)

        if cavities is None and self.detect_cavities:
            cavities = self._run_detection(faces, points)

        self._export_processed_topology_to_gmsh_geo(
            faces, points, str(path), volume_name=self.volume_name, cavities=cavities
        )

    def path_for(self, base_path: Path) -> Path:
        """Return the target .geo file path for the given base path.

        Mirrors other exporters which write next to the base with a
        deterministic suffix; ensures the pipeline always provides a
        concrete `.geo` filepath (avoids directory/file collisions on Windows).
        """
        base_path = Path(base_path)
        # ``base_path`` is an extension-less base; use ``.name`` (not ``.stem``)
        # so filenames containing dots (e.g. ``Vertigo_2.06_...``) are not
        # truncated by Path treating ``.06_...`` as a suffix.
        suffix = "_repaired.geo" if self.repaired else ".geo"
        return base_path.with_name(base_path.name + suffix)

    def _export_processed_topology_to_gmsh_geo(
        self,
        faces: List,
        unique_vertices: List[Tuple[float, float, float]],
        geo_file: str,
        volume_name: str = "RoomVolume",
        cavities: Optional[List[Cavity]] = None,
    ) -> Tuple[int, int]:
        """Export processed topology to Gmsh GEO file (inlined legacy impl).

        Returns (num_lines, num_surfaces).
        """
        # Build unique edges (Lines) + signed loops
        edge_to_line: Dict = {}
        line_orientation: Dict = {}
        next_line_id = 1
        face_line_loops: List[List[int]] = []

        def _face_verts(face):
            if hasattr(face, "vertex_indices"):
                return list(face.vertex_indices)
            if hasattr(face, "verts"):
                return list(face.verts)
            raise TypeError("Face must provide `vertex_indices` or legacy `verts`")

        for face in faces:
            loop_line_ids = []
            verts = _face_verts(face)
            n = len(verts)
            for i in range(n):
                a = verts[i]
                b = verts[(i + 1) % n]
                key = uedge(a, b)
                if key not in edge_to_line:
                    edge_to_line[key] = next_line_id
                    line_orientation[next_line_id] = (a, b)
                    next_line_id += 1
                lid = edge_to_line[key]
                ori = line_orientation[lid]
                loop_line_ids.append(lid if ori == (a, b) else -lid)
            face_line_loops.append(loop_line_ids)

        # looking for 3dm path
        base_path = Path(geo_file)
        three_dm_path = base_path.with_name(base_path.stem + ".3dm")

        model = None
        # Validate 3dm availability before trying to read it
        if three_dm_path.exists():
            try:
                model = rhino3dm.File3dm.Read(str(three_dm_path))
            except Exception as ex:
                logger.warning("Failed to read 3DM model at %s: %s", three_dm_path, ex)
                model = None
        else:
            logger.info(
                "3DM model not found at %s; continuing without material mapping", three_dm_path
            )

        # Material mapping for later use
        material_to_id = {}
        if model is not None:
            for obj in model.Objects:  # type: ignore[attr-defined]
                if isinstance(obj.Geometry, rhino3dm.Mesh):
                    material_name = obj.Geometry.GetUserString("material_name")  # type: ignore[attr-defined]
                    if material_name:
                        material_to_id[f"{obj.Attributes.Id}"] = material_name
        else:
            logger.debug("No 3DM model available; material_to_id mapping will be empty.")

        # Physical surface groups: material -> list of 0-based face indices
        physical_surfaces_dict: Dict = {}
        for idx, face in enumerate(faces):
            physical_surfaces_dict.setdefault(getattr(face, "material", None), []).append(idx)

        logger.info(
            "Physical surfaces: %s", {mat: len(ids) for mat, ids in physical_surfaces_dict.items()}
        )

        # Write GEO to a temporary file then atomically replace the target.
        target_path = Path(geo_file)
        tmp_fd = None
        tmp_path = None
        try:
            # Create a temp file in the same directory to ensure atomic replace works
            # across filesystems
            tmp_dir = target_path.parent if target_path.parent.exists() else None
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, dir=tmp_dir, prefix=target_path.stem + "_", suffix=".geo"
            ) as tf:
                tmp_fd = tf.fileno()
                tmp_path = Path(tf.name)

                # Points
                for i, v in enumerate(unique_vertices, start=1):
                    tf.write(f"Point({i}) = {{ {v[0]}, {v[1]}, {v[2]}, 1.0 }};\n")
                tf.write("\n")

                # Lines
                for lid in range(1, next_line_id):
                    a, b = line_orientation[lid]
                    tf.write(f"Line({lid}) = {{ {a}, {b} }};\n")
                tf.write("\n")

                # Line Loops
                for sid, (loop, face) in enumerate(zip(face_line_loops, faces), start=1):
                    loop_str = ", ".join(str(x) for x in loop)
                    tf.write(f"Line Loop({sid}) = {{ {loop_str} }};\n")
                tf.write("\n")

                # Plane Surfaces
                for sid in range(1, len(face_line_loops) + 1):
                    tf.write(f"Plane Surface({sid}) = {{ {sid} }};\n")
                tf.write("\n")

                # Surface Loop(s) + Volume(s)
                if not cavities:
                    total_surfaces = len(face_line_loops)
                    surf_list = ", ".join(str(i) for i in range(1, total_surfaces + 1))

                    tf.write(f"Surface Loop(1) = {{ {surf_list} }};\n")
                    tf.write("Volume(1) = { 1 };\n")
                    tf.write(f'Physical Volume("{volume_name}") = {{ 1 }};\n')

                else:
                    main_volume_surfaces: dict[int, int] = {}
                    separate_volumes: list[tuple[str, list[int]]] = []

                    for cav in cavities:
                        is_manifold = getattr(cav, "is_manifold", True)

                        if cav.id == 0 or not is_manifold:
                            for face_idx, sign in cav.oriented_faces:
                                sid = face_idx + 1
                                signed_sid = sid if sign > 0 else -sid
                                if sid not in main_volume_surfaces:
                                    main_volume_surfaces[sid] = signed_sid

                        else:
                            surf_ids: list[int] = []
                            for face_idx, sign in cav.oriented_faces:
                                sid = face_idx + 1
                                surf_ids.append(sid if sign > 0 else -sid)
                            separate_volumes.append((cav.name, surf_ids))

                    physical_volumes: list[tuple[str, int]] = []

                    if main_volume_surfaces:
                        surf_list = ", ".join(
                            str(signed_sid) for signed_sid in main_volume_surfaces.values()
                        )
                        tf.write(f"Surface Loop(1) = {{ {surf_list} }};\n")
                        tf.write("Volume(1) = { 1 };\n")
                        physical_volumes.append((volume_name, 1))
                        next_volume_id = 2
                    else:
                        next_volume_id = 1

                    for cav_name, surf_ids in separate_volumes:
                        surf_list = ", ".join(str(sid) for sid in surf_ids)
                        tf.write(f"Surface Loop({next_volume_id}) = {{ {surf_list} }};\n")
                        tf.write(f"Volume({next_volume_id}) = {{ {next_volume_id} }};\n")
                        physical_volumes.append((cav_name, next_volume_id))
                        next_volume_id += 1

                    for cav_name, volume_id in physical_volumes:
                        tf.write(f'Physical Volume("{cav_name}") = {{ {volume_id} }};\n')

                # Physical Surfaces
                ii = 1
                for grp in material_to_id:
                    tf.write(f'Physical Surface("{grp}") = {{ {str(ii)} }};\n')
                    ii = ii + 1

                # Physical Lines
                lines_all = ", ".join(str(i) for i in range(1, next_line_id))
                tf.write(f'Physical Line("default") = {{ {lines_all} }};\n')

                # Mesh options
                tf.write("Mesh.Algorithm = 6;\n")
                tf.write("Mesh.Algorithm3D = 1; // Delaunay3D\n")
                tf.write("Mesh.Optimize = 1;\n")
                tf.write("Mesh.CharacteristicLengthFromPoints = 1;\n")

                tf.flush()
                os.fsync(tmp_fd)

            # Replace the target atomically. On Windows this will overwrite existing files.
            os.replace(str(tmp_path), str(target_path))

        except Exception:
            # Clean up temp file on failure
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise

        return next_line_id - 1, len(face_line_loops)

    def _run_detection(self, faces, points) -> Optional[List[Cavity]]:
        mode = self.detection_mode
        if mode == "native":
            # In production we require the native detector; propagate errors
            # rather than silently falling back to the Python voxelizer.
            return self._run_native_detection(faces, points)
        elif mode == "voxel":
            return self._run_voxel_detection(faces, points)
        else:
            raise ValueError(f"Unsupported detection_mode: {mode}")

    def _run_native_detection(self, faces, points) -> Optional[List[Cavity]]:
        """Return cavities from the native detector.

        Raises when the native binary is missing or detection fails so the
        native binary is a hard requirement for cavity detection.
        """
        from ...cavity_detection.native_bridge import detect_cavities_native

        return detect_cavities_native(faces, points)

    def _run_voxel_detection(self, faces, points) -> Optional[List[Cavity]]:
        try:
            # Imported lazily so trimesh/scipy are only required when
            # cavity detection is actually enabled.
            from ...cavity_detection.cavity_detector import detect_cavities

            cavities = detect_cavities(
                faces,
                points,
                pitch=self.cavity_pitch,
                closing_iterations=self.cavity_closing_iterations,
            )
            if not cavities:
                return None
            return cavities
        except Exception:
            return None


# Backward compatibility: older callers import `GeoExporter` from this module.
GeoExporter = GmshGeoExporter

__all__ = ["GmshGeoExporter", "GeoExporter"]

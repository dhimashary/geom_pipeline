"""OBJ importer -> Mesh.

Wraps the legacy `parse_obj_file` + `process_and_instantiate_faces`
sequence. The resulting `Mesh` is in IR coordinates (Z-up) — the
SketchUp/Y-up flip is performed by `parse_obj_file` itself (tech-debt #6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

import rhino3dm

from geometry_pipeline.core.ir import Face, Mesh, Vertex

logger = logging.getLogger(__name__)


def _parse_obj_file(
    obj_file: str,
) -> Tuple[List[Tuple[float, float, float]], List[List[int]], List[str], List[str]]:
    vertices: List[Tuple[float, float, float]] = []
    raw_faces: List[List[int]] = []
    face_groups: List[str] = []
    face_group_materials: List[str] = []
    current_group = "default"
    current_group_material = "default"

    with open(obj_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("v "):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                # SketchUp (Y-up, left-handed) -> Gmsh (right-handed), flip Z.
                vertices.append((x, -z, y))
            elif line.startswith("g "):
                parts = line.split()[1:]
                parts = [p for p in parts if not p.startswith("Mesh") and not p.startswith("Model")]
                current_group = parts[0] if parts else "default"
            elif line.startswith("usemtl "):
                parts = line.split()[1:]
                current_group_material = parts[0] if parts else "default"
            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = [int(p.split("/")[0]) for p in parts]
                raw_faces.append(idxs)
                face_groups.append(current_group)
                face_group_materials.append(current_group_material)

    return vertices, raw_faces, face_groups, face_group_materials


def _deduplicate_vertices(
    vertices: List[Tuple[float, float, float]],
    tol: float = 1e-2,
) -> Tuple[List[Tuple[float, float, float]], Dict[int, int]]:
    unique_vertices: List[Tuple[float, float, float]] = []
    orig_to_unique: Dict[int, int] = {}

    for i, v in enumerate(vertices, start=1):
        found = None

        for j, uv in enumerate(unique_vertices, start=1):
            if abs(uv[0] - v[0]) < tol and abs(uv[1] - v[1]) < tol and abs(uv[2] - v[2]) < tol:
                found = j
                break
        if found is None:
            unique_vertices.append(v)
            orig_to_unique[i] = len(unique_vertices)
        else:
            orig_to_unique[i] = found
    return unique_vertices, orig_to_unique


def _clean_face_loop(verts: List[int]) -> List[int]:
    if not verts:
        return verts

    cleaned = [verts[0]]

    for v in verts[1:]:
        if v != cleaned[-1]:
            cleaned.append(v)

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    return cleaned


def _process_and_instantiate_faces(
    raw_faces: List[List[int]],
    face_groups: List[str],
    face_group_materials: List[str],
    material_id_array: List[str],
    orig_to_unique: Optional[Dict[int, int]] = None,
) -> List[Face]:
    faces: List[Face] = []
    face_id = 0

    if orig_to_unique is None:
        max_vertex_id = max((vid for face in raw_faces for vid in face), default=0)
        orig_to_unique = {i: i for i in range(1, max_vertex_id + 1)}

    for idx, raw_face in enumerate(raw_faces):
        grp = face_groups[idx] if idx < len(face_groups) else "default"
        grp_mat = face_group_materials[idx] if idx < len(face_group_materials) else "default"
        if idx < len(material_id_array):
            mat = material_id_array[idx]
        elif idx < len(face_group_materials):
            mat = face_group_materials[idx]
        else:
            mat = "unknown"
        mapped = [orig_to_unique[i] for i in raw_face]
        mapped = _clean_face_loop(mapped)

        sub_faces = [mapped]

        if not sub_faces:
            logger.error("[FACE] face_id=%d produced 0 sub_faces (mapped=%s)", face_id, mapped)
        for verts in sub_faces:
            faces.append(Face(vertex_indices=list(verts), group=grp, material=mat))
            face_id += 1
    return faces


def _extract_rhino_materials(rhino3dm_path: str) -> List[str]:
    """Extract per-mesh `material_name` user strings from a Rhino 3DM file."""

    model = rhino3dm.File3dm.Read(str(Path(rhino3dm_path)))
    if model is None:
        return []

    materials: List[str] = []
    for obj in model.Objects:  # type: ignore[attr-defined]
        if isinstance(obj.Geometry, rhino3dm.Mesh):
            name = obj.Geometry.GetUserString("material_name")  # type: ignore[attr-defined]
            materials.append(name or "unknown")

    return materials


class ObjImporter:
    extensions: ClassVar[tuple[str, ...]] = (".obj",)

    def load(self, path: Path) -> Mesh:
        path_str = str(path)
        vertices, raw_faces, face_groups, face_group_materials = _parse_obj_file(path_str)

        # `process_and_instantiate_faces` needs a per-face material id list;
        # absent a Rhino sidecar we fall back to the OBJ's `usemtl` value.
        material_id_array = list(face_group_materials)
        face_records = _process_and_instantiate_faces(
            raw_faces=raw_faces,
            face_groups=face_groups,
            face_group_materials=face_group_materials,
            material_id_array=material_id_array,
        )

        return Mesh(
            vertices=[Vertex(x=v[0], y=v[1], z=v[2]) for v in vertices],
            faces=[
                Face(
                    vertex_indices=list(f.vertex_indices),
                    group=f.group,
                    material=f.material,
                )
                for f in face_records
            ],
            metadata={"source_path": path},
        )


# Temporary compatibility exports while repairs are still legacy-oriented.
clean_face_loop = _clean_face_loop
deduplicate_vertices = _deduplicate_vertices
parse_obj_file = _parse_obj_file
process_and_instantiate_faces = _process_and_instantiate_faces
extract_rhino_materials = _extract_rhino_materials

"""Shared pytest fixtures and synthetic-mesh helpers.

Per-validator tests build small, isolated meshes in code (one defect each) so
that a failing assertion points unambiguously at a single validator/repair.
The one real-world file, ``tests/models/vert2.0.6.obj``, carries every defect
kind at once and is reserved for the end-to-end smoke test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from geometry_pipeline.core.ir import Face, Mesh, Vertex


# --- Locations ---------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"


@pytest.fixture
def models_dir() -> Path:
    """Directory holding committed geometry fixtures."""
    return MODELS_DIR


@pytest.fixture
def real_room_obj() -> Path:
    """The real, all-defects room used only for the end-to-end smoke test."""
    path = MODELS_DIR / "vert2.0.6.obj"
    if not path.exists():
        pytest.skip(f"real geometry fixture missing: {path}")
    return path


# --- Mesh construction helpers -----------------------------------------------

def make_mesh(
    vertices: list[tuple[float, float, float]],
    faces: list[list[int]],
    *,
    group: str = "default",
    material: str | None = None,
) -> Mesh:
    """Build a :class:`Mesh` from plain coordinate tuples and index lists."""
    return Mesh(
        vertices=[Vertex(*v) for v in vertices],
        faces=[Face(vertex_indices=list(f), group=group, material=material) for f in faces],
    )


# Eight corners of the unit cube [0,1]^3.
_CUBE_VERTS = [
    (0.0, 0.0, 0.0),  # 0
    (1.0, 0.0, 0.0),  # 1
    (1.0, 1.0, 0.0),  # 2
    (0.0, 1.0, 0.0),  # 3
    (0.0, 0.0, 1.0),  # 4
    (1.0, 0.0, 1.0),  # 5
    (1.0, 1.0, 1.0),  # 6
    (0.0, 1.0, 1.0),  # 7
]

# Six quad faces of the cube, outward-oriented.
_CUBE_FACES = [
    [0, 3, 2, 1],  # bottom (z=0), normal -z
    [4, 5, 6, 7],  # top    (z=1), normal +z
    [0, 1, 5, 4],  # front  (y=0), normal -y
    [2, 3, 7, 6],  # back   (y=1), normal +y
    [1, 2, 6, 5],  # right  (x=1), normal +x
    [0, 4, 7, 3],  # left   (x=0), normal -x
]


@pytest.fixture
def cube_verts() -> list[tuple[float, float, float]]:
    return list(_CUBE_VERTS)


@pytest.fixture
def cube_faces() -> list[list[int]]:
    return [list(f) for f in _CUBE_FACES]


@pytest.fixture
def unit_cube() -> Mesh:
    """A clean, closed, watertight unit cube (no defects)."""
    return make_mesh(_CUBE_VERTS, _CUBE_FACES)

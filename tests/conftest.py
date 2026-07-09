"""Shared pytest fixtures and synthetic-mesh helpers.

Per-validator tests build small, isolated meshes in code (one defect each) so
that a failing assertion points unambiguously at a single validator/repair.
The one real-world file, ``tests/models/vert2.0.6.obj``, carries every defect
kind at once and is reserved for the end-to-end smoke test.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Face, Mesh, Vertex
from geometry_pipeline.core.tolerances import Tolerances


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
    """Build a :class:`Mesh` from plain coordinate tuples and index lists.

    Face indices are **1-based**, matching the IR convention used throughout
    the pipeline (``points[vid - 1]``) and produced by the OBJ importer.
    """
    return Mesh(
        vertices=[Vertex(*v) for v in vertices],
        faces=[Face(vertex_indices=list(f), group=group, material=material) for f in faces],
    )


@pytest.fixture
def ctx() -> Context:
    """A default per-run context (canonical tolerances, throwaway logger)."""
    return Context(
        tolerances=Tolerances(),
        logger=logging.getLogger("geometry_pipeline.tests"),
        profile_name="test",
    )


# Eight corners of the unit cube [0,1]^3 (vertex ids 1..8, 1-based).
_CUBE_VERTS = [
    (0.0, 0.0, 0.0),  # 1
    (1.0, 0.0, 0.0),  # 2
    (1.0, 1.0, 0.0),  # 3
    (0.0, 1.0, 0.0),  # 4
    (0.0, 0.0, 1.0),  # 5
    (1.0, 0.0, 1.0),  # 6
    (1.0, 1.0, 1.0),  # 7
    (0.0, 1.0, 1.0),  # 8
]

# Six quad faces of the cube (1-based indices), outward-oriented.
_CUBE_FACES = [
    [1, 4, 3, 2],  # bottom (z=0), normal -z
    [5, 6, 7, 8],  # top    (z=1), normal +z
    [1, 2, 6, 5],  # front  (y=0), normal -y
    [3, 4, 8, 7],  # back   (y=1), normal +y
    [2, 3, 7, 6],  # right  (x=1), normal +x
    [1, 5, 8, 4],  # left   (x=0), normal -x
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


@pytest.fixture
def open_cube() -> Mesh:
    """A unit cube with the top face removed.

    The four top edges become open boundary edges forming a single closed loop
    whose interior is empty — exactly one boundary loop / possible hole.
    """
    faces_without_top = [f for f in _CUBE_FACES if f != [5, 6, 7, 8]]
    return make_mesh(_CUBE_VERTS, faces_without_top)

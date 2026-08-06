"""Unit tests for cavity detection.

The production detector is a compiled C++ ``volume_detector`` binary invoked as
a subprocess (:mod:`geometry_pipeline.cavity_detection.native_bridge`). These
tests **mock the native code** — the binary path and ``subprocess.run`` — so the
Python glue (mesh JSON serialisation, process invocation, JSON → ``Cavity``
conversion, error handling) is covered without compiling or shelling out.

The pure-Python voxel detector (:mod:`cavity_detector`) is also exercised on a
watertight unit cube; it is skipped when its optional native deps (trimesh /
scipy) are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from geometry_pipeline.cavity_detection import native_bridge
from geometry_pipeline.cavity_detection.native_bridge import (
    _cavities_from_json,
    _write_mesh_json,
    detect_cavities_native,
    is_native_detector_available,
    native_detector_path,
)
from geometry_pipeline.core.ir import Cavity, Face

# --- helpers -----------------------------------------------------------------


def _face(*vertex_indices: int) -> Face:
    """A minimal FaceRecord-like face with 1-based vertex indices."""
    return Face(vertex_indices=list(vertex_indices), group="default", material=None)


def _fake_run_factory(
    payload: dict | None,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    write_output: bool = True,
):
    """Build a ``subprocess.run`` stand-in that simulates the native tool.

    The real detector receives ``[binary, mesh_path, json_path]`` and writes its
    result to ``json_path``. The fake mirrors that contract so the bridge can
    read the output back.
    """

    def _fake_run(cmd, **kwargs):
        if write_output and payload is not None:
            json_out = Path(cmd[2])
            json_out.write_text(json.dumps(payload))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return _fake_run


# --- _write_mesh_json --------------------------------------------------------


def test_write_mesh_json_serialises_zero_based_faces(tmp_path):
    faces = [_face(1, 2, 3), _face(3, 4, 1)]
    vertices = [(0.0, 0.0, 0.0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    out = tmp_path / "mesh.json"

    _write_mesh_json(faces, vertices, out)
    payload = json.loads(out.read_text())

    # Vertices are coerced to float triples.
    assert payload["vertices"] == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    # 1-based IR indices become 0-based for the C++ tool; face order preserved.
    assert payload["faces"] == [[0, 1, 2], [2, 3, 0]]


# --- _cavities_from_json -----------------------------------------------------


def test_cavities_from_json_maps_ids_signs_and_names():
    payload = {
        "volumes": [
            {
                "volume_id": 0,
                "is_manifold": True,
                "faces": [{"face_id": 0, "sign": 1}, {"face_id": 1, "sign": -1}],
            },
            {"volume_id": 2, "faces": [{"face_id": 5}]},  # sign + manifold default
        ]
    }

    cavities = _cavities_from_json(payload)

    assert [c.id for c in cavities] == [0, 2]
    # id == 0 is the enclosing room; positive ids are inner cavities.
    assert cavities[0].name == "RoomVolume"
    assert cavities[1].name == "Cavity_2"
    # Signs are normalised to strictly +1 / -1.
    assert cavities[0].oriented_faces == [(0, 1), (1, -1)]
    # Missing sign defaults to +1; missing is_manifold defaults to True.
    assert cavities[1].oriented_faces == [(5, 1)]
    assert cavities[1].is_manifold is True
    # Native tool does not compute a metric volume.
    assert all(c.volume == 0.0 for c in cavities)


def test_cavities_from_json_skips_volumes_without_faces():
    payload = {"volumes": [{"volume_id": 1, "faces": []}]}
    assert _cavities_from_json(payload) == []


def test_cavities_from_json_empty_payload():
    assert _cavities_from_json({}) == []


# --- native_detector_path / availability -------------------------------------


def test_native_detector_path_none_when_absent(monkeypatch):
    monkeypatch.delenv("VOLUME_DETECTOR_BIN", raising=False)
    # Point the default location at somewhere guaranteed not to exist.
    monkeypatch.setattr(native_bridge, "_DEFAULT_BINARY", Path("/nonexistent/volume_detector"))
    assert native_detector_path() is None
    assert is_native_detector_available() is False


def test_native_detector_path_uses_env_override(tmp_path, monkeypatch):
    fake_bin = tmp_path / "volume_detector"
    fake_bin.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VOLUME_DETECTOR_BIN", str(fake_bin))

    assert native_detector_path() == fake_bin
    assert is_native_detector_available() is True


def test_native_detector_path_ignores_missing_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VOLUME_DETECTOR_BIN", str(tmp_path / "does_not_exist"))
    monkeypatch.setattr(native_bridge, "_DEFAULT_BINARY", Path("/nonexistent/volume_detector"))
    assert native_detector_path() is None


# --- detect_cavities_native (mocked binary + subprocess) ---------------------


def test_detect_cavities_native_happy_path(monkeypatch, tmp_path):
    binary = tmp_path / "volume_detector"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(native_bridge, "native_detector_path", lambda: binary)

    payload = {
        "volumes": [
            {
                "volume_id": 0,
                "is_manifold": True,
                "faces": [{"face_id": 0, "sign": 1}, {"face_id": 1, "sign": -1}],
            },
        ]
    }
    monkeypatch.setattr(native_bridge.subprocess, "run", _fake_run_factory(payload, stdout="done"))

    cavities = detect_cavities_native(
        [_face(1, 2, 3), _face(3, 2, 4)], [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    )

    assert len(cavities) == 1
    assert isinstance(cavities[0], Cavity)
    assert cavities[0].name == "RoomVolume"
    assert cavities[0].oriented_faces == [(0, 1), (1, -1)]


def test_detect_cavities_native_raises_when_binary_missing(monkeypatch):
    monkeypatch.setattr(native_bridge, "native_detector_path", lambda: None)
    with pytest.raises(FileNotFoundError):
        detect_cavities_native([_face(1, 2, 3)], [(0, 0, 0), (1, 0, 0), (0, 1, 0)])


def test_detect_cavities_native_raises_on_nonzero_exit(monkeypatch, tmp_path):
    binary = tmp_path / "volume_detector"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(native_bridge, "native_detector_path", lambda: binary)
    monkeypatch.setattr(
        native_bridge.subprocess,
        "run",
        _fake_run_factory(None, returncode=3, stderr="boom", write_output=False),
    )

    with pytest.raises(RuntimeError, match="exit 3"):
        detect_cavities_native([_face(1, 2, 3)], [(0, 0, 0), (1, 0, 0), (0, 1, 0)])


def test_detect_cavities_native_raises_when_no_output(monkeypatch, tmp_path):
    binary = tmp_path / "volume_detector"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(native_bridge, "native_detector_path", lambda: binary)
    # Exit 0 but the tool never wrote the JSON output file.
    monkeypatch.setattr(
        native_bridge.subprocess, "run", _fake_run_factory(None, write_output=False)
    )

    with pytest.raises(RuntimeError, match="no JSON output"):
        detect_cavities_native([_face(1, 2, 3)], [(0, 0, 0), (1, 0, 0), (0, 1, 0)])


def test_detect_cavities_native_raises_on_bad_json(monkeypatch, tmp_path):
    binary = tmp_path / "volume_detector"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(native_bridge, "native_detector_path", lambda: binary)

    def _bad_run(cmd, **kwargs):
        Path(cmd[2]).write_text("{ this is not json")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native_bridge.subprocess, "run", _bad_run)

    with pytest.raises(RuntimeError, match="Could not parse"):
        detect_cavities_native([_face(1, 2, 3)], [(0, 0, 0), (1, 0, 0), (0, 1, 0)])


def test_detect_cavities_native_timeout(monkeypatch, tmp_path):
    import subprocess as _sp

    binary = tmp_path / "volume_detector"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(native_bridge, "native_detector_path", lambda: binary)

    def _raise_timeout(cmd, **kwargs):
        raise _sp.TimeoutExpired(cmd, timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(native_bridge.subprocess, "run", _raise_timeout)

    with pytest.raises(RuntimeError, match="timed out"):
        detect_cavities_native([_face(1, 2, 3)], [(0, 0, 0), (1, 0, 0), (0, 1, 0)])


def test_detect_cavities_native_returns_empty_for_no_volumes(monkeypatch, tmp_path):
    binary = tmp_path / "volume_detector"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(native_bridge, "native_detector_path", lambda: binary)
    monkeypatch.setattr(native_bridge.subprocess, "run", _fake_run_factory({"volumes": []}))

    assert detect_cavities_native([_face(1, 2, 3)], [(0, 0, 0), (1, 0, 0), (0, 1, 0)]) == []


# --- voxel detector (real trimesh/scipy; skipped if unavailable) -------------


def test_voxel_detect_cavities_on_unit_cube(unit_cube):
    pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    from geometry_pipeline.cavity_detection.cavity_detector import detect_cavities

    vertices = [(v.x, v.y, v.z) for v in unit_cube.vertices]
    cavities = detect_cavities(unit_cube.faces, vertices, pitch=0.1)

    assert len(cavities) >= 1
    room = cavities[0]
    assert room.name == "Room"
    assert room.volume > 0
    assert room.oriented_faces  # every bounding face is assigned to the room


def test_voxel_detect_cavities_empty_input():
    pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    from geometry_pipeline.cavity_detection.cavity_detector import detect_cavities

    assert detect_cavities([], []) == []

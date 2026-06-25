"""Bridge to the native CGAL cavity detector.

This module runs the compiled C++ detector as a subprocess and converts its
JSON output to `Cavity` objects. The native detector is the production
detector; if it's not built, callers should surface an error (no fallback).

The compiled binary and its `VOLUME_DETECTOR_BIN` env var keep the historical
``volume_detector`` name (that is the build/Docker contract); only the Python
layer uses the canonical ``cavity`` term.

Data contract
-------------
We serialize the mesh IR directly to a small JSON document (no OBJ round-trip):

        {
            "vertices": [[x, y, z], ...],
            "faces":    [[i0, i1, i2, ...], ...]   # 0-based indices into vertices
        }

Face order matches the index order of the ``faces`` list, so the C++ ``face_id``
maps directly back to the Python face index. The tool writes JSON describing
per-detected bounded volume which original faces bound it and with what
orientation sign. We convert that into :class:`Cavity`.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from geometry_pipeline.core.ir import Cavity

logger = logging.getLogger(__name__)

# Repository root: src/geometry_pipeline/cavity_detection/native_bridge.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Allow overriding the binary location (e.g. in Docker) via env var.
_DEFAULT_BINARY = _REPO_ROOT / "bin" / "volume_detector"


def native_detector_path() -> Optional[Path]:
    """Return the path to the compiled native detector, or None if absent."""
    override = os.environ.get("VOLUME_DETECTOR_BIN")
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.append(_DEFAULT_BINARY)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def is_native_detector_available() -> bool:
    """True when the optional native detector binary is built and runnable."""
    return native_detector_path() is not None


def _write_mesh_json(
    faces,
    unique_vertices: Sequence[Tuple[float, float, float]],
    json_path: Path,
) -> None:
    """Serialize the mesh IR to JSON consumed by the native detector.

    ``faces[i]`` -> the *i*-th entry of ``"faces"`` -> C++ ``face_id == i``.
    ``FaceRecord.vertex_indices`` are 1-based vertex indices; we emit 0-based indices.
    """
    payload = {
        "vertices": [[float(x), float(y), float(z)] for (x, y, z) in unique_vertices],
        "faces": [[int(v) - 1 for v in face.vertex_indices] for face in faces],
    }
    json_path.write_text(json.dumps(payload))


def _cavities_from_json(payload: dict) -> List[Cavity]:
    """Convert the native tool's JSON into ``Cavity`` objects."""
    cavities: List[Cavity] = []
    for vol in payload.get("volumes", []):
        vid = int(vol["volume_id"])
        oriented_faces: List[Tuple[int, int]] = []
        is_manifold = bool(vol.get("is_manifold", True))
        for face in vol.get("faces", []):
            face_idx = int(face["face_id"])
            sign = int(face.get("sign", 1))
            oriented_faces.append((face_idx, 1 if sign >= 0 else -1))
        if not oriented_faces:
            continue
        cavities.append(
            Cavity(
                id=vid,
                name=f"Cavity_{vid}" if vid > 0 else "RoomVolume",
                volume=0.0,  # native tool does not compute metric volume
                oriented_faces=oriented_faces,
                is_manifold=is_manifold,
            )
        )
    return cavities


def detect_cavities_native(
    faces,
    unique_vertices: Sequence[Tuple[float, float, float]],
    *,
    timeout: float = 120.0,
) -> List[Cavity]:
    """Run the native detector and return detected cavities.

    Raises
    ------
    FileNotFoundError
        If the native binary is not available.
    RuntimeError
        If the native tool fails or produces no parseable output.
    """
    binary = native_detector_path()
    if binary is None:
        raise FileNotFoundError(
            "Native volume detector not found. Build it with ./app/geometry/volume/_native/build.sh "
            "or set VOLUME_DETECTOR_BIN."
        )

    logger.info(
        "Native volume detector: binary=%s, faces=%d, vertices=%d",
        binary, len(faces), len(unique_vertices),
    )

    with tempfile.TemporaryDirectory(prefix="volume_detector_") as tmp:
        tmp_dir = Path(tmp)
        mesh_path = tmp_dir / "mesh.json"
        json_path = tmp_dir / "volumes.json"
        _write_mesh_json(faces, unique_vertices, mesh_path)
        mesh_raw = mesh_path.read_text()

        try:
            proc = subprocess.run(
                [str(binary), str(mesh_path), str(json_path)],
                cwd=tmp_dir,  # keep generated-obj-volume/ out of the repo
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Native volume detector timed out after {timeout}s."
            ) from exc

        # Always surface the tool's own diagnostics; the C++ detector prints
        # progress/decisions to stdout/stderr which are essential when the
        # result differs between environments (e.g. local vs Docker).
        if proc.stdout and proc.stdout.strip():
            logger.debug("Native detector stdout:\n%s", proc.stdout.strip())
        if proc.stderr and proc.stderr.strip():
            logger.warning("Native detector stderr:\n%s", proc.stderr.strip())

        if proc.returncode != 0:
            raise RuntimeError(
                "Native volume detector failed "
                f"(exit {proc.returncode}): {proc.stderr.strip()}"
            )

        if not json_path.is_file():
            raise RuntimeError(
                "Native volume detector produced no JSON output."
            )

        raw = json_path.read_text()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Could not parse native detector JSON: {exc}"
            ) from exc

    cavities = _cavities_from_json(payload)
    n_volumes = len(payload.get("volumes", []))

    # Optional: persist the exact input/output for cross-environment debugging.
    # Set VOLUME_DETECTOR_DEBUG_DIR=/app/uploads to capture mesh.json + volumes.json.
    debug_dir = os.environ.get("VOLUME_DETECTOR_DEBUG_DIR")
    if debug_dir:
        try:
            out = Path(debug_dir)
            out.mkdir(parents=True, exist_ok=True)
            # (out / "last_mesh.json").write_text(mesh_raw)
            # (out / "last_volumes.json").write_text(raw)
            # logger.info("Wrote detector debug files to %s", out)
        except Exception as exc:  # never let debugging break the pipeline
            logger.warning("Could not write detector debug files: %s", exc)

    if not cavities:
        logger.warning(
            "Native volume detector ran successfully (exit 0) but reported "
            "%d volume(s) and yielded 0 usable cavities. The GEO will fall "
            "back to a single volume. Raw output: %s",
            n_volumes, raw[:2000],
        )
    else:
        logger.info(
            "Native volume detector produced %d cavity/cavities from %d "
            "reported volume(s).",
            len(cavities), n_volumes,
        )
    return cavities

"""Cavity detection.

Houses detector implementations plus native C++ sources under `_native/`
(`volume_detector.cpp`, `build.sh`, `CMakeLists.txt`). The native binary keeps
its historical ``volume_detector`` name for the build/Docker contract; the
Python API standardizes on the ``cavity`` term.
"""
from __future__ import annotations

from geometry_pipeline.cavity_detection.cavity_detector import detect_cavities
from geometry_pipeline.cavity_detection.native_bridge import (
    detect_cavities_native,
    is_native_detector_available,
    native_detector_path,
)

__all__ = [
    "detect_cavities",
    "detect_cavities_native",
    "is_native_detector_available",
    "native_detector_path",
]

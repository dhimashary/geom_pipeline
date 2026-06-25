#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${REPO_ROOT}/bin"

echo "[build] configuring (build dir: ${BUILD_DIR})"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo "[build] compiling"
cmake --build "${BUILD_DIR}" --config Release -j

mkdir -p "${BIN_DIR}"
cp "${BUILD_DIR}/volume_detector" "${BIN_DIR}/volume_detector"

echo "[build] done -> ${BIN_DIR}/volume_detector"

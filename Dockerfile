# syntax=docker/dockerfile:1

###############################################################################
# Stage 1 — build the wheel. scikit-build-core compiles the native CGAL
# "volume_detector" kernel into it, so this is the only stage that needs a
# C++ toolchain + CGAL/Eigen/GMP/MPFR headers.
###############################################################################
FROM python:3.11-slim AS wheel-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libcgal-dev \
        libeigen3-dev \
        libgmp-dev \
        libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /src/
COPY src/ /src/src/
RUN pip wheel --no-cache-dir --no-deps -w /wheels /src

###############################################################################
# Stage 2 — runtime image with the Python package installed.
###############################################################################
FROM python:3.11-slim AS runtime

# Runtime libs needed by the wheels (gmsh needs GL/GLU/gomp; CGAL kernel needs
# the GMP/MPFR shared libraries at run time).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa \
        libgomp1 \
        libgmp10 \
        libmpfr6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wheel-build /wheels /wheels
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

# Smoke test: the package imports and the native kernel was built into it.
# The CMake build skips the kernel quietly when CGAL is missing, so fail here.
RUN python -c "from geometry_pipeline.cavity_detection.native_bridge import native_detector_path as p; assert p(), 'volume_detector missing'; print('geometry_pipeline OK:', p())"

CMD ["python", "-c", "import geometry_pipeline; print(geometry_pipeline.__name__)"]

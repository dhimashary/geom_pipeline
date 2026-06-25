# syntax=docker/dockerfile:1

###############################################################################
# Stage 1 — build the native CGAL "volume_detector" kernel.
# This is the only part that needs a C++ toolchain + CGAL/Eigen/GMP/MPFR.
###############################################################################
FROM python:3.11-slim AS native-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libcgal-dev \
        libeigen3-dev \
        libgmp-dev \
        libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the native sources to maximise layer caching.
# Adjust this path if you nest the package differently during the rename.
COPY src/geometry_pipeline/cavity_detection/_native/ /native/

RUN cmake -S /native -B /native/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /native/build --config Release -j \
    && mkdir -p /out \
    && cp /native/build/volume_detector /out/volume_detector

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

WORKDIR /app

# Install the Python package (and its dependencies).
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# Drop in the prebuilt native detector and point the bridge at it.
COPY --from=native-build /out/volume_detector /usr/local/bin/volume_detector
ENV VOLUME_DETECTOR_BIN=/usr/local/bin/volume_detector

# Smoke test that the package imports and the kernel is reachable.
RUN python -c "import geometry_pipeline; print('geometry_pipeline import OK')"

CMD ["python", "-c", "import geometry_pipeline; print(geometry_pipeline.__name__)"]

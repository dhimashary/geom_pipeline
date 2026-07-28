# geometry_pipeline

A detached geometry-processing pipeline for **CHORAS**. It imports room geometry
from CAD formats, **validates** it, **repairs** common mesh defects, and
**exports** solver-ready files (OBJ, Rhino 3DM, Gmsh `.geo`) together with a
structured issue report.

The pipeline is organised as **ports-and-adapters**: a small public facade wraps
a registry-driven set of importers, validators, repairs, and exporters that are
wired together by a *profile* (one profile per simulation method).

---

## Installation

Requires **Python ≥ 3.11**.

```powershell
# from the geometry-pipeline/ folder
pip install -e .
```

For development (tests, linting, type checking):

```powershell
pip install -e ".[dev]"
```

Key runtime dependencies (pinned in [pyproject.toml](pyproject.toml)): `numpy`,
`scipy`, `shapely`, `rhino3dm`, `ezdxf`, `trimesh`, `numpy-stl`, `gmsh`,
`meshio`.

> **Native cavity detector (optional).** Cavity detection has a fast C++ kernel
> under `src/geometry_pipeline/cavity_detection/_native/`. The Docker image
> compiles it and sets `VOLUME_DETECTOR_BIN`. Without the binary the pipeline
> falls back to the pure-Python voxel detector — no action needed for local use.

---

## Usage

The **only** supported public surface is the facade in
`geometry_pipeline.api` (everything else is an implementation detail).

Supported input formats: **`.obj`**, **`.3dm`**, **`.dxf`**
(`geometry_pipeline.SUPPORTED_INPUTS`).

### Repair + export

```python
from geometry_pipeline import repair_geometry

result = repair_geometry(
    input_path="room.3dm",
    output_dir="out/",
    volume_name="RoomVolume",
    detect_cavities=True,   # emit one Gmsh volume per enclosed region
)

print(result.issue_count)   # total detected issues
print(result.outputs)       # {"obj": "out/room.obj", "geo": "out/room.geo", ...}
print(result.report)        # {"pre": {...}, "post": {...}, "repairs": {...}}
```

Both functions return a `GeometryResult`:

| Field          | Meaning                                                        |
| -------------- | ------------------------------------------------------------- |
| `outputs`      | `dict[str, str]` of exporter name → written file path         |
| `issue_report` | frontend-shaped issue dict (from the *initial* detection pass) |
| `report`       | `{pre, post, repairs}` snapshot across the run                |
| `issue_count`  | total number of detected issues                               |

All failures are raised as a single `GeometryError`.

---

## Folder structure

```
geometry-pipeline/
├─ pyproject.toml              # packaging, deps, pytest config
├─ Dockerfile                  # multi-stage build (compiles native kernel)
├─ src/geometry_pipeline/
│  ├─ api.py                   # PUBLIC facade: repair_geometry / process_geometry
│  ├─ __init__.py              # re-exports the facade
│  ├─ py.typed                 # PEP 561 type marker
│  ├─ core/                    # domain layer (stdlib only)
│  │  ├─ ir.py                 #   geometry IR: Mesh, BRep, Face, Cavity, …
│  │  ├─ issues.py             #   Issue, IssueKind, DetectionStage
│  │  ├─ profile.py            #   SimulationProfile + Stage dataclasses
│  │  ├─ tolerances.py         #   numeric thresholds / caps
│  │  ├─ context.py            #   per-run Context (tolerances, logger)
│  │  └─ report.py             #   PipelineResult / RepairResult
│  ├─ io/                      # adapters (registry-based)
│  │  ├─ registry.py           #   ImporterRegistry / ExporterRegistry
│  │  ├─ importers/            #   obj, rhino (3dm), dxf
│  │  └─ exporters/            #   obj, 3dm, gmsh .geo
│  ├─ validators/              # detection (read-only)
│  │  ├─ base.py               #   Validator protocol + BaseValidator
│  │  └─ mesh/                 #   mesh-specific validators + _common helpers
│  ├─ repairs/                 # mutation steps
│  │  ├─ base.py               #   RepairStep protocol + BaseRepair
│  │  └─ mesh/                 #   mesh-specific repairs + _common helpers
│  ├─ profiles/                # explicit stage wiring, one per solver
│  │  ├─ wave_based.py         #   FEM/FDTD profile (full + inspect)
│  │  └─ ray_tracing.py
│  ├─ pipeline/runner.py       # executes a profile against geometry
│  ├─ conversion/              # IR ↔ IR converters (e.g. brep → mesh)
│  ├─ cavity_detection/        # enclosed-region detection (Python + native C++)
│  ├─ geometry_math/           # pure geometric predicates / triangulation
│  └─ reporting/               # Issue → frontend JSON translation + writers
└─ tests/
```

**Layering rule:** `core/` depends on nothing but the stdlib. Validators and
repairs are split by **geometry kind** (currently only `mesh/`); add a
`brep/` sibling when you implement Brep-native steps. Each validator/repair
declares `accepts = {"mesh"}`, and a profile's `__post_init__` rejects any
component whose `accepts` doesn't match its `target_ir.kind`.

---

## Adding a new profile

A **profile** (`SimulationProfile`) is the explicit wiring of validators,
repair stages, and exporters for one simulation method. There is no registry —
you build and return the object from a factory function.

A profile has five moving parts:

- `pre_validators` — detection on the raw input (before any repair)
- `stages` — ordered list of `Stage(repairs=[...], post_validators=[...])`
- `final_validators` — detection after all stages
- `exporters` — what gets written to disk
- `tolerances` — numeric thresholds for this run

### Example: `examples/my_profile.py`

```python
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.profile import SimulationProfile, Stage
from geometry_pipeline.core.tolerances import Tolerances
from geometry_pipeline.io.registry import ExporterRegistry

# Validators / repairs live under the per-kind subpackage.
from geometry_pipeline.validators.mesh.duplicate_vertices import DuplicateVerticesValidator
from geometry_pipeline.validators.mesh.degenerate_faces import ZeroAreaFaceValidator
from geometry_pipeline.repairs.mesh.deduplicate_vertices import DeduplicateVerticesRepair
from geometry_pipeline.repairs.mesh.remove_degenerate_faces import RemoveZeroAreaFaceRepair


def my_profile() -> SimulationProfile:
    """A minimal clean-up profile: dedupe + drop degenerate faces, then export OBJ."""
    return SimulationProfile(
        name="my_profile",
        target_ir=Mesh,                       # this profile operates on meshes
        pre_validators=[
            DuplicateVerticesValidator(),
            ZeroAreaFaceValidator(),
        ],
        stages=[
            Stage(
                name="cleanup",
                repairs=[
                    DeduplicateVerticesRepair(),
                    RemoveZeroAreaFaceRepair(),
                ],
                # re-detect after the repairs in this stage:
                post_validators=[ZeroAreaFaceValidator()],
            ),
        ],
        final_validators=[ZeroAreaFaceValidator()],
        exporters=[
            ExporterRegistry.get("obj", Mesh.kind),
        ],
        tolerances=Tolerances(),
    )
```

Run it through the pipeline directly:

```python
import logging
from pathlib import Path
from geometry_pipeline.core.context import Context
from geometry_pipeline.core.tolerances import Tolerances
from geometry_pipeline.io.registry import ImporterRegistry
from geometry_pipeline.pipeline.runner import run_pipeline
from examples.my_profile import my_profile

geom = ImporterRegistry.for_extension(".obj").load(Path("room.obj"))
profile = my_profile()
ctx = Context(tolerances=Tolerances(), logger=logging.getLogger("choras_geometry"),
              profile_name=profile.name)
result = run_pipeline(geom, profile, Path("out/room"), ctx)
```

**Tips**

- **Stage order matters.** Detectors only produce meaningful results once their
  preconditions are repaired (e.g. T-junctions must be fixed before
  intersection/hole detection — see `profiles/wave_based.py`).
- **Match the IR kind.** Every component must `accept` the profile's
  `target_ir.kind`, or `SimulationProfile.__post_init__` raises a `ValueError`
  listing the mismatches.
- **Reuse exporters via the registry:** `ExporterRegistry.get("obj", Mesh.kind)`.
- To expose the profile through the facade, add a wrapper in
  `geometry_pipeline/api.py` (mirroring `repair_geometry`).

---

## Development

```powershell
# run the test suite (scoped to this package)
python -m pytest tests/ -q

# type-check and lint (with the dev extras installed)
mypy src/geometry_pipeline
ruff check src
```

## License

MIT — see [LICENSE](../LICENSE).

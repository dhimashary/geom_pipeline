# Contributing & development guide

This guide covers how to extend and develop **geometry_pipeline**. For the
package description, installation, and usage see [README.new.md](README.new.md).

The pipeline is organised as **ports-and-adapters**: a small public facade wraps
a registry-driven set of importers, validators, repairs, and exporters that are
wired together by a *profile*.

**Layering rule:** `core/` depends on nothing but the stdlib. Validators and
repairs are split by **geometry kind** (currently only `mesh/`). Each
validator/repair declares `accepts = {"mesh"}`, and a profile's `__post_init__`
rejects any component whose `accepts` doesn't match its `target_ir.kind`.

---

## Adding a new validator

A **validator** is read-only detection: it inspects geometry and returns a list
of `Issue`s. Subclass `BaseValidator` and implement `detect_raw`, which returns
legacy-style detector dicts (the base class caps, summarises, and converts them
into `Issue`s for you).

1. Add a new `IssueKind` to `core/issues.py` if you are detecting a defect that
   does not already have one.
2. Create the validator under `validators/mesh/` (or a new per-kind subpackage):

```python
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import IssueKind
from geometry_pipeline.validators.base import BaseValidator


class MyDefectValidator(BaseValidator):
    name: ClassVar[str] = "my_defect"
    accepts: ClassVar[set[str]] = {"mesh"}
    kind: ClassVar[IssueKind] = IssueKind.MY_DEFECT

    def detect_raw(self, geom: Mesh, ctx: Context) -> list[dict]:
        # return one dict per detected instance, e.g.:
        # {"elements": {"type": "face", "points": [[x, y, z], ...]}}
        return []
```

3. Wire it into a profile's `pre_validators` / stage `post_validators` /
   `final_validators` (see `profiles/default.py`).

---

## Adding a new repair

A **repair** mutates geometry to remove a defect. Subclass `BaseRepair`,
declare the `IssueKind`s it `handles`, and implement `apply`, returning the new
geometry plus a `RepairResult`.

1. Create the repair under `repairs/mesh/`:

```python
from typing import ClassVar

from geometry_pipeline.core.context import Context
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.issues import Issue, IssueKind
from geometry_pipeline.core.report import RepairResult
from geometry_pipeline.repairs.base import BaseRepair


class MyDefectRepair(BaseRepair):
    name: ClassVar[str] = "fix_my_defect"
    accepts: ClassVar[set[str]] = {"mesh"}
    handles: ClassVar[set[IssueKind]] = {IssueKind.MY_DEFECT}

    def apply(
        self,
        geom: Mesh,
        issues: list[Issue],
        ctx: Context,
        stage_name: str = "",
    ) -> tuple[Mesh, RepairResult]:
        before = len(self.affected_ids(issues))
        new_mesh = geom  # ... perform the mutation ...
        result = self.make_result(
            stage_name=stage_name,
            before_count=before,
            after_count=0,
            details={},
            issues=issues,
        )
        return new_mesh, result
```

2. Export the class from `repairs/mesh/<module>.py` and add it to the
   `__all__` list in `repairs/__init__.py`. This is what lets
   `list_issue_kinds()` report the defect as `repairable`.
3. Add the repair to a `Stage` in `profiles/default.py`. **Stage order
   matters** — detectors only produce meaningful results once their
   preconditions are repaired (e.g. T-junctions must be fixed before
   intersection/hole detection).

---

## Adding a new exporter

An **exporter** writes a geometry IR to disk. It implements the `Exporter`
protocol (`write(geom, path)`), optionally with a `path_for(base_path)` helper
that decides the final filename.

1. Create the exporter under `io/exporters/`:

```python
from pathlib import Path

from geometry_pipeline.core.ir import Mesh


class MyFormatExporter:
    def path_for(self, base_path: Path) -> Path:
        base_path = Path(base_path)
        return base_path.with_name(base_path.name + ".myfmt")

    def write(self, geom: Mesh, path: Path) -> None: ...  # serialise `geom` to `path`
```

2. Register it with the `ExporterRegistry` keyed by `(format, IR kind)` — either
   call `ExporterRegistry.register("myfmt", Mesh.kind, MyFormatExporter)` at
   import time, or add it to `ExporterRegistry._register_builtins` in
   `io/registry.py`.
3. Add it to the `exporters` list of a profile:
   `ExporterRegistry.get("myfmt", Mesh.kind)`.

---

## Wiring it into the default profile

A class does nothing until a profile references it. `default_profile()` in
`profiles/default.py` builds four lists — append yours to the right one:

| List | Runs on | Built by |
| --- | --- | --- |
| `pre_validators` | raw input (baseline report) | `_default_pre_validators` |
| `stages[].repairs` / `.post_validators` | mid-pipeline, in order | `_default_stages` |
| `final_validators` | repaired mesh (final report) | `_default_final_validators` |
| `exporters` | end of run | inline in `default_profile` |

```python
# profiles/default.py

# validator — add to any list that should detect it (often several):
def _default_pre_validators(tjunc, intersect):
    return [..., MyDefectValidator()]  # baseline report


# repair — add to a Stage; ORDER MATTERS (fix preconditions first):
Stage(
    name="my_defect", repairs=[MyDefectRepair()], post_validators=[MyDefectValidator()]
)  # measure right after the fix

# exporter — end-of-run artifact:
exporters = [..., ExporterRegistry.get("myfmt", Mesh.kind)]

# ...or an INTERMEDIATE artifact: attach to a Stage (like the tjunc checkpoint):
Stage(name="t_junctions", repairs=[...], exporters=[MyFormatExporter()])
```

Every component's `accepts` must include the profile's `target_ir.kind`
(`"mesh"`), else `SimulationProfile.__post_init__` raises `ValueError`.

---

## Adding a new profile

A **profile** (`SimulationProfile`) wires one full run: `target_ir`,
`pre_validators`, ordered `stages`, `final_validators`, `exporters`,
`tolerances`. `default_profile()` is the only one shipped today.

Add a *new* profile (rather than editing `default_profile`) only when the run
needs a different **shape**:

- a different **stage graph / order** (e.g. inspect-only: detect + export, no repairs);
- a different **component set or tolerances** for a specific solver;
- a **new IR kind** — a profile is bound to one `target_ir`, so a non-`Mesh` IR
  needs its own profile ([Option 2](#option-2--declare-a-new-ir-kind)).

Just adding one validator/repair/exporter to the `Mesh` flow? Wire it into
`default_profile` instead (above).

```python
# profiles/inspect_only.py — mirror default_profile, return a SimulationProfile
def inspect_only_profile() -> SimulationProfile:
    tjunc, intersect = TJunctionsValidator(), IntersectionsValidator()
    return SimulationProfile(
        name="inspect_only",
        target_ir=Mesh,  # one IR kind for the whole profile
        pre_validators=_default_pre_validators(tjunc, intersect),
        stages=[],  # no repairs -> detect + export only
        final_validators=[],
        exporters=[JsonReportWriter(issue_suffix="_issue.json", issue_source="composite")],
        tolerances=Tolerances(),
    )  # __post_init__ validates IR-kind agreement on construction
```

Profiles aren't auto-discovered — `api.py` hardcodes the default in **both**
`repair_geometry` and `process_geometry`:

```python
# api.py — change this call site to reach a new profile:
profile = default_profile(detect_cavities=detect_cavities, volume_name=volume_name)
```

Either add a `profile_factory=...` parameter, or select by `geom.kind` via a
registry ([Option 2](#option-2--declare-a-new-ir-kind), step 6).

---

## IR support: currently Mesh-only

The pipeline's **IR** lives in `core/ir.py`. Today it defines one geometry
variant, `Mesh` (vertices + faces) with `kind = "mesh"`. It's a *tagged union*,
so more variants (e.g. a `BRep` or `PointCloud`) could be added later — but
those are illustrative, not implemented.

Everything runnable is wired to `Mesh`: every validator/repair declares
`accepts = {"mesh"}`, `default_profile` sets `target_ir = Mesh`, importers
return `Mesh` (only the OBJ importer works; `ConverterRegistry` is a stub), and
the facade hardcodes the profile:

```python
# api.py — the profile is fixed, not chosen from the loaded geometry:
profile = default_profile(detect_cavities=detect_cavities, volume_name=volume_name)
```

To support a new input, pick one of the two options below.

### Option 1 — Convert the new input to `Mesh` (preferred)

Reuse the entire `Mesh` pipeline; just add an importer that returns a `Mesh`.

```python
# io/importers/skp.py
from pathlib import Path
from typing import ClassVar

from geometry_pipeline.core.ir import Mesh


class SkpImporter:
    extensions: ClassVar[tuple[str, ...]] = (".skp",)

    def load(self, path: Path) -> Mesh:
        # parse straight into Mesh(vertices=[...], faces=[...]), OR parse to an
        # intermediate rep and tessellate: ConverterRegistry.convert(geom, Mesh)
        # (that registry is still a stub — implement the converter first).
        ...  # return a Mesh in IR coordinates (Z-up)
```

1. **Register** — add to `ImporterRegistry._register_builtins` in `io/registry.py` (or call `ImporterRegistry.register(SkpImporter)`).
2. **Advertise** — add `".skp"` to `SUPPORTED_INPUTS` in `api.py`.

Done: `default_profile` and every mesh validator/repair run unchanged, because
the loaded geometry is still a `Mesh`.

### Option 2 — Declare a new IR kind

Only when the geometry genuinely can't be a face/vertex mesh. Detection, repair,
export **and** profile selection all become per-kind:

1. **IR variant** in `core/ir.py` — a dataclass with a unique `kind` (like `Mesh.kind = "mesh"`).
2. **Validators/repairs** under `validators/<kind>/` and `repairs/<kind>/`, each `accepts = {"<kind>"}`; export repairs from `repairs/__init__.py` so `list_issue_kinds()` sees them.
3. **Exporters** registered with `ExporterRegistry` keyed by `(format, "<kind>")`.
4. **Importer** returning the new IR, registered in `ImporterRegistry`, extension in `SUPPORTED_INPUTS`.
5. **Profile factory** with `target_ir = <NewIR>` wiring the above (`__post_init__` enforces IR-kind agreement).
6. **Make the facade IR-aware** — select the profile by `geom.kind` in **both** entrypoints:

```python
# api.py — replaces the hardcoded default_profile(...):
PROFILE_BY_IR_KIND = {Mesh.kind: default_profile, NewIR.kind: new_ir_profile}

geom = ImporterRegistry.for_extension(in_path.suffix).load(in_path)
profile = PROFILE_BY_IR_KIND[geom.kind](detect_cavities=detect_cavities, volume_name=volume_name)
```

**Rule of thumb:** tessellable into vertices + faces -> Option 1 (one importer,
reuse everything). Otherwise Option 2.

---

## Development

Install the dev extras first (`pip install -e ".[dev]"`), then:

```powershell
# run the test suite (scoped to this package)
python -m pytest tests/ -q

# type-check and lint
mypy src/geometry_pipeline
ruff check src
```

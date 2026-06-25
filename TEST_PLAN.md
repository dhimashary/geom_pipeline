# Geometry Pipeline — Test Plan

A staged plan for testing the geometry processing pipeline. The pipeline follows a
layered flow: **import → validate → repair → export**, orchestrated by
`pipeline/runner.py` and exposed through the `api.py` facade.

The plan is ordered by **value per effort**: pure-function unit tests first, then
validators/repairs, then integration and contract tests.

---

## 1. Goals & Principles

- **Determinism first.** Geometry math and IR transforms are pure — test exact
  (or `pytest.approx`) outputs.
- **Test our code, not third-party libs.** Wrap `trimesh`, `gmsh`, `rhino3dm`,
  `shapely` thinly; test the wrappers, not the libraries.
- **One defect per fixture.** Each validator/repair fixture isolates a single
  issue so failures point to a single cause.
- **Guard against false positives.** Every validator must also be tested against
  a clean mesh that reports **zero** issues.
- **Pin contracts.** The frontend report schema and `issue_count` are external
  contracts — snapshot and assert them.

---

## 2. Tooling

| Need | Tool | Status |
|------|------|--------|
| Test runner | `pytest` | configured (`[tool.pytest.ini_options]`) |
| Coverage | `pytest-cov` | dev dependency present |
| Parametrized cases | `@pytest.mark.parametrize` | built-in |
| Float assertions | `pytest.approx` | built-in |
| Shared mesh builders | `tests/conftest.py` | **to add** |
| Property-based tests (optional) | `hypothesis` | **to add (optional)** |

Run commands:

```bash
pytest                       # full suite
pytest --cov=geometry_pipeline --cov-report=term-missing
pytest tests/validators -q   # one layer
```

---

## 3. Proposed Test Layout

```
tests/
  conftest.py                 # shared mesh builders & fixtures
  fixtures/                   # small real files: *.obj, *.3dm, *.dxf + golden reports
  unit/
    test_geometry_math.py
    test_predicates.py
    test_triangulation.py
    test_tolerances.py
    test_diff.py
    test_ir.py
    test_jsonable.py          # exists — move here
  validators/
    test_boundary_edges.py
    test_degenerate_faces.py
    test_duplicate_vertices.py
    test_intersections.py
    test_non_planar_faces.py
    test_overlapping_faces.py
    test_possible_holes.py
    test_small_faces.py
    test_t_junctions.py
  repairs/
    test_deduplicate_vertices.py
    test_compact_vertices.py
    test_fix_t_junctions.py
    test_orient_consistent.py
    test_orient_outward.py
    test_remove_degenerate_faces.py
    test_repair_intersections.py
    test_sort_vertices.py
  integration/
    test_pipeline_runner.py
    test_profiles.py
  contract/
    test_api_facade.py
    test_frontend_schema.py    # exists — move here
    test_json_writer.py
```

---

## 4. Test Stages

### Stage 1 — Unit tests (highest priority)
Pure, dependency-light modules in `geometry_math/` and `core/`.

- `geometry_math/predicates.py` — point/face predicates on known inputs.
- `geometry_math/geometry_math.py` — areas, normals, distances (use `approx`).
- `geometry_math/triangulation.py` — triangulation of known polygons.
- `core/tolerances.py` — default/override resolution and boundary behavior.
- `core/diff.py` — IR diffing produces expected change sets.
- `core/ir.py` — IR construction/round-trip invariants.
- `core/jsonable.py` — numpy → JSON-native coercion *(already covered)*.

**Exit criteria:** every public function in `geometry_math/` and `core/` has at
least one happy-path and one edge-case test.

### Stage 2 — Validator tests (table-driven)
Each `validators/mesh/*.py` is parametrized over crafted meshes.

For each validator:
- **Positive:** a mesh with exactly one instance of the defect → detected, with
  correct count / `fid` / severity.
- **Negative:** a clean mesh → zero issues (no false positives).
- **Tolerance boundary:** a near-threshold case on both sides of the configured
  tolerance (directly relevant to `possible_holes` and `overlapping_faces`).

**Exit criteria:** every validator has positive, negative, and boundary cases.

### Stage 3 — Repair tests (invariant / property based)
Each `repairs/mesh/*.py` is tested by invariant, not byte-equality.

- **Round-trip:** validate → repair → validate; the targeted issue is gone and
  **no new issues** are introduced.
- **Invariants per repair:**
  - `deduplicate_vertices`, `compact_vertices` → vertex count only decreases.
  - `orient_consistent`, `orient_outward` → consistent/outward normals after.
  - `remove_degenerate_faces` → no zero-area faces remain.
  - `fix_t_junctions` → no t-junctions remain; topology otherwise preserved.
  - `repair_intersections` → no self-intersections remain.
- Optional `hypothesis` strategies generating random valid meshes to assert
  invariants hold broadly.

**Exit criteria:** each repair has a round-trip test and its specific invariant
assertions.

### Stage 4 — Integration / pipeline tests
- `pipeline/runner.py` — stages execute in order (PRE/POST), issue report
  accumulates correctly across stages.
- `profiles/wave_based.py`, `profiles/ray_tracing.py` — run end-to-end on a small
  in-memory mesh; assert selected validators/repairs are applied.

**Exit criteria:** at least one full pipeline run per profile on a tiny fixture.

### Stage 5 — Contract / facade tests
- `api.repair_geometry()` on small real fixture files (`.obj`, `.3dm`, `.dxf`):
  - returns a well-formed `GeometryResult`,
  - writes output files to the target dir,
  - **`issue_count` matches a pinned golden value** (regression guard for the
    `_count_issues` bug fixed in `80fcea2`).
- `reporting/frontend_schema.py` + `reporting/json_writer.py` — snapshot the
  emitted report against a committed golden JSON; schema changes must be
  intentional.

**Exit criteria:** one end-to-end facade test per supported input format and a
pinned schema snapshot.

---

## 5. Fixtures Strategy

- **Synthetic meshes** built in `conftest.py` (cube, quad with a hole, degenerate
  triangle, t-junction pair, intersecting faces) — fast, explicit, no I/O.
- **Real files** under `tests/fixtures/` — one small `.obj`, `.3dm`, `.dxf` each,
  plus their golden reports. Keep them tiny to stay in-repo.
- Golden JSON reports stored alongside fixtures; regenerate intentionally via a
  documented command, never silently.

---

## 6. CI & Coverage

- Run `pytest --cov=geometry_pipeline` in CI on every push/PR.
- Coverage focus order: `geometry_math` → `core` → `validators` → `repairs` →
  `pipeline` → `api`/`reporting`.
- Add `ruff` and `mypy` checks as separate CI steps (both are dev deps).
- Suggested initial gate: fail CI under **70%** line coverage; raise over time.

---

## 7. Suggested Implementation Order

1. Add `tests/conftest.py` with reusable mesh builders.
2. Seed Stage 1 unit tests for `geometry_math` + `core`.
3. Add one validator + one repair test as a pattern, then fan out.
4. Add the `api.repair_geometry` contract test with a pinned `issue_count`.
5. Wire `pytest --cov` into CI with a coverage gate.

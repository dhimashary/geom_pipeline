"""Wave-based (FEM/FDTD) simulation profile.

The stage order encodes geometric dependencies:
  1. dedup / degenerate / orient — preconditions for the topology checks below.
  2. fix T-junctions FIRST: an unfixed T-junction looks like a hole AND like
     an intersection, so detecting those before this stage produces noise.
  3. fix intersections (PLC violations).
  4. detect remaining holes / boundary edges only at the end.
"""
from __future__ import annotations

from geometry_pipeline.io.exporters.mesh_geo import GmshGeoExporter
from geometry_pipeline.io.registry import ExporterRegistry
from geometry_pipeline.reporting.json_writer import JsonReportWriter
from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.profile import SimulationProfile, Stage
from geometry_pipeline.repairs.mesh.deduplicate_vertices import DeduplicateVerticesRepair
from geometry_pipeline.repairs.mesh.fix_t_junctions import FixTJunctionsIterativeRepair
from geometry_pipeline.repairs.mesh.orient_outward import FlipFacesIfMajorityInwardRepair
from geometry_pipeline.repairs.mesh.remove_degenerate_faces import RemoveDegenerateFacesRepair
from geometry_pipeline.repairs.mesh.repair_intersections import (
    RepairPlcByOffsetRepair,
    RepairPlcSingleSplitsRepair,
    TrimSegmentFaceIntersectionsRepair,
)
from geometry_pipeline.repairs.mesh.sort_vertices import SortVerticesDeterministicallyRepair
from geometry_pipeline.core.tolerances import Tolerances
from geometry_pipeline.validators.mesh.boundary_edges import BoundaryEdgesValidator
from geometry_pipeline.validators.mesh.degenerate_faces import DegenerateFacesValidator
from geometry_pipeline.validators.mesh.duplicate_vertices import DuplicateVerticesValidator
from geometry_pipeline.validators.mesh.intersections import IntersectionsValidator
from geometry_pipeline.validators.mesh.non_planar_faces import NonPlanarFacesValidator
from geometry_pipeline.validators.mesh.overlapping_faces import OverlappingFacesValidator
from geometry_pipeline.validators.mesh.possible_holes import PossibleHolesValidator
from geometry_pipeline.validators.mesh.small_faces import SmallFacesValidator
from geometry_pipeline.validators.mesh.collinear_faces import CollinearFacesValidator
from geometry_pipeline.validators.mesh.t_junctions import TJunctionsValidator


def _wave_based_pre_validators(tjunc, intersect) -> list:
    return [
        NonPlanarFacesValidator(),
        DuplicateVerticesValidator(),
        DegenerateFacesValidator(),
        OverlappingFacesValidator(),
        SmallFacesValidator(),
        CollinearFacesValidator(),
        tjunc,
        intersect,
        BoundaryEdgesValidator(),
        PossibleHolesValidator(),
    ]


def _wave_based_stages(tjunc, intersect, *, inspect: bool = False) -> list[Stage]:
    """Shared stage list: dedupe → tj → intersect → topology.

    The order is what makes detection meaningful (T-junctions only
    detectable after dedupe; intersections only meaningful after tj fix).
    Both the full and inspect-only profiles consume this same list so
    the diagnostic order can never drift between them.

    ``inspect=True`` reshapes the stage tail for diagnostic-only runs:
      - the ``orient`` stage gains a tjunc post-validator (so the report
        captures the *original* tjunc count, before any fix)
      - the ``t_junctions`` stage drops its tjunc post-validator (the fix
        runs only to denoise the intersection detector — we don't care
        about the residual count)
      - the ``intersections`` stage drops its repairs (detection only)
    """
    return [
        Stage(name="deduplication", repairs=[DeduplicateVerticesRepair()]),
        Stage(name="degenerate", repairs=[RemoveDegenerateFacesRepair()]),
        Stage(name="sort", repairs=[SortVerticesDeterministicallyRepair()]),
        Stage(
            name="orient",
            repairs=[FlipFacesIfMajorityInwardRepair()],
        ),
        Stage(
            name="t_junctions",
            repairs=[] if inspect else [FixTJunctionsIterativeRepair(detector=tjunc)],
            post_validators=[tjunc] if inspect else [],
        ),
        Stage(
            name="intersections",
            repairs=(
                [FixTJunctionsIterativeRepair(detector=tjunc)]
                if inspect
                else [
                    TrimSegmentFaceIntersectionsRepair(detector=intersect),
                    RepairPlcSingleSplitsRepair(detector=intersect),
                    # RepairPlcByOffsetRepair(detector=intersect),
                ]
            ),
            post_validators=[intersect, OverlappingFacesValidator()],
        ),
        Stage(
            name="topology",
            post_validators=[BoundaryEdgesValidator(), PossibleHolesValidator()],
        ),
    ]


def _wave_based_final_validators(tjunc, intersect) -> list:
    return [
        NonPlanarFacesValidator(),
        DegenerateFacesValidator(),
        OverlappingFacesValidator(),
        SmallFacesValidator(),
        CollinearFacesValidator(),
        tjunc,
        intersect,
        BoundaryEdgesValidator(),
        PossibleHolesValidator(),
    ]


def wave_based_profile(
    volume_name: str = "RoomVolume",
    *,
    detect_cavities: bool = False,
    cavity_pitch: float = 0.05,
    cavity_closing_iterations: int = 0,
) -> SimulationProfile:
    """Full profile: detect → repair → emit OBJ + GEO.

    When `detect_cavities=True`, the GEO exporter runs the voxel-based cavity
    detector and emits one `Volume` per enclosed region (required by Gmsh
    when the geometry contains nested/attached enclosed objects).
    """
    tjunc = TJunctionsValidator()
    intersect = IntersectionsValidator()
    return SimulationProfile(
        name="wave_based",
        target_ir=Mesh,
        pre_validators=_wave_based_pre_validators(tjunc, intersect),
        stages=_wave_based_stages(tjunc, intersect),
        final_validators=_wave_based_final_validators(tjunc, intersect),
        exporters=[
            ExporterRegistry.get("obj", Mesh.kind),
            # 3DM exporter consumes the OBJ produced by the OBJ exporter
            # and converts it to a Rhino 3DM using the existing converter.
            # Placed after the OBJ exporter so the .obj file is available on disk.
            ExporterRegistry.get("3dm", Mesh.kind),
            GmshGeoExporter(
                volume_name=volume_name,
                repaired=True,
                detect_cavities=detect_cavities,
                cavity_pitch=cavity_pitch,
                cavity_closing_iterations=cavity_closing_iterations,
            ),
            JsonReportWriter(issue_suffix="_remaining_issue.json", issue_source="final"),
        ],
        tolerances=Tolerances(),
    )


def wave_based_inspect_profile() -> SimulationProfile:
    """Inspect-only profile: same stages run (repairs still happen so each
    detector sees a clean mesh), but no geometry exporters are wired.

    Each issue kind is detected at its authoritative moment via the stage
    ``post_validators`` (T-junctions *before* their fix, intersections and
    overlaps *after* it, boundary/holes at the end). The PRE pass covers the
    repair-independent kinds (duplicate/degenerate/non-planar/small).

    ``final_validators`` is intentionally empty: re-validating the
    fully-processed mesh would report post-repair counts and override the
    per-stage diagnostics in ``PipelineResult.composite_issues``. The writer
    therefore consumes the composite view (``issue_source="composite"``) and
    skips the ``_report.json`` artifact (``write_report=False``).
    """
    tjunc = TJunctionsValidator()
    intersect = IntersectionsValidator()
    return SimulationProfile(
        name="wave_based_inspect",
        target_ir=Mesh,
        pre_validators=_wave_based_pre_validators(tjunc, intersect),
        stages=_wave_based_stages(tjunc, intersect, inspect=True),
        final_validators=[],
        exporters=[
            GmshGeoExporter(),
            JsonReportWriter(
                issue_suffix="_inspect_issue.json",
                issue_source="composite",
                write_report=False,
            ),
        ],
        tolerances=Tolerances(),
    )

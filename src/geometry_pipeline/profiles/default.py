"""Default geometry-repair simulation profile.

This is the single profile CHORAS runs for every uploaded geometry.

The stage order encodes geometric dependencies:
  1. dedup / degenerate / orient — preconditions for the topology checks below.
  2. fix T-junctions FIRST: an unfixed T-junction looks like a hole AND like
     an intersection, so detecting those before this stage produces noise.
  3. fix intersections (PLC violations).
  4. detect remaining holes / boundary edges only at the end.
"""

from __future__ import annotations

from geometry_pipeline.core.ir import Mesh
from geometry_pipeline.core.profile import SimulationProfile, Stage
from geometry_pipeline.core.tolerances import Tolerances
from geometry_pipeline.io.exporters.mesh_geo import GmshGeoExporter
from geometry_pipeline.io.registry import ExporterRegistry
from geometry_pipeline.repairs.mesh.deduplicate_vertices import DeduplicateVerticesRepair
from geometry_pipeline.repairs.mesh.fix_t_junctions import FixTJunctionsIterativeRepair
from geometry_pipeline.repairs.mesh.orient_outward import FlipFacesIfMajorityInwardRepair
from geometry_pipeline.repairs.mesh.remove_degenerate_faces import RemoveZeroAreaFaceRepair
from geometry_pipeline.repairs.mesh.repair_intersections import (
    RepairPlcSingleSplitsRepair,
    TrimSegmentFaceIntersectionsRepair,
)
from geometry_pipeline.repairs.mesh.sort_vertices import SortVerticesDeterministicallyRepair
from geometry_pipeline.reporting.json_writer import JsonReportWriter
from geometry_pipeline.validators.mesh.boundary_edges import BoundaryEdgesValidator
from geometry_pipeline.validators.mesh.collinear_faces import CollinearFacesValidator
from geometry_pipeline.validators.mesh.degenerate_faces import ZeroAreaFaceValidator
from geometry_pipeline.validators.mesh.duplicate_vertices import DuplicateVerticesValidator
from geometry_pipeline.validators.mesh.intersections import IntersectionsValidator
from geometry_pipeline.validators.mesh.non_planar_faces import NonPlanarFacesValidator
from geometry_pipeline.validators.mesh.overlapping_faces import OverlappingFacesValidator
from geometry_pipeline.validators.mesh.possible_holes import PossibleHolesValidator
from geometry_pipeline.validators.mesh.small_faces import SmallFacesValidator
from geometry_pipeline.validators.mesh.t_junctions import TJunctionsValidator


def _default_pre_validators(tjunc, intersect) -> list:
    return [
        NonPlanarFacesValidator(),
        DuplicateVerticesValidator(),
        ZeroAreaFaceValidator(),
        OverlappingFacesValidator(),
        SmallFacesValidator(),
        CollinearFacesValidator(),
        tjunc,
        intersect,
        BoundaryEdgesValidator(),
        PossibleHolesValidator(),
    ]


def _default_stages(
    tjunc,
    intersect,
    *,
    checkpoint_exporters=(),
) -> list[Stage]:
    """Single source of truth for the default stage order.

    The order is what makes detection meaningful (T-junctions only detectable
    after dedupe; intersections only meaningful after the tj fix).

    The ``t_junctions`` stage is the INSPECT CHECKPOINT. Its
    ``checkpoint_exporters`` emit ``<stem>.geo`` + ``<stem>_inspect_issue.json``
    on the tjunc-fixed, pre-intersection-repair mesh. Detectors are placed so
    the interim ``composite_issues`` matches the historical inspect report
    exactly: ``tjunc`` *before* the fix (``orient`` post-validator), everything
    else *after* it (``t_junctions`` post-validators).

    Note: the initial ``<stem>.3dm``/``.zip`` bundle is NOT produced here — it
    is created by the backend ``map_to_3dm_and_geo`` (model-creation flow),
    which also writes the ``File`` row and ``Geometry.outputModelId`` the
    frontend depends on. The pipeline only writes files, so reproducing it here
    would drop that DB linkage and overwrite the bundle with a different
    converter.
    """
    stages = [
        Stage(name="deduplication", repairs=[DeduplicateVerticesRepair()]),  # type: ignore[list-item]
        Stage(name="zero_area_face", repairs=[RemoveZeroAreaFaceRepair()]),  # type: ignore[list-item]
        Stage(name="sort", repairs=[SortVerticesDeterministicallyRepair()]),  # type: ignore[list-item]
        # tjunc measured here = ORIGINAL count (orient does not change tjuncs)
        Stage(
            name="orient",
            repairs=[FlipFacesIfMajorityInwardRepair()],  # type: ignore[list-item]
            post_validators=[tjunc],
        ),
        # === INSPECT CHECKPOINT ===
        # Real T-junction fix, then measure intersect/overlap/boundary/holes on
        # the tjunc-fixed, pre-intersection-repair mesh (denoised so a single
        # T-junction is not double-counted as an intersection AND a hole).
        Stage(
            name="t_junctions",
            repairs=[FixTJunctionsIterativeRepair(detector=tjunc)],  # type: ignore[list-item]
            post_validators=[
                intersect,
                OverlappingFacesValidator(),
                BoundaryEdgesValidator(),
                PossibleHolesValidator(),
            ],
            exporters=list(checkpoint_exporters),
            checkpoint=True,
        ),
        Stage(
            name="intersections",
            repairs=[
                TrimSegmentFaceIntersectionsRepair(detector=intersect),  # type: ignore[list-item]
                RepairPlcSingleSplitsRepair(detector=intersect),  # type: ignore[list-item]
                # RepairPlcByOffsetRepair(detector=intersect),
            ],
            post_validators=[intersect, OverlappingFacesValidator()],
        ),
        Stage(
            name="topology",
            post_validators=[BoundaryEdgesValidator(), PossibleHolesValidator()],
        ),
    ]
    return stages


def _default_final_validators(tjunc, intersect) -> list:
    return [
        NonPlanarFacesValidator(),
        ZeroAreaFaceValidator(),
        OverlappingFacesValidator(),
        SmallFacesValidator(),
        CollinearFacesValidator(),
        tjunc,
        intersect,
        BoundaryEdgesValidator(),
        PossibleHolesValidator(),
    ]


def _inspect_checkpoint_exporters(*, detect_cavities: bool = True) -> list:
    """Exporters fired at the ``t_junctions`` checkpoint (initial geo + report).

    ``GmshGeoExporter(repaired=False)`` keeps today's inspect defaults
    (``detect_cavities=True``) and writes ``<stem>.geo``; the writer emits
    ``<stem>_inspect_issue.json`` from the interim ``composite_issues`` and
    skips the ``_report.json`` artifact. ``detect_cavities`` is threaded so a
    caller can disable the native cavity kernel (e.g. tests/CI); production
    keeps it ``True`` so the checkpoint geo matches the historical inspect run.
    """
    return [
        GmshGeoExporter(repaired=False, detect_cavities=detect_cavities),
        JsonReportWriter(
            issue_suffix="_inspect_issue.json",
            issue_source="composite",
            write_report=False,
        ),
    ]


def default_profile(
    volume_name: str = "RoomVolume",
    *,
    detect_cavities: bool = False,
    cavity_pitch: float = 0.05,
    cavity_closing_iterations: int = 0,
) -> SimulationProfile:
    """Merged default profile: one pass emits both the inspect and the
    fully-repaired artifacts.

    Export points:
      * ``t_junctions`` -> ``<stem>.geo`` + ``<stem>_inspect_issue.json``
        (checkpoint; tjunc-fixed, pre-intersection-repair mesh).
      * end of pipeline -> ``<stem>_repaired.{obj,3dm,geo,zip}``
        + ``<stem>_remaining_issue.json`` + ``<stem>_report.json``.

    The initial ``<stem>.3dm``/``.zip`` bundle is produced separately by the
    backend ``map_to_3dm_and_geo`` (model-creation flow), which also creates
    the ``File`` row + ``Geometry.outputModelId`` the frontend relies on, so it
    is intentionally NOT emitted here.

    When `detect_cavities=True`, the repaired GEO exporter emits one `Volume`
    per enclosed region (required by Gmsh for nested/attached enclosed objects).
    """
    tjunc = TJunctionsValidator()
    intersect = IntersectionsValidator()
    return SimulationProfile(
        name="default",
        target_ir=Mesh,
        pre_validators=_default_pre_validators(tjunc, intersect),
        stages=_default_stages(
            tjunc,
            intersect,
            checkpoint_exporters=_inspect_checkpoint_exporters(detect_cavities=detect_cavities),
        ),
        final_validators=_default_final_validators(tjunc, intersect),
        exporters=[
            ExporterRegistry.get("obj", Mesh.kind),
            # 3DM exporter consumes the OBJ produced by the OBJ exporter
            # and converts it to a Rhino 3DM using the existing converter.
            # Placed after the OBJ exporter so the .obj file is available on disk.
            ExporterRegistry.get("3dm", Mesh.kind),
            GmshGeoExporter(  # type: ignore[list-item]
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

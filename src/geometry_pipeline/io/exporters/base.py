"""Exporter Protocol: Geometry IR -> file."""

from __future__ import annotations

from geometry_pipeline.core.ir import Exporter as ExporterProtocol

# Re-export the protocol under the historical module name. Other modules
# import `Exporter` from `geometry_pipeline.io.exporters.base`; keep that
# symbol but source it from the core IR module.
Exporter = ExporterProtocol

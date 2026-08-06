"""Per-run context passed to validators and repair steps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from geometry_pipeline.core.tolerances import Tolerances


@dataclass
class Context:
    tolerances: Tolerances
    logger: logging.Logger
    profile_name: str
    extras: dict = field(default_factory=dict)

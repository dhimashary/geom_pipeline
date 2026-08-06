"""Snapshot diffing — compare two validation snapshots by Issue.id."""

from __future__ import annotations

from dataclasses import dataclass, field

from geometry_pipeline.core.issues import Issue
from geometry_pipeline.core.report import ValidationSnapshot


@dataclass
class SnapshotDiff:
    fixed: list[Issue] = field(default_factory=list)
    introduced: list[Issue] = field(default_factory=list)
    remaining: list[Issue] = field(default_factory=list)


def diff_snapshots(before: ValidationSnapshot, after: ValidationSnapshot) -> SnapshotDiff:
    b = {i.id: i for i in before.issues}
    a = {i.id: i for i in after.issues}
    return SnapshotDiff(
        fixed=[b[i] for i in b.keys() - a.keys()],
        introduced=[a[i] for i in a.keys() - b.keys()],
        remaining=[a[i] for i in a.keys() & b.keys()],
    )

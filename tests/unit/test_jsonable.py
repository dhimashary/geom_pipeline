"""Tests for JSON-native coercion of numpy payloads (tech-debt #7)."""
from __future__ import annotations

import json

import numpy as np

from geometry_pipeline.core.issues import (
    DetectionStage,
    Issue,
    IssueKind,
    Severity,
)
from geometry_pipeline.core.jsonable import to_jsonable


def test_to_jsonable_coerces_numpy_scalars_and_arrays():
    out = to_jsonable(
        {
            "f32": np.float32(2.5),
            "f64": np.float64(0.25),
            "i64": np.int64(7),
            "arr": np.array([1.0, 2.0, 3.0]),
            "nested": {"pt": (np.float32(1.0), np.int64(2))},
        }
    )

    assert type(out["f32"]) is float
    assert type(out["f64"]) is float
    assert type(out["i64"]) is int
    assert out["arr"] == [1.0, 2.0, 3.0] and all(type(v) is float for v in out["arr"])
    assert type(out["nested"]["pt"][0]) is float
    assert type(out["nested"]["pt"][1]) is int


def test_numpy_payload_round_trips_as_json_numbers():
    payload = {
        "point": (np.float32(0.1), np.float64(0.2), 3.0),
        "fid": np.int64(7),
        "vec": np.array([1.0, 2.0]),
    }
    issue = Issue.create(
        IssueKind.INTERSECTION,
        Severity.WARN,
        DetectionStage.PRE,
        payload=payload,
    )

    # default json.dumps (no default=str): only succeeds with native types.
    parsed = json.loads(json.dumps(issue.payload))

    assert isinstance(parsed["point"][0], float)
    assert isinstance(parsed["fid"], int) and parsed["fid"] == 7
    assert isinstance(parsed["vec"], list) and parsed["vec"] == [1.0, 2.0]
    # No numpy types must survive into the stored payload.
    assert type(issue.payload["fid"]) is int
    assert isinstance(issue.payload["vec"], list)


def test_issue_id_deterministic_for_identical_numpy_input():
    def make() -> Issue:
        return Issue.create(
            IssueKind.DEGENERATE_FACE,
            Severity.FATAL,
            DetectionStage.PRE,
            payload={"a": np.float32(2.5), "b": np.int64(7), "v": np.array([1.0, 2.0])},
        )

    assert make().id == make().id

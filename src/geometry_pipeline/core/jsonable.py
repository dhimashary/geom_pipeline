"""Coerce arbitrary values into JSON-native builtins.

Validators and repairs occasionally place ``numpy`` scalars/arrays into issue
payloads or ``RepairResult.details``. ``json.dumps(..., default=str)`` only
serialises ``np.float64`` correctly (it subclasses ``float``); ``np.float32``,
numpy integers and ``ndarray`` fall through to ``default=str`` and silently
become *quoted strings* (e.g. ``"0.1"`` / ``"[1. 2.]"``), which downstream
JSON consumers must not have to special-case.

``to_jsonable`` walks containers and converts numpy (and any object exposing
``.item()`` / ``.tolist()``) into plain ``int`` / ``float`` / ``list`` so the
serialised output always contains native JSON types, independent of the numpy
version in use.
"""
from __future__ import annotations

from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-native Python builtins."""
    # Fast path for the exact builtin types (note: ``type() in`` rather than
    # ``isinstance`` so numpy subclasses such as ``np.float64``/``np.bool_``
    # are *not* short-circuited and get coerced below).
    if obj is None or type(obj) in (bool, int, float, str):
        return obj

    if isinstance(obj, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]

    # numpy ndarray (and anything else array-like exposing tolist()); numpy
    # scalars also expose tolist()/item() returning a native Python scalar.
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist())

    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except (ValueError, TypeError):
            pass

    # Builtin subclasses we don't otherwise handle (e.g. str-based Enums).
    if isinstance(obj, (bool, int, float, str)):
        return obj

    return obj


__all__ = ["to_jsonable"]

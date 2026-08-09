"""Shared helpers for chroma-backed stores."""

from __future__ import annotations

from typing import Any


def to_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata values to types chroma can store; drop None values.

    Chroma rejects None, lists, dicts and numpy scalars inside metadata, so
    everything that is not a scalar bool/int/float/str is stringified.
    """
    output: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            output[key] = value
        else:
            output[key] = str(value)
    return output


__all__ = ["to_chroma_metadata"]

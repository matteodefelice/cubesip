"""Geohash helpers (pure Python, no dependency).

Implemented in milestone M1.
"""

from __future__ import annotations

from typing import NamedTuple


class BBox(NamedTuple):
    """Bounding box in EPSG:4326 degrees."""

    west: float
    south: float
    east: float
    north: float


def bbox(geohash: str) -> BBox:
    """Exact bounds of a geohash cell in EPSG:4326."""
    raise NotImplementedError("M1")


def ancestors(geohash: str, include_self: bool = True) -> list[str]:
    """Cells containing ``geohash``, longest prefix first: 'qem7' -> ['qem7', 'qem', 'qe', 'q']."""
    raise NotImplementedError("M1")


def covers(parent: str, child: str) -> bool:
    """True if cell ``parent`` contains cell ``child`` (prefix rule)."""
    raise NotImplementedError("M1")

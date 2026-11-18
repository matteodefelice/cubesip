"""Geohash helpers (pure Python, no dependency)."""

from __future__ import annotations

from typing import NamedTuple

ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
MAX_LENGTH = 12
_DECODE = {char: index for index, char in enumerate(ALPHABET)}


class BBox(NamedTuple):
    """Bounding box in EPSG:4326 degrees."""

    west: float
    south: float
    east: float
    north: float


def validate(geohash: str) -> str:
    """Return the normalised (stripped, lower-case) geohash or raise ValueError."""
    value = geohash.strip().lower()
    if not 1 <= len(value) <= MAX_LENGTH:
        raise ValueError(f"geohash must have 1-{MAX_LENGTH} characters, got {value!r}")
    bad = sorted(set(value) - set(ALPHABET))
    if bad:
        raise ValueError(f"geohash {value!r} contains invalid characters: {bad}")
    return value


def bbox(geohash: str) -> BBox:
    """Exact bounds of a geohash cell in EPSG:4326."""
    lon = [-180.0, 180.0]
    lat = [-90.0, 90.0]
    is_lon = True  # bits alternate, starting with longitude
    for char in validate(geohash):
        bits = _DECODE[char]
        for shift in range(4, -1, -1):
            interval = lon if is_lon else lat
            mid = (interval[0] + interval[1]) / 2
            if (bits >> shift) & 1:
                interval[0] = mid
            else:
                interval[1] = mid
            is_lon = not is_lon
    return BBox(west=lon[0], south=lat[0], east=lon[1], north=lat[1])


def encode(lat: float, lon: float, precision: int = 6) -> str:
    """Geohash of the cell containing a point (handy to find the cell of a site)."""
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"invalid coordinates lat={lat}, lon={lon}")
    if not 1 <= precision <= MAX_LENGTH:
        raise ValueError(f"precision must be 1-{MAX_LENGTH}")
    lon_iv = [-180.0, 180.0]
    lat_iv = [-90.0, 90.0]
    is_lon = True
    chars = []
    bits = 0
    nbits = 0
    while len(chars) < precision:
        interval, value = (lon_iv, lon) if is_lon else (lat_iv, lat)
        mid = (interval[0] + interval[1]) / 2
        if value >= mid:
            bits = (bits << 1) | 1
            interval[0] = mid
        else:
            bits <<= 1
            interval[1] = mid
        is_lon = not is_lon
        nbits += 1
        if nbits == 5:
            chars.append(ALPHABET[bits])
            bits = 0
            nbits = 0
    return "".join(chars)


def ancestors(geohash: str, include_self: bool = True) -> list[str]:
    """Cells containing ``geohash``, longest prefix first: 'qem7' -> ['qem7', 'qem', 'qe', 'q']."""
    value = validate(geohash)
    start = len(value) if include_self else len(value) - 1
    return [value[:n] for n in range(start, 0, -1)]


def covers(parent: str, child: str) -> bool:
    """True if cell ``parent`` contains cell ``child`` (prefix rule; a cell covers itself)."""
    return validate(child).startswith(validate(parent))

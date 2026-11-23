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
    """Clean up a geohash (trim spaces, lower case) and check that it is valid: the right length and
    only allowed characters. Raises ValueError otherwise.
    """
    value = geohash.strip().lower()
    if not 1 <= len(value) <= MAX_LENGTH:
        raise ValueError(f"geohash must have 1-{MAX_LENGTH} characters, got {value!r}")
    bad = sorted(set(value) - set(ALPHABET))
    if bad:
        raise ValueError(f"geohash {value!r} contains invalid characters: {bad}")
    return value


def bbox(geohash: str) -> BBox:
    """Work out the edges of a geohash cell in degrees. Each character narrows the area down,
    splitting longitude and latitude in half in turn.
    """
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
    """Find the geohash, of the given length, of the cell that contains a point. Handy to find the
    cell of a site.
    """
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


def containing(box: BBox, max_precision: int = MAX_LENGTH) -> str:
    """Find the smallest geohash cell that fully contains a box, using at most ``max_precision``
    characters.

    A box lying on a cell's edge belongs to that cell, so the box of a cell gives the cell
    back, and a point gives the same cell as :func:`encode`. Raises ValueError when no cell
    contains the box: level-1 cells are 45 x 45 degrees, so a box crossing the equator, the
    prime meridian or any other multiple of 45 degrees is not inside any geohash.
    """
    west, south, east, north = box
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise ValueError(f"invalid box {tuple(box)}: need west <= east, south <= north")
    if not 1 <= max_precision <= MAX_LENGTH:
        raise ValueError(f"max_precision must be 1-{MAX_LENGTH}")
    lon_iv = [-180.0, 180.0]
    lat_iv = [-90.0, 90.0]
    is_lon = True
    chars = []
    bits = 0
    nbits = 0
    while len(chars) < max_precision:
        interval, low, high = (lon_iv, west, east) if is_lon else (lat_iv, south, north)
        mid = (interval[0] + interval[1]) / 2
        if low >= mid:  # upper half checked first, like encode() for a point on an edge
            bits = (bits << 1) | 1
            interval[0] = mid
        elif high <= mid:
            bits <<= 1
            interval[1] = mid
        else:  # the box straddles this split: the cell so far is the smallest
            break
        is_lon = not is_lon
        nbits += 1
        if nbits == 5:
            chars.append(ALPHABET[bits])
            bits = 0
            nbits = 0
    if not chars:
        raise ValueError(
            f"no geohash contains the box {tuple(box)}: it crosses a level-1 cell edge "
            "(multiples of 45 degrees, such as the equator or the prime meridian)"
        )
    return "".join(chars)


def ancestors(geohash: str, include_self: bool = True) -> list[str]:
    """List the cells that contain a geohash, from the cell itself (unless ``include_self`` is
    False) up to the largest one: 'qem7' gives ['qem7', 'qem', 'qe', 'q'].
    """
    value = validate(geohash)
    start = len(value) if include_self else len(value) - 1
    return [value[:n] for n in range(start, 0, -1)]


def children(geohash: str) -> list[str]:
    """The 32 cells one character longer that together make up a geohash: 'qem' gives 'qem0' ...
    'qemz'.
    """
    value = validate(geohash)
    if len(value) == MAX_LENGTH:
        raise ValueError(f"{value!r} is already at the maximum length ({MAX_LENGTH})")
    return [value + char for char in ALPHABET]


def covers(parent: str, child: str) -> bool:
    """True if the cell ``parent`` contains the cell ``child``. A geohash contains another when the
    other starts with it, so a cell also contains itself.
    """
    return validate(child).startswith(validate(parent))

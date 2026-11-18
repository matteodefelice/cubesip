"""Shared xarray subsetting used by the edh/local drivers and by cache reuse.

One code path for: finding time/x/y dimensions, detecting the CRS, transforming the geohash
box into the native CRS, handling 0..360 longitudes and descending coordinates, and keeping
every cell that intersects the box (so coarse grids such as ERA5 0.25 deg never come back empty).

Implemented in milestone M1.
"""

from __future__ import annotations

from typing import NamedTuple

import xarray as xr

from cubesip.geohash import BBox
from cubesip.request import TimeRange

TIME_NAMES = ("time", "valid_time", "t")
Y_NAMES = ("y", "latitude", "lat")
X_NAMES = ("x", "longitude", "lon")


class CubeDims(NamedTuple):
    time: str | None
    y: str
    x: str


def find_dims(ds: xr.Dataset) -> CubeDims:
    """Identify time / y / x dimension names (CF attributes first, then common names)."""
    raise NotImplementedError("M1")


def native_crs(ds: xr.Dataset) -> str:
    """CRS of the dataset as 'EPSG:xxxx' (rioxarray / CF grid_mapping); EPSG:4326 if lat/lon."""
    raise NotImplementedError("M1")


def to_native_bounds(box: BBox, crs: str) -> BBox:
    """Envelope of the (densified) EPSG:4326 box transformed into ``crs``."""
    raise NotImplementedError("M1")


def subset(
    ds: xr.Dataset,
    box: BBox,
    time: TimeRange | None,
    variables: list[str],
) -> xr.Dataset:
    """Lazily select variables, intersecting cells and days (end day inclusive)."""
    raise NotImplementedError("M1")

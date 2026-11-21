"""Shared xarray subsetting used by the edh/local drivers and by cache reuse.

One code path for: finding time/x/y dimensions, detecting the CRS, transforming the geohash
box into the native CRS, handling 0..360 longitudes and descending coordinates, and keeping
every cell that intersects the box (so coarse grids such as ERA5 0.25 deg never come back empty).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from cubesip.geohash import BBox
from cubesip.request import TimeRange

TIME_NAMES = ("time", "valid_time", "t")
Y_NAMES = ("y", "latitude", "lat")
X_NAMES = ("x", "longitude", "lon")
_GEOGRAPHIC_Y = ("latitude", "lat")
_GEOGRAPHIC_X = ("longitude", "lon")


class EmptySelectionError(ValueError):
    """The request area or period does not overlap the dataset."""


class CubeDims(NamedTuple):
    time: str | None
    y: str
    x: str


def _match_dim(
    ds: xr.Dataset, axis: str, standard_names: tuple[str, ...], names: tuple[str, ...]
) -> str | None:
    """Find which dimension of the dataset is the given axis (time, y or x). The labels in the
    coordinates' metadata are trusted first; then usual names such as 'lat' or 'time' are tried.
    Returns None if nothing matches.
    """
    dims = list(ds.dims)
    for dim in dims:  # CF metadata first
        if dim in ds.coords:
            attrs = ds[dim].attrs
            if str(attrs.get("axis", "")).upper() == axis:
                return str(dim)
            if attrs.get("standard_name") in standard_names:
                return str(dim)
    for name in names:  # then conventional names
        for dim in dims:
            if str(dim).lower() == name:
                return str(dim)
    return None


def find_dims(ds: xr.Dataset) -> CubeDims:
    """Work out the names of the time, y and x dimensions of a dataset. Raises ValueError if x or y
    cannot be found; time may be missing (None).
    """
    time = _match_dim(ds, "T", ("time",), TIME_NAMES)
    y = _match_dim(ds, "Y", ("latitude", "projection_y_coordinate"), Y_NAMES)
    x = _match_dim(ds, "X", ("longitude", "projection_x_coordinate"), X_NAMES)
    if y is None or x is None:
        raise ValueError(f"cannot identify x/y dimensions among {list(ds.dims)}")
    return CubeDims(time=time, y=y, x=x)


def native_crs(ds: xr.Dataset) -> str:
    """Work out the dataset's coordinate reference system, e.g. 'EPSG:32632'. When no CRS is
    recorded but the coordinates are named latitude/longitude, plain lat/lon (EPSG:4326) is assumed.
    """
    crs = _crs_from_metadata(ds)
    if crs is None:
        dims = find_dims(ds)
        if dims.y.lower() in _GEOGRAPHIC_Y and dims.x.lower() in _GEOGRAPHIC_X:
            return "EPSG:4326"
        raise ValueError("dataset has projected x/y coordinates but no CRS metadata")
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_wkt()


def _crs_from_metadata(ds: xr.Dataset) -> CRS | None:
    """Look for the CRS in the dataset's metadata: the 'grid mapping' variable that CF-style files
    use, such as 'spatial_ref' or 'crs'. Returns None if there is none.
    """
    candidates: list[str] = []
    for var in ds.data_vars.values():
        gm = var.encoding.get("grid_mapping") or var.attrs.get("grid_mapping")
        if gm:
            candidates.append(str(gm))
    candidates += ["spatial_ref", "crs"]
    for name in candidates:
        if name not in ds.variables:
            continue
        attrs = ds[name].attrs
        for key in ("crs_wkt", "spatial_ref"):
            if attrs.get(key):
                return CRS.from_user_input(attrs[key])
        if "grid_mapping_name" in attrs:
            return CRS.from_cf(attrs)
    return None


def grid_mapping_names(ds: xr.Dataset, variables: list[str] | None = None) -> set[str]:
    """Names of the helper variables that only hold CRS information, for the given variables (all
    data variables by default).
    """
    names = set()
    for name in variables if variables is not None else list(ds.data_vars):
        var = ds[name]
        gm = var.encoding.get("grid_mapping") or var.attrs.get("grid_mapping")
        if gm and str(gm) in ds.variables:
            names.add(str(gm))
    return names


def data_variables(ds: xr.Dataset) -> list[str]:
    """Names of the variables that hold real data, leaving out the helper variables that only hold
    CRS information.
    """
    skip = grid_mapping_names(ds)
    return [str(name) for name in ds.data_vars if name not in skip]


def to_native_bounds(box: BBox, crs: str) -> BBox:
    """Convert a lon/lat box into the dataset's own coordinates. Several points along each edge are
    converted, so the result is a rectangle that fully encloses the converted shape.
    """
    target = CRS.from_user_input(crs)
    if target.is_geographic:
        return box
    transformer = Transformer.from_crs("EPSG:4326", target, always_xy=True)
    west, south, east, north = transformer.transform_bounds(*box, densify_pts=21)
    return BBox(west, south, east, north)


def _contiguous(idx: np.ndarray) -> slice | np.ndarray:
    """A slice covering the same positions when they are side by side, otherwise the positions
    themselves.
    """
    if idx[-1] - idx[0] + 1 == idx.size:
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def _cell_indices(
    values: np.ndarray, lo: float, hi: float, periodic: bool = False
) -> slice | np.ndarray:
    """Find which grid cells along one axis overlap the range [lo, hi]. Each value is taken as the
    centre of a cell, so a cell is kept as soon as part of it falls inside the range. For
    longitudes, values shifted by 360 degrees are tried too. Returns a slice when the cells are side
    by side, otherwise their positions.
    """
    if values.size == 1:
        return slice(None)
    half = float(np.median(np.abs(np.diff(values)))) / 2
    shifts = (-360.0, 0.0, 360.0) if periodic else (0.0,)
    mask = np.zeros(values.shape, dtype=bool)
    for shift in shifts:
        shifted = values + shift
        mask |= (shifted > lo - half) & (shifted < hi + half)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise EmptySelectionError(
            f"area [{lo}, {hi}] is outside the dataset extent [{values.min()}, {values.max()}]"
        )
    return _contiguous(idx)


def subset(
    ds: xr.Dataset,
    box: BBox,
    time: TimeRange | None,
    variables: list[str],
) -> xr.Dataset:
    """Cut a dataset down to the requested variables, the cells that touch the box and the requested
    days (the last day included). Nothing is loaded yet: the data are read only when they are used.

    With ``time=None`` every time step is kept; a time range on a dataset without a time
    dimension is an error.
    """
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise KeyError(f"variables not in dataset: {missing}; available: {data_variables(ds)}")
    crs = native_crs(ds)
    keep_gm = sorted(
        n for n in grid_mapping_names(ds, variables) if n in ds.data_vars and n not in variables
    )
    selected = ds[list(variables) + keep_gm]
    dims = find_dims(selected)

    geographic = CRS.from_user_input(crs).is_geographic
    native = to_native_bounds(box, crs)
    x_values = np.asarray(selected[dims.x].values, dtype=float)
    y_values = np.asarray(selected[dims.y].values, dtype=float)
    indexers: dict[str, slice | np.ndarray] = {
        dims.x: _cell_indices(x_values, native.west, native.east, periodic=geographic),
        dims.y: _cell_indices(y_values, native.south, native.north),
    }

    if time is not None:
        if dims.time is None:
            raise ValueError("dataset has no time dimension: use time=null in the request")
        times = pd.DatetimeIndex(selected[dims.time].values)
        start = pd.Timestamp(time.start)
        stop = pd.Timestamp(time.end_exclusive)
        idx = np.flatnonzero((times >= start) & (times < stop))
        if idx.size == 0:
            raise EmptySelectionError(
                f"no time steps between {time.start} and {time.end} "
                f"(dataset covers {times.min()} to {times.max()})"
            )
        indexers[dims.time] = _contiguous(idx)

    return selected.isel(indexers)

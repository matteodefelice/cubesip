"""Processing helpers for cubesip results.

- :func:`containing_geohash`: the smallest geohash cell holding a geometry, to request data
  for a field or a region drawn elsewhere.
- :func:`regrid`: put one or more results on the grid (CRS, cells) of a reference result,
  e.g. ERA5 on a Sentinel-2 grid, or Sentinel-2 averaged on a TESSERA grid.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from functools import partial
from typing import NamedTuple

import numpy as np
import shapely
import xarray as xr
from pyproj import CRS
from rasterio.crs import CRS as RasterioCRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds

from cubesip import geohash, spatial
from cubesip.api import Result

Gridded = Result | xr.Dataset | xr.DataArray

_UNNAMED = "__cubesip_regrid__"  # temporary name when a DataArray goes through a Dataset
_SPACING_RTOL = 1e-3  # tolerated relative deviation from an even coordinate spacing
# rasterio hides a spurious "no geotransform" warning with warnings.catch_warnings, which is
# not thread-safe: concurrent warps (dask threads) let it through, so warps run one at a time.
_WARP_LOCK = threading.Lock()


# -- geohash of a geometry ----------------------------------------------------------------


def containing_geohash(wkt: str, max_precision: int = geohash.MAX_LENGTH) -> str:
    """Find the smallest geohash cell that fully contains a shape given as WKT text.

    The WKT is in EPSG:4326 with longitude first (``POLYGON ((lon lat, ...))``). The cell
    contains the geometry's bounding box, edges included, so a polygon traced along a cell's
    border returns that cell. ``max_precision`` caps the length: a point would otherwise get
    a 12-character cell. Raises ValueError for unreadable or empty WKT, and when no cell
    contains the geometry (it crosses the equator, the prime meridian or another level-1
    edge; see :func:`cubesip.geohash.containing`).
    """
    try:
        geometry = shapely.from_wkt(wkt)
    except (shapely.errors.ShapelyError, TypeError) as exc:
        raise ValueError(f"cannot read WKT {wkt!r}: {exc}") from exc
    if geometry is None or geometry.is_empty:
        raise ValueError(f"WKT {wkt!r} is an empty geometry")
    return geohash.containing(geohash.BBox(*geometry.bounds), max_precision)


# -- regridding ---------------------------------------------------------------------------


class _Grid(NamedTuple):
    """Target grid, warped north-up (x ascending, y descending) and then put in the
    reference's own coordinate order."""

    crs: RasterioCRS
    transform: Affine
    height: int
    width: int
    y: str  # dimension names of the output
    x: str
    y_index: np.ndarray | None  # north-up row/column -> reference order (None: same order)
    x_index: np.ndarray | None
    coords: dict[str, xr.Variable]  # reference coordinates copied onto every output
    grid_mapping: str | None


def regrid(
    sources: Gridded | Iterable[Gridded],
    reference: Gridded,
    *,
    method: str = "nearest",
) -> xr.Dataset | xr.DataArray | list[xr.Dataset | xr.DataArray]:
    """Put one or more datasets on the same grid (same CRS and cells) as a reference dataset.

    ``sources`` and ``reference`` are :class:`cubesip.Result` objects, Datasets or
    DataArrays, in any CRS cubesip knows (a CF grid mapping, or plain lat/lon). One source
    gives one output, a list gives a list; a Result or Dataset comes back as a Dataset, a
    DataArray as a DataArray with the same name.

    Each output has the reference's x/y coordinates and grid mapping, and keeps the source's
    other dimensions (time, band...), which are not aligned with the reference's. Values are
    floating point (integers are converted), NaN where the source has no data; NaN source
    cells are ignored. The computation is lazy when the source is (``Result.dataset`` is).

    ``method`` is a rasterio resampling name: ``nearest`` (default; right for classes and
    masks), ``bilinear`` or ``cubic`` to interpolate onto a finer grid, ``average`` (or
    ``min``, ``max``, ``med``, ``mode``, ``sum``) to aggregate onto a coarser one.

    A source with a single cell along x or y (ERA5 for a small geohash) carries no cell
    size, so that cell is taken to cover the whole reference along that axis. cubesip only
    returns a single cell when the request's area lies inside it, so this holds when the
    reference is for the same geohash or one inside it.

    The reference needs at least two evenly spaced cells along x and y.
    """
    resampling = _resampling(method)
    grid = _reference_grid(_to_dataset(reference))
    if isinstance(sources, Result | xr.Dataset | xr.DataArray):
        return _regrid_one(sources, grid, resampling)
    return [_regrid_one(source, grid, resampling) for source in sources]


def _resampling(method: str) -> Resampling:
    """Turn a method name such as 'nearest' or 'average' into the matching rasterio setting, with a
    clear error if the name is unknown.
    """
    try:
        return Resampling[method]
    except KeyError:
        names = ", ".join(r.name for r in Resampling)
        raise ValueError(f"unknown method {method!r}; use one of: {names}") from None


def _to_dataset(obj: Gridded) -> xr.Dataset:
    """Get a Dataset out of a Result, Dataset or DataArray. A DataArray is wrapped in a Dataset
    under a temporary name.
    """
    if isinstance(obj, Result):
        return obj.dataset
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=_UNNAMED)
    if isinstance(obj, xr.Dataset):
        return obj
    raise TypeError(f"expected a cubesip Result, Dataset or DataArray, got {type(obj).__name__}")


def _regrid_one(source: Gridded, grid: _Grid, resampling: Resampling) -> xr.Dataset | xr.DataArray:
    """Regrid one source and give it back in the form it came in (a DataArray stays a DataArray)."""
    out = _regrid_dataset(_to_dataset(source), grid, resampling)
    if isinstance(source, xr.DataArray):
        return out[_UNNAMED].rename(source.name)
    return out


def _grid_mapping_vars(ds: xr.Dataset) -> set[str]:
    """Names of the variables that only hold CRS information: those the data variables point to,
    plus 'spatial_ref' or 'crs' when present.
    """
    names = spatial.grid_mapping_names(ds)
    names |= {n for n in ("spatial_ref", "crs") if n in ds.variables and ds[n].ndim == 0}
    return names


def _x_values(ds: xr.Dataset, x: str, crs: str) -> np.ndarray:
    """The x coordinates as numbers. For lon/lat data, longitudes written as 0..360 are converted to
    -180..180.
    """
    values = np.asarray(ds[x].values, dtype=float)
    if CRS.from_user_input(crs).is_geographic:  # 0..360 longitudes -> -180..180
        values = (values + 180) % 360 - 180
    return values


def _sort_axis(values: np.ndarray, name: str, descending: bool) -> tuple[np.ndarray, float | None]:
    """Work out how to put an axis in order (ascending or descending) and the distance between
    neighbouring cells. Raises ValueError if the cells are not evenly spaced; the distance is None
    when there is a single cell.
    """
    order = np.argsort(values, kind="stable")
    if descending:
        order = order[::-1]
    ordered = values[order]
    if ordered.size < 2:
        return order, None
    steps = np.diff(ordered)
    step = float(np.median(steps))
    if step == 0 or not np.allclose(steps, step, rtol=_SPACING_RTOL, atol=0):
        raise ValueError(f"coordinate {name!r} is not evenly spaced: cannot regrid")
    return order, step


def _inverse(order: np.ndarray) -> np.ndarray | None:
    """The reordering that puts things back as they were. Returns None when the order changes
    nothing.
    """
    if np.array_equal(order, np.arange(order.size)):
        return None
    return np.argsort(order)


def _reference_grid(ds: xr.Dataset) -> _Grid:
    """Describe the grid of the reference dataset: its CRS, where its cells are and how big they
    are, its dimension names and order, and the coordinates to copy onto every regridded output.
    """
    dims = spatial.find_dims(ds)
    crs = spatial.native_crs(ds)
    x = _x_values(ds, dims.x, crs)
    y = np.asarray(ds[dims.y].values, float)
    x_order, dx = _sort_axis(x, dims.x, descending=False)
    y_order, dy = _sort_axis(y, dims.y, descending=True)
    if dx is None or dy is None:
        raise ValueError(
            "the reference has a single cell along x or y, so its cell size is unknown: "
            "use a reference with at least two cells along each axis"
        )
    x0, y0 = float(x[x_order[0]]), float(y[y_order[0]])
    gm = _grid_mapping_vars(ds)
    coords = {  # grid mapping, x/y, and coordinates on x/y only (e.g. 2-D lat/lon)
        str(name): var
        for name, var in ds.variables.items()
        if name in gm
        or (name not in ds.data_vars and var.dims and set(var.dims) <= {dims.y, dims.x})
    }
    return _Grid(
        crs=RasterioCRS.from_user_input(crs),
        transform=Affine(dx, 0.0, x0 - dx / 2, 0.0, dy, y0 - dy / 2),
        height=ds.sizes[dims.y],
        width=ds.sizes[dims.x],
        y=dims.y,
        x=dims.x,
        y_index=_inverse(y_order),
        x_index=_inverse(x_order),
        coords=coords,
        grid_mapping=min(gm) if gm else None,
    )


def _source_transform(
    x: np.ndarray, dx: float | None, y: np.ndarray, dy: float | None, crs: str, grid: _Grid
) -> Affine:
    """Describe where the source's cells are and how big they are. If the source has a single cell
    along an axis, its size is unknown, so that cell is stretched to cover the whole reference grid.
    """
    if dx is None or dy is None:
        left, top = grid.transform.c, grid.transform.f
        right = left + grid.transform.a * grid.width
        bottom = top + grid.transform.e * grid.height
        west, south, east, north = transform_bounds(
            grid.crs, RasterioCRS.from_user_input(crs), left, bottom, right, top, densify_pts=21
        )
        if dx is None:
            dx = 2 * max(abs(x[0] - west), abs(east - x[0]))
        if dy is None:
            dy = -2 * max(abs(y[0] - south), abs(north - y[0]))
    return Affine(dx, 0.0, x[0] - dx / 2, 0.0, dy, y[0] - dy / 2)


def _regrid_dataset(ds: xr.Dataset, grid: _Grid, resampling: Resampling) -> xr.Dataset:
    """Regrid every variable of a dataset that lies on the x/y grid onto the reference grid. The
    source is first put north-up, as the warping step expects. Variables without x/y are kept as
    they are, and the source's own CRS information is replaced by the reference's.
    """
    dims = spatial.find_dims(ds)
    crs = spatial.native_crs(ds)
    ds = ds.drop_vars(_grid_mapping_vars(ds))  # the output takes the reference's CRS

    x_order, dx = _sort_axis(_x_values(ds, dims.x, crs), dims.x, descending=False)
    y_order, dy = _sort_axis(np.asarray(ds[dims.y].values, float), dims.y, descending=True)
    reorder = {d: o for d, o in ((dims.x, x_order), (dims.y, y_order)) if _inverse(o) is not None}
    if reorder:
        ds = ds.isel(reorder)  # north-up, as the transform says
    x = _x_values(ds, dims.x, crs)
    y = np.asarray(ds[dims.y].values, float)
    src_transform = _source_transform(x, dx, y, dy, crs, grid)

    spatial_vars = []
    for name, var in ds.data_vars.items():
        found = {dims.y, dims.x} & set(var.dims)
        if not found:
            continue
        if len(found) == 1:
            raise ValueError(f"variable {name!r} has only one of the dimensions {dims.y}/{dims.x}")
        if var.dtype.kind not in "biuf":
            raise TypeError(f"variable {name!r} is not numeric ({var.dtype}): cannot regrid")
        spatial_vars.append(name)

    # a partial keeps the parameters opaque to dask (Affine is a namedtuple)
    warp = partial(
        _warp_block,
        src_transform=src_transform,
        src_crs=RasterioCRS.from_user_input(crs),
        grid=grid,
        resampling=resampling,
    )
    regridded = {name: _regrid_var(ds[name], dims.y, dims.x, warp, grid) for name in spatial_vars}
    out = ds.drop_dims([dims.y, dims.x]).assign(regridded).assign_coords(grid.coords)
    return out[list(ds.data_vars)]


def _regrid_var(
    var: xr.DataArray, y: str, x: str, warp: Callable[[np.ndarray], np.ndarray], grid: _Grid
) -> xr.DataArray:
    """Regrid one variable. It is turned into floating point (so missing cells can be NaN), and the
    work is done block by block, lazily when the data are lazy.
    """
    dtype = np.result_type(var.dtype, np.float32)  # float32 for float32 and <= 16-bit ints
    var = var.astype(dtype)
    if var.chunks is not None:
        var = var.chunk({y: -1, x: -1})  # every block needs whole rows and columns
    out = xr.apply_ufunc(
        warp,
        var,
        input_core_dims=[[y, x]],
        output_core_dims=[[grid.y, grid.x]],
        exclude_dims={y, x},
        dask="parallelized",
        output_dtypes=[dtype],
        dask_gufunc_kwargs={"output_sizes": {grid.y: grid.height, grid.x: grid.width}},
        keep_attrs=True,
    )
    out.attrs.pop("grid_mapping", None)
    out.encoding = {"grid_mapping": grid.grid_mapping} if grid.grid_mapping else {}
    return out


def _warp_block(
    data: np.ndarray,
    *,
    src_transform: Affine,
    src_crs: RasterioCRS,
    grid: _Grid,
    resampling: Resampling,
) -> np.ndarray:
    """Warp a block of data shaped (..., y, x) onto the reference grid with rasterio, then put rows
    and columns in the reference's order. Cells without data become NaN. Only one warp runs at a
    time, because rasterio is not thread-safe here.
    """
    lead = data.shape[:-2]
    source = np.ascontiguousarray(data.reshape((-1, *data.shape[-2:])))
    dest = np.full((source.shape[0], grid.height, grid.width), np.nan, dtype=data.dtype)
    if source.shape[0]:
        with _WARP_LOCK:
            reproject(
                source,
                dest,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
    if grid.y_index is not None:
        dest = dest[:, grid.y_index, :]
    if grid.x_index is not None:
        dest = dest[:, :, grid.x_index]
    return dest.reshape((*lead, grid.height, grid.width))

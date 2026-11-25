"""Beeldmateriaal driver: high-resolution (8 cm) aerial orthophotos of the Netherlands.

The Dutch government programme Beeldmateriaal Nederland (https://www.beeldmateriaal.nl) flies
the whole country every spring (leafless season) and publishes the orthophoto mosaic as open
data (CC BY 4.0): an RGB and a colour-infrared (CIR) product, each cut into 1 x 1 km GeoTIFF
tiles in RD New (EPSG:28992) at 8 cm (5 cm in some areas). The same imagery is behind PDOK's
"Luchtfoto RGB" / "Luchtfoto CIR" web services, which only serve rendered JPEG images; this
driver reads the GeoTIFF tiles themselves, window by window over HTTP, without downloading
whole tiles. No account is needed.

- dataset: ``ortho_hr`` (the spring high-resolution orthophoto mosaic)
- variables: ``red``, ``green``, ``blue`` (RGB tiles) and ``nir`` (CIR tiles, whose bands are
  near-infrared, red, green). Only the products the variables need are read: ``["nir",
  "red"]`` reads the CIR tiles only.
- time: whole calendar years, one step per year stamped on 1 January (the flight is in
  early spring). Tiles are published for 2025 onwards; earlier high-resolution years exist
  only as PDOK web-service renderings.
- options: ``resolution`` (metres, default 0.08; 1000 must be a multiple of it, e.g. 0.05,
  0.1, 0.25, 1). Pixels are copied untouched when the tile has that resolution, otherwise
  averaged (or repeated, when finer).
- grid: RD New (EPSG:28992), pixel edges on multiples of ``resolution``, uint8. Where no tile
  exists (sea, outside the Netherlands, not delivered yet) the values are 0.

Tiles are located through the GeoJSON index the Beeldmateriaal download portal uses
(https://basisdata.nl/hwh-portal/download/index.html), one per year and product. The index
(~40 MB) is downloaded once and kept as a small CSV under ``<data_dir>/_index/``; it is
refreshed after a week, since the current year's tiles are delivered over several months.
"""

from __future__ import annotations

import csv
import logging
import math
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import CRS

from cubesip import spatial
from cubesip.drivers.base import (
    DatasetInfo,
    DatasetNotFoundError,
    Driver,
    EmptyResultError,
    SizeEstimate,
)
from cubesip.geohash import BBox
from cubesip.output import provenance_attrs, write_netcdf
from cubesip.request import Request

log = logging.getLogger(__name__)

DATASET = "ortho_hr"
CRS_RD = "EPSG:28992"
INDEX_BASE = "https://basisdata.nl/hwh-portal/20230609_tmp/links/nationaal/Nederland"
INDEX_MAX_AGE_S = 7 * 24 * 3600
TILE_M = 1000.0
DEFAULT_RESOLUTION = 0.08
# Which product holds each variable, and at which (1-based) band.
BANDS = {
    "RGB": {"red": 1, "green": 2, "blue": 3},
    "CIR": {"nir": 1, "red": 2, "green": 3},
}
VARIABLES = ("red", "green", "blue", "nir")
LONG_NAMES = {
    "red": "red reflectance (digital number)",
    "green": "green reflectance (digital number)",
    "blue": "blue reflectance (digital number)",
    "nir": "near-infrared reflectance (digital number)",
}
# The whole cube is held in memory before writing; warn above this estimate.
MAX_REQUEST_BYTES = 2_000_000_000
READ_WORKERS = 4
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


@dataclass(frozen=True)
class Tile:
    """One 1 x 1 km GeoTIFF: its lower-left corner in RD metres and its URL."""

    x0: int
    y0: int
    url: str


@dataclass(frozen=True)
class Grid:
    """The output grid: pixel edges in RD metres and the pixel size."""

    west: float
    south: float
    east: float
    north: float
    resolution: float

    @property
    def width(self) -> int:
        """Number of columns."""
        return round((self.east - self.west) / self.resolution)

    @property
    def height(self) -> int:
        """Number of rows."""
        return round((self.north - self.south) / self.resolution)


class BeeldmateriaalDriver(Driver):
    name = "beeldmateriaal"
    supported_options = frozenset({"resolution"})

    def __init__(self, config) -> None:
        """Set up the driver with an empty memory of tile indexes already read."""
        super().__init__(config)
        self._indexes: dict[tuple[int, str], dict[tuple[int, int], Tile]] = {}

    @property
    def backend_url(self) -> str:
        """The folder of the tile indexes on the Beeldmateriaal portal."""
        return INDEX_BASE

    @property
    def max_request_bytes(self) -> int | None:
        """Size above which the check warns, since the whole cube is held in memory."""
        return MAX_REQUEST_BYTES

    def dataset_key(self, dataset: str) -> str:
        """The dataset name in lower case without spaces."""
        return dataset.strip().lower()

    def validate(self, request: Request) -> None:
        """Refuse requests this driver cannot serve: an unknown dataset or variable, a time range
        that is not whole calendar years, or a resolution that does not divide a 1 km tile.
        """
        super().validate(request)
        if self.dataset_key(request.dataset) != DATASET:
            raise ValueError(
                f"unknown beeldmateriaal dataset {request.dataset!r}; available: {DATASET!r}"
            )
        time_ = request.time
        if (
            time_ is None
            or (time_.start.month, time_.start.day) != (1, 1)
            or (time_.end.month, time_.end.day) != (12, 31)
        ):
            raise ValueError(
                "Beeldmateriaal orthophotos are flown once a year: give time as whole calendar "
                "years, e.g. {'start': '2025-01-01', 'end': '2025-12-31'}"
            )
        resolution(request)

    # -- index ------------------------------------------------------------------------

    def index_dir(self) -> Path:
        """Folder where the tile indexes are kept."""
        return Path(self.config.data_dir) / "_index"

    def tiles(self, year: int, product: str) -> dict[tuple[int, int], Tile]:
        """The tiles of one year and product, keyed by lower-left corner. Raises
        DatasetNotFoundError when the portal has no tile index for that year.
        """
        key = (year, product)
        if key not in self._indexes:
            path = self.index_dir() / f"{index_name(year, product)}.csv"
            fresh = path.is_file() and time.time() - path.stat().st_mtime < INDEX_MAX_AGE_S
            if not fresh:
                try:
                    features = download_index(year, product)
                except DatasetNotFoundError:
                    if not path.is_file():
                        raise
                    log.warning("could not refresh %s; using the cached copy", path)
                else:
                    write_index_csv(path, select_tiles(features))
            self._indexes[key] = read_index_csv(path)
        return self._indexes[key]

    # -- check ------------------------------------------------------------------------

    def describe(self, request: Request) -> DatasetInfo:
        """Check that every requested year has tiles, and report missing tiles as notes."""
        info = DatasetInfo(
            dataset=self.dataset_key(request.dataset),
            available_variables=list(VARIABLES),
            missing_variables=[v for v in request.variables if v not in VARIABLES],
            crs=CRS_RD,
        )
        if info.missing_variables:
            return info
        corners = tile_corners(rd_bounds(request.bbox))
        for year in years(request):
            for product in products(request.variables):
                found = self.tiles(year, product)
                missing = [c for c in corners if c not in found]
                if len(missing) == len(corners):
                    raise ValueError(
                        f"no {product} tiles for {year} at {request.geohash}: outside the "
                        "Netherlands or not delivered yet"
                    )
                if missing:
                    info.notes.append(
                        f"{len(missing)} of {len(corners)} {product} tiles missing for {year} "
                        "(sea or not delivered yet): those pixels will be 0"
                    )
        return info

    def estimate_size(self, request: Request) -> SizeEstimate:
        """Exact size of the output: pixels times variables times years, one byte per value."""
        grid = output_grid(request.bbox, resolution(request))
        n_vars, n_years = len(request.variables), len(years(request))
        return SizeEstimate(
            nbytes=grid.width * grid.height * n_vars * n_years,
            method="exact",
            detail=(
                f"{grid.width}x{grid.height} px at {grid.resolution:g} m, {n_vars} variable(s), "
                f"{n_years} year(s), uint8"
            ),
        )

    # -- fetch ------------------------------------------------------------------------

    def fetch(self, request: Request, out_path: Path) -> Path:
        """Read the requested window of every tile the geohash overlaps, for each year and each
        product needed, mosaic them on the output grid and save the result as NetCDF.
        """
        grid = output_grid(request.bbox, resolution(request))
        corners = tile_corners((grid.west, grid.south, grid.east, grid.north))
        sources = band_sources(request.variables)
        shape = (len(years(request)), grid.height, grid.width)
        data = {v: np.zeros(shape, "uint8") for v in sources}
        used: list[str] = []
        for t, year in enumerate(years(request)):
            for product, wanted in group_by_product(sources).items():
                index = self.tiles(year, product)
                tiles = [index[c] for c in corners if c in index]
                log.info(
                    "Beeldmateriaal %s %d: reading %d tile(s) for %s",
                    product,
                    year,
                    len(tiles),
                    request.geohash,
                )
                bands = [BANDS[product][v] for v in wanted]
                for tile, block, (rows, cols) in read_tiles(tiles, grid, bands):
                    for i, variable in enumerate(wanted):
                        data[variable][t, rows, cols] = block[i]
                    used.append(tile.url)
        if not used:
            raise EmptyResultError(f"no Beeldmateriaal tiles at {request.geohash}")
        ds = orthophoto_dataset(data, years(request), grid)
        attrs = provenance_attrs(request, self.dataset_key(request.dataset), self.backend_url)
        attrs["source_tiles"] = "\n".join(sorted(set(used)))
        attrs["license"] = "CC BY 4.0, Beeldmateriaal Nederland (https://www.beeldmateriaal.nl)"
        return write_netcdf(ds, out_path, attrs)


# -- request helpers ----------------------------------------------------------------------


def years(request: Request) -> list[int]:
    """The calendar years covered by the request."""
    assert request.time is not None  # validate() requires it
    return list(range(request.time.start.year, request.time.end.year + 1))


def resolution(request: Request) -> float:
    """The output pixel size in metres (option 'resolution', default 0.08). Raises ValueError
    unless it is positive and divides a 1 km tile into a whole number of pixels.
    """
    value = request.options.get("resolution", DEFAULT_RESOLUTION)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError("options.resolution must be a positive number of metres")
    per_tile = TILE_M / value
    if abs(per_tile - round(per_tile)) > 1e-6:
        raise ValueError(
            f"options.resolution {value} does not divide a 1 km tile into whole pixels "
            "(use e.g. 0.05, 0.08, 0.1, 0.25, 1)"
        )
    return float(value)


def band_sources(variables: Iterable[str]) -> dict[str, str]:
    """Which product each variable is read from. Red and green are in both products: they come
    from the RGB tiles when those are read anyway (blue requested, or no nir), otherwise from
    the CIR tiles, so a request never reads more products than it needs.
    """
    variables = list(variables)
    unknown = [v for v in variables if v not in VARIABLES]
    if unknown:
        raise ValueError(f"unknown variables {unknown}; available: {list(VARIABLES)}")
    use_rgb = "blue" in variables or "nir" not in variables
    sources = {}
    for v in variables:
        if v == "nir":
            sources[v] = "CIR"
        elif v == "blue":
            sources[v] = "RGB"
        else:
            sources[v] = "RGB" if use_rgb else "CIR"
    return sources


def products(variables: Iterable[str]) -> list[str]:
    """The products (RGB, CIR) needed for these variables."""
    return sorted(set(band_sources(variables).values()), reverse=True)


def group_by_product(sources: dict[str, str]) -> dict[str, list[str]]:
    """Variables grouped by the product they are read from, in request order."""
    groups: dict[str, list[str]] = {}
    for variable, product in sources.items():
        groups.setdefault(product, []).append(variable)
    return groups


# -- grid helpers -------------------------------------------------------------------------


def rd_bounds(box: BBox) -> tuple[float, float, float, float]:
    """The geohash box in RD New metres (west, south, east, north)."""
    native = spatial.to_native_bounds(box, CRS_RD)
    return native.west, native.south, native.east, native.north


def output_grid(box: BBox, res: float) -> Grid:
    """The grid that covers the geohash box, with edges snapped outwards to multiples of the
    resolution (so they line up with tile edges, which are on whole kilometres).
    """
    west, south, east, north = rd_bounds(box)
    return Grid(
        west=math.floor(round(west / res, 6)) * res,
        south=math.floor(round(south / res, 6)) * res,
        east=math.ceil(round(east / res, 6)) * res,
        north=math.ceil(round(north / res, 6)) * res,
        resolution=res,
    )


def tile_corners(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """Lower-left corners (RD metres) of the 1 km tiles that overlap the bounds."""
    west, south, east, north = bounds
    xs = range(int(math.floor(west / TILE_M)), int(math.ceil(east / TILE_M)))
    ys = range(int(math.floor(south / TILE_M)), int(math.ceil(north / TILE_M)))
    return [(int(x * TILE_M), int(y * TILE_M)) for y in ys for x in xs]


def tile_slices(tile: Tile, grid: Grid) -> tuple[tuple[slice, slice], tuple[float, ...]]:
    """Where a tile lands on the output grid: the (rows, cols) slices, and the RD bounds of
    the overlap (west, south, east, north) to read from the tile.
    """
    west = max(grid.west, tile.x0)
    east = min(grid.east, tile.x0 + TILE_M)
    south = max(grid.south, tile.y0)
    north = min(grid.north, tile.y0 + TILE_M)
    res = grid.resolution
    col0 = round((west - grid.west) / res)
    col1 = round((east - grid.west) / res)
    row0 = round((grid.north - north) / res)
    row1 = round((grid.north - south) / res)
    return (slice(row0, row1), slice(col0, col1)), (west, south, east, north)


def read_tiles(tiles: list[Tile], grid: Grid, bands: list[int]):
    """Read the part of each tile that falls on the grid, in parallel. Yields (tile, array of
    shape (len(bands), rows, cols), (row slice, col slice)).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    def read_one(tile: Tile):
        (rows, cols), (west, south, east, north) = tile_slices(tile, grid)
        shape = (len(bands), rows.stop - rows.start, cols.stop - cols.start)
        path = f"/vsicurl/{tile.url}" if "://" in tile.url else tile.url
        with rasterio.Env(**GDAL_ENV), rasterio.open(path) as src:
            window = from_bounds(west, south, east, north, src.transform)
            same = math.isclose(src.res[0], grid.resolution, rel_tol=1e-6)
            block = src.read(
                bands,
                window=window,
                out_shape=shape,
                resampling=Resampling.nearest if same else Resampling.average,
            )
        return tile, block, (rows, cols)

    if not tiles:
        return
    with ThreadPoolExecutor(max_workers=min(READ_WORKERS, len(tiles))) as pool:
        yield from pool.map(read_one, tiles)


def orthophoto_dataset(data: dict[str, np.ndarray], years_: list[int], grid: Grid) -> xr.Dataset:
    """Wrap the per-variable arrays (time, y, x) into a dataset with pixel-centre RD coordinates
    and the CRS recorded.
    """
    res = grid.resolution
    x = grid.west + res * (np.arange(grid.width) + 0.5)
    y = grid.north - res * (np.arange(grid.height) + 0.5)
    crs = CRS.from_user_input(CRS_RD)
    grid_mapping = {**crs.to_cf(), "spatial_ref": crs.to_wkt()}
    ds = xr.Dataset(
        {
            name: (
                ("time", "y", "x"),
                values,
                {
                    "long_name": LONG_NAMES[name],
                    "units": "1",
                    "grid_mapping": "spatial_ref",
                    "comment": "0 where no tile exists",
                },
            )
            for name, values in data.items()
        },
        coords={
            "time": pd.to_datetime([f"{year}-01-01" for year in years_]),
            "y": y,
            "x": x,
            "spatial_ref": ((), 0, grid_mapping),
        },
    )
    ds["time"].attrs["long_name"] = "flight year (spring flight, stamped on 1 January)"
    ds["x"].attrs.update(standard_name="projection_x_coordinate", units="m")
    ds["y"].attrs.update(standard_name="projection_y_coordinate", units="m")
    return ds


# -- tile index ---------------------------------------------------------------------------


def index_name(year: int, product: str) -> str:
    """Name of the portal's tile index for one year and product, e.g. BM_HRL2025O_CIR_TIF."""
    return f"BM_HRL{year}O_{product}_TIF"


def download_index(year: int, product: str) -> list[dict]:
    """Download the GeoJSON tile index of one year and product and return its features. Raises
    DatasetNotFoundError when the portal has none (it answers 403 for unknown files).
    """
    url = f"{INDEX_BASE}/{index_name(year, product)}.json"
    log.info("downloading the Beeldmateriaal tile index %s", url)
    response = requests.get(url, timeout=300)
    if response.status_code in (403, 404):
        raise DatasetNotFoundError(
            f"no Beeldmateriaal {product} orthophoto tiles published for {year} ({url})"
        )
    response.raise_for_status()
    return response.json().get("features", [])


_TILE_NAME = re.compile(r"[/\\]\d{4}_\d+_\d+_(RGB|CIR)(_JPEG)?_hrl\.tif$")


def select_tiles(features: list[dict]) -> list[Tile]:
    """Pick one GeoTIFF per 1 km tile from the index features.

    The index also lists sidecar files (.aux.xml), and about 2% of the tiles appear twice,
    from two neighbouring flight blocks. For those the most recently modified file is kept
    (then the larger one, then the URL, so the choice is always the same).
    """
    best: dict[tuple[int, int], tuple[tuple, Tile]] = {}
    for feature in features:
        props = feature.get("properties") or {}
        url = props.get("file", "")
        if not _TILE_NAME.search(url):
            continue
        ring = feature["geometry"]["coordinates"][0]
        x0 = int(round(min(p[0] for p in ring)))
        y0 = int(round(min(p[1] for p in ring)))
        rank = (props.get("mtime", ""), props.get("size", 0), url)
        tile = Tile(x0, y0, url)
        if (x0, y0) not in best or rank > best[(x0, y0)][0]:
            best[(x0, y0)] = (rank, tile)
    return [tile for _, tile in sorted(best.values(), key=lambda item: (item[1].x0, item[1].y0))]


def write_index_csv(path: Path, tiles: list[Tile]) -> None:
    """Save the selected tiles as a CSV (x0, y0, url), replacing the file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x0", "y0", "url"])
        writer.writerows((t.x0, t.y0, t.url) for t in tiles)
    tmp.replace(path)


def read_index_csv(path: Path) -> dict[tuple[int, int], Tile]:
    """Read a tile CSV written by write_index_csv, keyed by lower-left corner."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = csv.DictReader(fh)
        tiles = (Tile(int(r["x0"]), int(r["y0"]), r["url"]) for r in rows)
        return {(t.x0, t.y0): t for t in tiles}

"""TESSERA driver: annual embeddings from the TESSERA Earth-observation foundation model.

TESSERA (https://github.com/ucam-eo/tessera) compresses a year of Sentinel-1 and Sentinel-2
observations into a 128-channel embedding for every 10 m pixel. cubesip reads them with
geotessera's Zarr API (``GeoTesseraZarr.read_region``) from the public stores on Source
Cooperative; no account is needed. Install with ``pip install 'cubesip[tessera]'``.

- dataset: a store version (``v1.1``, the global default; ``v1``; ``v2``), an explicit store
  such as ``v2-2B-L~beta1``, or a store URL / local path. Versions are resolved with
  ``geotessera.registry.zarr_store_url``; the resolved URL is the dataset in the manifest.
- variables: ``["embedding"]``, float32 ``(time, band, y, x)``, dequantised by geotessera,
  NaN where there is no embedding (open water, outside coverage)
- time: whole calendar years (``2024-01-01`` .. ``2024-12-31``), one step per year stamped
  on 1 January. Partial years are refused: an embedding summarises a whole year, and a cached
  file could not serve a partial period.
- options: ``bands`` (zero-based channels to keep, in order), ``depth`` (a published
  embedding prefix of v2 stores, e.g. 16; ``bands`` then index into that prefix)
- grid: the native UTM grid (10 m) of the zone holding the geohash centre, pixels untouched

Docs: https://github.com/ucam-eo/geotessera
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from cubesip.output import provenance_attrs, write_netcdf
from cubesip.request import Request

log = logging.getLogger(__name__)

VARIABLE = "embedding"
PIXEL_M = 10.0
BYTES_PER_VALUE = 4  # float32 after dequantisation
# The whole cube is read into memory before writing; warn above this estimate.
MAX_REQUEST_BYTES = 2_000_000_000
INSTALL_HINT = "the tessera driver needs geotessera: pip install 'cubesip[tessera]'"


def _require_geotessera() -> None:
    """Check that the optional geotessera package is installed; if not, say how to install it."""
    try:
        import geotessera  # noqa: F401
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc


class TesseraDriver(Driver):
    name = "tessera"
    supported_options = frozenset({"bands", "depth"})

    def __init__(self, config) -> None:
        """Set up the driver with an empty memory of stores already opened."""
        super().__init__(config)
        self._stores: dict[str, Any] = {}

    @property
    def max_request_bytes(self) -> int | None:
        """Size above which the check warns, since the whole cube is held in memory before it is
        written.
        """
        return MAX_REQUEST_BYTES

    def dataset_key(self, dataset: str) -> str:
        """Turn a store version such as 'v1.1' into the store's URL. A URL or an existing local path
        is kept as it is.
        """
        value = dataset.strip().rstrip("/")
        if "://" in value or Path(value).expanduser().exists():
            return value
        _require_geotessera()
        from geotessera.registry import zarr_store_url

        return zarr_store_url(value).rstrip("/")

    def validate(self, request: Request) -> None:
        """Refuse requests TESSERA cannot serve: the time must cover whole calendar years, 'bands'
        must be a list of different channel numbers (0 or more), and 'depth' a positive whole
        number.
        """
        super().validate(request)
        _require_geotessera()
        time = request.time
        if (
            time is None
            or (time.start.month, time.start.day) != (1, 1)
            or (time.end.month, time.end.day) != (12, 31)
        ):
            raise ValueError(
                "TESSERA embeddings are annual: give time as whole calendar years, "
                "e.g. {'start': '2024-01-01', 'end': '2024-12-31'}"
            )
        bands = request.options.get("bands")
        if bands is not None and not (
            isinstance(bands, list)
            and bands
            and all(isinstance(b, int) and not isinstance(b, bool) and b >= 0 for b in bands)
            and len(set(bands)) == len(bands)
        ):
            raise ValueError("options.bands must be a non-empty list of distinct channel indices")
        depth = request.options.get("depth")
        if depth is not None and (
            not isinstance(depth, int) or isinstance(depth, bool) or depth < 1
        ):
            raise ValueError("options.depth must be a positive integer")

    # -- store ------------------------------------------------------------------------

    def store(self, request: Request):
        """Open the TESSERA store of the request once, and reuse it afterwards. Raises
        DatasetNotFoundError if the store does not exist.
        """
        key = self.dataset_key(request.dataset)
        if key not in self._stores:
            from geotessera.store import GeoTesseraZarr

            try:
                self._stores[key] = GeoTesseraZarr(key)
            except ValueError as exc:
                raise DatasetNotFoundError(f"TESSERA store not found: {key}") from exc
        return self._stores[key]

    @staticmethod
    def years(request: Request) -> list[int]:
        """The calendar years covered by the request."""
        assert request.time is not None  # validate() requires it
        return list(range(request.time.start.year, request.time.end.year + 1))

    def channels(self, request: Request) -> list[int]:
        """The embedding channels the output will hold (all of them unless 'bands' is given),
        checked against what the store offers.
        """
        store = self.store(request)
        depth = request.options.get("depth")
        if depth is not None and depth not in store.depths:
            raise ValueError(
                f"depth {depth} is not published in this store; available: {sorted(store.depths)}"
            )
        width = depth or store.n_bands
        bands = request.options.get("bands") or list(range(width))
        outside = [b for b in bands if b >= width]
        if outside:
            raise ValueError(f"options.bands {outside} outside 0..{width - 1}")
        return bands

    # -- check ------------------------------------------------------------------------

    def describe(self, request: Request) -> DatasetInfo:
        """Check that the store has every requested year and the requested channels, and report the
        one available variable ('embedding') and the CRS.
        """
        store = self.store(request)
        missing_years = [y for y in self.years(request) if y not in store.years]
        if missing_years:
            raise ValueError(
                f"years {missing_years} not in the TESSERA store; available: {store.years}"
            )
        self.channels(request)
        return DatasetInfo(
            dataset=self.dataset_key(request.dataset),
            available_variables=[VARIABLE],
            missing_variables=[v for v in request.variables if v != VARIABLE],
            crs=zone_crs(request),
        )

    def estimate_size(self, request: Request) -> SizeEstimate:
        """Estimate the download size: the pixels in the area (10 m each) times the channels times
        the years, at 4 bytes per value.
        """
        box = spatial.to_native_bounds(request.bbox, zone_crs(request))
        nx = max(1, math.ceil((box.east - box.west) / PIXEL_M))
        ny = max(1, math.ceil((box.north - box.south) / PIXEL_M))
        channels, years = len(self.channels(request)), len(self.years(request))
        return SizeEstimate(
            nbytes=nx * ny * channels * years * BYTES_PER_VALUE,
            method="estimate",
            detail=f"{nx}x{ny} px at 10 m, {channels} channel(s), {years} year(s), float32",
        )

    # -- fetch ------------------------------------------------------------------------

    def fetch(self, request: Request, out_path: Path) -> Path:
        """Read the embeddings of each year, keep the requested channels, combine them into one
        dataset and save it as NetCDF. Raises an error if the grid differs between years, or if
        there is no data at all (e.g. open water).
        """
        store = self.store(request)
        channels = self.channels(request)
        depth = request.options.get("depth")
        box = request.bbox
        mosaics, grid = [], None
        for year in self.years(request):
            log.info("TESSERA %s: reading %s for %d", store.url, request.geohash, year)
            mosaic, transform, crs = store.read_region(
                (box.west, box.south, box.east, box.north), year, depth=depth
            )
            if grid is None:
                grid = (transform, crs, mosaic.shape[:2])
            elif grid != (transform, crs, mosaic.shape[:2]):
                raise RuntimeError(f"TESSERA grid changed between years at {request.geohash}")
            mosaics.append(mosaic[..., channels])
        assert grid is not None
        transform, crs, _ = grid
        ds = embedding_dataset(mosaics, self.years(request), transform, crs, channels)
        if not np.isfinite(ds[VARIABLE].values).any():
            raise EmptyResultError(
                f"no TESSERA embedding at {request.geohash} for {self.years(request)} "
                "(open water or outside coverage)"
            )
        attrs = provenance_attrs(request, self.dataset_key(request.dataset), self.backend_url)
        attrs["tessera_model"] = str(getattr(store, "model_version", ""))
        return write_netcdf(ds, out_path, attrs)


# -- helpers (pure, unit tested) ----------------------------------------------------------


def zone_crs(request: Request) -> str:
    """The UTM CRS of the TESSERA zone that holds the centre of the geohash cell.

    TESSERA zones use the northern UTM CRS everywhere (negative northings south of the
    equator), so this is always EPSG:326NN.
    """
    box = request.bbox
    lon = (box.west + box.east) / 2
    zone = max(1, min(60, math.floor((lon + 180) / 6) + 1))
    return f"EPSG:{32600 + zone}"


def embedding_dataset(
    mosaics: list[np.ndarray],
    years: list[int],
    transform,
    crs: str,
    channels: list[int],
) -> xr.Dataset:
    """Combine the per-year embedding arrays into one dataset with dimensions (time, band, y, x),
    pixel-centre coordinates and the CRS recorded.
    """
    data = np.stack([np.moveaxis(m, -1, 0) for m in mosaics]).astype("float32", copy=False)
    height, width = data.shape[-2:]
    x = transform.c + transform.a * (np.arange(width) + 0.5)
    y = transform.f + transform.e * (np.arange(height) + 0.5)
    crs_obj = CRS.from_user_input(crs)
    grid_mapping = {**crs_obj.to_cf(), "spatial_ref": crs_obj.to_wkt()}
    ds = xr.Dataset(
        {
            VARIABLE: (
                ("time", "band", "y", "x"),
                data,
                {
                    "long_name": "TESSERA embedding",
                    "units": "1",
                    "grid_mapping": "spatial_ref",
                },
            )
        },
        coords={
            "time": pd.to_datetime([f"{year}-01-01" for year in years]),
            "band": np.asarray(channels, dtype="int16"),
            "y": y,
            "x": x,
            "spatial_ref": ((), 0, grid_mapping),
        },
    )
    ds["time"].attrs["long_name"] = "embedding year (stamped on 1 January)"
    ds["band"].attrs["long_name"] = "embedding channel (zero-based)"
    ds["x"].attrs.update(standard_name="projection_x_coordinate", units="m")
    ds["y"].attrs.update(standard_name="projection_y_coordinate", units="m")
    return ds

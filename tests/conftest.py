"""Synthetic datasets shared by the tests (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer


def era5_like(
    lat_step: float = -1.0,
    lon_0_360: bool = True,
    hours: int = 72,
    start: str = "2024-01-01",
) -> xr.Dataset:
    """Hourly lat/lon grid like ERA5 on Earth Data Hub (valid_time, descending latitude)."""
    lat = np.arange(60, 30 + lat_step / 2, lat_step) if lat_step < 0 else np.arange(30, 60.5, 1)
    lon = np.arange(0, 360, 1.0) if lon_0_360 else np.arange(-180, 180, 1.0)
    time = pd.date_range(start, periods=hours, freq="h")
    rng = np.random.default_rng(0)
    data = rng.normal(280, 5, (time.size, lat.size, lon.size)).astype("float32")
    return xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), data, {"units": "K"}),
            "tp": (("valid_time", "latitude", "longitude"), data / 1000, {"units": "m"}),
        },
        coords={"valid_time": time, "latitude": lat, "longitude": lon},
    )


UTM_CENTER = (45.3, -3.7)  # lat, lon


def utm_like(epsg: int = 32630, days: int = 5, res: float = 100.0) -> xr.Dataset:
    """Daily projected grid (pixel centres) with a CF grid-mapping coordinate.

    Covers ~20 x 20 km around UTM_CENTER, in UTM 30N.
    """
    x0, y0 = Transformer.from_crs("EPSG:4326", epsg, always_xy=True).transform(
        UTM_CENTER[1], UTM_CENTER[0]
    )
    x0, y0 = round(x0, -3), round(y0, -3)
    x = np.arange(x0 - 10_000 + res / 2, x0 + 10_000, res)
    y = np.arange(y0 + 10_000 - res / 2, y0 - 10_000, -res)
    time = pd.date_range("2024-06-01", periods=days, freq="D")
    rng = np.random.default_rng(1)
    shape = (time.size, y.size, x.size)
    crs = CRS.from_epsg(epsg)
    gm_attrs = {**crs.to_cf(), "spatial_ref": crs.to_wkt()}
    ds = xr.Dataset(
        {
            "B04": (("t", "y", "x"), rng.integers(0, 10_000, shape).astype("float32")),
            "B08": (("t", "y", "x"), rng.integers(0, 10_000, shape).astype("float32")),
        },
        coords={"t": time, "y": y, "x": x, "spatial_ref": ((), 0, gm_attrs)},
    )
    for name in ("B04", "B08"):
        ds[name].attrs["grid_mapping"] = "spatial_ref"
    return ds


def static_like() -> xr.Dataset:
    """Static lat/lon field (e.g. a DEM) with no time dimension."""
    lat = np.arange(30, 60, 0.1)
    lon = np.arange(-10, 10, 0.1)
    data = np.random.default_rng(2).uniform(0, 2000, (lat.size, lon.size)).astype("float32")
    return xr.Dataset(
        {"elevation": (("lat", "lon"), data, {"units": "m"})},
        coords={"lat": lat, "lon": lon},
    )


@pytest.fixture
def era5_ds() -> xr.Dataset:
    return era5_like()


@pytest.fixture
def utm_ds() -> xr.Dataset:
    return utm_like()


@pytest.fixture
def static_ds() -> xr.Dataset:
    return static_like()

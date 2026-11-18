from datetime import date

import numpy as np
import pytest
from conftest import UTM_CENTER, era5_like
from pyproj import Transformer

from cubesip import geohash as gh
from cubesip.request import TimeRange
from cubesip.spatial import (
    EmptySelectionError,
    find_dims,
    native_crs,
    subset,
    to_native_bounds,
)

DAY1 = TimeRange(start=date(2024, 1, 1), end=date(2024, 1, 1))


def _lon180(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values) + 180) % 360) - 180


def test_find_dims(era5_ds, utm_ds, static_ds) -> None:
    assert find_dims(era5_ds) == ("valid_time", "latitude", "longitude")
    assert find_dims(utm_ds) == ("t", "y", "x")
    assert find_dims(static_ds) == (None, "lat", "lon")


def test_native_crs(era5_ds, utm_ds, static_ds) -> None:
    assert native_crs(era5_ds) == "EPSG:4326"
    assert native_crs(static_ds) == "EPSG:4326"
    assert native_crs(utm_ds) == "EPSG:32630"
    with pytest.raises(ValueError, match="no CRS"):
        native_crs(utm_ds.drop_vars("spatial_ref"))


@pytest.mark.parametrize("lon_0_360", [True, False])
@pytest.mark.parametrize("lat_step", [-1.0, 1.0])
def test_subset_selects_intersecting_cells(lon_0_360: bool, lat_step: float) -> None:
    ds = era5_like(lat_step=lat_step, lon_0_360=lon_0_360)
    code = gh.encode(45.3, -3.7, 3)  # ~1.4 x 1.4 deg, west of Greenwich
    box = gh.bbox(code)
    out = subset(ds, box, DAY1, ["t2m"])

    lons = _lon180(out["longitude"].values)
    lats = out["latitude"].values
    half = 0.5
    all_lons = _lon180(ds["longitude"].values)
    expected_lons = all_lons[(all_lons > box.west - half) & (all_lons < box.east + half)]
    all_lats = ds["latitude"].values
    expected_lats = all_lats[(all_lats > box.south - half) & (all_lats < box.north + half)]
    assert sorted(lons) == sorted(expected_lons)
    assert sorted(lats) == sorted(expected_lats)
    assert list(out.data_vars) == ["t2m"]
    assert out.sizes["valid_time"] == 24  # end day inclusive, hourly


def test_tiny_geohash_never_empty(era5_ds) -> None:
    out = subset(era5_ds, gh.bbox(gh.encode(45.3, -3.7, 7)), DAY1, ["t2m"])
    assert out.sizes["latitude"] >= 1 and out.sizes["longitude"] >= 1
    assert out.sizes["latitude"] <= 2 and out.sizes["longitude"] <= 2


def test_subset_wraps_around_greenwich(era5_ds) -> None:
    # cell touching Greenwich from the west: on a 0..360 grid its cells sit at both array ends
    box = gh.bbox(gh.encode(50.0, -0.01, 2))
    assert box.east == 0.0 and box.west == -11.25
    out = subset(era5_ds, box, DAY1, ["t2m"])
    assert sorted(_lon180(out["longitude"].values)) == list(range(-11, 1))


def test_subset_projected_grid(utm_ds) -> None:
    code = gh.encode(*UTM_CENTER, 5)  # ~4.9 x 4.9 km, inside the synthetic 20 km UTM tile
    box = gh.bbox(code)
    native = to_native_bounds(box, "EPSG:32630")
    out = subset(utm_ds, box, None, ["B04"])
    x, y = out["x"].values, out["y"].values
    res = 100.0
    # covers the transformed box and adds at most one pixel
    assert x.min() - res / 2 <= native.west and x.max() + res / 2 >= native.east
    assert y.min() - res / 2 <= native.south and y.max() + res / 2 >= native.north
    assert x.min() > native.west - res and x.max() < native.east + res
    assert out.sizes["t"] == utm_ds.sizes["t"]  # time=None keeps every step
    assert "spatial_ref" in out.coords
    assert native_crs(out) == "EPSG:32630"


def test_to_native_bounds_contains_corners() -> None:
    box = gh.bbox("ezs42")
    native = to_native_bounds(box, "EPSG:32630")
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    for lon in (box.west, box.east):
        for lat in (box.south, box.north):
            x, y = tr.transform(lon, lat)
            assert native.west <= x <= native.east and native.south <= y <= native.north
    assert to_native_bounds(box, "EPSG:4326") == box


def test_static_dataset(static_ds) -> None:
    box = gh.bbox(gh.encode(45.3, -3.7, 4))
    out = subset(static_ds, box, None, ["elevation"])
    assert set(out.dims) == {"lat", "lon"}
    with pytest.raises(ValueError, match="no time dimension"):
        subset(static_ds, box, DAY1, ["elevation"])


def test_errors(era5_ds) -> None:
    box = gh.bbox(gh.encode(45.3, -3.7, 4))
    with pytest.raises(KeyError, match="u10"):
        subset(era5_ds, box, DAY1, ["u10"])
    with pytest.raises(EmptySelectionError):
        subset(era5_ds, gh.bbox(gh.encode(-30.0, 20.0, 4)), DAY1, ["t2m"])
    with pytest.raises(EmptySelectionError, match="no time steps"):
        subset(era5_ds, box, TimeRange(start=date(2030, 1, 1), end=date(2030, 1, 2)), ["t2m"])

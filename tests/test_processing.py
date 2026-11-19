"""cubesip.processing: geohash of a WKT geometry, and regridding onto a reference grid."""

from datetime import date
from pathlib import Path

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr
from conftest import UTM_CENTER, era5_like, utm_like
from pyproj import Transformer

from cubesip import geohash as gh
from cubesip.api import Result
from cubesip.processing import containing_geohash, regrid
from cubesip.request import TimeRange
from cubesip.spatial import native_crs, subset

LAT, LON = UTM_CENTER
DAY1 = TimeRange(start=date(2024, 1, 1), end=date(2024, 1, 1))


def _polygon(west: float, south: float, east: float, north: float) -> str:
    ring = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    return "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in ring) + "))"


# -- containing_geohash -------------------------------------------------------------------


def test_containing_geohash_of_a_field() -> None:
    code = gh.encode(LAT, LON, 6)
    box = gh.bbox(code)
    cx, cy = (box.west + box.east) / 2, (box.south + box.north) / 2
    dx, dy = (box.east - box.west) / 4, (box.north - box.south) / 4
    # straddles the cell's next split, so no level-7 child holds it
    assert containing_geohash(_polygon(cx - dx, cy - dy, cx + dx, cy + dy)) == code


def test_containing_geohash_of_a_cell_outline_is_the_cell() -> None:
    assert containing_geohash(_polygon(*gh.bbox("gbj3ey"))) == "gbj3ey"


def test_containing_geohash_of_a_point() -> None:
    point = f"POINT ({LON} {LAT})"
    assert containing_geohash(point, max_precision=7) == gh.encode(LAT, LON, 7)
    assert len(containing_geohash(point)) == gh.MAX_LENGTH


def test_containing_geohash_of_a_multipoint_is_the_common_parent() -> None:
    a, b = gh.bbox("ezs42"), gh.bbox("ezs43")
    centres = [((c.west + c.east) / 2, (c.south + c.north) / 2) for c in (a, b)]
    wkt = "MULTIPOINT (" + ", ".join(f"({x} {y})" for x, y in centres) + ")"
    assert containing_geohash(wkt) == "ezs4"


@pytest.mark.parametrize(
    ("wkt", "match"),
    [
        ("POLYGON ((0 0, 1 0", "cannot read WKT"),
        ("POLYGON EMPTY", "empty"),
        ("LINESTRING (-1 51, 1 52)", "level-1"),  # crosses the prime meridian
    ],
)
def test_containing_geohash_errors(wkt: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        containing_geohash(wkt)


# -- regrid -------------------------------------------------------------------------------


def latlon_like(step: float = 0.05, lon_0_360: bool = True) -> xr.Dataset:
    """Lat/lon grid (descending latitude, like ERA5) around UTM_CENTER, finer than ERA5 so
    the UTM test grid spans several cells."""
    lat = np.round(np.arange(45.6, 45.0 - step / 2, -step), 6)
    lon = np.round(np.arange(-4.1, -3.3 + step / 2, step), 6)
    if lon_0_360:
        lon = lon % 360
    rng = np.random.default_rng(3)
    data = rng.normal(280, 5, (2, lat.size, lon.size)).astype("float32")
    return xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), data, {"units": "K"})},
        coords={
            "valid_time": np.array(["2024-06-01", "2024-06-02"], dtype="datetime64[ns]"),
            "latitude": lat,
            "longitude": lon,
        },
    )


def _pixel_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    to_geo = Transformer.from_crs(native_crs(ds), "EPSG:4326", always_xy=True)
    x, y = np.meshgrid(ds["x"].values, ds["y"].values)
    return to_geo.transform(x, y)


def _off_edges(values: np.ndarray, centres: np.ndarray, margin: float) -> np.ndarray:
    """True where ``values`` are farther than ``margin`` from the edges of the cells."""
    step = abs(centres[1] - centres[0])
    offset = (values - (centres[0] - step / 2)) % step
    return np.minimum(offset, step - offset) > margin


def _broadcast(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(values[:, None, None], shape)


@pytest.mark.parametrize("lon_0_360", [True, False])
def test_regrid_latlon_onto_utm_takes_the_cell_under_each_pixel(lon_0_360: bool) -> None:
    source = latlon_like(lon_0_360=lon_0_360).assign(hour=("valid_time", [0, 24]))
    ref = utm_like()
    out = regrid(source, ref)

    lon, lat = _pixel_lonlat(ref)
    lons = (source["longitude"].values + 180) % 360 - 180
    col = np.abs(lons[:, None, None] - lon).argmin(axis=0)
    row = np.abs(source["latitude"].values[:, None, None] - lat).argmin(axis=0)
    assert len(np.unique(col)) > 3 and len(np.unique(row)) > 3  # several source cells
    # GDAL's approximate transformer is exact to 1/8 of a reference pixel (12.5 m here), so
    # pixels on a source cell edge may fall either side: compare the others (~50 m away)
    away = _off_edges(lon, lons, 5e-4) & _off_edges(lat, source["latitude"].values, 5e-4)
    assert away.mean() > 0.95
    expected = source["t2m"].values[:, row, col]
    np.testing.assert_array_equal(out["t2m"].values[:, away], expected[:, away])

    assert out["t2m"].dims == ("valid_time", "y", "x")
    np.testing.assert_array_equal(out["x"], ref["x"])
    np.testing.assert_array_equal(out["y"], ref["y"])
    assert native_crs(out) == native_crs(ref)
    assert out["t2m"].encoding["grid_mapping"] == "spatial_ref"
    assert out["t2m"].attrs == {"units": "K"}
    xr.testing.assert_identical(out["hour"].reset_coords(drop=True), source["hour"])


def test_regrid_utm_onto_latlon_average() -> None:
    fine = utm_like()
    easting = (0 * fine["B04"] + fine["x"]).astype("float64").rename("easting")
    ref = latlon_like(step=0.02).sel(latitude=slice(45.36, 45.24), longitude=slice(356.24, 356.36))
    out = regrid(easting, ref, method="average")

    lon, lat = np.meshgrid(ref["longitude"].values - 360, ref["latitude"].values)
    to_utm = Transformer.from_crs("EPSG:4326", native_crs(fine), always_xy=True)
    expected, _ = to_utm.transform(lon, lat)
    assert out.dims == ("t", "latitude", "longitude")
    np.testing.assert_allclose(out.isel(t=0).values, expected, atol=20)  # cells are ~1.6 km


def test_regrid_onto_its_own_grid_is_identity() -> None:
    ds = utm_like()
    xr.testing.assert_equal(regrid(ds, ds), ds)


def test_regrid_average_onto_a_coarser_grid() -> None:
    fine = utm_like()
    coarse = fine.coarsen(x=10, y=10).mean()
    out = regrid(fine, coarse, method="average")
    np.testing.assert_allclose(out["B04"].values, coarse["B04"].values, rtol=1e-5)


def test_regrid_follows_the_reference_coordinate_order() -> None:
    fine = utm_like()
    coarse = fine.coarsen(x=10, y=10).mean()
    flipped = coarse.isel(y=slice(None, None, -1), x=slice(None, None, -1))  # y ascending
    out = regrid(fine, flipped, method="average")
    np.testing.assert_array_equal(out["y"], flipped["y"])
    np.testing.assert_allclose(out["B04"].values, flipped["B04"].values, rtol=1e-5)


def test_regrid_single_cell_source_covers_the_reference() -> None:
    code = gh.encode(LAT, LON, 6)
    era5 = subset(era5_like(), gh.bbox(code), DAY1, ["t2m"])
    assert era5.sizes["latitude"] == era5.sizes["longitude"] == 1
    ref = subset(utm_like(), gh.bbox(code), None, ["B04"])  # e.g. Sentinel-2 for the cell
    out = regrid(era5, ref)
    expected = era5["t2m"].values[:, 0, 0]
    np.testing.assert_array_equal(out["t2m"].values, _broadcast(expected, out["t2m"].shape))


def test_regrid_single_row_is_stretched_over_the_reference() -> None:
    strip = era5_like().sel(latitude=[45.0], longitude=[356.0, 357.0])  # cells at -4 and -3
    out = regrid(strip, utm_like())  # the UTM grid lies in the -4 cell
    expected = strip["t2m"].sel(latitude=45.0, longitude=356.0).values
    np.testing.assert_array_equal(out["t2m"].values, _broadcast(expected, out["t2m"].shape))


def test_regrid_integers_become_float_with_nan_outside_the_source() -> None:
    ref = utm_like()
    west = ref.isel(x=slice(0, 100)).astype("uint16")
    out = regrid(west, ref)
    assert out["B04"].dtype == np.float32
    np.testing.assert_array_equal(out["B04"].values[..., :100], ref["B04"].values[..., :100])
    assert np.isnan(out["B04"].values[..., 100:]).all()


def test_regrid_keeps_extra_dimensions() -> None:
    fine = utm_like()
    embedding = fine["B04"].expand_dims(band=[0, 1, 2], axis=1).to_dataset(name="embedding")
    coarse = fine.coarsen(x=10, y=10).mean()
    out = regrid(embedding, coarse, method="average")
    assert out["embedding"].dims == ("t", "band", "y", "x")
    expected = embedding.coarsen(x=10, y=10).mean()["embedding"].values
    np.testing.assert_allclose(out["embedding"].values, expected, rtol=1e-5)


def test_regrid_is_lazy_for_dask_sources() -> None:
    fine = utm_like().chunk({"t": 1, "y": 50, "x": 50})
    coarse = utm_like().coarsen(x=10, y=10).mean()
    out = regrid(fine, coarse, method="average")
    assert isinstance(out["B04"].data, dsa.Array)
    np.testing.assert_allclose(out["B04"].values, coarse["B04"].values, rtol=1e-5)


def test_regrid_accepts_results_dataarrays_and_lists(tmp_path: Path) -> None:
    fine = utm_like()
    coarse = fine.coarsen(x=10, y=10).mean()
    ndvi = ((fine["B08"] - fine["B04"]) / (fine["B08"] + fine["B04"])).rename("ndvi")

    out = regrid(ndvi, coarse["B04"], method="average")  # DataArray in, DataArray out
    assert isinstance(out, xr.DataArray) and out.name == "ndvi"
    assert regrid(ndvi.rename(None), coarse).name is None

    result = Result(path=tmp_path / "s2.nc", dataset=fine, from_cache=False)
    outs = regrid([result, ndvi], coarse, method="average")
    assert isinstance(outs, list) and len(outs) == 2
    assert isinstance(outs[0], xr.Dataset) and isinstance(outs[1], xr.DataArray)
    np.testing.assert_allclose(outs[0]["B04"].values, coarse["B04"].values, rtol=1e-5)


def test_regrid_errors() -> None:
    ds = utm_like()
    with pytest.raises(ValueError, match="unknown method"):
        regrid(ds, ds, method="linear")
    with pytest.raises(ValueError, match="single cell"):
        regrid(ds, ds.isel(x=[0]))
    with pytest.raises(ValueError, match="not evenly spaced"):
        regrid(ds, ds.isel(x=[0, 1, 3]))
    with pytest.raises(TypeError, match="Result, Dataset or DataArray"):
        regrid(ds, "s2.nc")

"""Local driver end to end, fully offline: Zarr, NetCDF and GeoTIFF sources."""

import warnings
from pathlib import Path

import pytest
import xarray as xr
from conftest import UTM_CENTER, era5_like, static_like, utm_like

import cubesip
from cubesip import geohash as gh
from cubesip.config import Config
from cubesip.drivers.local import LocalDriver, open_geotiff


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(manifest=tmp_path / "manifest.csv")


@pytest.fixture
def zarr_store(tmp_path: Path) -> Path:
    path = tmp_path / "era5.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zarr v3 consolidated-metadata notice
        era5_like().to_zarr(path)
    return path


def _request(dataset: Path | str, tmp_path: Path, **extra) -> dict:
    return {
        "driver": "local",
        "dataset": str(dataset),
        "variables": ["t2m"],
        "geohash": gh.encode(45.3, -3.7, 4),
        "time": {"start": "2024-01-02", "end": "2024-01-02"},
        "target_dir": str(tmp_path / "out"),
        **extra,
    }


def test_zarr_end_to_end(config: Config, zarr_store: Path, tmp_path: Path) -> None:
    request = _request(zarr_store, tmp_path)
    report = cubesip.check(request, config)
    assert report.ok, report.summary()
    assert report.dataset.available_variables == ["t2m", "tp"]
    assert report.dataset.crs == "EPSG:4326"

    result = cubesip.run(request, config)
    assert not result.from_cache
    assert result.path.name.startswith("local_era5_")
    assert result.dataset.sizes["valid_time"] == 24
    assert result.manifest_rows[0]["dataset"] == zarr_store.resolve().as_posix()

    again = cubesip.run(request, config)
    assert again.from_cache and again.path == result.path


def test_relative_path_matches_cache(
    config: Config, zarr_store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cubesip.run(_request("era5.zarr", tmp_path), config)
    assert cubesip.check(_request(zarr_store, tmp_path), config).cache_hit is not None


def test_static_netcdf(config: Config, tmp_path: Path) -> None:
    path = tmp_path / "dem.nc"
    static_like().to_netcdf(path)
    request = _request(path, tmp_path, variables=["elevation"], time=None)
    result = cubesip.run(request, config)
    assert set(result.dataset.dims) == {"lat", "lon"}
    assert result.manifest_rows[0]["time_start"] == ""
    assert cubesip.run(request, config).from_cache

    dated = {**request, "time": {"start": "2024-01-01", "end": "2024-01-02"}}
    report = cubesip.check(dated, config)
    assert not report.ok and "no time dimension" in report.error


def _write_geotiff(path: Path, names: tuple[str, ...] | None) -> None:
    ds = utm_like().isel(t=0)
    da = xr.concat([ds["B04"], ds["B08"]], dim="band").assign_coords(band=[1, 2])
    da = da.drop_vars("t").rio.write_crs("EPSG:32630")
    if names:
        da.attrs["long_name"] = names
    da.rio.to_raster(path)


@pytest.mark.parametrize("names", [("red", "nir"), None])
def test_geotiff_bands_become_variables(tmp_path: Path, names) -> None:
    path = tmp_path / "s2.tif"
    _write_geotiff(path, names)
    ds = open_geotiff(path)
    expected = list(names) if names else ["band_1", "band_2"]
    assert list(ds.data_vars) == expected


def test_geotiff_end_to_end(config: Config, tmp_path: Path) -> None:
    path = tmp_path / "s2.tif"
    _write_geotiff(path, ("red", "nir"))
    request = _request(
        path, tmp_path, variables=["red", "nir"], time=None, geohash=gh.encode(*UTM_CENTER, 6)
    )
    result = cubesip.run(request, config)
    assert result.manifest_rows[0]["crs"] == "EPSG:32630"
    assert set(result.dataset.data_vars) >= {"red", "nir"}
    assert result.dataset.sizes["x"] <= 15  # ~1.2 km cell at 100 m


def test_missing_dataset(config: Config, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path / "nope.zarr", tmp_path), config)
    assert not report.ok and "not found" in report.error


def test_dataset_key_is_absolute(tmp_path: Path) -> None:
    driver = LocalDriver(Config(manifest=tmp_path / "m.csv"))
    assert Path(driver.dataset_key("x/y.zarr")).is_absolute()

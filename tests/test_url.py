"""URL driver end to end, offline: files are served by a local HTTP server that answers range
requests, as the real data servers do.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import warnings
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from conftest import era5_like

import cubesip
from cubesip import geohash as gh
from cubesip.config import Config
from cubesip.drivers.base import DatasetNotFoundError
from cubesip.drivers.url import UrlDriver, gdal_path, sniff_format, split_member
from cubesip.output import dataset_slug
from cubesip.request import Request

SITE = (52.37, 4.90)  # lat, lon: Amsterdam


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """A folder and the base URL it is served at. The server runs in its own process (see
    range_server.py), on a new port per test so GDAL's cache of remote files never mixes tests.
    """
    root = tmp_path / "www"
    root.mkdir()
    server = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("range_server.py")), str(root)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        port = int(server.stdout.readline())
        yield root, f"http://127.0.0.1:{port}"
    finally:
        server.kill()
        server.wait()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(manifest=tmp_path / "manifest.csv")


def hazard_like() -> xr.DataArray:
    """A static 0.1 degree lat/lon layer over western Europe, like a flood depth map, with a
    nodata value.
    """
    lat = np.arange(60 - 0.05, 40, -0.1)
    lon = np.arange(-10 + 0.05, 20, 0.1)
    data = np.random.default_rng(3).uniform(0, 5, (lat.size, lon.size)).astype("float32")
    data[:10, :10] = -9999
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": lat, "x": lon})
    return da.rio.write_crs("EPSG:4326").rio.write_nodata(-9999)


def write_tif(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    hazard_like().rio.to_raster(path, driver="GTiff", tiled=True, blockxsize=64, blockysize=64)
    return path


def write_zip(path: Path, members: dict[str, Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, source in members.items():
            zf.write(source, name)
    return path


def _request(dataset: str, tmp_path: Path, **extra) -> dict:
    return {
        "driver": "url",
        "dataset": dataset,
        "variables": ["band_1"],
        "geohash": gh.encode(*SITE, 4),
        "time": None,
        "target_dir": str(tmp_path / "out"),
        **extra,
    }


# -- request handling, no server -------------------------------------------------------------


def test_dataset_key_normalises(config: Config) -> None:
    driver = UrlDriver(config)
    assert driver.dataset_key(" https://h/a/store.zarr/ ") == "https://h/a/store.zarr"
    assert (
        driver.dataset_key("https://h/a.zip!/maps/rp100.tif/") == "https://h/a.zip!maps/rp100.tif"
    )
    assert split_member("https://h/a.zip!maps/rp100.tif") == ("https://h/a.zip", "maps/rp100.tif")
    assert split_member("https://h/a.tif") == ("https://h/a.tif", None)


def test_gdal_paths() -> None:
    assert gdal_path("https://h/a.tif") == "/vsicurl/https://h/a.tif"
    assert gdal_path("https://h/file/123", "x.tif") == "/vsizip/{/vsicurl/https://h/file/123}/x.tif"


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        ("./maps/rp100.tif", "takes an http"),
        ("s3://bucket/rp100.tif", "takes an http"),
        ("https://h/a.zip!maps/readme.txt", "only GeoTIFF"),
    ],
)
def test_validate_refuses(config: Config, tmp_path: Path, dataset: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        UrlDriver(config).validate(Request.model_validate(_request(dataset, tmp_path)))


def test_validate_refuses_options(config: Config, tmp_path: Path) -> None:
    request = Request.model_validate(_request("https://h/a.tif", tmp_path, options={"x": 1}))
    with pytest.raises(ValueError, match="options not supported"):
        UrlDriver(config).validate(request)


def test_slug_uses_archive_member() -> None:
    assert dataset_slug("https://h/floodMapGL_rp100y.zip!floodMapGL_rp100y.tif") == (
        "floodMapGL-rp100y"
    )


# -- over HTTP -------------------------------------------------------------------------------


def test_geotiff_end_to_end(config: Config, served, tmp_path: Path) -> None:
    root, base = served
    write_tif(root / "maps" / "rp100.tif")
    request = _request(f"{base}/maps/rp100.tif", tmp_path)

    report = cubesip.check(request, config)
    assert report.ok, report.summary()
    assert report.dataset.available_variables == ["band_1"]
    assert report.dataset.crs == "EPSG:4326"

    result = cubesip.run(request, config)
    assert not result.from_cache
    assert result.path.name.startswith("url_rp100_")
    assert set(result.dataset.dims) == {"y", "x"}
    assert 0 < result.dataset.sizes["x"] * result.dataset.sizes["y"] <= 100  # ~40 km at 0.1 deg
    assert float(result.dataset["band_1"].max()) <= 5
    assert result.manifest_rows[0]["dataset"] == f"{base}/maps/rp100.tif"

    child = {**request, "geohash": gh.encode(*SITE, 6)}
    again = cubesip.run(child, config)
    assert again.from_cache and again.path == result.path
    assert again.dataset.sizes["x"] >= 1 and again.dataset.sizes["y"] >= 1  # pixel > cell


@pytest.mark.parametrize("archive", ["maps/floodmaps.zip", "file/4f1c/9a2e"])
def test_geotiff_inside_zip(config: Config, served, tmp_path: Path, archive: str) -> None:
    """Both an archive URL ending in .zip and one without an extension (data repositories)."""
    root, base = served
    tif = write_tif(tmp_path / "src" / "rp100.tif")
    write_zip(root / archive, {"readme.txt": tif, "maps/rp100.tif": tif})

    request = _request(f"{base}/{archive}!maps/rp100.tif", tmp_path)
    result = cubesip.run(request, config)
    assert result.path.name.startswith("url_rp100_")
    expected = hazard_like().rio.clip_box(*result.dataset.rio.bounds()).values
    np.testing.assert_allclose(result.dataset["band_1"].values, expected)


def test_extensionless_geotiff_is_sniffed(config: Config, served, tmp_path: Path) -> None:
    root, base = served
    write_tif(root / "file" / "abc")
    assert sniff_format(f"{base}/file/abc") == "geotiff"
    assert cubesip.check(_request(f"{base}/file/abc", tmp_path), config).ok


def test_zip_without_member_is_explained(config: Config, served, tmp_path: Path) -> None:
    root, base = served
    tif = write_tif(tmp_path / "src" / "rp100.tif")
    write_zip(root / "file" / "xyz", {"rp100.tif": tif})
    report = cubesip.check(_request(f"{base}/file/xyz", tmp_path), config)
    assert not report.ok and "zip archive" in report.error


def test_unknown_format(config: Config, served, tmp_path: Path) -> None:
    root, base = served
    (root / "notes").write_text("hello world")
    with pytest.raises(ValueError, match="cannot tell the format"):
        sniff_format(f"{base}/notes")


def test_missing_files(config: Config, served, tmp_path: Path) -> None:
    _, base = served
    with pytest.raises(DatasetNotFoundError):
        sniff_format(f"{base}/file/none")
    for dataset in (f"{base}/missing.tif", f"{base}/missing.zip!a.tif"):
        report = cubesip.check(_request(dataset, tmp_path), config)
        assert not report.ok and "cannot open GeoTIFF" in report.error


def test_netcdf_over_http(config: Config, served, tmp_path: Path) -> None:
    root, base = served
    era5_like(hours=24).to_netcdf(root / "era5.nc")
    request = _request(
        f"{base}/era5.nc",
        tmp_path,
        variables=["t2m"],
        time={"start": "2024-01-01", "end": "2024-01-01"},
    )
    result = cubesip.run(request, config)
    assert result.dataset.sizes["valid_time"] == 24
    assert result.manifest_rows[0]["crs"] == "EPSG:4326"


def test_zarr_over_http(config: Config, served, tmp_path: Path) -> None:
    root, base = served
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zarr v3 consolidated-metadata notice
        era5_like(hours=24).to_zarr(tmp_path / "era5.zarr")
    shutil.copytree(tmp_path / "era5.zarr", root / "era5.zarr")
    request = _request(
        f"{base}/era5.zarr/",
        tmp_path,
        variables=["tp"],
        time={"start": "2024-01-01", "end": "2024-01-01"},
    )
    result = cubesip.run(request, config)
    assert result.dataset.sizes["valid_time"] == 24
    assert result.manifest_rows[0]["dataset"] == f"{base}/era5.zarr"

"""Live tests against the real services. Run with: uv run pytest -m live

They read cubesip.yaml (or $CUBESIP_CONFIG) and are skipped when credentials are missing.
The openeo, cdse_federation and stac tests log in (device-code flow the first time; the
federation has its own login client, so it asks once more) and download a small cube.
Run with -s to see the login link.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import cubesip
from cubesip.config import load_config

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def config():
    try:
        return load_config()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _isolated(config, tmp_path: Path):
    """Same credentials, but a throw-away manifest."""
    return config.model_copy(update={"manifest": tmp_path / "manifest.csv"})


def test_edh_era5(config, tmp_path: Path) -> None:
    if config.earthdatahub.token is None:
        pytest.skip("earthdatahub.token not set")
    cfg = _isolated(config, tmp_path)
    request = {
        "driver": "edh",
        "dataset": "era5/era5-single-levels-atmosphere-v0",
        "variables": ["t2m"],
        "geohash": "u0nd9",
        "time": {"start": "2024-01-01", "end": "2024-01-02"},
        "target_dir": str(tmp_path),
    }
    report = cubesip.check(request, cfg)
    assert report.ok, report.summary()
    result = cubesip.run(request, cfg)
    assert result.dataset.sizes["valid_time"] == 48
    assert float(result.dataset["t2m"].mean()) > 200  # Kelvin


def test_openeo_sentinel2(config, tmp_path: Path) -> None:
    cfg = _isolated(config, tmp_path)
    request = {
        "driver": "openeo",
        "dataset": "SENTINEL2_L2A",
        "variables": ["B04", "B08"],
        "geohash": "u0nd9w",
        "time": {"start": "2024-06-01", "end": "2024-06-15"},
        "target_dir": str(tmp_path),
        "options": {"max_cloud_cover": 80},
    }
    report = cubesip.check(request, cfg)
    assert report.ok, report.summary()
    result = cubesip.run(request, cfg)
    assert {"B04", "B08"} <= set(result.dataset.data_vars)
    assert result.manifest_rows[0]["crs"].startswith("EPSG:326")  # native UTM kept
    child = {**request, "geohash": "u0nd9w1", "variables": ["B08"]}
    assert cubesip.run(child, cfg).from_cache


def test_stac_planetary_computer(config, tmp_path: Path) -> None:
    """Planetary Computer assets are signed; the CDSE back-end must still read them."""
    cfg = _isolated(config, tmp_path)
    request = {
        "driver": "stac",
        "dataset": "https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2",
        "variables": ["red", "nir08"],
        "geohash": "u0nd9w",
        "time": {"start": "2024-06-01", "end": "2024-07-31"},
        "target_dir": str(tmp_path),
        "options": {"max_cloud_cover": 50},
    }
    report = cubesip.check(request, cfg)
    assert report.ok, report.summary()
    result = cubesip.run(request, cfg)
    red = result.dataset["red"].load()
    assert int(red.notnull().sum()) > 0  # real pixels, not fill values
    assert result.manifest_rows[0]["crs"].startswith("EPSG:326")


FEDERATION_REQUEST = {
    "driver": "cdse_federation",
    "dataset": "TERRASCOPE_S2_NDVI_V2",  # Terrascope collection, only in the federation
    "variables": ["NDVI"],
    "geohash": "u0nd9w",
    # July: 5 clear scenes. Early June has one fully cloudy scene, and the federation then
    # answers with an empty body (EmptyResultError).
    "time": {"start": "2024-07-01", "end": "2024-07-31"},
    "options": {"max_cloud_cover": 80},
}


def test_cdse_federation_check(config, tmp_path: Path) -> None:
    """Metadata and size estimate are public: no login needed."""
    request = {**FEDERATION_REQUEST, "target_dir": str(tmp_path)}
    report = cubesip.check(request, _isolated(config, tmp_path))
    assert report.ok, report.summary()
    assert report.dataset.available_variables and "NDVI" in report.dataset.available_variables
    assert "STAC search" in report.size.detail


def test_cdse_federation_download(config, tmp_path: Path) -> None:
    request = {**FEDERATION_REQUEST, "target_dir": str(tmp_path)}
    result = cubesip.run(request, _isolated(config, tmp_path))
    assert "NDVI" in result.dataset.data_vars
    assert int(result.dataset["NDVI"].notnull().sum()) > 0

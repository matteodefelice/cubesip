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


def test_tessera_embeddings(config, tmp_path: Path) -> None:
    """Public TESSERA v1.1 store: no credentials needed."""
    pytest.importorskip("geotessera")
    cfg = _isolated(config, tmp_path)
    request = {
        "driver": "tessera",
        "dataset": "v1.1",
        "variables": ["embedding"],
        "geohash": "u094v8",
        "time": {"start": "2024-01-01", "end": "2024-12-31"},
        "target_dir": str(tmp_path),
    }
    report = cubesip.check(request, cfg)
    assert report.ok, report.summary()
    result = cubesip.run(request, cfg)
    emb = result.dataset["embedding"]
    assert emb.sizes["band"] == 128 and emb.sizes["time"] == 1
    assert int(emb.isel(band=0).notnull().sum()) > 0
    assert result.manifest_rows[0]["crs"] == "EPSG:32631"
    child = {**request, "geohash": "u094v8k"}
    assert cubesip.run(child, cfg).from_cache


def test_beeldmateriaal_orthophoto(config, tmp_path: Path) -> None:
    """Public Beeldmateriaal tiles: no credentials needed (downloads two ~40 MB tile indexes)."""
    cfg = _isolated(config, tmp_path).model_copy(update={"data_dir": tmp_path})
    request = {
        "driver": "beeldmateriaal",
        "dataset": "ortho_hr",
        "variables": ["red", "nir"],
        "geohash": "u178kdc",
        "time": {"start": "2025-01-01", "end": "2025-12-31"},
        "target_dir": str(tmp_path),
    }
    report = cubesip.check(request, cfg)
    assert report.ok, report.summary()
    result = cubesip.run(request, cfg)
    ds = result.dataset
    assert ds.sizes["x"] > 1000 and float(ds["x"][1] - ds["x"][0]) == pytest.approx(0.08)
    assert int((ds["nir"] > 0).sum()) > 0
    assert result.manifest_rows[0]["crs"] == "EPSG:28992"
    assert cubesip.run({**request, "geohash": "u178kdck"}, cfg).from_cache


def test_beeldmateriaal_older_years(config, tmp_path: Path) -> None:
    """8 cm RGB before 2025 (RGB-only index) and 25 cm RGB + near-infrared, both 2024."""
    cfg = _isolated(config, tmp_path).model_copy(update={"data_dir": tmp_path})
    base = {
        "driver": "beeldmateriaal",
        "geohash": "u177rxt",  # inside the RD tile 122000_518000
        "time": {"start": "2024-01-01", "end": "2024-12-31"},
        "target_dir": str(tmp_path),
    }
    hr = cubesip.run({**base, "dataset": "ortho_hr", "variables": ["red", "blue"]}, cfg)
    assert float(hr.dataset["x"][1] - hr.dataset["x"][0]) == pytest.approx(0.08)
    assert int((hr.dataset["red"] > 0).sum()) > 0
    lr = cubesip.run({**base, "dataset": "ortho_lr", "variables": ["nir", "red"]}, cfg)
    assert float(lr.dataset["x"][1] - lr.dataset["x"][0]) == pytest.approx(0.25)
    assert float(lr.dataset["nir"].mean()) > float(lr.dataset["red"].mean())  # vegetation
    report = cubesip.check({**base, "dataset": "ortho_hr", "variables": ["nir"]}, cfg)
    assert not report.ok and "ortho_lr" in report.error


JRC_FLOOD_RP100 = (
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/"
    "floodMapGL_rp100y.zip!floodMapGL_rp100y.tif"
)
# STORM present-climate wind speeds per return period, one zip on 4TU.ResearchData whose URL
# has no extension
STORM_WIND_PRESENT = (
    "https://data.4tu.nl/file/0ea98bdd-5772-4da8-ae97-99735e891aff/"
    "0b98c6f5-c7af-45bb-bb6e-dac53e1b8d55"
)


def test_url_jrc_flood_depth(config, tmp_path: Path) -> None:
    """JRC global river flood depth (100-year return period), read inside the remote zip."""
    cfg = _isolated(config, tmp_path)
    request = {
        "driver": "url",
        "dataset": JRC_FLOOD_RP100,
        "variables": ["band_1"],
        "geohash": "u173",  # Amsterdam and its surroundings
        "time": None,
        "target_dir": str(tmp_path),
    }
    report = cubesip.check(request, cfg)
    assert report.ok, report.summary()
    assert report.dataset.crs == "EPSG:4326"
    result = cubesip.run(request, cfg)
    assert result.path.name.startswith("url_floodMapGL-rp100y_")
    assert float(result.dataset["band_1"].max()) > 0  # some flood depth in metres
    assert cubesip.run({**request, "geohash": "u173z"}, cfg).from_cache


def test_url_storm_wind_speed(config, tmp_path: Path) -> None:
    """STORM 100-year cyclone wind speed, western Pacific, from an extensionless zip URL."""
    cfg = _isolated(config, tmp_path)
    request = {
        "driver": "url",
        "dataset": f"{STORM_WIND_PRESENT}!STORM_FIXED_RETURN_PERIODS_WP_100_YR_RP.tif",
        "variables": ["band_1"],
        "geohash": "wdw5",  # Manila
        "time": None,
        "target_dir": str(tmp_path),
    }
    result = cubesip.run(request, cfg)
    assert 20 < float(result.dataset["band_1"].max()) < 100  # m/s

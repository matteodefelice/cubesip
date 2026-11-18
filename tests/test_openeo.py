"""openEO driver, offline: metadata parsing, size estimate, process building, auth dispatch.

Metadata dicts are trimmed copies of the real CDSE / Planetary Computer responses.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openeo.rest import OpenEoApiError
from pydantic import SecretStr

import cubesip
from cubesip.config import Config, OpenEOAuth, OpenEOConfig
from cubesip.drivers import DatasetNotFoundError
from cubesip.drivers import openeo as oe
from cubesip.request import Request

S2_META = {
    "id": "SENTINEL2_L2A",
    "cube:dimensions": {
        "bands": {"type": "bands", "values": ["B02", "B04", "B08", "B11", "SCL"]},
        "t": {"type": "temporal"},
        "x": {"type": "spatial", "axis": "x"},
        "y": {"type": "spatial", "axis": "y"},
    },
    "summaries": {
        "eo:bands": [
            {"name": "B02", "gsd": 10},
            {"name": "B04", "gsd": 10},
            {"name": "B08", "gsd": 10},
            {"name": "B11", "gsd": 20},
            {"name": "SCL", "gsd": 20},
        ],
        "gsd": [10, 20, 60],
    },
    "links": [
        {"rel": "derived_from", "href": "https://stac.example.eu/v1/collections/sentinel-2-l2a"},
        {"rel": "root", "href": "https://openeo.example.eu/openeo/1.2/collections"},
    ],
}

PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2"
PC_META = {
    "type": "Collection",
    "id": "landsat-c2-l2",
    "item_assets": {
        "red": {"raster:bands": [{"spatial_resolution": 30, "data_type": "uint16"}]},
        "nir08": {"raster:bands": [{"spatial_resolution": 30}]},
        "qa_pixel": {"raster:bands": [{"spatial_resolution": 30}]},
    },
    "summaries": {"gsd": [30, 100]},
    "links": [{"rel": "root", "href": "https://planetarycomputer.microsoft.com/api/stac/v1/"}],
}


def _config(tmp_path: Path, **auth) -> Config:
    return Config(
        manifest=tmp_path / "manifest.csv",
        openeo=OpenEOConfig(url="https://openeo.example.eu", auth=OpenEOAuth(**auth)),
    )


def _request(tmp_path: Path, **extra) -> Request:
    return Request.from_json(
        {
            "driver": "openeo",
            "dataset": "SENTINEL2_L2A",
            "variables": ["B04", "B08"],
            "geohash": "qem7k2",
            "time": {"start": "2024-01-01", "end": "2024-01-31"},
            "target_dir": str(tmp_path / "out"),
            "options": {"max_cloud_cover": 60},
            **extra,
        }
    )


@pytest.fixture
def driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> oe.OpenEODriver:
    drv = oe.OpenEODriver(_config(tmp_path))
    drv._metadata = {"SENTINEL2_L2A": S2_META, PC_URL: PC_META}
    return drv


def test_band_and_resolution_parsing() -> None:
    assert oe.available_bands(S2_META) == ["B02", "B04", "B08", "B11", "SCL"]
    assert oe.resolution_m(S2_META, ["B04", "B11"]) == 10
    assert oe.resolution_m(S2_META, ["B11"]) == 20
    assert oe.available_bands(PC_META) == ["red", "nir08", "qa_pixel"]
    assert oe.resolution_m(PC_META, ["red"]) == 30
    assert oe.resolution_m({"summaries": {"gsd": 0.0001}}, ["x"]) == pytest.approx(11.132)
    assert oe.resolution_m({}, ["x"]) is None


def test_stac_search_target() -> None:
    assert oe.stac_search_target(S2_META, "SENTINEL2_L2A") == (
        "https://stac.example.eu/v1",
        "sentinel-2-l2a",
    )
    assert oe.stac_search_target(PC_META, PC_URL) == (
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        "landsat-c2-l2",
    )
    assert oe.stac_search_target({"links": []}, "OTHER") is None


def test_describe_reports_missing_bands(driver, tmp_path: Path) -> None:
    info = driver.describe(_request(tmp_path, variables=["B04", "B99"]))
    assert info.missing_variables == ["B99"]


def test_describe_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drv = oe.OpenEODriver(_config(tmp_path))
    conn = MagicMock()
    conn.describe_collection.side_effect = OpenEoApiError(
        http_status_code=404, code="CollectionNotFound", message="nope"
    )
    monkeypatch.setattr(drv, "connection", lambda authenticate=False: conn)
    with pytest.raises(DatasetNotFoundError, match="NOPE"):
        drv.describe(_request(tmp_path, dataset="NOPE"))


def test_size_estimate(driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(driver, "_count_time_steps", lambda request, meta: 7)
    request = _request(tmp_path)
    width, height = oe.box_size_m(request)
    size = driver.estimate_size(request)
    expected = -(-width // 10) * -(-height // 10) * 2 * 7 * 4
    assert size.nbytes == expected and size.method == "estimate"
    assert "7 time step(s) (STAC search)" in size.detail

    monkeypatch.setattr(driver, "_count_time_steps", lambda request, meta: None)
    size = driver.estimate_size(request)
    assert "31 time step(s)" in size.detail and "upper bound" in size.detail


def test_checker_warns_above_sync_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    config.openeo.max_sync_bytes = 1000
    drv = oe.OpenEODriver(config)
    drv._metadata = {"SENTINEL2_L2A": S2_META}
    monkeypatch.setattr(drv, "_count_time_steps", lambda request, meta: 3)
    monkeypatch.setattr(cubesip.checker, "get_driver", lambda name, cfg: drv)
    report = cubesip.check(_request(tmp_path), config)
    assert report.ok
    assert any("exceeds the openeo limit" in w for w in report.warnings)


def test_load_collection_arguments(driver, tmp_path: Path, monkeypatch) -> None:
    conn = MagicMock()
    monkeypatch.setattr(driver, "connection", lambda authenticate=False: conn)
    request = _request(tmp_path, options={"max_cloud_cover": 40})
    driver.fetch(request, tmp_path / "x.nc")
    _, kwargs = conn.load_collection.call_args
    assert conn.load_collection.call_args.args == ("SENTINEL2_L2A",)
    box = request.bbox
    assert kwargs["spatial_extent"] == {
        "west": box.west,
        "south": box.south,
        "east": box.east,
        "north": box.north,
        "crs": 4326,
    }
    assert kwargs["temporal_extent"] == ["2024-01-01", "2024-02-01"]  # end exclusive
    assert kwargs["bands"] == ["B04", "B08"]
    assert kwargs["max_cloud_cover"] == 40
    assert kwargs["properties"] is None
    conn.load_collection.return_value.download.assert_called_once_with(
        tmp_path / "x.nc", format="netCDF"
    )


def test_load_stac_turns_cloud_cover_into_property(driver, tmp_path: Path, monkeypatch) -> None:
    conn = MagicMock()
    monkeypatch.setattr(driver, "connection", lambda authenticate=False: conn)
    request = _request(
        tmp_path,
        dataset=PC_URL,
        variables=["red", "nir08"],
        options={"max_cloud_cover": 50, "properties": {"platform": {"eq": "landsat-9"}}},
    )
    driver.load(request)
    args, kwargs = conn.load_stac.call_args
    assert args == (PC_URL,)
    props = kwargs["properties"]
    assert props["eo:cloud_cover"](50) is True and props["eo:cloud_cover"](51) is False
    assert props["platform"]("landsat-9") is True


@pytest.mark.parametrize(
    "bad",
    [{"properties": {"eo:cloud_cover": {"lte": 1, "gte": 0}}}, {"properties": {"x": {"like": 1}}}],
)
def test_malformed_property_filters(driver, tmp_path: Path, bad: dict) -> None:
    with pytest.raises(ValueError):
        driver.validate_options(_request(tmp_path, options=bad))


@pytest.mark.parametrize(
    ("auth", "method", "expected"),
    [
        ({"method": "oidc", "provider_id": "CDSE"}, "authenticate_oidc", {"provider_id": "CDSE"}),
        (
            {"method": "client_credentials", "client_id": "id", "client_secret": SecretStr("s")},
            "authenticate_oidc_client_credentials",
            {"client_id": "id", "client_secret": "s", "provider_id": None},
        ),
    ],
)
def test_authentication_dispatch(tmp_path, monkeypatch, auth, method, expected) -> None:
    conn = MagicMock()
    import openeo

    monkeypatch.setattr(openeo, "connect", lambda url: conn)
    drv = oe.OpenEODriver(_config(tmp_path, **auth))
    assert drv.connection(authenticate=True) is conn
    drv.connection(authenticate=True)
    getattr(conn, method).assert_called_once_with(**expected)


def test_basic_auth_requires_credentials(tmp_path, monkeypatch) -> None:
    import openeo

    monkeypatch.setattr(openeo, "connect", lambda url: MagicMock())
    with pytest.raises(PermissionError):
        oe.OpenEODriver(_config(tmp_path, method="basic")).connection(authenticate=True)

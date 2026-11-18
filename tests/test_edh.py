"""Earth Data Hub driver against a local HTTP server that mimics EDH basic auth.

This exercises the real path (xarray -> zarr -> fsspec HTTP -> aiohttp basic auth) with a
Zarr store served from a temporary directory; no network or token needed.
"""

from __future__ import annotations

import base64
import threading
import warnings
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import era5_like
from pydantic import SecretStr

import cubesip
from cubesip import geohash as gh
from cubesip.config import Config, EarthDataHubConfig
from cubesip.drivers.edh import EarthDataHubDriver

TOKEN = "s3cr3t-token"
DATASET = "era5/era5-single-levels-atmosphere-v0"


class _AuthHandler(SimpleHTTPRequestHandler):
    expected = "Basic " + base64.b64encode(f"edh:{TOKEN}".encode()).decode()

    def _authorised(self) -> bool:
        if self.headers.get("Authorization") == self.expected:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="edh"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_GET(self) -> None:  # noqa: N802
        if self._authorised():
            super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if self._authorised():
            super().do_HEAD()

    def log_message(self, *args) -> None:
        pass


@pytest.fixture(scope="module", params=[2, 3], ids=["zarr-v2", "zarr-v3"])
def edh_server(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"edh-v{request.param}")
    store = root / f"{DATASET}.zarr"
    store.parent.mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        era5_like(hours=96).chunk({"valid_time": 24}).to_zarr(
            store, zarr_format=request.param, consolidated=True
        )
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_AuthHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _config(tmp_path: Path, url: str, token: str | None = TOKEN) -> Config:
    return Config(
        manifest=tmp_path / "manifest.csv",
        earthdatahub=EarthDataHubConfig(url=url, token=SecretStr(token) if token else None),
    )


def _request(tmp_path: Path, **extra) -> dict:
    return {
        "driver": "edh",
        "dataset": DATASET,
        "variables": ["t2m"],
        "geohash": gh.encode(45.3, -3.7, 4),
        "time": {"start": "2024-01-02", "end": "2024-01-03"},
        "target_dir": str(tmp_path / "out"),
        **extra,
    }


def test_dataset_key_normalisation(tmp_path: Path) -> None:
    driver = EarthDataHubDriver(_config(tmp_path, "https://data.earthdatahub.destine.eu/"))
    expected = DATASET
    for variant in (
        DATASET,
        f"{DATASET}.zarr",
        f"/{DATASET}.zarr/",
        f"https://data.earthdatahub.destine.eu/{DATASET}.zarr",
    ):
        assert driver.dataset_key(variant) == expected
    assert driver.store_url(expected) == f"https://data.earthdatahub.destine.eu/{DATASET}.zarr"
    other = "https://example.org/x/store.zarr"
    assert driver.store_url(driver.dataset_key(other)) == other


def test_check_and_run_through_http(edh_server: str, tmp_path: Path) -> None:
    config = _config(tmp_path, edh_server)
    request = _request(tmp_path)
    report = cubesip.check(request, config)
    assert report.ok, report.summary()
    assert report.dataset.available_variables == ["t2m", "tp"]
    assert report.size.method == "exact"

    result = cubesip.run(request, config)
    assert result.dataset.sizes["valid_time"] == 48
    assert result.manifest_rows[0]["dataset"] == DATASET
    assert TOKEN not in result.path.read_bytes().decode("latin-1")
    assert cubesip.run(request, config).from_cache


def test_wrong_token_is_reported_without_leaking(edh_server: str, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path), _config(tmp_path, edh_server, token="wrong"))
    assert not report.ok and "HTTP 401" in report.error
    assert "wrong" not in report.error


def test_missing_token(edh_server: str, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path), _config(tmp_path, edh_server, token=None))
    assert not report.ok and "token missing" in report.error


def test_unknown_dataset(edh_server: str, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path, dataset="era5/nope"), _config(tmp_path, edh_server))
    assert not report.ok and "not found" in report.error

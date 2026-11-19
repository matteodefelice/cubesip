"""TESSERA driver, offline: a fake store stands in for geotessera's GeoTesseraZarr."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

import cubesip
from cubesip import api, checker
from cubesip.config import Config
from cubesip.drivers import EmptyResultError, get_driver
from cubesip.drivers import tessera as ts
from cubesip.spatial import native_crs

pytest.importorskip("geotessera")

STORE_URL = "https://example.invalid/tessera/v1.1"
GEOHASH = "u094v8"  # Beauce, France: UTM zone 31


class FakeStore:
    """Same surface as GeoTesseraZarr: years, n_bands, depths, read_region."""

    url = STORE_URL
    years = [2023, 2024]
    n_bands = 6
    depths = {2: "d2", 6: "embeddings"}
    model_version = "test-model"

    def __init__(self, fill: float | None = None) -> None:
        self.fill = fill
        self.calls: list[tuple[int, int | None]] = []

    def read_region(self, bbox, year, *, depth=None):
        self.calls.append((year, depth))
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
        xs, ys = to_utm.transform([bbox[0], bbox[2]], [bbox[1], bbox[3]])
        x0, x1 = math.floor(min(xs) / 10) * 10, math.ceil(max(xs) / 10) * 10
        y0, y1 = math.floor(min(ys) / 10) * 10, math.ceil(max(ys) / 10) * 10
        height, width, bands = (y1 - y0) // 10, (x1 - x0) // 10, depth or self.n_bands
        # value = channel + year fraction, so tests can tell channels and years apart
        values = np.arange(bands, dtype="float32") + (year - 2000) / 100
        mosaic = np.broadcast_to(values, (height, width, bands)).copy()
        if self.fill is not None:
            mosaic[:] = self.fill
        return mosaic, Affine(10, 0, x0, 0, -10, y1), "EPSG:32631"


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: FakeStore) -> Config:
    config = Config(manifest=tmp_path / "manifest.csv")
    driver = ts.TesseraDriver(config)
    driver._stores[STORE_URL] = store
    monkeypatch.setattr(api, "get_driver", lambda name, cfg: driver)
    monkeypatch.setattr(checker, "get_driver", lambda name, cfg: driver)
    return config


def _request(tmp_path: Path, **extra) -> dict:
    return {
        "driver": "tessera",
        "dataset": STORE_URL,
        "variables": ["embedding"],
        "geohash": GEOHASH,
        "time": {"start": "2023-01-01", "end": "2024-12-31"},
        "target_dir": str(tmp_path / "out"),
        **extra,
    }


def test_check_run_and_cache(config: Config, store: FakeStore, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path), config)
    assert report.ok, report.summary()
    assert report.dataset.crs == "EPSG:32631"
    assert report.size.method == "estimate"
    assert "6 channel(s), 2 year(s)" in report.size.detail

    result = cubesip.run(_request(tmp_path), config)
    ds = result.dataset
    assert ds["embedding"].dims == ("time", "band", "y", "x")
    assert ds.sizes["time"] == 2 and ds.sizes["band"] == 6
    assert [str(t)[:10] for t in ds["time"].values] == ["2023-01-01", "2024-01-01"]
    assert ds["band"].values.tolist() == list(range(6))
    assert float(ds["embedding"].sel(band=4).isel(time=1, x=0, y=0)) == pytest.approx(4.24)
    assert native_crs(ds) == "EPSG:32631"
    assert ds.attrs["tessera_model"] == "test-model"
    assert np.allclose(np.diff(ds["x"].values), 10) and np.allclose(np.diff(ds["y"].values), -10)
    assert result.manifest_rows[0]["dataset"] == STORE_URL
    assert result.manifest_rows[0]["crs"] == "EPSG:32631"
    assert [c[0] for c in store.calls] == [2023, 2024]

    # A child cell and a single whole year are cut from the cached file.
    child = cubesip.run(
        _request(
            tmp_path, geohash=GEOHASH + "k", time={"start": "2024-01-01", "end": "2024-12-31"}
        ),
        config,
    )
    assert child.from_cache and child.path == result.path
    assert child.dataset.sizes["time"] == 1 and child.dataset.sizes["band"] == 6
    assert child.dataset.sizes["x"] < ds.sizes["x"]
    assert len(store.calls) == 2  # nothing read again


def test_bands_and_depth(config: Config, store: FakeStore, tmp_path: Path) -> None:
    result = cubesip.run(_request(tmp_path, options={"bands": [5, 1]}), config)
    emb = result.dataset["embedding"]
    assert emb["band"].values.tolist() == [5, 1]
    assert float(emb.isel(time=0, band=0, x=0, y=0)) == pytest.approx(5.23)

    result = cubesip.run(_request(tmp_path, options={"depth": 2}), config)
    assert result.dataset.sizes["band"] == 2
    assert store.calls[-1] == (2024, 2)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"time": {"start": "2016-01-01", "end": "2016-12-31"}}, "not in the TESSERA store"),
        ({"options": {"depth": 3}}, "depth 3 is not published"),
        ({"options": {"bands": [6]}}, "outside 0..5"),
        ({"options": {"depth": 2, "bands": [2]}}, "outside 0..1"),
        ({"variables": ["ndvi"]}, "ndvi"),
    ],
)
def test_errors_reported_by_check(config: Config, tmp_path: Path, change, message) -> None:
    report = cubesip.check(_request(tmp_path, **change), config)
    assert not report.ok and message in report.error


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"time": None}, "whole calendar years"),
        ({"time": {"start": "2024-06-01", "end": "2024-12-31"}}, "whole calendar years"),
        ({"time": {"start": "2024-01-01", "end": "2024-06-30"}}, "whole calendar years"),
        ({"options": {"bands": []}}, "options.bands"),
        ({"options": {"bands": [1, 1]}}, "options.bands"),
        ({"options": {"bands": ["a"]}}, "options.bands"),
        ({"options": {"depth": 0}}, "options.depth"),
        ({"options": {"max_cloud_cover": 10}}, "not supported"),
    ],
)
def test_invalid_requests(config: Config, tmp_path: Path, change, message) -> None:
    with pytest.raises(ValueError, match=message):
        cubesip.run(_request(tmp_path, **change), config)


def test_no_embedding_is_an_error(tmp_path: Path, monkeypatch) -> None:
    config = Config(manifest=tmp_path / "manifest.csv")
    driver = ts.TesseraDriver(config)
    driver._stores[STORE_URL] = FakeStore(fill=np.nan)  # open water
    monkeypatch.setattr(api, "get_driver", lambda name, cfg: driver)
    with pytest.raises(EmptyResultError, match="open water"):
        cubesip.run(_request(tmp_path), config)
    assert not config.manifest.exists()
    assert not list((tmp_path / "out").glob("*.nc"))


def test_dataset_key(tmp_path: Path) -> None:
    from geotessera.registry import zarr_store_url

    driver = get_driver("tessera", Config(manifest=tmp_path / "m.csv"))
    assert driver.dataset_key("v1.1") == zarr_store_url("v1.1")
    assert driver.dataset_key(" v2 ") == zarr_store_url("v2")
    assert driver.dataset_key(STORE_URL + "/") == STORE_URL
    assert driver.dataset_key(str(tmp_path)) == str(tmp_path)  # existing local store


def test_zone_crs() -> None:
    def request(geohash: str) -> cubesip.Request:
        return cubesip.Request.from_json({**_request(Path(".")), "geohash": geohash})

    assert ts.zone_crs(request("u094v8")) == "EPSG:32631"  # France
    lima = cubesip.geohash.encode(-12.05, -77.04, 6)
    assert ts.zone_crs(request(lima)) == "EPSG:32618"  # southern hemisphere: still 326NN


def test_missing_geotessera_is_explained(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "geotessera", None)  # import now raises ImportError
    driver = get_driver("tessera", Config(manifest=tmp_path / "m.csv"))
    with pytest.raises(ImportError, match=r"cubesip\[tessera\]"):
        driver.dataset_key("v1.1")

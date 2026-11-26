"""Beeldmateriaal driver, offline: small local GeoTIFFs stand in for the orthophoto tiles
(1 km for ortho_hr, 4 km on a grid offset by 2 km for ortho_lr), and a fake index download
lists them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import cubesip
from cubesip import api, checker
from cubesip.config import Config
from cubesip.drivers import EmptyResultError
from cubesip.drivers import beeldmateriaal as bm
from cubesip.request import Request
from cubesip.spatial import native_crs

GEOHASH = "u178kdc"  # Utrecht centre, inside the RD tile 136000_455000
RES = 10.0  # tests use 10 m pixels so a 1 km tile is only 100 x 100


def _write_tile(
    folder: Path, year: int, product: str, x0: int, y0: int, tile_m: int = 1000, res: float = RES
) -> str:
    """A 3-band GeoTIFF tile; band b holds 10*b + (year - 2020), so tests can tell bands, years
    and products apart (CIR adds 100, 4 km tiles add 50).
    """
    size = round(tile_m / res)
    kind = "hrl" if tile_m == 1000 else "lrl"
    path = folder / f"{year}_{x0:06d}_{y0:06d}_{product}_{kind}.tif"
    offset = (100 if product == "CIR" else 0) + (50 if tile_m != 1000 else 0)
    data = np.stack(
        [np.full((size, size), offset + 10 * b + (year - 2020), "uint8") for b in (1, 2, 3)]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs=bm.CRS_RD,
        transform=from_origin(x0, y0 + tile_m, res, res),
    ) as dst:
        dst.write(data)
    return str(path)


def _feature(
    url: str, x0: int, y0: int, mtime: str = "2025-07-01", size: int = 1, tile_m: int = 1000
) -> dict:
    x1, y1 = x0 + tile_m, y0 + tile_m
    ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return {
        "properties": {"file": url, "size": size, "mtime": mtime},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


@pytest.fixture
def tiles(tmp_path: Path) -> dict[tuple[str, int, str], list[dict]]:
    """Index features around the geohash, as published: ortho_hr RGB 2024 (no CIR before
    2025), RGB + CIR 2025 and 2026 (2026 lacks the CIR tile at 137000); ortho_lr RGB + CIR
    2024 in 4 km tiles on the grid offset by 2 km.
    """
    folder = tmp_path / "tiles"
    folder.mkdir()
    index: dict[tuple[str, int, str], list[dict]] = {}
    hr = [(2024, "RGB"), (2025, "RGB"), (2025, "CIR"), (2026, "RGB"), (2026, "CIR")]
    for year, product in hr:
        feats = []
        for x0 in (135000, 136000, 137000):
            for y0 in (454000, 455000, 456000):
                if (year, product, x0) == (2026, "CIR", 137000):
                    continue
                url = _write_tile(folder, year, product, x0, y0)
                feats += [_feature(url, x0, y0), _feature(url + ".aux.xml", x0, y0)]
        index[("ortho_hr", year, product)] = feats
    for product in ("RGB", "CIR"):
        feats = []
        for x0 in (130000, 134000, 138000):
            for y0 in (450000, 454000, 458000):
                url = _write_tile(folder, 2024, product, x0, y0, tile_m=4000)
                feats.append(_feature(url, x0, y0, tile_m=4000))
        index[("ortho_lr", 2024, product)] = feats
    return index


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, tiles) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []

    def fake_download(mosaic: bm.Mosaic, year: int, product: str) -> list[dict]:
        calls.append((year, product))
        if (mosaic.name, year, product) not in tiles:
            raise bm.DatasetNotFoundError(f"no {mosaic.name} {product} tiles for {year}")
        return tiles[(mosaic.name, year, product)]

    monkeypatch.setattr(bm, "download_index", fake_download)
    return calls


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, downloads) -> Config:
    config = Config(manifest=tmp_path / "manifest.csv", data_dir=tmp_path / "data")
    driver = bm.BeeldmateriaalDriver(config)
    monkeypatch.setattr(api, "get_driver", lambda name, cfg: driver)
    monkeypatch.setattr(checker, "get_driver", lambda name, cfg: driver)
    return config


def _request(tmp_path: Path, **extra) -> dict:
    return {
        "driver": "beeldmateriaal",
        "dataset": "ortho_hr",
        "variables": ["red", "nir"],
        "geohash": GEOHASH,
        "time": {"start": "2025-01-01", "end": "2025-12-31"},
        "target_dir": str(tmp_path / "out"),
        "options": {"resolution": RES},
        **extra,
    }


def test_check_run_and_cache(config: Config, downloads, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path), config)
    assert report.ok, report.summary()
    assert report.dataset.crs == "EPSG:28992"
    assert report.size.method == "exact"
    assert downloads == [(2025, "CIR")]  # red + nir: the CIR tiles alone are enough

    result = cubesip.run(_request(tmp_path), config)
    ds = result.dataset
    assert set(ds.data_vars) == {"red", "nir"}
    assert ds["nir"].dims == ("time", "y", "x") and ds["nir"].dtype == "uint8"
    assert native_crs(ds) == "EPSG:28992"
    assert np.allclose(np.diff(ds["x"].values), RES) and np.allclose(np.diff(ds["y"].values), -RES)
    # CIR band 1 is nir, band 2 red; 2025 adds 5
    assert set(np.unique(ds["nir"].values)) == {115}
    assert set(np.unique(ds["red"].values)) == {125}
    assert report.size.nbytes == ds["red"].size + ds["nir"].size
    assert result.manifest_rows[0]["crs"] == "EPSG:28992"
    assert "136000_455000_CIR" in ds.attrs["source_tiles"]
    assert "spring" in ds["time"].attrs["long_name"]
    assert downloads == [(2025, "CIR")]  # index read from the CSV, not downloaded again

    child = cubesip.run(_request(tmp_path, geohash=GEOHASH + "k"), config)
    assert child.from_cache and child.path == result.path


def test_rgb_variables_read_rgb_tiles(config: Config, downloads, tmp_path: Path) -> None:
    req = _request(tmp_path, variables=["red", "green", "blue", "nir"])
    ds = cubesip.run(req, config).dataset
    assert [int(ds[v].max()) for v in ("red", "green", "blue", "nir")] == [15, 25, 35, 115]
    assert sorted(downloads) == [(2025, "CIR"), (2025, "RGB")]


def test_two_years_and_missing_tile(config: Config, tmp_path: Path) -> None:
    # u178kd spans the tiles at x=136000 and x=137000; the 2026 CIR one at 137000 is missing
    req = _request(
        tmp_path,
        geohash="u178kd",
        variables=["nir"],
        time={"start": "2025-01-01", "end": "2026-12-31"},
    )
    report = cubesip.check(req, config)
    assert report.ok, report.summary()
    assert any("has no CIR tile for 2026" in w for w in report.warnings)
    ds = cubesip.run(req, config).dataset
    assert ds.sizes["time"] == 2
    nir25, nir26 = ds["nir"].isel(time=0).values, ds["nir"].isel(time=1).values
    assert (nir25 == 115).all()
    east = ds["x"].values >= 137000
    assert (nir26[:, east] == 0).all() and (nir26[:, ~east] == 116).all()


def test_coarser_resolution_averages(config: Config, tmp_path: Path) -> None:
    ds = cubesip.run(_request(tmp_path, options={"resolution": 50}), config).dataset
    assert np.allclose(np.diff(ds["x"].values), 50)
    assert set(np.unique(ds["nir"].values)) == {115}


def test_hr_rgb_before_2025(config: Config, tmp_path: Path) -> None:
    req = _request(
        tmp_path, variables=["red", "blue"], time={"start": "2024-01-01", "end": "2024-12-31"}
    )
    ds = cubesip.run(req, config).dataset
    assert [int(ds[v].max()) for v in ("red", "blue")] == [14, 34]
    # 8 cm near-infrared only exists from 2025
    report = cubesip.check({**req, "variables": ["nir"]}, config)
    assert not report.ok and "CIR" in report.error


def test_lr_mosaic_on_offset_grid(config: Config, tmp_path: Path) -> None:
    # u178kd lies inside the 4 km tile 134000-138000 of the grid offset by 2 km, which the
    # 1 km corner arithmetic of ortho_hr would not find
    req = _request(
        tmp_path,
        dataset="ortho_lr",
        geohash="u178kd",
        variables=["nir", "red"],
        time={"start": "2024-01-01", "end": "2024-12-31"},
    )
    report = cubesip.check(req, config)
    assert report.ok, report.summary()
    assert report.warnings == []
    ds = cubesip.run(req, config).dataset
    assert set(np.unique(ds["nir"].values)) == {164}
    assert set(np.unique(ds["red"].values)) == {174}
    assert "ortho_lr" in ds.attrs["beeldmateriaal_mosaic"]
    assert "summer" in ds["time"].attrs["long_name"]
    assert ds.attrs["source_tiles"].endswith("2024_134000_454000_CIR_lrl.tif")


def test_outside_netherlands(config: Config, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path, geohash="u0nd9w"), config)
    assert not report.ok and "outside the Netherlands" in report.error


def test_year_without_index(config: Config, tmp_path: Path) -> None:
    req = _request(tmp_path, time={"start": "2020-01-01", "end": "2020-12-31"})
    report = cubesip.check(req, config)
    assert not report.ok and "2020" in report.error


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"dataset": "ortho_xl"}, "unknown beeldmateriaal dataset"),
        ({"time": {"start": "2025-03-01", "end": "2025-12-31"}}, "whole calendar years"),
        ({"time": None}, "whole calendar years"),
        ({"options": {"resolution": 0.3}}, "does not divide"),
        ({"options": {"resolution": -1}}, "positive"),
        ({"options": {"bands": [1]}}, "options not supported"),
    ],
)
def test_validate_refuses(config: Config, tmp_path: Path, extra: dict, message: str) -> None:
    report = cubesip.check(_request(tmp_path, **extra), config)
    assert not report.ok and message in report.error


def test_unknown_variable(config: Config, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path, variables=["swir"]), config)
    assert not report.ok and "swir" in report.error


def test_empty_result(config: Config, tmp_path: Path) -> None:
    driver = bm.BeeldmateriaalDriver(config)
    req = Request.model_validate(_request(tmp_path, geohash="u0nd9w"))
    with pytest.raises(EmptyResultError):
        driver.fetch(req, tmp_path / "x.nc")


# -- pure helpers --------------------------------------------------------------------------


def test_default_resolution_follows_the_mosaic(tmp_path: Path) -> None:
    base = _request(tmp_path, options={})
    assert bm.resolution(Request.model_validate(base)) == 0.08
    assert bm.resolution(Request.model_validate({**base, "dataset": "ortho_lr"})) == 0.25


def test_download_index_tries_the_older_name(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    features = [{"properties": {"file": "a.tif"}}]

    class Response:
        def __init__(self, status: int) -> None:
            self.status_code = status

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"features": features}

    def fake_get(url: str, timeout: float) -> Response:
        name = url.rsplit("/", 1)[1]
        asked.append(name)
        return Response(200 if name in ("BM_HRL2024O_TIF.json", "BM_LRL2024O_CIR.json") else 403)

    monkeypatch.setattr(bm.requests, "get", fake_get)
    hr = bm.MOSAICS["ortho_hr"]
    assert bm.download_index(hr, 2024, "RGB") == features
    assert asked == ["BM_HRL2024O_RGB_TIF.json", "BM_HRL2024O_TIF.json"]
    asked.clear()
    with pytest.raises(bm.DatasetNotFoundError, match="ortho_lr"):
        bm.download_index(hr, 2024, "CIR")
    assert asked == ["BM_HRL2024O_CIR_TIF.json"]  # the RGB-only index is not tried for CIR
    asked.clear()
    assert bm.download_index(bm.MOSAICS["ortho_lr"], 2024, "CIR") == features
    assert asked == ["BM_LRL2024O_CIR.json"]


def test_band_sources() -> None:
    assert bm.band_sources(["nir", "red"]) == {"nir": "CIR", "red": "CIR"}
    assert bm.band_sources(["red", "green"]) == {"red": "RGB", "green": "RGB"}
    assert bm.band_sources(["blue", "nir", "red"]) == {"blue": "RGB", "nir": "CIR", "red": "RGB"}


def test_select_tiles_skips_sidecars_and_picks_one_duplicate() -> None:
    base = "https://x/2025/Ortho/{b}/beelden_RGB_jpeg_tegels/2025_{x}_455000_RGB_JPEG_hrl.tif"
    old = base.format(b="1/1", x="057000")
    new = base.format(b="8/1", x="57000")
    lower = "https://x/2021/1/01/beelden_tif_tegels/2021_058000_455000_rgb_hrl.tif"
    tiles = bm.select_tiles(
        [
            _feature(old, 57000, 455000, mtime="2025-07-16"),
            _feature(old + ".aux.xml", 57000, 455000, mtime="2025-12-01"),
            _feature(old.replace(".tif", ".tfw"), 57000, 455000, mtime="2025-12-01"),
            _feature(new, 57000, 455000, mtime="2025-10-03"),
            _feature(lower, 58000, 455000),
        ]
    )
    assert sorted((t.west, t.south, t.east, t.north, t.url) for t in tiles) == [
        (57000, 455000, 58000, 456000, new),
        (58000, 455000, 59000, 456000, lower),
    ]


def test_output_grid_snaps_to_tile_edges() -> None:
    req = Request.model_validate(
        {
            "driver": "beeldmateriaal",
            "dataset": "ortho_hr",
            "variables": ["red"],
            "geohash": GEOHASH,
            "time": {"start": date(2025, 1, 1), "end": date(2025, 12, 31)},
        }
    )
    grid = bm.output_grid(req.bbox, 0.08)
    for edge in grid.bounds:
        assert abs(edge / 0.08 - round(edge / 0.08)) < 1e-6
    tile = bm.Tile(136000, 455000, 137000, 456000, "")
    area = (grid.east - grid.west) * (grid.north - grid.south)
    assert tile.overlap(grid.bounds) == pytest.approx(area)
    (rows, cols), _ = bm.tile_slices(tile, grid)
    assert (rows.stop - rows.start, cols.stop - cols.start) == (grid.height, grid.width)

"""Beeldmateriaal driver, offline: small local GeoTIFFs stand in for the 1 km orthophoto tiles,
and a fake index download lists them.
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


def _write_tile(folder: Path, year: int, product: str, x0: int, y0: int, res: float = RES) -> str:
    """A 3-band GeoTIFF tile; band b holds 10*b + (year - 2020), so tests can tell bands, years
    and products apart (CIR adds 100).
    """
    size = round(bm.TILE_M / res)
    path = folder / f"{year}_{x0:06d}_{y0:06d}_{product}_JPEG_hrl.tif"
    offset = 100 if product == "CIR" else 0
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
        transform=from_origin(x0, y0 + bm.TILE_M, res, res),
    ) as dst:
        dst.write(data)
    return str(path)


def _feature(url: str, x0: int, y0: int, mtime: str = "2025-07-01", size: int = 1) -> dict:
    ring = [[x0, y0], [x0 + 1000, y0], [x0 + 1000, y0 + 1000], [x0, y0 + 1000], [x0, y0]]
    return {
        "properties": {"file": url, "size": size, "mtime": mtime},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


@pytest.fixture
def tiles(tmp_path: Path) -> dict[tuple[int, str], list[dict]]:
    """Tiles around the geohash for 2025 and 2026; 2026 lacks the CIR tile at 137000."""
    folder = tmp_path / "tiles"
    folder.mkdir()
    index: dict[tuple[int, str], list[dict]] = {}
    for year in (2025, 2026):
        for product in ("RGB", "CIR"):
            feats = []
            for x0 in (135000, 136000, 137000):
                for y0 in (454000, 455000, 456000):
                    if (year, product, x0) == (2026, "CIR", 137000):
                        continue
                    url = _write_tile(folder, year, product, x0, y0)
                    feats += [_feature(url, x0, y0), _feature(url + ".aux.xml", x0, y0)]
            index[(year, product)] = feats
    return index


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, tiles) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []

    def fake_download(year: int, product: str) -> list[dict]:
        calls.append((year, product))
        if (year, product) not in tiles:
            raise bm.DatasetNotFoundError(f"no {product} tiles for {year}")
        return tiles[(year, product)]

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
    assert any("tiles missing for 2026" in w for w in report.warnings)
    ds = cubesip.run(req, config).dataset
    assert ds.sizes["time"] == 2
    nir25, nir26 = ds["nir"].isel(time=0).values, ds["nir"].isel(time=1).values
    assert set(np.unique(nir25)) <= {115, 0} and (nir25 == 115).all()
    east = ds["x"].values >= 137000
    assert (nir26[:, east] == 0).all() and (nir26[:, ~east] == 116).all()


def test_coarser_resolution_averages(config: Config, tmp_path: Path) -> None:
    ds = cubesip.run(_request(tmp_path, options={"resolution": 50}), config).dataset
    assert np.allclose(np.diff(ds["x"].values), 50)
    assert set(np.unique(ds["nir"].values)) == {115}


def test_outside_netherlands(config: Config, tmp_path: Path) -> None:
    report = cubesip.check(_request(tmp_path, geohash="u0nd9w"), config)
    assert not report.ok and "outside the Netherlands" in report.error


def test_year_without_index(config: Config, tmp_path: Path) -> None:
    req = _request(tmp_path, time={"start": "2024-01-01", "end": "2024-12-31"})
    report = cubesip.check(req, config)
    assert not report.ok and "2024" in report.error


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"dataset": "ortho_lr"}, "unknown beeldmateriaal dataset"),
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


def test_band_sources() -> None:
    assert bm.band_sources(["nir", "red"]) == {"nir": "CIR", "red": "CIR"}
    assert bm.band_sources(["red", "green"]) == {"red": "RGB", "green": "RGB"}
    assert bm.band_sources(["blue", "nir", "red"]) == {"blue": "RGB", "nir": "CIR", "red": "RGB"}


def test_select_tiles_skips_sidecars_and_picks_one_duplicate() -> None:
    base = "https://x/2025/Ortho/{b}/beelden_RGB_jpeg_tegels/2025_{x}_455000_RGB_JPEG_hrl.tif"
    old = base.format(b="1/1", x="057000")
    new = base.format(b="8/1", x="57000")
    tiles = bm.select_tiles(
        [
            _feature(old, 57000, 455000, mtime="2025-07-16"),
            _feature(old + ".aux.xml", 57000, 455000, mtime="2025-12-01"),
            _feature(new, 57000, 455000, mtime="2025-10-03"),
            _feature(
                "https://x/2025/Ortho/5/6/beelden_RGB_tif_tegels/2025_058000_455000_RGB_hrl.tif",
                58000,
                455000,
            ),
        ]
    )
    assert [(t.x0, t.y0, t.url) for t in tiles] == [
        (57000, 455000, new),
        (58000, 455000, tiles[1].url),
    ]
    assert tiles[1].url.endswith("_RGB_hrl.tif")


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
    for edge in (grid.west, grid.south, grid.east, grid.north):
        assert abs(edge / 0.08 - round(edge / 0.08)) < 1e-6
    assert bm.tile_corners((grid.west, grid.south, grid.east, grid.north)) == [(136000, 455000)]
    (rows, cols), _ = bm.tile_slices(bm.Tile(136000, 455000, ""), grid)
    assert (rows.stop - rows.start, cols.stop - cols.start) == (grid.height, grid.width)

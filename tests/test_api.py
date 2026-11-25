"""check() / run() orchestration and cache rules, with an in-memory fake driver."""

from pathlib import Path

import pytest
import xarray as xr
from conftest import era5_like

import cubesip
from cubesip import api, checker
from cubesip import geohash as gh
from cubesip.config import Config
from cubesip.drivers import XarraySourceDriver

GEOHASH = gh.encode(45.3, -3.7, 3)


class FakeDriver(XarraySourceDriver):
    name = "edh"
    supported_options = frozenset({"foo"})
    fetches = 0
    fail = False
    backend = ""  # what backend_url reports; tests switch it to mimic another back-end

    @property
    def backend_url(self) -> str:
        return FakeDriver.backend

    def _open(self, dataset_key: str) -> xr.Dataset:
        return era5_like()

    def fetch(self, request, out_path):
        FakeDriver.fetches += 1
        if FakeDriver.fail:
            Path(out_path).write_bytes(b"half a file")
            raise RuntimeError("connection lost")
        return super().fetch(request, out_path)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    FakeDriver.fetches = 0
    FakeDriver.fail = False
    FakeDriver.backend = ""
    monkeypatch.setattr(api, "get_driver", lambda name, cfg: FakeDriver(cfg))
    monkeypatch.setattr(checker, "get_driver", lambda name, cfg: FakeDriver(cfg))
    return Config(manifest=tmp_path / "manifest.csv", data_dir=tmp_path / "default")


@pytest.fixture
def request_dict(tmp_path: Path) -> dict:
    return {
        "driver": "edh",
        "dataset": "fake/era5",
        "variables": ["t2m", "tp"],
        "geohash": GEOHASH,
        "time": {"start": "2024-01-01", "end": "2024-01-02"},
        "target_dir": str(tmp_path / "out"),
    }


def test_check_then_run_then_cache(config: Config, request_dict: dict) -> None:
    report = cubesip.check(request_dict, config)
    assert report.ok and report.cache_hit is None
    assert report.size.method == "exact" and report.size.nbytes > 0
    assert "ready to download" in str(report)

    first = cubesip.run(request_dict, config)
    assert not first.from_cache and first.path.is_file()
    assert first.path.parent == Path(request_dict["target_dir"]).resolve()
    assert [r["variable"] for r in first.manifest_rows] == ["t2m", "tp"]
    assert first.manifest_rows[0]["crs"] == "EPSG:4326"
    assert first.dataset.attrs["cubesip_geohash"] == GEOHASH
    assert first.dataset.sizes["valid_time"] == 48
    assert not list(first.path.parent.glob("*.partial.nc"))
    assert FakeDriver.fetches == 1

    report = cubesip.check(request_dict, config)
    assert report.ok and report.cache_hit is not None
    assert report.cache_hit.path == first.path
    assert "cached" in str(report)

    second = cubesip.run(request_dict, config)
    assert second.from_cache and second.path == first.path
    assert second.manifest_rows == []
    assert FakeDriver.fetches == 1
    xr.testing.assert_allclose(second.dataset["t2m"].load(), first.dataset["t2m"].load())


def test_without_target_dir_writes_to_config_data_dir(
    config: Config, request_dict: dict, tmp_path: Path
) -> None:
    del request_dict["target_dir"]
    result = cubesip.run(request_dict, config)
    assert result.path.parent == (tmp_path / "default").resolve()
    assert result.path.is_file()


def test_target_dir_overrides_config_data_dir(config: Config, request_dict: dict) -> None:
    result = cubesip.run(request_dict, config)
    assert result.path.parent == Path(request_dict["target_dir"]).resolve()
    assert not config.data_dir.exists()


def test_child_geohash_shorter_period_reuses_parent(config: Config, request_dict: dict) -> None:
    parent = cubesip.run(request_dict, config)
    child = {
        **request_dict,
        "geohash": GEOHASH + "k",
        "variables": ["tp"],
        "time": {"start": "2024-01-02", "end": "2024-01-02"},
    }
    result = cubesip.run(child, config)
    assert result.from_cache and result.path == parent.path
    assert list(result.dataset.data_vars) == ["tp"]
    assert result.dataset.sizes["valid_time"] == 24
    assert result.dataset.sizes["longitude"] <= parent.dataset.sizes["longitude"]
    assert FakeDriver.fetches == 1
    assert len(cubesip.manifest.Manifest(config.manifest).read()) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"options": {"foo": 1}},
        {"time": {"start": "2024-01-01", "end": "2024-01-03"}},
        {"geohash": GEOHASH[:-1]},
        {"variables": ["t2m", "tp", "missing_in_cache"]},
    ],
    ids=["other-options", "longer-period", "parent-area", "more-variables"],
)
def test_cache_misses(config: Config, request_dict: dict, change: dict) -> None:
    cubesip.run(request_dict, config)
    report = cubesip.check({**request_dict, **change}, config)
    assert report.cache_hit is None


def test_cache_is_per_backend(config: Config, request_dict: dict) -> None:
    """Same request, two back-ends: two downloads, two files, each served to its own."""
    FakeDriver.backend = "https://a.example.eu"
    first = cubesip.run(request_dict, config)
    assert first.manifest_rows[0]["backend"] == "https://a.example.eu"
    assert first.dataset.attrs["cubesip_backend"] == "https://a.example.eu"

    FakeDriver.backend = "https://b.example.eu"
    assert cubesip.check(request_dict, config).cache_hit is None
    second = cubesip.run(request_dict, config)
    assert not second.from_cache and second.path != first.path
    assert first.path.is_file()  # not overwritten
    assert FakeDriver.fetches == 2

    FakeDriver.backend = "https://a.example.eu"
    again = cubesip.run(request_dict, config)
    assert again.from_cache and again.path == first.path
    assert FakeDriver.fetches == 2


def test_rows_without_backend_are_still_reused(config: Config, request_dict: dict) -> None:
    """Rows written before back-ends were recorded keep serving, with a warning."""
    cubesip.run(request_dict, config)  # backend "" mimics an old row
    FakeDriver.backend = "https://a.example.eu"
    report = cubesip.check(request_dict, config)
    assert report.cache_hit is not None and report.cache_hit.backend == ""
    assert any("before cubesip tracked back-ends" in w for w in report.warnings)
    assert cubesip.run(request_dict, config).from_cache
    assert FakeDriver.fetches == 1


def test_deleted_file_is_not_a_hit_and_force_refetches(config: Config, request_dict) -> None:
    first = cubesip.run(request_dict, config)
    first.dataset.close()
    again = cubesip.run(request_dict, config, force=True)
    assert FakeDriver.fetches == 2 and not again.from_cache
    again.dataset.close()
    first.path.unlink()
    assert cubesip.check(request_dict, config).cache_hit is None


def test_force_while_previous_result_open_is_explained(config: Config, request_dict) -> None:
    first = cubesip.run(request_dict, config)
    first.dataset["t2m"].load()  # keep the file open
    try:
        cubesip.run(request_dict, config, force=True)
    except PermissionError as exc:  # Windows refuses to replace an open file
        assert "close" in str(exc)
    finally:
        first.dataset.close()


def test_failed_fetch_leaves_nothing(config: Config, request_dict: dict) -> None:
    FakeDriver.fail = True
    with pytest.raises(RuntimeError):
        cubesip.run(request_dict, config)
    assert not list(Path(request_dict["target_dir"]).glob("*"))
    assert not config.manifest.exists()


def test_errors_are_reported(config: Config, request_dict: dict) -> None:
    report = cubesip.check({**request_dict, "variables": ["u10"]}, config)
    assert not report.ok and "u10" in report.error
    with pytest.raises(KeyError, match="u10"):
        cubesip.run({**request_dict, "variables": ["u10"]}, config)

    report = cubesip.check({**request_dict, "options": {"bar": 1}}, config)
    assert not report.ok and "bar" in report.error
    with pytest.raises(ValueError, match="bar"):
        cubesip.run({**request_dict, "options": {"bar": 1}}, config)

    outside = gh.encode(-30.0, 20.0, 3)
    report = cubesip.check({**request_dict, "geohash": outside}, config)
    assert not report.ok and "outside" in report.error


def test_static_request_on_temporal_dataset_warns(config: Config, request_dict: dict) -> None:
    report = cubesip.check({**request_dict, "time": None}, config)
    assert report.ok
    assert any("every time step" in w for w in report.warnings)


class FakeDbutils:
    """Stands in for Databricks dbutils: records calls and moves files on the local disk."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fs = self

    def mkdirs(self, path: str) -> None:
        self.calls.append(("mkdirs", path))
        Path(path).mkdir(parents=True, exist_ok=True)

    def mv(self, src: str, dst: str) -> None:
        self.calls.append(("mv", src, dst))
        assert src.startswith("file:")
        Path(src.removeprefix("file:")).replace(dst)


def test_stage_locally_moves_with_dbutils(
    config: Config, request_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeDbutils()
    monkeypatch.setattr(api.databricks, "get_dbutils", lambda: fake)
    result = cubesip.run({**request_dict, "stage_locally": True}, config)
    try:
        assert result.path.exists() and not result.from_cache
        assert result.path.parent == Path(request_dict["target_dir"]).resolve()
        assert [c[0] for c in fake.calls] == ["mkdirs", "mv"]
        assert fake.calls[1][2] == result.path.as_posix()
        assert not Path(fake.calls[1][1].removeprefix("file:")).parent.exists()  # tmp cleaned
        assert set(result.dataset.data_vars) == {"t2m", "tp"}
    finally:
        result.dataset.close()
    # same file name as without staging, so the cache is shared
    again = cubesip.run(request_dict, config)
    assert again.from_cache and again.path == result.path
    again.dataset.close()


def test_stage_locally_failed_fetch_moves_nothing(
    config: Config, request_dict: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeDbutils()
    monkeypatch.setattr(api.databricks, "get_dbutils", lambda: fake)
    FakeDriver.fail = True
    with pytest.raises(RuntimeError):
        cubesip.run({**request_dict, "stage_locally": True}, config)
    assert fake.calls == []
    assert not config.manifest.exists()


def test_stage_locally_outside_databricks_is_explained(config: Config, request_dict: dict) -> None:
    with pytest.raises(RuntimeError, match="dbutils"):
        cubesip.run({**request_dict, "stage_locally": True}, config)

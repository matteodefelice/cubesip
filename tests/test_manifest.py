from pathlib import Path

import pytest

from cubesip.manifest import COLUMNS, Manifest, make_rows, path_to_uri, uri_to_path
from cubesip.request import Request

BASE = {
    "driver": "edh",
    "dataset": "era5/era5-single-levels-atmosphere-v0",
    "variables": ["t2m", "tp"],
    "geohash": "qem7",
    "time": {"start": "2024-01-01", "end": "2024-01-31"},
    "target_dir": "./data",
}


def test_uri_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "a dir" / "file.nc"
    assert uri_to_path(path_to_uri(path)) == path.resolve()
    with pytest.raises(ValueError):
        uri_to_path("https://example.com/x.nc")


def test_read_missing_manifest_is_empty(tmp_path: Path) -> None:
    frame = Manifest(tmp_path / "manifest.csv").read()
    assert frame.empty and list(frame.columns) == COLUMNS


def test_make_rows_and_append(tmp_path: Path) -> None:
    nc = tmp_path / "out.nc"
    nc.write_bytes(b"x" * 123)
    request = Request.from_json(BASE)
    rows = make_rows(request, dataset_key="era5/x", path=nc, crs="EPSG:4326")
    assert [r["variable"] for r in rows] == ["t2m", "tp"]
    assert len({r["request_id"] for r in rows}) == 1
    assert rows[0]["file_size_bytes"] == "123"
    assert rows[0]["options"] == "{}"

    manifest = Manifest(tmp_path / "sub" / "manifest.csv")
    manifest.append(rows)
    manifest.append(rows)
    text = manifest.path.read_text(encoding="utf-8")
    assert text.count("request_id") == 1  # header written once
    frame = manifest.read()
    assert len(frame) == 4
    assert frame.loc[0, "time_start"] == "2024-01-01"


def test_append_rejects_unknown_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bogus"):
        Manifest(tmp_path / "m.csv").append([{"bogus": "1"}])

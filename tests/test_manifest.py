import csv
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


def test_old_manifest_is_upgraded_before_appending(tmp_path: Path) -> None:
    """A manifest without the backend column is rewritten, so new rows line up."""
    nc = tmp_path / "out.nc"
    nc.write_bytes(b"x")
    request = Request.from_json(BASE)
    old_columns = [c for c in COLUMNS if c != "backend"]
    old_row = make_rows(request, dataset_key="era5/x", path=nc, crs="EPSG:4326")[0]
    del old_row["backend"]
    manifest = Manifest(tmp_path / "manifest.csv")
    with manifest.path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow(old_row)

    new_rows = make_rows(
        request, dataset_key="era5/x", path=nc, crs="EPSG:4326", backend="https://edh.example"
    )
    manifest.append(new_rows)
    with manifest.path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == COLUMNS
    frame = manifest.read()
    assert frame["backend"].tolist() == ["", "https://edh.example", "https://edh.example"]
    assert frame["variable"].tolist() == ["t2m", "t2m", "tp"]
    assert frame["geohash"].tolist() == ["qem7"] * 3


def test_append_rejects_unknown_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bogus"):
        Manifest(tmp_path / "m.csv").append([{"bogus": "1"}])


def test_summarize_empty_manifest(tmp_path: Path) -> None:
    summary = Manifest(tmp_path / "manifest.csv").summarize()
    assert summary["manifest"].endswith("manifest.csv")
    assert summary["rows"] == 0
    assert summary["files"] == 0
    assert summary["missing_files"] == 0
    assert summary["requests"] == 0
    assert summary["datasets"] == 0
    assert summary["variables"] == 0
    assert summary["geohashes"] == 0
    assert summary["drivers"] == []
    assert summary["users"] == []
    assert summary["hosts"] == []
    assert summary["total_size_bytes"] == 0
    assert summary["missing_size_bytes"] == 0
    assert summary["first_entry"] == ""
    assert summary["last_entry"] == ""
    assert summary["time_start"] == ""
    assert summary["time_end"] == ""
    assert summary["by_dataset"].empty
    assert Manifest(tmp_path / "manifest.csv").find_missing().empty


def test_summarize_and_find_missing(tmp_path: Path) -> None:
    """Two files recorded, one of them then deleted: the summary counts it as missing and
    find_missing reports exactly that file.
    """
    manifest = Manifest(tmp_path / "manifest.csv")
    first = tmp_path / "first.nc"
    first.write_bytes(b"x" * 100)
    second = tmp_path / "second.nc"
    second.write_bytes(b"x" * 40)
    manifest.append(
        make_rows(Request.from_json(BASE), dataset_key="era5/x", path=first, crs="EPSG:4326")
    )
    manifest.append(
        make_rows(
            Request.from_json(
                {
                    **BASE,
                    "dataset": "sentinel2/y",
                    "variables": ["ndvi"],
                    "geohash": "qem",
                    "time": {"start": "2024-06-01", "end": "2024-06-30"},
                }
            ),
            dataset_key="sentinel2/y",
            path=second,
            crs="EPSG:32630",
        )
    )
    second.unlink()

    summary = manifest.summarize()
    assert summary["rows"] == 3
    assert summary["files"] == 2
    assert summary["missing_files"] == 1
    assert summary["requests"] == 2
    assert summary["datasets"] == 2
    assert summary["variables"] == 3
    assert summary["geohashes"] == 2
    assert summary["drivers"] == ["edh"]
    assert summary["total_size_bytes"] == 140
    assert summary["missing_size_bytes"] == 40
    assert summary["first_entry"] != ""
    assert summary["last_entry"] != ""
    assert summary["time_start"] == "2024-01-01"
    assert summary["time_end"] == "2024-06-30"

    by_dataset = summary["by_dataset"]
    assert len(by_dataset) == 2
    era5 = by_dataset[by_dataset["dataset"] == "era5/x"].iloc[0]
    assert (era5["files"], era5["missing_files"], era5["rows"]) == (1, 0, 2)
    assert era5["variables"] == "t2m, tp"
    assert era5["size_bytes"] == 100
    assert (era5["time_start"], era5["time_end"]) == ("2024-01-01", "2024-01-31")
    s2 = by_dataset[by_dataset["dataset"] == "sentinel2/y"].iloc[0]
    assert (s2["files"], s2["missing_files"], s2["rows"]) == (1, 1, 1)
    assert s2["size_bytes"] == 40

    missing = manifest.find_missing()
    assert missing["uri"].tolist() == [path_to_uri(second)]
    assert missing["variables"].tolist() == ["ndvi"]
    assert missing["rows"].tolist() == [1]
    assert missing["file_size_bytes"].tolist() == [40]


def test_prune_drops_missing_rows_only(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifest.csv")
    keep = tmp_path / "keep.nc"
    keep.write_bytes(b"x" * 10)
    gone = tmp_path / "gone.nc"
    gone.write_bytes(b"x" * 20)
    request = Request.from_json(BASE)
    manifest.append(make_rows(request, dataset_key="era5/x", path=keep, crs="EPSG:4326"))
    manifest.append(make_rows(request, dataset_key="era5/x", path=gone, crs="EPSG:4326"))
    gone.unlink()

    removed = manifest.prune()
    assert removed["uri"].tolist() == [path_to_uri(gone)]
    frame = manifest.read()
    assert len(frame) == 2  # only the two rows of the file that is still there
    assert set(frame["variable"]) == {"t2m", "tp"}
    assert manifest.find_missing().empty
    assert manifest.summarize()["files"] == 1
    assert manifest.prune().empty  # nothing more to remove, nothing rewritten


def test_prune_keeps_extra_columns(tmp_path: Path) -> None:
    """Rows added by hand (or a newer cubesip) keep their extra columns when pruning."""
    kept = tmp_path / "out.nc"
    kept.write_bytes(b"x")
    gone = tmp_path / "gone.nc"
    gone.write_bytes(b"x" * 5)
    request = Request.from_json(BASE)
    rows = make_rows(request, dataset_key="era5/x", path=kept, crs="EPSG:4326")
    gone_rows = make_rows(request, dataset_key="era5/x", path=gone, crs="EPSG:4326")
    gone.unlink()
    manifest = Manifest(tmp_path / "manifest.csv")
    for row in (*rows, *gone_rows):
        row["note"] = "hand-added"
    with manifest.path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS + ["note"])
        writer.writeheader()
        writer.writerows([*rows, *gone_rows])

    assert len(manifest.prune()) == 1
    frame = manifest.read()
    assert len(frame) == 2
    assert set(frame["note"]) == {"hand-added"}

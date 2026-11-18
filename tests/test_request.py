import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cubesip.request import Request, request_json_schema

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("request_*.json"))

BASE = {
    "driver": "edh",
    "dataset": "era5/era5-single-levels-atmosphere-v0",
    "variables": ["t2m"],
    "geohash": "qem7",
    "time": {"start": "2024-01-01", "end": "2024-01-31"},
    "target_dir": "./data",
}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_are_valid(path: Path) -> None:
    Request.from_json(path)


def test_from_json_accepts_dict_string_and_path(tmp_path: Path) -> None:
    file = tmp_path / "r.json"
    file.write_text(json.dumps(BASE))
    a = Request.from_json(BASE)
    b = Request.from_json(json.dumps(BASE))
    c = Request.from_json(file)
    assert a == b == c


def test_geohash_is_normalised() -> None:
    assert Request.from_json({**BASE, "geohash": " QEM7 "}).geohash == "qem7"


@pytest.mark.parametrize("bad", ["", "qem7a", "qemi", "1234567890123"])
def test_invalid_geohash_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        Request.from_json({**BASE, "geohash": bad})


def test_time_can_be_null() -> None:
    assert Request.from_json({**BASE, "time": None}).time is None


def test_time_order_enforced() -> None:
    with pytest.raises(ValidationError):
        Request.from_json({**BASE, "time": {"start": "2024-02-01", "end": "2024-01-01"}})


def test_unknown_fields_and_duplicate_variables_rejected() -> None:
    with pytest.raises(ValidationError):
        Request.from_json({**BASE, "bands": ["t2m"]})
    with pytest.raises(ValidationError):
        Request.from_json({**BASE, "variables": ["t2m", "t2m"]})


def test_committed_schema_is_up_to_date() -> None:
    committed = json.loads((ROOT / "schemas" / "request.schema.json").read_text())
    assert committed == request_json_schema(), (
        "Regenerate with: uv run python scripts/export_schema.py"
    )

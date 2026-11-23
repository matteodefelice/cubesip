"""Splitting a request into its 32 child geohash cells."""

import pytest

import cubesip
from cubesip import geohash as gh


def test_children_tile_the_parent() -> None:
    kids = gh.children("QEM")
    assert len(kids) == 32 and len(set(kids)) == 32
    assert all(len(k) == 4 and gh.covers("qem", k) for k in kids)
    parent = gh.bbox("qem")
    area = sum((b.east - b.west) * (b.north - b.south) for b in map(gh.bbox, kids))
    assert area == pytest.approx((parent.east - parent.west) * (parent.north - parent.south))
    with pytest.raises(ValueError):
        gh.children("a" * gh.MAX_LENGTH)


def test_split_keeps_everything_but_the_geohash() -> None:
    req = cubesip.Request.from_json(
        {
            "driver": "openeo",
            "dataset": "SENTINEL2_L2A",
            "variables": ["B04", "B08"],
            "geohash": "u0q",
            "time": {"start": "2024-06-01", "end": "2024-06-30"},
            "options": {"max_cloud_cover": 60},
        }
    )
    parts = cubesip.split(req)
    assert [p.geohash for p in parts] == gh.children("u0q")
    for part in parts:
        assert part.model_dump(exclude={"geohash"}) == req.model_dump(exclude={"geohash"})
    assert req.geohash == "u0q"  # the original is untouched

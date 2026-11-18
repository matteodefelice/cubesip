import random

import pytest

from cubesip import geohash as gh


def test_bbox_known_value() -> None:
    # Wikipedia example: ezs42 ~ (42.6, -5.6)
    box = gh.bbox("ezs42")
    assert box.west == pytest.approx(-5.625)
    assert box.east == pytest.approx(-5.5810546875)
    assert box.south == pytest.approx(42.5830078125)
    assert box.north == pytest.approx(42.626953125)


def test_bbox_first_level() -> None:
    assert gh.bbox("u") == gh.BBox(0.0, 45.0, 45.0, 90.0)
    assert gh.bbox("0") == gh.BBox(-180.0, -90.0, -135.0, -45.0)


def test_encode_known_value() -> None:
    assert gh.encode(42.605, -5.603, 5) == "ezs42"


def test_encode_bbox_roundtrip() -> None:
    rnd = random.Random(42)
    for _ in range(500):
        lat, lon = rnd.uniform(-89.9, 89.9), rnd.uniform(-179.9, 179.9)
        code = gh.encode(lat, lon, rnd.randint(1, 10))
        box = gh.bbox(code)
        assert box.south <= lat <= box.north and box.west <= lon <= box.east


def test_children_nest_inside_parent() -> None:
    parent = gh.bbox("qem")
    child = gh.bbox("qem7")
    assert parent.west <= child.west and child.east <= parent.east
    assert parent.south <= child.south and child.north <= parent.north


def test_ancestors_and_covers() -> None:
    assert gh.ancestors("qem7") == ["qem7", "qem", "qe", "q"]
    assert gh.ancestors("QEM7", include_self=False) == ["qem", "qe", "q"]
    assert gh.covers("qem", "qem7")
    assert gh.covers("qem7", "qem7")
    assert not gh.covers("qem7", "qem")
    assert not gh.covers("qe7", "qem7")


@pytest.mark.parametrize("bad", ["", "a", "qemi", "x" * 13])
def test_validate_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        gh.validate(bad)

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


def test_containing_gives_a_cell_back_from_its_box() -> None:
    for code in ("u", "ezs42", "gbj3ey", "u09tvw0q"):
        assert gh.containing(gh.bbox(code)) == code


def test_containing_a_point_matches_encode() -> None:
    rnd = random.Random(7)
    for _ in range(300):
        lat, lon = rnd.uniform(-89.9, 89.9), rnd.uniform(-179.9, 179.9)
        precision = rnd.randint(1, gh.MAX_LENGTH)
        point = gh.BBox(lon, lat, lon, lat)
        assert gh.containing(point, precision) == gh.encode(lat, lon, precision)


def test_containing_a_box_over_two_siblings_gives_the_parent() -> None:
    a, b = gh.bbox("ezs42"), gh.bbox("ezs43")  # east-west neighbours inside ezs4
    assert gh.containing(gh.BBox(a.west, a.south, b.east, b.north)) == "ezs4"


def test_containing_rejects_boxes_no_cell_holds() -> None:
    with pytest.raises(ValueError, match="level-1"):
        gh.containing(gh.BBox(-1.0, 51.0, 1.0, 52.0))  # crosses the prime meridian
    with pytest.raises(ValueError, match="invalid box"):
        gh.containing(gh.BBox(2.0, 51.0, 1.0, 52.0))
    with pytest.raises(ValueError, match="max_precision"):
        gh.containing(gh.bbox("u0"), max_precision=13)

"""Tests for the naive travel distance.

What these check is the arithmetic and the missing-venue rule. Whether the
coordinates are *right* is not something a unit test can tell you -- they
were typed in by hand, and the module says so.
"""

import pytest

from .travel import VENUES, distance_km


def test_a_team_travelling_to_itself_goes_nowhere() -> None:
    assert distance_km("Duke Blue Devils", "Duke Blue Devils") == pytest.approx(0.0)


def test_distance_is_symmetric() -> None:
    there = distance_km("Oregon Ducks", "Miami Hurricanes")
    back = distance_km("Miami Hurricanes", "Oregon Ducks")

    assert there == pytest.approx(back)


def test_an_unknown_venue_is_none_rather_than_zero() -> None:
    """0 is "they were already there", which is a claim about a team we have
    never heard of."""
    assert distance_km("Nobody State Nobodies", "Duke Blue Devils") is None
    assert distance_km("Duke Blue Devils", "Nobody State Nobodies") is None


def test_the_distances_land_in_the_right_order_of_magnitude() -> None:
    """A coarse sanity check on the table, not a survey.

    A cross-town rivalry is tens of km, a neighbouring state is hundreds,
    and Hawai'i to New England is most of the way across the Pacific. A
    typo that put a team in the wrong hemisphere would fail here.
    """
    crosstown = distance_km("USC Trojans", "UCLA Bruins")
    instate = distance_km("Duke Blue Devils", "NC State Wolfpack")
    coast_to_coast = distance_km("Oregon Ducks", "Miami Hurricanes")
    pacific = distance_km("Hawai'i Rainbow Warriors", "UConn Huskies")

    assert crosstown is not None and crosstown < 50
    assert instate is not None and instate < 100
    assert coast_to_coast is not None and 3500 < coast_to_coast < 5000
    assert pacific is not None and 7000 < pacific < 9000


def test_every_venue_is_somewhere_plausible() -> None:
    """Latitudes and longitudes in range, and all in the western hemisphere.

    The cheapest guard against a hand-entered table: a dropped minus sign
    puts a team in Kazakhstan, and every school here is in the Americas or
    the Pacific.
    """
    for team, (lat, lon) in VENUES.items():
        assert -90 <= lat <= 90, team
        assert -180 <= lon <= 0, team
        # No FBS program is south of the equator or north of the Arctic.
        assert 15 < lat < 60, team

"""Tests for the naive starting-QB parser.

The sample sentences are real ncaafb play text, copied from
`ncaafb 2025 week 5`, because a regex written against imagined text is a
regex that works on imagined text.
"""

import pytest

from .qb import attempt_share, passer, starters


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Taylen Green pass complete to O'Mega Blake for 16 yds to the ARK 38 "
            "for a 1ST down",
            "Taylen Green",
        ),
        ("Brendon Lewis pass incomplete", "Brendon Lewis"),
        ("Brendon Lewis pass incomplete to Cortez Braham Jr.", "Brendon Lewis"),
        (
            "Taylen Green pass complete to Rohan Jones for 62 yds for a TD "
            "(Scott Starzyk KICK)",
            "Taylen Green",
        ),
    ],
)
def test_the_passer_is_the_name_before_the_verb(text: str, expected: str) -> None:
    assert passer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Sutton Smith run for 10 yds to the MEM 48 for a 1ST down",
        "Taylen Green run for a loss of 3 yards to the ARK 22",
        "",
        None,
    ],
)
def test_a_play_with_no_pass_in_it_has_no_passer(text: str | None) -> None:
    """A rush is not a quarterback sighting, even when a quarterback ran it.

    Deliberate: the point is to identify the *passer*, and counting scrambles
    would credit a wildcat back with a start.
    """
    assert passer(text) is None


def test_the_receiver_is_not_mistaken_for_the_passer() -> None:
    """ "complete to X" is the one place a second name appears."""
    text = "Carson Beck pass complete to Cortez Braham Jr. for 13 yds"

    assert passer(text) == "Carson Beck"


def test_the_starter_is_whoever_threw_the_most() -> None:
    plays = [
        ("g1", "10", "Starter Name pass incomplete"),
        ("g1", "10", "Starter Name pass complete to X for 5 yds"),
        ("g1", "10", "Backup Name pass incomplete"),
        ("g1", "20", "Other Guy pass complete to Y for 9 yds"),
        ("g1", "10", "Somebody run for 3 yds"),
    ]
    found = starters(*zip(*plays))

    assert found == {("g1", "10"): "Starter Name", ("g1", "20"): "Other Guy"}


def test_a_team_that_never_threw_is_absent() -> None:
    """ "We don't know" is not a quarterback."""
    plays = [("g1", "10", "Somebody run for 3 yds"), ("g1", "10", "Punt for 40 yds")]

    assert starters(*zip(*plays)) == {}


def test_a_play_with_no_offense_is_skipped() -> None:
    """Kickoffs and timeouts arrive with a null offense."""
    found = starters(["g1"], [None], ["Starter Name pass incomplete"])

    assert found == {}


def test_a_tie_breaks_by_name_so_a_rerun_agrees() -> None:
    plays = [
        ("g1", "10", "Bravo Passer pass incomplete"),
        ("g1", "10", "Alpha Passer pass incomplete"),
    ]

    assert starters(*zip(*plays)) == {("g1", "10"): "Alpha Passer"}


def test_attempt_share_says_how_much_to_trust_a_start() -> None:
    """30 of 32 is a starter; 9 of 17 is a quarterback controversy."""
    assert attempt_share({"A": 30, "B": 2}) == pytest.approx(30 / 32)
    assert attempt_share({"A": 9, "B": 8}) == pytest.approx(9 / 17)
    assert attempt_share({}) == 0.0

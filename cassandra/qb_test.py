"""Tests for the naive starting-QB parser.

The sample sentences are real ncaafb play text, copied from
`ncaafb 2025 week 5`, because a regex written against imagined text is a
regex that works on imagined text.
"""

import pytest

from .qb import attempt_share, ball_carrier, name_key, passer, starters, team_games


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


def test_an_even_split_goes_to_whoever_threw_first() -> None:
    plays = [
        ("g1", "10", "Bravo Passer pass incomplete"),
        ("g1", "10", "Alpha Passer pass incomplete"),
    ]

    assert starters(*zip(*plays)) == {("g1", "10"): "Bravo Passer"}


def test_a_starter_hurt_early_is_still_the_starter() -> None:
    """Three passes and off to the tent; the reliever throws thirty.

    Read the reliever as the starter and next week's flag turns inside out:
    the injured man's absence is expected and his return is "the backup
    went missing".
    """
    plays = [("g1", "10", "Starter Name pass incomplete")] * 3 + [
        ("g1", "10", "Backup Name pass incomplete")
    ] * 30

    (found,) = team_games(*zip(*plays)).values()

    assert found.starter == "Starter Name"
    assert found.share == pytest.approx(3 / 33)


def test_two_early_passes_are_not_enough_to_be_the_starter() -> None:
    """Less rigid than "the first three": a trick play and a change of
    mind on the opening drive shouldn't hand the start to a receiver."""
    plays = [("g1", "10", "Trick Play pass incomplete")] * 2 + [
        ("g1", "10", "Real Starter pass incomplete")
    ] * 30

    (found,) = team_games(*zip(*plays)).values()

    assert found.starter == "Real Starter"


def test_attempt_share_says_how_much_to_trust_a_start() -> None:
    """30 of 32 is a starter; 9 of 17 is a quarterback controversy."""
    assert attempt_share({"A": 30, "B": 2}) == pytest.approx(30 / 32)
    assert attempt_share({"A": 9, "B": 8}) == pytest.approx(9 / 17)
    assert attempt_share({}) == 0.0


# ------------------------------------------------------- the second format

# Real ncaafb text, clock-prefixed and `Last,First`. About 1.6% of plays and
# present in roughly half of games, never as a whole game.
_ALT_PASS = (
    "(05:55) Shotgun Nussmeier,Garrett pass incomplete deep left to Hilton Jr.,Chris"
)
_ALT_RUSH = "(04:47) No Huddle-Shotgun Van Buren Jr.,Michael rush left for 2 yards gain"
_ALT_SURNAME = (
    "(01:02) No Huddle-Shotgun Del Rio-Wilson,Angel pass complete short right to Y"
)


def test_the_clock_and_formation_come_off_before_the_name() -> None:
    """Or the passer is "Shotgun Nussmeier,Garrett", who does not exist."""
    assert passer(_ALT_PASS) == "Nussmeier,Garrett"
    assert ball_carrier(_ALT_RUSH) == "Van Buren Jr.,Michael"


def test_a_surname_with_spaces_survives_the_prefix_strip() -> None:
    """The formation has to come off by name, not by counting tokens."""
    assert passer(_ALT_SURNAME) == "Del Rio-Wilson,Angel"


def test_the_two_formats_normalize_to_the_same_person() -> None:
    """Without this a quarterback looks absent when he was only described
    differently, which is an injury that didn't happen."""
    assert name_key("Garrett Nussmeier") == name_key("Nussmeier,Garrett")
    assert name_key("Taylen Green") == name_key("Green,Taylen")


def test_a_suffix_is_not_part_of_the_key() -> None:
    assert name_key("Michael Penix Jr.") == name_key("Michael Penix")
    assert name_key("Emanuel, Jr.,Bert") == name_key("Bert Emanuel")
    assert name_key("JR Wilson") == "wilson j"  # a first name, not a suffix
    assert name_key("Ekleinpeter Jr.") == "ekleinpeter"


def test_the_team_is_not_a_passer() -> None:
    assert passer("TEAM pass incomplete") is None


def test_a_surname_with_spaces_is_the_same_person_in_every_format() -> None:
    assert (
        name_key("Michael Van Buren Jr.")
        == name_key("Van Buren Jr.,Michael")
        == name_key("M.Van Buren Jr.")
        == "van buren m"
    )
    assert name_key("Carlos Del Rio-Wilson") == name_key("C.Del Rio-Wilson")
    assert name_key("Keali'i Ah Yat") == name_key("K.Ah Yat")


# -------------------------------------------------------- the third format


# Real ncaafb text from 2025 week 10 on, jersey-numbered and `F.Last`. The
# names that run past one token are the ones the first cut of the pattern
# lost entirely.
@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "(15:00) No Huddle-Shotgun #7 K.Jackson pass complete short left to "
            "#3 C.Brown caught at UNC35, for 6 yards",
            "K.Jackson",
        ),
        (
            "(14:22) Shotgun #2 D.Williams Jr. pass complete short right to "
            "#12 D.Boston caught at WASH31, for 8 yards",
            "D.Williams Jr.",
        ),
        (
            "(06:26) Shotgun #11 M.Van Buren Jr. pass complete short left to "
            "#1 A.Anderson caught at LSU15",
            "M.Van Buren Jr.",
        ),
        (
            "(14:25) No Huddle-Shotgun #7 C.Del Rio-Wilson pass incomplete short "
            "right to #0 D.Tamarez thrown to MAR30",
            "C.Del Rio-Wilson",
        ),
        (
            "(12:22) No Huddle-Shotgun #8 K.Ah Yat pass incomplete short middle "
            "to #6 M.Wortham thrown to SAC20",
            "K.Ah Yat",
        ),
        (
            "(14:59) Shotgun #12 J.French IV pass complete short left to #22 O.Arnold",
            "J.French IV",
        ),
    ],
)
def test_a_numbered_name_runs_to_the_verb(text: str, expected: str) -> None:
    assert passer(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "(05:40) Shotgun #11 M.Van Buren Jr. rush left for 3 yards gain to the LSU19",
            "M.Van Buren Jr.",
        ),
        (
            "(04:58) Shotgun #12 J.French IV sacked for loss of 3 yards to the GSO24",
            "J.French IV",
        ),
        (
            "No Huddle-Shotgun #7 A.St. Louis rush middle for 9 yards gain",
            "A.St. Louis",
        ),
    ],
)
def test_a_numbered_carrier_runs_to_the_verb(text: str, expected: str) -> None:
    assert ball_carrier(text) == expected


def test_a_numbered_receiver_is_not_the_passer() -> None:
    """The lazy tail stops at the first verb, so `to #12 D.Boston` is never
    pulled into the passer's name."""
    text = "(14:22) Shotgun #2 D.Williams Jr. pass complete short right to #12 D.Boston"
    thrower = passer(text)

    assert thrower is not None
    assert name_key(thrower) == "williams d"


def test_the_nfl_feed_keeps_a_suffix_too() -> None:
    assert (
        passer("(Shotgun) A.Richardson Sr. pass short middle to J.Downs")
        == "A.Richardson Sr."
    )
    assert (
        ball_carrier("K.Walker III up the middle to SEA 40 for 5 yards")
        == "K.Walker III"
    )


def test_a_sack_counts_as_a_snap_taken() -> None:
    """It is not a rushing attempt, and availability isn't asking about that."""
    assert (
        ball_carrier("Brendon Lewis sacked by Quincy Rhodes Jr. for a loss of 9 yards")
        == "Brendon Lewis"
    )


def test_a_kneel_down_names_nobody() -> None:
    assert ball_carrier("(00:06) Kneel down by Southeastern La. at SLU20") is None


def test_snap_keys_hold_everyone_who_touched_it() -> None:
    """Runners included -- the question is whether a given man played."""
    plays = [
        ("g1", "10", "Starter Name pass complete to X for 5 yds"),
        ("g1", "10", "Runner Person run for 12 yds"),
        ("g1", "10", "Starter Name sacked by Somebody Else for a loss of 4 yards"),
    ]

    (found,) = team_games(*zip(*plays)).values()

    assert found.starter == "Starter Name"
    assert found.snap_keys == {name_key("Starter Name"), name_key("Runner Person")}
    assert found.attempts == 1
    assert found.share == 1.0

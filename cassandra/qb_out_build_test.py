"""Tests for deriving availability from parsed quarterback data."""

from cassandra.qb import TeamGameQb
from cassandra.qb_out_build import out_teams


def _qb(starter: str, snaps: tuple[str, ...]) -> TeamGameQb:
    return TeamGameQb(
        starter=starter,
        starter_key=starter,
        attempts=20,
        share=1.0,
        snap_keys=frozenset(snaps),
    )


def test_a_starter_who_took_no_snap_is_out() -> None:
    parsed = {
        ("g1", "A"): _qb("ace", ("ace", "rb")),
        ("g2", "A"): _qb("backup", ("backup", "rb")),
    }

    assert out_teams([("g1", "A"), ("g2", "A")], parsed) == {"g2": {"A"}}


def test_a_starter_who_played_but_lost_the_job_is_not_out() -> None:
    """A benching mid-game is a quarterback change, not an absence.

    The distinction is the whole point: the model is asking who was
    available, and a man who threw eight passes was available.
    """
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        # ace still took snaps, backup just threw more of them
        ("g2", "A"): _qb("backup", ("backup", "ace")),
    }

    assert out_teams([("g1", "A"), ("g2", "A")], parsed) == {}


def test_the_first_game_of_a_season_has_no_expected_starter() -> None:
    """ "Is he missing?" needs somebody to be missing."""
    parsed = {("g1", "A"): _qb("ace", ("ace",))}

    assert out_teams([("g1", "A")], parsed) == {}


def test_a_game_with_no_passing_at_all_is_not_an_absence() -> None:
    """Otherwise every wildcat afternoon reads as an injury."""
    parsed = {("g1", "A"): _qb("ace", ("ace",))}

    assert out_teams([("g1", "A"), ("g2", "A")], parsed) == {}


def test_the_expected_starter_rolls_forward() -> None:
    """After a change, the new man is who the next game expects."""
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("backup", ("backup",)),  # ace out
        ("g3", "A"): _qb("backup", ("backup",)),  # settled, nobody out
    }

    assert out_teams([("g1", "A"), ("g2", "A"), ("g3", "A")], parsed) == {"g2": {"A"}}


def test_two_teams_are_tracked_apart() -> None:
    parsed = {
        ("g1", "A"): _qb("ace_a", ("ace_a",)),
        ("g1", "B"): _qb("ace_b", ("ace_b",)),
        ("g2", "A"): _qb("sub_a", ("sub_a",)),
        ("g2", "B"): _qb("ace_b", ("ace_b",)),
    }
    ordered = [("g1", "A"), ("g2", "A")]

    assert out_teams(ordered, parsed) == {"g2": {"A"}}
    assert out_teams([("g1", "B"), ("g2", "B")], parsed) == {}

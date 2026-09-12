"""Tests for deriving availability from parsed quarterback data."""

from cassandra.qb import TeamGameQb
from cassandra.qb_out_build import home_away_ids, out_teams


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

    assert out_teams([("g1", "A"), ("g2", "A")], parsed, 3) == {"g2": {"A"}}


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

    assert out_teams([("g1", "A"), ("g2", "A")], parsed, 3) == {}


def test_the_first_game_of_a_season_has_no_expected_starter() -> None:
    """ "Is he missing?" needs somebody to be missing."""
    parsed = {("g1", "A"): _qb("ace", ("ace",))}

    assert out_teams([("g1", "A")], parsed, 3) == {}


def test_a_game_with_no_passing_at_all_is_not_an_absence() -> None:
    """Otherwise every wildcat afternoon reads as an injury."""
    parsed = {("g1", "A"): _qb("ace", ("ace",))}

    assert out_teams([("g1", "A"), ("g2", "A")], parsed, 3) == {}


def test_every_game_of_an_absence_is_an_absence() -> None:
    """Not just the first. The starter is still the starter while he is out."""
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("backup", ("backup",)),  # ace out
        ("g3", "A"): _qb("backup", ("backup",)),  # still out
    }

    assert out_teams([("g1", "A"), ("g2", "A"), ("g3", "A")], parsed, 3) == {
        "g2": {"A"},
        "g3": {"A"},
    }


def test_the_starter_coming_back_is_not_the_backup_being_out() -> None:
    """The week after an absence, the man who covered sits back down.

    The one-game rule read that as a second absence -- the backup started
    the previous game, so he was the expected starter -- and penalized the
    team for getting its quarterback back.
    """
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("backup", ("backup",)),  # ace out
        ("g3", "A"): _qb("ace", ("ace",)),  # ace back, backup sits
    }

    assert out_teams([("g1", "A"), ("g2", "A"), ("g3", "A")], parsed, 3) == {
        "g2": {"A"}
    }


def _settled(starter: str = "ace", weeks: int = 4) -> dict[tuple[str, str], TeamGameQb]:
    """A starter with the job for `weeks` games, so seniority is not in
    question and the handover is what a test exercises."""
    return {(f"g{n}", "A"): _qb(starter, (starter,)) for n in range(1, weeks + 1)}


def test_the_job_is_handed_over_after_three_straight_starts() -> None:
    """A starter gone for the year stops costing his team once the man
    behind him has held the job for three games."""
    parsed = _settled()
    parsed.update({(f"g{n}", "A"): _qb("backup", ("backup",)) for n in range(5, 10)})
    ordered = [(f"g{n}", "A") for n in range(1, 10)]

    assert out_teams(ordered, parsed, 3) == {f"g{n}": {"A"} for n in (5, 6, 7)}


def test_the_handover_needs_one_man_in_a_row() -> None:
    """Two backups alternating are a team without a starter, not a new one."""
    parsed = _settled()
    parsed.update(
        {
            ("g5", "A"): _qb("backup", ("backup",)),
            ("g6", "A"): _qb("backup", ("backup",)),
            ("g7", "A"): _qb("third", ("third",)),
            ("g8", "A"): _qb("backup", ("backup",)),
        }
    )
    ordered = [(f"g{n}", "A") for n in range(1, 9)]

    assert out_teams(ordered, parsed, 3) == {f"g{n}": {"A"} for n in range(5, 9)}


def test_a_snap_by_the_starter_restarts_the_count() -> None:
    parsed = _settled()
    parsed.update(
        {
            ("g5", "A"): _qb("backup", ("backup",)),
            ("g6", "A"): _qb("backup", ("backup",)),
            ("g7", "A"): _qb("backup", ("backup", "ace")),  # ace back, in relief
            ("g8", "A"): _qb("backup", ("backup",)),
            ("g9", "A"): _qb("backup", ("backup",)),
        }
    )
    ordered = [(f"g{n}", "A") for n in range(1, 10)]

    assert out_teams(ordered, parsed, 3) == {f"g{n}": {"A"} for n in (5, 6, 8, 9)}


def test_a_second_absence_is_an_absence() -> None:
    """The backup having covered before does not make him the starter."""
    parsed = _settled()
    parsed.update(
        {
            ("g5", "A"): _qb("backup", ("backup",)),
            ("g6", "A"): _qb("ace", ("ace",)),
            ("g7", "A"): _qb("backup", ("backup",)),
        }
    )
    ordered = [(f"g{n}", "A") for n in range(1, 8)]

    assert out_teams(ordered, parsed, 3) == {"g5": {"A"}, "g7": {"A"}}


def test_the_backup_can_be_missing_too() -> None:
    """Once the job is his, the backup is the man expected, and a third
    quarterback covering for him is an absence."""
    parsed = _settled()
    parsed.update({(f"g{n}", "A"): _qb("backup", ("backup",)) for n in range(5, 8)})
    parsed[("g8", "A")] = _qb("third", ("third",))  # backup out
    parsed[("g9", "A")] = _qb("backup", ("backup",))  # backup back
    ordered = [(f"g{n}", "A") for n in range(1, 10)]

    assert out_teams(ordered, parsed, 3) == {f"g{n}": {"A"} for n in (5, 6, 7, 8)}


def test_an_early_absence_hands_over_sooner() -> None:
    """One start to the starter's name and two to the cover's, and the
    cover is the more established man: the handover is a ceiling, not the
    only way the job changes hands."""
    parsed = {("g1", "A"): _qb("ace", ("ace",))}
    parsed.update({(f"g{n}", "A"): _qb("backup", ("backup",)) for n in range(2, 6)})
    ordered = [(f"g{n}", "A") for n in range(1, 6)]

    assert out_teams(ordered, parsed, 3) == {"g2": {"A"}, "g3": {"A"}}


def test_two_teams_are_tracked_apart() -> None:
    parsed = {
        ("g1", "A"): _qb("ace_a", ("ace_a",)),
        ("g1", "B"): _qb("ace_b", ("ace_b",)),
        ("g2", "A"): _qb("sub_a", ("sub_a",)),
        ("g2", "B"): _qb("ace_b", ("ace_b",)),
    }
    ordered = [("g1", "A"), ("g2", "A")]

    assert out_teams(ordered, parsed, 3) == {"g2": {"A"}}
    assert out_teams([("g1", "B"), ("g2", "B")], parsed, 3) == {}


def test_the_established_man_starting_again_is_not_an_absence() -> None:
    """One start apiece after two games: the opener's starter is the
    coach's pick, and his return is not the other man's absence. Found by
    hand -- Illinois State, 2025, the starter hurt in the opener."""
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("sub", ("sub",)),  # ace out; sub becomes the starter
        ("g3", "A"): _qb("ace", ("ace",)),  # ace back, sub sits
    }

    assert out_teams([("g1", "A"), ("g2", "A"), ("g3", "A")], parsed, 1) == {
        "g2": {"A"}
    }


def test_a_college_change_of_starter_is_one_flag() -> None:
    """With the handover at one game, the man who takes over is the
    starter from his second start on: a settled battle is not an injury."""
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("ace", ("ace",)),
        ("g3", "A"): _qb("sub", ("sub",)),
        ("g4", "A"): _qb("sub", ("sub",)),
    }
    ordered = [(g, "A") for g in ("g1", "g2", "g3", "g4")]

    assert out_teams(ordered, parsed, 1) == {"g3": {"A"}}


def test_a_more_established_man_taking_over_is_not_an_absence() -> None:
    """Even mid-absence: three starts to one, and the man with three
    starting again means the cover sat, not that anybody is missing."""
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("ace", ("ace",)),
        ("g3", "A"): _qb("ace", ("ace",)),
        ("g4", "A"): _qb("sub", ("sub",)),  # ace out
        ("g5", "A"): _qb("sub", ("sub",)),  # sub is now the starter (h=1)
        ("g6", "A"): _qb("ace", ("ace",)),  # ace back, and the senior man
    }
    ordered = [(f"g{n}", "A") for n in range(1, 7)]

    assert out_teams(ordered, parsed, 1) == {"g4": {"A"}}


# ------------------------------------------------------ reading the sides


def test_the_home_side_is_voted_from_the_scoring() -> None:
    """The registry can't do this: `TeamNamer.for_league("nfl")` is empty."""
    # Home (id 10) scores first, then the away side (id 20) does.
    offense = ["10", "10", "20", "20"]
    home = [0, 7, 7, 7]
    away = [0, 0, 0, 7]

    assert home_away_ids(offense, home, away) == ("10", "20")


def test_a_defensive_score_is_outvoted_rather_than_special_cased() -> None:
    """A pick six credits the score to a play whose offense is the other
    team. Majority absorbs it without anyone having to know which plays
    those were."""
    # Scores after each play. The home side (10) scores twice on offense; in
    # between, the away side's defence takes one back while 10 has the ball,
    # which lands a vote *against* 10 on a play it ran.
    offense = ["10", "10", "10", "10", "20"]
    home = [0, 7, 7, 14, 14]
    away = [0, 0, 7, 7, 14]

    assert home_away_ids(offense, home, away) == ("10", "20")


def test_a_game_with_nothing_to_vote_on_places_nobody() -> None:
    """Better to skip the game than to guess which side was at home."""
    assert home_away_ids(["10", "20"], [0, 0], [0, 0]) == (None, None)
    assert home_away_ids([], [], []) == (None, None)


def test_a_missing_score_column_is_skipped_not_counted() -> None:
    offense = ["10", "10", "20"]
    home = [None, 7, 7]
    away = [None, 0, 7]

    assert home_away_ids(offense, home, away) == ("10", "20")

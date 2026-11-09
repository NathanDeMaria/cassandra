"""Tests for deriving availability from parsed quarterback data."""

import asyncio
from datetime import UTC, datetime

import pyarrow as pa
from call_it_what_you_want import TeamNamer
from endgame.types import Game, Season, Week
from say_youll_remember_me.qb import TeamGameQb

from cassandra.qb_out_build import (
    build_all,
    game_qbs,
    home_away_ids,
    out_teams,
    registry_ids,
    with_season,
)


def _qb(starter: str, snaps: tuple[str, ...]) -> TeamGameQb:
    return TeamGameQb(
        starter=starter,
        starter_key=starter,
        attempts=20,
        share=1.0,
        snap_keys=frozenset(snaps),
    )


def _flagged(
    ordered: list[tuple[str, str]],
    parsed: dict[tuple[str, str], TeamGameQb],
    handover: int,
    keep_unseen: bool = False,
) -> dict[str, set[str]]:
    """`out_teams` read as who was out, without which game of it each was."""
    return {
        game: set(teams)
        for game, teams in out_teams(ordered, parsed, handover, keep_unseen).items()
    }


def test_a_starter_who_took_no_snap_is_out() -> None:
    parsed = {
        ("g1", "A"): _qb("ace", ("ace", "rb")),
        ("g2", "A"): _qb("backup", ("backup", "rb")),
    }

    assert _flagged([("g1", "A"), ("g2", "A")], parsed, 3) == {"g2": {"A"}}


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

    assert _flagged([("g1", "A"), ("g2", "A")], parsed, 3) == {}


def test_the_first_game_of_a_season_has_no_expected_starter() -> None:
    """ "Is he missing?" needs somebody to be missing."""
    parsed = {("g1", "A"): _qb("ace", ("ace",))}

    assert _flagged([("g1", "A")], parsed, 3) == {}


def test_a_game_with_no_passing_at_all_is_not_an_absence() -> None:
    """Otherwise every wildcat afternoon reads as an injury."""
    parsed = {("g1", "A"): _qb("ace", ("ace",))}

    assert _flagged([("g1", "A"), ("g2", "A")], parsed, 3) == {}


def test_every_game_of_an_absence_is_an_absence() -> None:
    """Not just the first. The starter is still the starter while he is out."""
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("backup", ("backup",)),  # ace out
        ("g3", "A"): _qb("backup", ("backup",)),  # still out
    }

    assert _flagged([("g1", "A"), ("g2", "A"), ("g3", "A")], parsed, 3) == {
        "g2": {"A"},
        "g3": {"A"},
    }


def test_each_flag_says_which_game_of_the_absence_it_was() -> None:
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("backup", ("backup",)),
        ("g3", "A"): _qb("backup", ("backup",)),
        ("g4", "A"): _qb("ace", ("ace",)),
        ("g5", "A"): _qb("backup", ("backup",)),
    }
    ordered = [(f"g{n}", "A") for n in range(1, 6)]

    assert out_teams(ordered, parsed, 3) == {
        "g2": {"A": 1},
        "g3": {"A": 2},
        "g5": {"A": 1},
    }


def test_an_unseen_starter_keeps_the_job_through_his_absence() -> None:
    """Hurt, not benched: with `keep_unseen` a one-game handover waits for
    him to be seen, so every game he misses is flagged."""
    parsed = _settled()
    parsed.update({(f"g{n}", "A"): _qb("backup", ("backup",)) for n in (5, 6, 7)})
    ordered = [(f"g{n}", "A") for n in range(1, 8)]

    assert _flagged(ordered, parsed, 1) == {"g5": {"A"}}
    assert out_teams(ordered, parsed, 1, keep_unseen=True) == {
        "g5": {"A": 1},
        "g6": {"A": 2},
        "g7": {"A": 3},
    }


def test_a_starter_seen_on_the_bench_hands_the_job_over() -> None:
    """In relief one week and gone the next reads as a benching: the cover
    has the job, and the game after is nobody's absence."""
    parsed = _settled()
    parsed.update(
        {
            ("g5", "A"): _qb("backup", ("backup", "ace")),  # ace in relief
            ("g6", "A"): _qb("backup", ("backup",)),  # ace gone: benched
            ("g7", "A"): _qb("backup", ("backup",)),
        }
    )
    ordered = [(f"g{n}", "A") for n in range(1, 8)]

    assert out_teams(ordered, parsed, 1, keep_unseen=True) == {"g6": {"A": 1}}


def test_a_cover_who_out_starts_him_takes_the_job_anyway() -> None:
    """Out for the season: once the man covering has started more games than
    the starter did, the team is his. A tie goes to the earlier start."""
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("backup", ("backup",)),
        ("g3", "A"): _qb("backup", ("backup",)),
        ("g4", "A"): _qb("backup", ("backup",)),
    }
    ordered = [(f"g{n}", "A") for n in range(1, 5)]

    assert out_teams(ordered, parsed, 1, keep_unseen=True) == {
        "g2": {"A": 1},
        "g3": {"A": 2},
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

    assert _flagged([("g1", "A"), ("g2", "A"), ("g3", "A")], parsed, 3) == {"g2": {"A"}}


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

    assert _flagged(ordered, parsed, 3) == {f"g{n}": {"A"} for n in (5, 6, 7)}


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

    assert _flagged(ordered, parsed, 3) == {f"g{n}": {"A"} for n in range(5, 9)}


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

    assert _flagged(ordered, parsed, 3) == {f"g{n}": {"A"} for n in (5, 6, 8, 9)}


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

    assert _flagged(ordered, parsed, 3) == {"g5": {"A"}, "g7": {"A"}}


def test_the_backup_can_be_missing_too() -> None:
    """Once the job is his, the backup is the man expected, and a third
    quarterback covering for him is an absence."""
    parsed = _settled()
    parsed.update({(f"g{n}", "A"): _qb("backup", ("backup",)) for n in range(5, 8)})
    parsed[("g8", "A")] = _qb("third", ("third",))  # backup out
    parsed[("g9", "A")] = _qb("backup", ("backup",))  # backup back
    ordered = [(f"g{n}", "A") for n in range(1, 10)]

    assert _flagged(ordered, parsed, 3) == {f"g{n}": {"A"} for n in (5, 6, 7, 8)}


def test_an_early_absence_hands_over_sooner() -> None:
    """One start to the starter's name and two to the cover's, and the
    cover is the more established man: the handover is a ceiling, not the
    only way the job changes hands."""
    parsed = {("g1", "A"): _qb("ace", ("ace",))}
    parsed.update({(f"g{n}", "A"): _qb("backup", ("backup",)) for n in range(2, 6)})
    ordered = [(f"g{n}", "A") for n in range(1, 6)]

    assert _flagged(ordered, parsed, 3) == {"g2": {"A"}, "g3": {"A"}}


def test_two_teams_are_tracked_apart() -> None:
    parsed = {
        ("g1", "A"): _qb("ace_a", ("ace_a",)),
        ("g1", "B"): _qb("ace_b", ("ace_b",)),
        ("g2", "A"): _qb("sub_a", ("sub_a",)),
        ("g2", "B"): _qb("ace_b", ("ace_b",)),
    }
    ordered = [("g1", "A"), ("g2", "A")]

    assert _flagged(ordered, parsed, 3) == {"g2": {"A"}}
    assert _flagged([("g1", "B"), ("g2", "B")], parsed, 3) == {}


def test_the_established_man_starting_again_is_not_an_absence() -> None:
    """One start apiece after two games: the opener's starter is the
    coach's pick, and his return is not the other man's absence. Found by
    hand -- Illinois State, 2025, the starter hurt in the opener."""
    parsed = {
        ("g1", "A"): _qb("ace", ("ace",)),
        ("g2", "A"): _qb("sub", ("sub",)),  # ace out; sub becomes the starter
        ("g3", "A"): _qb("ace", ("ace",)),  # ace back, sub sits
    }

    assert _flagged([("g1", "A"), ("g2", "A"), ("g3", "A")], parsed, 1) == {"g2": {"A"}}


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

    assert _flagged(ordered, parsed, 1) == {"g3": {"A"}}


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

    assert _flagged(ordered, parsed, 1) == {"g4": {"A"}}


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


def test_a_fresh_season_replaces_only_its_own_games() -> None:
    def game(game_id: str) -> Game:
        return Game(
            home="Ohio",
            home_score=0,
            away="Kent",
            away_score=0,
            neutral_site=False,
            completed=True,
            date=datetime(2026, 10, 17, tzinfo=UTC),
            game_id=game_id,
        )

    season = Season([Week([game("new1"), game("new2")], 1)], 2026)
    stored = {"old": {"Kent"}, "new1": {"Ohio"}, "new2": {"Kent"}}
    # The fresh build re-read new1 as nobody out, and flags new2 for Ohio.
    merged = with_season(stored, season, {"new2": {"Ohio"}})
    assert merged == {"old": {"Kent"}, "new2": {"Ohio"}}
    # The runs merge the same way.
    runs = with_season(
        {"old": {"Kent": 2}, "new1": {"Ohio": 1}}, season, {"new2": {"Ohio": 3}}
    )
    assert runs == {"old": {"Kent": 2}, "new2": {"Ohio": 3}}


def test_game_qbs_counts_passes_and_the_last_snap_per_man() -> None:
    """South Florida on Oct 31: the starter's last snap came early in a game
    his team trailed, and the man after him threw the rest."""
    plays = [
        # (offense, text, period, home score, away score)
        ("1", "Alan Ace pass complete to Rex Wide for 16 yds", 1, 0, 0),
        ("2", "Kay Kicker 30 yd FG GOOD", 1, 0, 3),
        ("1", "Alan Ace run for 5 yds", 1, 0, 3),
        ("2", "Dee Other pass complete to Tee End for 20 yds for a TD", 1, 0, 10),
        ("1", "Ben Backup pass incomplete to Rex Wide", 2, 0, 10),
        ("1", "Ben Backup pass complete to Rex Wide for 75 yds for a TD", 3, 7, 10),
    ]
    offense, texts, periods, home, away = map(list, zip(*plays))
    parsed = {
        "1": TeamGameQb(
            "Alan Ace", "ace a", 3, 1 / 3, frozenset({"ace a", "backup b"})
        ),
        "2": TeamGameQb("Dee Other", "other d", 1, 1.0, frozenset({"other d"})),
    }

    qbs = game_qbs(offense, texts, periods, home, away, "1", parsed)

    assert qbs["1"].starter == "ace a"
    assert qbs["1"].attempts == {"ace a": 1, "backup b": 2}
    assert qbs["1"].names == {"ace a": "Alan Ace", "backup b": "Ben Backup"}
    assert qbs["1"].snaps == ["ace a", "backup b"]
    # Home: down 3 at his run, from the score before it.
    assert qbs["1"].last_snap == {"ace a": (1, -3), "backup b": (3, -3)}
    assert qbs["2"].last_snap == {"other d": (1, 10)}


class _Plays:
    def __init__(self, weeks: dict[tuple[int, int], pa.Table]) -> None:
        self.weeks = weeks

    async def load_week(self, league: str, season: int, week: int) -> pa.Table:
        return self.weeks.get((season, week), pa.table({}))


def _table(game_id: str, plays: list[tuple[str, str, int, int]]) -> pa.Table:
    """(offense id, text, home score, away score); home is team 1."""
    return pa.table(
        {
            "game_id": [game_id] * len(plays),
            "sequence_number": list(range(len(plays))),
            "offense_team_id": [p[0] for p in plays],
            "text": [p[1] for p in plays],
            "period": [1] * len(plays),
            "home_score": [p[2] for p in plays],
            "away_score": [p[3] for p in plays],
        }
    )


def test_build_all_keeps_the_latest_seasons_quarterbacks() -> None:
    def season(year: int, game_id: str) -> Season:
        game = Game(
            home="Ohio",
            home_score=7,
            away="Kent",
            away_score=0,
            neutral_site=False,
            completed=True,
            date=datetime(year, 10, 17, tzinfo=UTC),
            game_id=game_id,
        )
        return Season([Week([game], 1)], year)

    # The touchdown is the vote that team 1 is home.
    plays = [
        ("1", "Alan Ace run for 3 yds", 0, 0),
        ("1", "Alan Ace pass complete to Rex Wide for 7 yds for a TD", 7, 0),
        ("2", "Dee Other pass incomplete to Tee End", 7, 0),
    ]
    weeks = {(2025, 1): _table("old", plays), (2026, 1): _table("new", plays)}
    namer = TeamNamer(None)

    built = asyncio.run(
        build_all(
            "ncaafb", [season(2025, "old"), season(2026, "new")], _Plays(weeks), namer
        )
    )

    assert built.missing == {}
    assert built.starts is not None
    assert built.starts.season == 2026
    assert built.starts.games.keys() == {"new"}
    assert built.starts.games["new"]["Ohio"].starter == "ace a"
    assert built.starts.games["new"]["Kent"].attempts == {"other d": 1}


def test_the_registry_places_the_teams_when_it_knows_both() -> None:
    namer = TeamNamer.for_league("ncaafb")
    offense = ["97", "24", "97"]
    assert registry_ids(
        namer, "Louisville Cardinals", "Stanford Cardinal", offense
    ) == (
        "97",
        "24",
    )
    # A school it can't place, or one that ran no play here, leaves the vote.
    assert registry_ids(namer, "Nowhere Tech", "Stanford Cardinal", offense) is None
    assert (
        registry_ids(namer, "Louisville Cardinals", "Ohio State Buckeyes", offense)
        is None
    )
    assert (
        registry_ids(
            TeamNamer(None), "Louisville Cardinals", "Stanford Cardinal", offense
        )
        is None
    )

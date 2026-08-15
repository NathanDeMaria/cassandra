from datetime import datetime

from call_it_what_you_want import NCAAFB, default_teams, record, teams_from_csv
from endgame.types import Game

from .aliases import TeamNamer

# Two observations of one team under two names, plus a second team that
# shares a nickname with nobody, plus a third that collides with the first
# team's old name.
_CSV = [
    "espn_id,name,year,source",
    "1,Old Name Tigers,2012,espn",
    "1,New Name Tigers,2025,espn",
    "2,Rival Bears,2025,espn",
]


def _namer(*rows: str) -> TeamNamer:
    return TeamNamer(teams_from_csv([*_CSV, *rows]))


def _game(home: str, away: str) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=1,
        away_score=0,
        neutral_site=False,
        completed=True,
        date=datetime(2023, 1, 1),
        game_id="1",
    )


def test_an_old_name_resolves_to_the_current_one() -> None:
    assert _namer().canonical("Old Name Tigers") == "New Name Tigers"


def test_the_current_name_stays_put() -> None:
    assert _namer().canonical("New Name Tigers") == "New Name Tigers"


def test_an_unknown_name_passes_through_and_is_counted() -> None:
    """A replay can't stop because the registry has a gap."""
    namer = _namer()
    assert namer.canonical("Never Heard Of Them") == "Never Heard Of Them"
    assert namer.unknown["Never Heard Of Them"] == 1


def test_an_ambiguous_name_passes_through_and_is_counted() -> None:
    """Two schools sharing a name is normal; renaming would merge them."""
    namer = _namer("3,Old Name Tigers,2019,espn")
    assert namer.canonical("Old Name Tigers") == "Old Name Tigers"
    assert namer.ambiguous["Old Name Tigers"] == 1


def test_apply_renames_both_sides() -> None:
    renamed = _namer().apply(_game("Old Name Tigers", "Rival Bears"))
    assert (renamed.home, renamed.away) == ("New Name Tigers", "Rival Bears")
    assert renamed.game_id == "1"
    assert renamed.home_score == 1


def test_apply_returns_the_same_game_when_nothing_changes() -> None:
    game = _game("New Name Tigers", "Rival Bears")
    assert _namer().apply(game) is game


def test_apply_refuses_to_make_a_team_play_itself() -> None:
    """Two programs under one ESPN id would turn a real game into nonsense."""
    namer = _namer("2,Rival Bears Old,2011,espn")
    game = _game("Rival Bears", "Rival Bears Old")
    assert namer.apply(game) is game
    assert namer.ambiguous


def test_empty_namer_renames_nothing() -> None:
    namer = TeamNamer.empty()
    assert namer.canonical("Anyone") == "Anyone"
    assert not namer


def test_a_league_with_no_registry_renames_nothing() -> None:
    namer = TeamNamer.for_league("no-such-league")
    assert namer.canonical("Anyone") == "Anyone"


def test_a_recorded_name_is_usable_immediately(tmp_path, monkeypatch) -> None:
    """A rename found today, usable today, without waiting on a release.

    cassandra keeps no copy of the registry: `record` stages the row in the
    package's own local file and `default_teams` layers it on, so
    `for_league` picks it up with nothing else to wire.
    """
    monkeypatch.setenv("CIWYW_DATA_DIR", str(tmp_path))
    default_teams.cache_clear()
    try:
        # Army (349) is in the package's bundled data.
        record("349", "Army Cadets", 2005, league=NCAAFB)
        namer = TeamNamer.for_league("ncaafb")
        assert namer.canonical("Army Cadets") == namer.canonical("Army Black Knights")
    finally:
        default_teams.cache_clear()


# One school, two sports, two names in the same season -- which is the case
# a registry keyed by id and scoped by league exists to handle.
_UNLV = [
    "espn_id,name,year,source,league,same_as",
    "2439,UNLV Rebels,2025,espn,ncaafb,",
    "2439,UNLV Lady Rebels,2025,espn,ncaawbb,",
]


def test_a_name_is_answered_in_the_league_that_asked() -> None:
    teams = teams_from_csv(_UNLV)
    football = TeamNamer(teams, "ncaafb")
    womens = TeamNamer(teams, "womens")

    # Either spelling finds the school; the answer depends on who's asking.
    assert football.canonical("UNLV Lady Rebels") == "UNLV Rebels"
    assert womens.canonical("UNLV Rebels") == "UNLV Lady Rebels"


def test_a_league_with_no_name_on_record_leaves_it_alone() -> None:
    """Football has no name for a basketball-only team, and won't invent one."""
    namer = TeamNamer(teams_from_csv(_UNLV[:1] + [_UNLV[2]]), "ncaafb")
    assert namer.canonical("UNLV Lady Rebels") == "UNLV Lady Rebels"
    assert namer.ambiguous["UNLV Lady Rebels"] == 1


def test_the_nfl_does_not_borrow_college_ids() -> None:
    """ESPN id 2 is both the Buffalo Bills and the Auburn Tigers."""
    namer = TeamNamer.for_league("nfl")
    assert not namer
    assert namer.canonical("Buffalo Bills") == "Buffalo Bills"

"""The control sweep, against plays built here rather than against a bucket.

Which number comes off a game is what this checks; that the sweep walks the
weeks and drops the games it should is `pbp_sweep`'s half and is checked
through here too, since this is the build that has been run for real. The
plays come from `cassandra.conftest` -- see there for why they look the way
they do.
"""

import asyncio

import pytest
from endgame.types import Season, Week
from lucky_ones import MODELS, group_by_game

from cassandra.conftest import (
    LEAGUE as _LEAGUE,
)
from cassandra.conftest import (
    FakeSource as _Source,
)
from cassandra.conftest import (
    game_for as _game,
)
from cassandra.conftest import (
    plays_for as _plays,
)
from cassandra.conftest import (
    season_for as _season,
)
from cassandra.game_control_build import (
    build,
    current_fit,
    sweep,
    weeks_in,
    write,
)
from cassandra.predictor.game_control import (
    ControlFit,
    load_game_control,
    read_game_control_file,
)

_LEAGUE = "nfl"
_HOME_ID, _AWAY_ID = "100", "200"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Point the artifact at tmp_path, and clear the cached read around it.

    Both bindings, not just the one: `game_control_build` imports the name
    rather than the module, so patching only the definition would leave the
    build writing into a real `~/.cassandra` while the reads came from
    tmp_path -- which passes some of these and corrupts a developer's disk.
    """
    for module in (
        "cassandra.predictor.game_control",
        "cassandra.game_control_build",
    ):
        monkeypatch.setattr(
            f"{module}.game_control_path",
            lambda league: tmp_path / f"{league}_game_control.json",
        )
    load_game_control.cache_clear()
    yield
    load_game_control.cache_clear()


def test_weeks_come_out_in_the_order_a_season_runs() -> None:
    seasons = [_season(2025, "a", week=2), _season(2024, "b", week=1)]

    assert weeks_in(seasons) == [(2024, 1), (2025, 2)]


def test_a_full_game_gets_control_covering_regulation() -> None:
    source = _Source({(2025, 1): _plays("g1")})

    control, stats = asyncio.run(sweep(_LEAGUE, [_season(2025, "g1")], source))

    assert set(control) == {"g1"}
    # The home team led from the first score on, so it controlled most of it.
    assert control["g1"].home > 0.5
    assert stats.scored == 1
    assert stats.played == 1
    assert stats.disagreed == 0


def test_the_number_is_the_luck_adjusted_reading() -> None:
    """Which of the two readings the sweep takes, which is the whole point.

    Asserted against `lucky_ones` on the same game rather than by direction,
    for the reason the module docstring gives: these fixtures are a snap
    every thirty seconds at midfield, and what a split fumble is worth in a
    game like that is the model's business. What is this module's business is
    that the number written is the adjusted one -- and on a game with a
    bounce in it the two readings differ, so the second assertion is what
    would catch a quiet revert to `game_control`.
    """
    plays = _plays("g1", fumble_on=4)
    game = next(iter(group_by_game(plays)))
    model = MODELS[_LEAGUE]
    adjusted = model.luck_adjusted_game_control(game)
    realized = model.game_control(game)
    assert adjusted is not None and realized is not None

    control, _ = asyncio.run(
        sweep(_LEAGUE, [_season(2025, "g1")], _Source({(2025, 1): plays}))
    )

    assert control["g1"].home == pytest.approx(adjusted.home)
    assert control["g1"].home != pytest.approx(realized.home)


def test_a_game_whose_plays_stop_early_is_left_out() -> None:
    """A half-covered curve is a well-formed number about a different game.

    And the clock can't tell you it happened: this game's plays stop at the
    half, and `game_control` still reports a full 3600 seconds, because the
    last snap is charged with everything left in regulation. The plays
    reaching 14-0 against a game that finished 28-0 is what gives it away.
    """
    source = _Source({(2025, 1): _plays("g1", periods=(1, 2))})

    control, stats = asyncio.run(sweep(_LEAGUE, [_season(2025, "g1")], source))

    assert control == {}
    assert stats.incomplete == 1
    assert stats.scored == 0


def test_plays_for_a_game_the_schedule_lacks_are_skipped() -> None:
    """A game still in progress when the plays were pulled looks like this."""
    source = _Source({(2025, 1): _plays("not_scheduled")})

    control, stats = asyncio.run(sweep(_LEAGUE, [_season(2025, "g1")], source))

    assert control == {}
    assert stats.unmatched == 1


def test_a_week_with_no_plays_is_not_an_error() -> None:
    """Most of an NCAAFB week, and every week of a season not yet played."""
    control, stats = asyncio.run(
        sweep(_LEAGUE, [_season(2025, "g1")], _Source({}))
    )

    assert control == {}
    assert stats.played == 1


def test_plays_that_contradict_the_stored_result_are_counted_and_dropped() -> None:
    """See `SweepStats.disagreed`.

    The two are supposed to agree by construction, both being ESPN's own
    home/away labelling, so a count above zero means an assumption this code
    rests on has stopped holding. Until someone has looked at why, the game
    doesn't get a control number -- flipping it would be guessing.
    """
    source = _Source({(2025, 1): _plays("g1", home_leads=False)})
    season = Season(
        year=2025, weeks=[Week(games=[_game("g1", 28, 0)], number=1)]
    )

    control, stats = asyncio.run(sweep(_LEAGUE, [season], source))

    assert stats.disagreed == 1
    assert control == {}


def test_a_first_build_writes_the_fit_that_made_it() -> None:
    source = _Source({(2025, 1): _plays("g1")})

    asyncio.run(build(_LEAGUE, [_season(2025, "g1")], source))

    stored = read_game_control_file(_LEAGUE)
    assert stored is not None
    assert stored.league == _LEAGUE
    assert stored.fit == current_fit(_LEAGUE)
    assert set(stored.games) == {"g1"}


def test_an_unchanged_fit_only_re_sweeps_the_newest_season() -> None:
    """The weekly case: twenty weeks of reads rather than four hundred."""
    seasons = [_season(2024, "old"), _season(2025, "new")]
    source = _Source({(2024, 1): _plays("old", season=2024), (2025, 1): _plays("new")})
    asyncio.run(build(_LEAGUE, seasons, source))
    source.loaded.clear()

    asyncio.run(build(_LEAGUE, seasons, source))

    assert source.loaded == [(2025, 1)]
    # ...and the season it didn't re-read is still in the file.
    stored = read_game_control_file(_LEAGUE)
    assert stored is not None
    assert set(stored.games) == {"old", "new"}


def test_a_different_fit_rebuilds_the_whole_league(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An index holding two models' numbers is one nobody can reproduce."""
    seasons = [_season(2024, "old"), _season(2025, "new")]
    source = _Source({(2024, 1): _plays("old", season=2024), (2025, 1): _plays("new")})
    asyncio.run(build(_LEAGUE, seasons, source))
    source.loaded.clear()

    monkeypatch.setattr(
        "cassandra.game_control_build.current_fit",
        lambda league: ControlFit(lucky_ones="deadbeef", run_id="retrained"),
    )
    asyncio.run(build(_LEAGUE, seasons, source))

    assert source.loaded == [(2024, 1), (2025, 1)]
    stored = read_game_control_file(_LEAGUE)
    assert stored is not None
    assert stored.fit.run_id == "retrained"


def test_changing_only_the_reading_rebuilds_the_league() -> None:
    """The near-term hazard, since comparing the two readings means switching.

    Neither the package rev nor the run id has to move for every number in
    the file to change meaning, so `reading` is what stands between a
    comparison run and an index that quietly holds half of each.
    """
    seasons = [_season(2024, "old"), _season(2025, "new")]
    source = _Source({(2024, 1): _plays("old", season=2024), (2025, 1): _plays("new")})
    asyncio.run(build(_LEAGUE, seasons, source))
    stored = read_game_control_file(_LEAGUE)
    assert stored is not None
    write(_LEAGUE, stored.fit.model_copy(update={"reading": "realized"}), stored.games)
    source.loaded.clear()

    asyncio.run(build(_LEAGUE, seasons, source))

    assert source.loaded == [(2024, 1), (2025, 1)]
    refreshed = read_game_control_file(_LEAGUE)
    assert refreshed is not None
    assert refreshed.fit == current_fit(_LEAGUE)


def test_rebuild_re_sweeps_a_league_that_is_already_current() -> None:
    seasons = [_season(2024, "old"), _season(2025, "new")]
    source = _Source({(2024, 1): _plays("old", season=2024), (2025, 1): _plays("new")})
    asyncio.run(build(_LEAGUE, seasons, source))
    source.loaded.clear()

    asyncio.run(build(_LEAGUE, seasons, source, rebuild=True))

    assert source.loaded == [(2024, 1), (2025, 1)]


def test_building_with_no_seasons_says_so() -> None:
    with pytest.raises(ValueError, match="No seasons"):
        asyncio.run(build(_LEAGUE, [], _Source({})))


def test_the_written_index_is_what_a_predictor_reads_back() -> None:
    """The artifact is the seam, so the round trip is worth one assertion."""
    source = _Source({(2025, 1): _plays("g1")})
    asyncio.run(build(_LEAGUE, [_season(2025, "g1")], source))
    load_game_control.cache_clear()

    control = load_game_control(_LEAGUE)

    assert set(control) == {"g1"}
    assert 0.0 < control["g1"].home < 1.0

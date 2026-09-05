"""The EPA sweep, against plays built here rather than against a bucket.

Which number comes off a game, and which header goes on the file. The week
loop and the coverage filters underneath are `cassandra.pbp_sweep`'s, checked
through `game_control_build_test` -- what's asserted here is the part that is
this build's own, and the part that differs: two fits behind one index.
"""

import asyncio

import pytest
from lucky_ones import MODELS, group_by_game

from cassandra.conftest import (
    LEAGUE as _LEAGUE,
)
from cassandra.conftest import (
    FakeSource as _Source,
)
from cassandra.conftest import (
    plays_for as _plays,
)
from cassandra.conftest import (
    season_for as _season,
)
from cassandra.epa_build import CLIP, build, current_fit, sweep, write
from cassandra.predictor.epa import EpaFit, load_epa, read_epa_file


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Point the artifact at tmp_path, and clear the cached read around it.

    Both bindings, not just the one: `epa_build` imports the name rather than
    the module, so patching only the definition would leave the build writing
    into a real `~/.cassandra` while the reads came from tmp_path -- which
    passes some of these and corrupts a developer's disk.
    """
    for module in ("cassandra.predictor.epa", "cassandra.epa_build"):
        monkeypatch.setattr(
            f"{module}.epa_path", lambda league: tmp_path / f"{league}_epa.json"
        )
    load_epa.cache_clear()
    yield
    load_epa.cache_clear()


def test_a_full_game_gets_epa_for_both_offenses() -> None:
    source = _Source({(2025, 1): _plays("g1")})

    epa, stats = asyncio.run(sweep(_LEAGUE, [_season(2025, "g1")], source))

    assert set(epa) == {"g1"}
    assert epa["g1"].home_plays > 0
    assert epa["g1"].away_plays > 0
    assert stats.scored == 1
    assert stats.played == 1
    assert stats.disagreed == 0


def test_the_number_is_the_unweighted_reading() -> None:
    """Which of the two averages the sweep takes, which is the whole point.

    Asserted against `lucky_ones` on the same game rather than by direction:
    what a crude fixture's expected points come to is the model's business,
    and what is this module's business is that the number written is the flat
    mean rather than the competitiveness-weighted one. The fixture is a
    runaway -- one side scores in every period -- so the two readings differ
    and a quiet swap would show up here.
    """
    plays = _plays("g1")
    game = next(iter(group_by_game(plays)))
    scored = MODELS[_LEAGUE].epa_per_play(game, clip=CLIP)
    assert scored.home_unweighted is not None and scored.home is not None

    epa, _ = asyncio.run(
        sweep(_LEAGUE, [_season(2025, "g1")], _Source({(2025, 1): plays}))
    )

    assert epa["g1"].home == pytest.approx(scored.home_unweighted)
    assert epa["g1"].home != pytest.approx(scored.home)


def test_the_play_counts_are_the_denominators_that_were_used() -> None:
    """The honesty field, and what turns the averages back into points."""
    plays = _plays("g1")
    game = next(iter(group_by_game(plays)))
    scored = MODELS[_LEAGUE].epa_per_play(game, clip=CLIP)

    epa, _ = asyncio.run(
        sweep(_LEAGUE, [_season(2025, "g1")], _Source({(2025, 1): plays}))
    )

    assert epa["g1"].home_plays == scored.home_plays
    assert epa["g1"].away_plays == scored.away_plays


def test_a_first_build_writes_the_fit_that_made_it() -> None:
    source = _Source({(2025, 1): _plays("g1")})

    asyncio.run(build(_LEAGUE, [_season(2025, "g1")], source))

    stored = read_epa_file(_LEAGUE)
    assert stored is not None
    assert stored.fit == current_fit(_LEAGUE)
    assert stored.league == _LEAGUE


def test_the_fit_names_both_models() -> None:
    """The difference from the control index, and the reason for `ep_run_id`.

    Expected points and win probability retrain separately, so an index built
    against one pair is stale if either half moves.
    """
    fit = current_fit(_LEAGUE)

    assert fit.run_id == MODELS[_LEAGUE].run_id
    assert fit.ep_run_id == MODELS[_LEAGUE].expected_points_release.run_id
    assert fit.clip == CLIP
    assert fit.reading == "unweighted"


def test_an_unchanged_fit_only_re_sweeps_the_newest_season() -> None:
    """The weekly case: history can't change while the models don't."""
    source = _Source({(2024, 1): _plays("old", season=2024), (2025, 1): _plays("new")})
    seasons = [_season(2024, "old"), _season(2025, "new")]
    asyncio.run(build(_LEAGUE, seasons, source))
    source.loaded.clear()

    asyncio.run(build(_LEAGUE, seasons, source))

    assert source.loaded == [(2025, 1)]
    stored = read_epa_file(_LEAGUE)
    assert stored is not None
    assert set(stored.games) == {"old", "new"}


@pytest.mark.parametrize(
    "changed",
    [
        {"lucky_ones": "something-else"},
        {"run_id": "20200101-000000"},
        {"ep_run_id": "20200101-000000"},
        {"clip": 5.0},
        {"reading": "weighted"},
    ],
)
def test_any_field_of_the_fit_moving_rebuilds_the_league(changed: dict) -> None:
    """Merging two models' numbers into one index is unreproducible.

    Every field, because each of the five moves the numbers on its own and
    none of them is visible in the others.
    """
    source = _Source({(2024, 1): _plays("old", season=2024), (2025, 1): _plays("new")})
    seasons = [_season(2024, "old"), _season(2025, "new")]
    stale = EpaFit(**{**current_fit(_LEAGUE).model_dump(), **changed})
    write(_LEAGUE, stale, {})

    asyncio.run(build(_LEAGUE, seasons, source))

    assert source.loaded == [(2024, 1), (2025, 1)]


def test_rebuild_re_sweeps_a_league_that_is_already_current() -> None:
    source = _Source({(2024, 1): _plays("old", season=2024), (2025, 1): _plays("new")})
    seasons = [_season(2024, "old"), _season(2025, "new")]
    asyncio.run(build(_LEAGUE, seasons, source))
    source.loaded.clear()

    asyncio.run(build(_LEAGUE, seasons, source, rebuild=True))

    assert source.loaded == [(2024, 1), (2025, 1)]


def test_building_with_no_seasons_says_so() -> None:
    with pytest.raises(ValueError, match="nothing to sweep"):
        asyncio.run(build(_LEAGUE, [], _Source({})))


def test_the_written_index_is_what_a_predictor_reads_back() -> None:
    """The seam: the sweep writes it, and `EpaIndex` is the other side."""
    from cassandra.predictor.epa import EpaIndex

    source = _Source({(2025, 1): _plays("g1")})
    asyncio.run(build(_LEAGUE, [_season(2025, "g1")], source))

    index = EpaIndex.for_league(_LEAGUE)

    assert len(index) == 1
    assert index.margin("g1") is not None

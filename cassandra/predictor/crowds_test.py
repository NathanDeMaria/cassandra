from datetime import date, datetime
from pathlib import Path

import pytest

from . import crowds
from .adjustments import MatchupSources
from .conftest import GameFactory
from .crowds import CrowdAsOf, CrowdIndex, CrowdsFile, latest_crowds, load_crowds
from .glicko import GlickoPredictor
from .qb_out import QbOutIndex
from .rest import RestLedger

# The test games are on 2023-01-01. Home U plays in a big building, by then.
INDEX = CrowdIndex(
    {
        "Home U": [
            CrowdAsOf(date(2022, 8, 1), None),
            CrowdAsOf(date(2022, 12, 20), 1.0),
            CrowdAsOf(date(2023, 1, 2), 3.0),
        ],
        "New U": [CrowdAsOf(date(2022, 8, 1), None)],
    }
)


def test_a_reading_is_the_latest_on_or_before_the_games_day(game: GameFactory) -> None:
    played = game("Home U", "Away St")
    # the row dated the day after this game is the one this game made
    assert INDEX.home_crowd(played) == 1.0
    assert INDEX.home_crowd(played._replace(date=datetime(2023, 1, 2))) == 3.0
    assert INDEX.home_crowd(played._replace(date=datetime(2022, 12, 1))) is None
    assert INDEX.home_crowd(played._replace(date=datetime(2022, 7, 1))) is None


def test_no_reading_without_enough_games_a_file_or_a_home_side(
    game: GameFactory,
) -> None:
    assert INDEX.home_crowd(game("New U", "Away St")) is None
    assert INDEX.home_crowd(game("Away St", "Home U")) is None
    assert (
        INDEX.home_crowd(game("Home U", "Away St")._replace(neutral_site=True)) is None
    )
    assert CrowdIndex().home_crowd(game("Home U", "Away St")) is None


def _file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "nfl_crowds.csv"
    path.write_text(
        "league,season,team_id,team,as_of,crowd,total,weight,games\n"
        "nfl,2022,1,Home U,2022-12-20,1.0,30,3,3\n"
        "nfl,2022,1,Home U,2022-08-01,,8,1,1\n"
        "nfl,2022,2,Away St,2022-12-21,-0.5,40,5,5\n"
        "nfl,2022,3,New U,2022-08-01,,8,1,1\n"
    )
    monkeypatch.setattr(crowds, "crowds_path", lambda league: path)
    load_crowds.cache_clear()


def test_the_file_reads_by_canonical_name_in_date_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _file(monkeypatch, tmp_path)
    try:
        assert load_crowds("nfl")["Home U"] == [
            CrowdAsOf(date(2022, 8, 1), None),
            CrowdAsOf(date(2022, 12, 20), 1.0),
        ]
    finally:
        load_crowds.cache_clear()


def test_the_published_file_is_each_teams_newest_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, game: GameFactory
) -> None:
    _file(monkeypatch, tmp_path)
    try:
        stored = latest_crowds("nfl")
    finally:
        load_crowds.cache_clear()
    # New U has no reading yet, so nothing to publish
    assert stored == CrowdsFile(
        league="nfl", as_of=date(2022, 12, 21), teams={"Home U": 1.0, "Away St": -0.5}
    )
    rebuilt = CrowdIndex.from_file(
        CrowdsFile.model_validate_json(stored.model_dump_json())
    )
    assert len(rebuilt) == 2
    # a consumer is predicting the games ahead: the reading holds on any day
    ahead = game("Away St", "Home U")._replace(date=datetime(2022, 12, 20))
    assert rebuilt.home_crowd(ahead) == -0.5


def _predictor(slope: float) -> GlickoPredictor:
    sources = MatchupSources(QbOutIndex(), RestLedger(), crowds=INDEX)
    return GlickoPredictor(
        "test_league",
        scoring_method="sigmoid",
        home_advantage=60.0,
        home_crowd_slope=slope,
        sources=sources,
    )


def test_the_home_edge_grows_with_the_crowd(game: GameFactory) -> None:
    big = game("Home U", "Away St")
    assert _predictor(20.0).home_edge(big) == pytest.approx(80.0)
    assert _predictor(0.0).home_edge(big) == pytest.approx(60.0)
    # a team with no reading gets the league's edge, and nobody one at a neutral site
    assert _predictor(20.0).home_edge(game("New U", "Away St")) == pytest.approx(60.0)
    assert _predictor(20.0).home_edge(big._replace(neutral_site=True)) == 0.0


def test_a_visitor_is_given_less_and_credited_more_in_a_big_building(
    game: GameFactory,
) -> None:
    plain, crowded = _predictor(0.0), _predictor(40.0)
    ahead = game("Home U", "Away St")
    assert crowded.predict_game(ahead) > plain.predict_game(ahead)
    upset = game("Home U", "Away St", 60, 70)
    plain.update_game(upset)
    crowded.update_game(upset)
    assert crowded.get_rating("Away St").rating > plain.get_rating("Away St").rating
    assert crowded.get_rating("Home U").rating < plain.get_rating("Home U").rating


def test_the_slope_is_kept_only_when_on_and_is_never_negative() -> None:
    restored = GlickoPredictor.from_state_dict(_predictor(15.0).state_dict())
    assert restored.state_dict()["home_crowd_slope"] == 15.0
    assert "home_crowd_slope" not in _predictor(0.0).state_dict()
    with pytest.raises(ValueError, match="home_crowd_slope"):
        GlickoPredictor("test_league", home_crowd_slope=-1.0)


def test_two_rows_of_one_day_keep_the_files_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "nfl_crowds.csv"
    path.write_text(
        "league,season,team_id,team,as_of,crowd,total,weight,games\n"
        "nfl,2022,1,Home U,2022-12-20,,30,3,4\n"
        "nfl,2022,1,Home U,2022-12-20,0.7,30,3,5\n"
    )
    monkeypatch.setattr(crowds, "crowds_path", lambda league: path)
    load_crowds.cache_clear()
    try:
        assert [row.crowd for row in load_crowds("nfl")["Home U"]] == [None, 0.7]
    finally:
        load_crowds.cache_clear()

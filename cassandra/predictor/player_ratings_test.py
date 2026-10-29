from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import pytest

from . import player_ratings
from .adjustments import MatchupSources
from .conftest import GameFactory
from .distribution import collapse
from .glicko import GlickoPredictor, _Rating
from .player_ratings import (
    PlayerRatingIndex,
    PlayerRatingsFile,
    TeamAsOf,
    TeamRow,
    load_player_ratings,
)
from .qb_out import QbOutIndex
from .rest import RestLedger
from .types import PlayerRates


class _Fixture(NamedTuple):
    home: str
    away: str
    date: datetime
    neutral_site: bool = False
    game_id: str = "g"


INDEX = PlayerRatingIndex(
    {
        # +10 per 100 at 70 a game, then +4 from the week after
        "Home U": [
            TeamAsOf(date(2024, 11, 4), 10.0, 70.0),
            TeamAsOf(date(2024, 11, 11), 4.0, 70.0),
        ],
        "Away St": [TeamAsOf(date(2024, 11, 4), -2.0, 66.0)],
    }
)


def test_a_team_is_its_latest_row_from_before_the_day() -> None:
    def net(team: str, day: date) -> float | None:
        row = INDEX.team(team, day)
        return None if row is None else row.net

    assert net("Home U", date(2024, 11, 4)) is None
    assert net("Home U", date(2024, 11, 5)) == 10.0
    # a row dated the game's day might have seen it
    assert net("Home U", date(2024, 11, 11)) == 10.0
    assert net("Home U", date(2024, 11, 12)) == 4.0
    assert net("Nobody", date(2024, 11, 12)) is None


def test_a_row_goes_stale() -> None:
    assert INDEX.team("Away St", date(2024, 11, 18)) is not None
    assert INDEX.team("Away St", date(2024, 11, 19)) is None
    assert INDEX.team("Away St", date(2025, 3, 1)) is None


def test_the_margin_is_the_average_pace_times_the_net_gap() -> None:
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 6))
    assert INDEX.margin(fixture) == pytest.approx(68 * 12 / 100)
    # the home edge is the model's: a neutral site reads the same
    assert INDEX.margin(fixture._replace(neutral_site=True)) == INDEX.margin(fixture)
    assert INDEX.margin(fixture._replace(away="Nobody")) is None


def test_a_published_file_prices_a_game_ahead_as_the_whole_file_does() -> None:
    stored = PlayerRatingsFile(
        league="mens",
        teams={
            "Home U": TeamRow(
                as_of=date(2024, 11, 11), season=2024, net=4.0, pace=70.0
            ),
            "Away St": TeamRow(
                as_of=date(2024, 11, 4), season=2024, net=-2.0, pace=66.0
            ),
        },
    )
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 14))
    assert PlayerRatingIndex.from_file(stored).margin(fixture) == INDEX.margin(fixture)


def test_glicko_blends_it_into_the_prediction_and_not_the_update(
    game: GameFactory,
) -> None:
    sources = MatchupSources(QbOutIndex(), RestLedger(), INDEX)
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 6))
    plain = GlickoPredictor("test_league", sources=sources)
    priced = GlickoPredictor("test_league", player_rating_weight=5.0, sources=sources)
    assert (
        priced.predict_game(fixture).team1_win_prob
        > plain.predict_game(fixture).team1_win_prob
    )
    played = game("Home U", "Away St", 70, 60)._replace(date=datetime(2024, 11, 6))
    plain.update_game(played)
    priced.update_game(played)
    assert priced.get_rating("Home U") == plain.get_rating("Home U")


def test_the_share_reads_glickos_gap_and_leaves_the_edge_whole(
    game: GameFactory,
) -> None:
    sources = MatchupSources(QbOutIndex(), RestLedger(), INDEX)
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 6))
    blended = GlickoPredictor(
        "test_league",
        home_advantage=50.0,
        glicko_share=0.5,
        player_rating_weight=2.0,
        sources=sources,
    )
    whole = GlickoPredictor("test_league", home_advantage=50.0, sources=sources)
    for predictor in (blended, whole):
        predictor.update_game(game("Home U", "Away St", 80, 60))
    gap = whole.get_rating("Home U").rating - whole.get_rating("Away St").rating
    players = 68 * 12 / 100
    want = collapse(whole.gap_distribution(0.5 * gap + 50.0 + 2.0 * players, 0.0))
    assert blended.predict_game(fixture).team1_win_prob == pytest.approx(
        want.team1_win_prob
    )
    # a game the file doesn't cover is glicko's alone
    uncovered = fixture._replace(away="Nobody")
    assert blended.predict_distribution(uncovered) == whole.predict_distribution(
        uncovered
    )


def test_a_ranking_reads_the_blend_the_way_a_neutral_game_does(
    game: GameFactory,
) -> None:
    sources = MatchupSources(QbOutIndex(), RestLedger(), INDEX)
    blended = GlickoPredictor(
        "test_league", glicko_share=0.5, player_rating_weight=2.0, sources=sources
    )
    plain = GlickoPredictor("test_league", sources=sources)
    for predictor in (blended, plain):
        for away, day in (("Away St", 6), ("Nobody", 8)):
            predictor.update_game(
                game("Home U", away, 80, 60)._replace(date=datetime(2024, 11, day))
            )
    ratings = blended.ratings
    home, away, nobody = ratings["Home U"], ratings["Away St"], ratings["Nobody"]
    neutral = _Fixture("Home U", "Away St", datetime(2024, 11, 9), neutral_site=True)
    players = INDEX.margin(neutral)
    assert players is not None
    gap = home.rating - away.rating
    assert home.overall - away.overall == pytest.approx(0.5 * gap + 2.0 * players)
    # the teams it covers keep their mean; one it doesn't is its rating
    assert home.overall + away.overall == pytest.approx(home.rating + away.rating)
    assert nobody.blended is None and nobody.overall == nobody.rating
    assert all(r.blended is None for r in plain.ratings.values())
    # each covered team carries the row its next game reads, for a reader
    assert home.players == PlayerRates(10.0, 70.0, None, date(2024, 11, 4))
    assert away.players == PlayerRates(-2.0, 66.0, None, date(2024, 11, 4))
    assert nobody.players is None
    assert all(r.players is None for r in plain.ratings.values())


def test_before_a_season_a_ranking_reads_its_newest_rows(game: GameFactory) -> None:
    index = PlayerRatingIndex(
        {
            team: [
                TeamAsOf(date(2025, 3, 3), 0.0, 70.0, 2024),
                TeamAsOf(date(2025, 10, 20), net, 70.0, 2025),
                TeamAsOf(date(2025, 10, 21), 2 * net, 70.0, 2025),
            ]
            for team, net in (("Home U", 5.0), ("Away St", -5.0))
        }
    )
    sources = MatchupSources(QbOutIndex(), RestLedger(), index)
    predictor = GlickoPredictor(
        "test_league", player_rating_weight=1.0, sources=sources
    )
    predictor.update_game(
        game("Home U", "Away St", 70, 70)._replace(date=datetime(2025, 3, 8))
    )
    ratings = predictor.ratings
    # in season, the rows before the day after: level by the season's last
    assert ratings["Home U"].overall - ratings["Away St"].overall == pytest.approx(
        ratings["Home U"].rating - ratings["Away St"].rating
    )
    predictor.pass_season()
    ratings = predictor.ratings
    assert ratings["Home U"].overall - ratings["Away St"].overall == pytest.approx(
        ratings["Home U"].rating - ratings["Away St"].rating + 70 * 20 / 100
    )


def test_the_blend_is_kept_only_when_on() -> None:
    assert "glicko_share" not in GlickoPredictor("test_league").state_dict()
    priced = GlickoPredictor("test_league", glicko_share=0.8, player_rating_weight=5.0)
    restored = GlickoPredictor.from_state_dict(priced.state_dict())
    assert restored.state_dict()["glicko_share"] == 0.8
    assert restored.state_dict()["player_rating_weight"] == 5.0
    with pytest.raises(ValueError):
        GlickoPredictor("test_league", player_rating_weight=-1.0)


def test_the_file_reads_by_canonical_name_in_date_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "nfl_player_ratings.csv"
    path.write_text(
        "league,season,as_of,team_id,team,offense,defense,net,pace,games,home_edge,"
        "available,net_sd\n"
        "nfl,2025,2024-11-11,1,Home U,3,1,4,70,1,1.5,2.5,4.5\n"
        "nfl,2025,2024-11-04,1,Home U,6,4,10,70,0,1.5,,\n"
        "nfl,2025,2024-11-04,9,,0,0,0,68,0,1.5,,\n"
    )
    monkeypatch.setattr(player_ratings, "player_ratings_path", lambda league: path)
    load_player_ratings.cache_clear()
    try:
        assert load_player_ratings("nfl") == {
            "Home U": [
                TeamAsOf(date(2024, 11, 4), 10.0, 70.0, 2025),
                TeamAsOf(date(2024, 11, 11), 4.0, 70.0, 2025, 2.5, 4.5),
            ]
        }
    finally:
        load_player_ratings.cache_clear()


def test_the_margin_reads_the_expected_net_where_the_row_has_one() -> None:
    index = PlayerRatingIndex(
        {
            # home's starter might sit: expected 4 against a healthy 10
            "Home U": [TeamAsOf(date(2024, 11, 4), 10.0, 70.0, 2024, 4.0)],
            # a row from before the column: its net
            "Away St": [TeamAsOf(date(2024, 11, 4), -2.0, 66.0, 2024, None)],
        }
    )
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 6))
    assert index.margin(fixture) == pytest.approx(68 * (4 - -2) / 100)


def test_the_absences_are_the_shift_from_the_net_at_the_pace() -> None:
    index = PlayerRatingIndex(
        {
            "Home U": [TeamAsOf(date(2024, 11, 4), 10.0, 70.0, 2024, 4.0)],
            "Away St": [TeamAsOf(date(2024, 11, 4), -2.0, 66.0, 2024, None)],
        }
    )
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 6))
    assert index.absences(fixture) == pytest.approx(68 * -6 / 100)
    assert index.absences(fixture._replace(away="Nobody")) is None
    # rows from before the column move nothing
    assert INDEX.absences(fixture) == 0.0


def test_glicko_prices_the_absences_on_top_of_the_margin(game: GameFactory) -> None:
    index = PlayerRatingIndex(
        {
            "Home U": [TeamAsOf(date(2024, 11, 4), 10.0, 70.0, 2024, 4.0)],
            "Away St": [TeamAsOf(date(2024, 11, 4), -2.0, 66.0, 2024, -2.0)],
        }
    )
    sources = MatchupSources(QbOutIndex(), RestLedger(), index)
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 6))
    margin_only = GlickoPredictor(
        "test_league", player_rating_weight=5.0, sources=sources
    )
    with_absences = GlickoPredictor(
        "test_league", player_rating_weight=5.0, absence_weight=20.0, sources=sources
    )
    assert (
        with_absences.predict_game(fixture).team1_win_prob
        < margin_only.predict_game(fixture).team1_win_prob
    )
    restored = GlickoPredictor.from_state_dict(with_absences.state_dict())
    assert restored.state_dict()["absence_weight"] == 20.0
    assert "absence_weight" not in margin_only.state_dict()


def _sure(net_sd: float | None) -> PlayerRatingIndex:
    return PlayerRatingIndex(
        {
            "Home U": [TeamAsOf(date(2024, 11, 4), 10.0, 70.0, 2024, None, net_sd)],
            "Away St": [TeamAsOf(date(2024, 11, 4), -2.0, 66.0, 2024, None, net_sd)],
        }
    )


def test_the_margins_variance_is_both_sides_sd_at_the_pace() -> None:
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 6))
    assert _sure(5.0).margin_variance(fixture) == pytest.approx(0.68**2 * (25 + 25))
    # a row from before the column: as sure as can be
    assert _sure(None).margin_variance(fixture) == 0.0
    assert _sure(5.0).margin_variance(fixture._replace(away="Nobody")) is None


def _weighed(index: PlayerRatingIndex, rd: float, **params) -> GlickoPredictor:
    return GlickoPredictor(
        "test_league",
        ratings={"Home U": _Rating(1500.0, rd), "Away St": _Rating(1500.0, rd)},
        player_rating_weight=20.0,
        home_advantage=0.0,
        sources=MatchupSources(QbOutIndex(), RestLedger(), index),
        **params,
    )


def test_a_blend_weighed_by_uncertainty_leans_on_the_surer_side() -> None:
    fixture = _Fixture("Home U", "Away St", datetime(2024, 11, 6))
    margin = 68 * 12 / 100
    params = {"player_variance_scale": 1.0, "player_floor": 2.0}
    # Glicko unsure: nearly all the players' margin
    unsure = _weighed(_sure(5.0), 600.0, **params)
    # Glicko sure: nearly none of it, the teams being level
    sure = _weighed(_sure(5.0), 20.0, **params)
    gap_unsure = unsure.predict_distribution(fixture).mean
    gap_sure = sure.predict_distribution(fixture).mean
    assert 0 < gap_sure < gap_unsure < 20.0 * margin
    assert gap_unsure == pytest.approx(20.0 * margin, rel=0.1)
    # the exact weight: the players' variance over both's
    mine = 20.0**2 * (0.68**2 * 50 + 4.0)
    glicko = mine / (mine + 2 * 600.0**2)
    assert gap_unsure == pytest.approx((1 - glicko) * 20.0 * margin)


def test_a_blend_weighed_by_uncertainty_round_trips_and_needs_a_weight() -> None:
    weighed = _weighed(_sure(5.0), 100.0, player_variance_scale=2.0, player_floor=3.0)
    state = weighed.state_dict()
    assert state["player_variance_scale"] == 2.0 and state["player_floor"] == 3.0
    restored = GlickoPredictor.from_state_dict(state)
    assert restored.state_dict()["player_floor"] == 3.0
    assert "player_floor" not in _weighed(_sure(5.0), 100.0).state_dict()
    with pytest.raises(ValueError, match="player_rating_weight"):
        GlickoPredictor("test_league", player_variance_scale=1.0)

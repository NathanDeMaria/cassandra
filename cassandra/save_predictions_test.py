from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Self

import pandas as pd
import pytest
from call_it_what_you_want import TeamNamer, teams_from_csv
from endgame.types import Game, OverlappingWeeksError, Season, Week
from pandas.testing import assert_frame_equal

from .predictor import (
    ControlGlickoPredictor,
    Elo538Predictor,
    EloPredictor,
    GameControl,
    GameControlIndex,
    GlickoPredictor,
    Predictor,
)
from .predictor.opponent_prior import OpponentPriorManager
from .predictor.types import Matchup, Prediction
from .save_predictions import (
    _Prediction,
    generate_predictions,
    predictions_frame,
    prepared_for_replay,
)

_HOME = "Team A"
_AWAY = "Team B"


def _repeated_matchup(n_weeks: int) -> Season:
    """The same two teams, same site, with the home team winning every time.

    Nothing but the home team's rating changes between games, so a predictor
    that never sees the results has to return the same probability every week.
    """
    return Season(
        year=2023,
        weeks=[
            Week(
                games=[
                    Game(
                        home=_HOME,
                        away=_AWAY,
                        home_score=10,
                        away_score=0,
                        neutral_site=False,
                        completed=True,
                        date=datetime(2023, 1, week),
                        game_id=str(week),
                    )
                ],
                number=week,
            )
            for week in range(1, n_weeks + 1)
        ],
    )


def _stateful_predictors() -> list[Predictor]:
    # A league nobody has run, so there are no priors on disk to pick up
    league = "test_league"
    return [
        EloPredictor(league),
        Elo538Predictor(league, opponent_prior_manager=OpponentPriorManager(league)),
        GlickoPredictor(league, opponent_prior_manager=OpponentPriorManager(league)),
    ]


@pytest.mark.parametrize(
    "predictor", _stateful_predictors(), ids=lambda p: type(p).__name__
)
def test_generate_predictions_feeds_results_back(predictor: Predictor) -> None:
    """The backtest has to update predictors as it walks the season.

    Predicting without updating leaves every rating at its initial value, which
    silently turns the ratings models into a constant.
    """
    results = list(generate_predictions(predictor, [_repeated_matchup(4)]))

    probs = [result.prediction.team1_win_prob for result in results]
    # Strictly increasing: a flat run is the exact shape of the bug this catches
    assert all(earlier < later for earlier, later in zip(probs, probs[1:])), (
        f"{type(predictor).__name__} should get more confident about a repeat "
        f"winner, got {probs}"
    )


class _RecordingPredictor(Predictor):
    """Records the order games are handed to it."""

    def __init__(self, league: str = "test_league") -> None:
        super().__init__(league)
        self.seen: list[str] = []

    def predict_game(self, matchup: Matchup) -> Prediction:
        self.seen.append(matchup.game_id)
        return Prediction(team1_win_prob=0.5)

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        raise NotImplementedError


def _game(game_id: str, day: int, month: int = 11, year: int = 2023) -> Game:
    return Game(
        home=_HOME,
        home_score=1,
        away=_AWAY,
        away_score=0,
        neutral_site=False,
        completed=True,
        date=datetime(year, month, day, tzinfo=timezone.utc),
        game_id=game_id,
    )


def test_generate_predictions_walks_games_chronologically() -> None:
    # Weeks within the season, and games within each week, are both
    # deliberately stored out of order.
    week1 = Week([_game("b", 8), _game("a", 6)], 1)
    week2 = Week([_game("d", 15), _game("c", 13)], 2)
    season = Season([week2, week1], 2023)

    predictor = _RecordingPredictor()
    list(generate_predictions(predictor, [season]))

    assert predictor.seen == ["a", "b", "c", "d"]


def test_generate_predictions_walks_seasons_chronologically() -> None:
    older = Season([Week([_game("older", 6, year=2022)], 1)], 2022)
    newer = Season([Week([_game("newer", 6, year=2023)], 1)], 2023)

    predictor = _RecordingPredictor()
    list(generate_predictions(predictor, [newer, older]))

    assert predictor.seen == ["older", "newer"]


class _CountingPredictor(_RecordingPredictor):
    """Records how many week and season rollovers it was put through."""

    def __init__(self) -> None:
        super().__init__()
        self.rollovers = 0
        self.weeks = 0

    def pass_week(self) -> None:
        self.weeks += 1

    def _roll_over(self) -> None:
        self.rollovers += 1


def test_the_last_seasons_rollover_can_be_left_off() -> None:
    """`roll_over_final_season=False` skips the last one and only the last one.

    The rollover between two seasons is part of the replay -- without it the
    later season is predicted by ratings that never cooled off -- so turning
    the flag off has to leave that one alone. Only the trailing one, applied
    to ratings nothing else in the replay reads, is optional; publish drops it
    until a month after the season ends.
    """
    seasons = [
        Season([Week([_game("older", 6, year=2022)], 1)], 2022),
        Season([Week([_game("newer", 6, year=2023)], 1)], 2023),
    ]

    kept = _CountingPredictor()
    list(generate_predictions(kept, seasons))
    skipped = _CountingPredictor()
    list(generate_predictions(skipped, seasons, roll_over_final_season=False))

    assert (kept.rollovers, skipped.rollovers) == (2, 1)


def _renamed_game(home: str, away: str, year: int) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=1,
        away_score=0,
        neutral_site=False,
        completed=True,
        date=datetime(year, 11, 6, tzinfo=timezone.utc),
        game_id=f"{year}",
    )


def test_generate_predictions_canonicalizes_team_names() -> None:
    """A school renamed mid-history is one team, with one rating.

    Without this the old name keeps its rating forever -- nothing ever plays
    it again -- and the new one starts over at 1500.
    """
    seasons = [
        Season([Week([_renamed_game("Old Name", "Rival", 2022)], 1)], 2022),
        Season([Week([_renamed_game("New Name", "Rival", 2023)], 1)], 2023),
    ]
    namer = TeamNamer(
        teams_from_csv(
            [
                "espn_id,name,year,source",
                "1,Old Name,2022,espn",
                "1,New Name,2023,espn",
                "2,Rival,2023,espn",
            ]
        )
    )

    predictor = EloPredictor("test_league")
    results = list(generate_predictions(predictor, seasons, namer=namer))

    assert [r.game.home for r in results] == ["New Name", "New Name"]
    # Both wins landed on one team rather than being split across two names.
    assert "Old Name" not in predictor.ratings
    assert predictor.get_rating("New Name") > 1500


def _rename_registry() -> TeamNamer:
    """One school under two names, and an opponent under one."""
    return TeamNamer(
        teams_from_csv(
            [
                "espn_id,name,year,source",
                "1,Old Name,2022,espn",
                "1,New Name,2023,espn",
                "2,Rival,2023,espn",
            ]
        )
    )


def test_prepared_for_replay_renames_once_up_front(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A search's seasons arrive canonical, and replay to the same games.

    The equivalence is the whole claim: renaming every game once and
    replaying with a namer that renames nothing has to produce exactly what
    renaming as the replay goes produced, or a thousand probes are faster
    and measuring a different league.
    """
    seasons = [
        Season([Week([_renamed_game("Old Name", "Rival", 2022)], 1)], 2022),
        Season([Week([_renamed_game("New Name", "Rival", 2023)], 1)], 2023),
    ]
    registry = _rename_registry()
    monkeypatch.setattr(
        TeamNamer, "for_league", classmethod(lambda cls, league: registry)
    )

    prepared, replay_namer = prepared_for_replay("test_league", seasons)

    # The renaming has already happened, before any predictor sees a game.
    assert [
        game.home for season in prepared for week in season.weeks for game in week.games
    ] == ["New Name", "New Name"]
    # And the namer that comes back with them renames nothing, so the games
    # are not resolved a second time.
    assert not replay_namer

    once = list(
        generate_predictions(EloPredictor("test_league"), prepared, namer=replay_namer)
    )
    as_it_goes = list(
        generate_predictions(EloPredictor("test_league"), seasons, namer=registry)
    )
    assert [r.game for r in once] == [r.game for r in as_it_goes]
    assert [r.prediction for r in once] == [r.prediction for r in as_it_goes]


def test_prepared_for_replay_leaves_the_seasons_it_was_given_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's seasons are shared -- `read_rated_seasons` hands the same
    list to the priors pass and to the search -- so the rename has to copy."""
    season = Season([Week([_renamed_game("Old Name", "Rival", 2022)], 1)], 2022)
    monkeypatch.setattr(
        TeamNamer, "for_league", classmethod(lambda cls, league: _rename_registry())
    )

    prepared, _ = prepared_for_replay("test_league", [season])

    assert prepared[0].weeks[0].games[0].home == "New Name"
    assert season.weeks[0].games[0].home == "Old Name"


def _two_predictions() -> list[_Prediction]:
    return [
        _Prediction(
            year=2023,
            week_number=week,
            home_score=21,
            away_score=17,
            team1_win=True,
            team1_win_prob=0.6,
            # None on one row and a number on the other: the mixed column is
            # the one a columnar build could type differently.
            spread=None if week == 1 else -3.5,
            home_team=_HOME,
            away_team=_AWAY,
            game_id=str(week),
            date=datetime(2023, 1, week, tzinfo=timezone.utc),
            neutral_site=False,
        )
        for week in (1, 2)
    ]


def test_the_frame_matches_the_row_by_row_build() -> None:
    """`predictions_frame` is a faster spelling of the frame, not a new one.

    Every objective reads the frame by column name, and
    `cassandra.residuals` and `serving.predictions` read the same shape, so
    a column that changed name, order or dtype here would be a silent change
    to what a search maximizes.
    """
    predictions = _two_predictions()

    assert_frame_equal(
        predictions_frame(predictions),
        pd.DataFrame([asdict(prediction) for prediction in predictions]),
    )


def test_the_frame_is_empty_when_nothing_was_predicted() -> None:
    """What the objectives recognize as "no games": see `brier_score_df`."""
    assert predictions_frame([]).empty


def test_a_narrowed_frame_is_the_wide_one_with_columns_removed() -> None:
    """What a search builds: the objective's columns and no others.

    Same values and same dtypes as the full frame's, so narrowing can't
    change what an objective reads -- only what it costs to hand over.
    """
    predictions = _two_predictions()
    reads = {"team1_win_prob", "home_score", "away_score"}

    narrowed = predictions_frame(predictions, reads)

    assert_frame_equal(narrowed, predictions_frame(predictions)[list(narrowed.columns)])
    # Declaration order, not the caller's set iteration order.
    assert list(narrowed.columns) == ["home_score", "away_score", "team1_win_prob"]


def test_a_narrowed_frame_can_be_a_single_column() -> None:
    """`attrgetter` of one name returns a bare value, not a 1-tuple.

    Nothing asks for one column today; this is here because the failure
    would be a transposed frame rather than an error.
    """
    frame = predictions_frame(_two_predictions(), {"team1_win_prob"})

    assert list(frame.columns) == ["team1_win_prob"]
    assert frame["team1_win_prob"].tolist() == [0.6, 0.6]


def test_a_frame_of_nothing_at_all_is_refused() -> None:
    """A column name that isn't a field would otherwise build an empty frame,
    which every objective reports as "No games to score" -- a league with no
    data, which is not what went wrong."""
    with pytest.raises(ValueError, match="no prediction columns"):
        predictions_frame(_two_predictions(), {"not_a_column"})


def test_generate_predictions_leaves_names_alone_without_a_registry() -> None:
    """The default path for a league the registry doesn't cover."""
    season = Season([Week([_renamed_game("Old Name", "Rival", 2022)], 1)], 2022)
    predictor = EloPredictor("no-such-league")

    (result,) = list(generate_predictions(predictor, [season]))

    assert result.game.home == "Old Name"


def test_generate_predictions_rejects_misgrouped_weeks() -> None:
    """A week holding games from both ends of the season is a grouping bug."""
    spans_season = Week([_game("nov", 6), _game("mar", 20, month=3, year=2024)], 1)
    normal = Week([_game("later_nov", 13)], 2)
    season = Season([spans_season, normal], 2023)

    with pytest.raises(OverlappingWeeksError):
        list(generate_predictions(_RecordingPredictor(), [season]))


def _fixture(game_id: str, day: int, month: int = 11, year: int = 2023) -> Game:
    """A game ESPN has listed but nobody has played yet.

    ESPN sends a 0-0 scoreline for a scheduled game, so a fixture is only
    told from a real scoreless result by `completed`.
    """
    return _game(game_id, day, month=month, year=year)._replace(
        home_score=0, away_score=0, completed=False, status="STATUS_SCHEDULED"
    )


def test_generate_predictions_skips_unplayed_games() -> None:
    """Season pickles carry fixtures now; they are not results to train on.

    A scheduled game comes back 0-0, so replaying one would teach the
    predictor that the two teams drew.
    """
    week = Week([_game("played", 6), _fixture("scheduled", 8)], 1)

    predictor = _RecordingPredictor()
    results = list(generate_predictions(predictor, [Season([week], 2023)]))

    assert predictor.seen == ["played"]
    assert [r.game.game_id for r in results] == ["played"]


def test_generate_predictions_doesnt_pass_a_week_that_hasnt_happened() -> None:
    """The clock stops at the last week that was played.

    `pass_week` is what ages a rating -- Glicko inflates every team's rd once
    per week passed -- so walking the fixtures at the end of a pickle would
    publish ratings aged by weeks nobody has played.
    """
    played = Week([_game("played", 6)], 1)
    upcoming = [Week([_fixture("later", 13)], 2), Week([_fixture("latest", 20)], 3)]

    predictor = _CountingPredictor()
    list(generate_predictions(predictor, [Season([played, *upcoming], 2023)]))

    assert (predictor.seen, predictor.weeks) == (["played"], 1)


def test_generate_predictions_still_passes_a_week_with_no_games() -> None:
    """A week the source had nothing for is not the future.

    The NFL's season is always all 22 weeks, and a bye week or a week ESPN
    returned nothing for arrives as a Week with no games at all. Those passed
    before fixtures existed, and telling them from an upcoming week is the
    only reason this looks at `week.games` rather than just the filter.
    """
    season = Season([Week([_game("played", 6)], 1), Week([], 2)], 2023)

    predictor = _CountingPredictor()
    list(generate_predictions(predictor, [season]))

    assert predictor.weeks == 2


def test_the_replay_records_the_game_as_it_was_played() -> None:
    """A control predictor learns from the plays. Scoring must not.

    `_build_prediction` takes the score, the winner and the margin every
    metric is computed against off `result.game` -- the brier score, the
    prob-to-margin fit, against-spread accuracy. Control has to stop at the
    predictor's own state: a model that learned from control and was then
    graded against control would look excellent and mean nothing.

    Both halves are asserted, because either one alone passes for the wrong
    reason -- a predictor that ignored control entirely would also record the
    real score.
    """
    played = Game(
        home=_HOME,
        away=_AWAY,
        home_score=20,
        away_score=17,
        neutral_site=False,
        completed=True,
        date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        game_id="1",
    )
    season = Season(year=2023, weeks=[Week(games=[played], number=1)])
    # Controlled 0.2 by the home team: the scoreboard scores this a 1.0 for
    # the home side and the plays score it 0.2, so at full weight the model
    # learns from a game the home team lost.
    predictor = ControlGlickoPredictor(
        "test_league",
        game_control=GameControlIndex({"1": GameControl(home=0.2, seconds=3600)}),
        control_weight=1.0,
    )

    (result,) = list(generate_predictions(predictor, [season]))

    assert (result.game.home_score, result.game.away_score) == (20, 17)
    assert predictor.get_rating(_HOME) < predictor.get_rating(_AWAY)

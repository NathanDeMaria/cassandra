from datetime import datetime, timezone
from typing import Any, Self

import pytest
from endgame.types import Game, OverlappingWeeksError, Season, Week

from .predictor import Elo538Predictor, EloPredictor, GlickoPredictor, Predictor
from .predictor.opponent_prior import OpponentPriorManager
from .predictor.types import Matchup, Prediction
from .save_predictions import generate_predictions

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


def test_generate_predictions_rejects_misgrouped_weeks() -> None:
    """A week holding games from both ends of the season is a grouping bug."""
    spans_season = Week([_game("nov", 6), _game("mar", 20, month=3, year=2024)], 1)
    normal = Week([_game("later_nov", 13)], 2)
    season = Season([spans_season, normal], 2023)

    with pytest.raises(OverlappingWeeksError):
        list(generate_predictions(_RecordingPredictor(), [season]))

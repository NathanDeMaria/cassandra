from datetime import datetime

import pytest
from endgame.types import Game, Season, Week

from .predictor import Elo538Predictor, EloPredictor, GlickoPredictor, Predictor
from .predictor.opponent_prior import OpponentPriorManager
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

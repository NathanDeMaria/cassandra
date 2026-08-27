import pytest

from .conftest import GameFactory
from .elo import EloPredictor

# What every predictor does -- ratings, anchors, regression, save/load -- is
# in contract_test.py. What's left here is Elo's own arithmetic.


def test_a_cross_tier_matchup_is_predicted_from_the_gap(game: GameFactory) -> None:
    """The 300-point gap the tier fit found is what the first meeting predicts."""
    predictor = EloPredictor(
        "test_league", home_advantage=0, anchors={"Small": 1200, "Big": 1500}
    )

    prediction = predictor.predict_game(game("Small", "Big"))

    assert prediction.team1_win_prob == pytest.approx(
        1 / (1 + 10 ** (300 / 400)), abs=1e-9
    )


def test_old_state_without_regression_still_loads() -> None:
    """A state file written before the parameter existed."""
    loaded = EloPredictor.from_state_dict(
        {
            "league": "test_league",
            "home_advantage": 80,
            "k": 15,
            "ratings": {"Team A": 1600},
        }
    )
    loaded.pass_season()
    assert loaded.get_rating("Team A") == 1600

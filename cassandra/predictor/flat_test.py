from .conftest import GameFactory
from .flat import FlatPredictor

# Save/load and the state dict are covered for every model in contract_test.py.


def test_flat_predicts_a_coin_flip(game: GameFactory) -> None:
    """The baseline every other model has to beat, and it has no opinions."""
    predictor = FlatPredictor("test_league")

    assert predictor.predict_game(game("Team A", "Team B")).team1_win_prob == 0.5

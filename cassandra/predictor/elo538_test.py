from .conftest import GameFactory
from .elo538 import Elo538Predictor

# Everything Elo538 shares with the other models lives in contract_test.py.


def test_a_blowout_moves_the_ratings_more_than_a_squeaker(
    game: GameFactory,
) -> None:
    """The margin-of-victory multiplier is the whole reason this isn't EloPredictor."""
    blowout = Elo538Predictor("test_league")
    blowout.update_game(game("Team A", "Team B", 42, 0))

    squeaker = Elo538Predictor("test_league")
    squeaker.update_game(game("Team A", "Team B", 1, 0))

    assert blowout.get_rating("Team A") > squeaker.get_rating("Team A")

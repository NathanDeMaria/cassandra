from endgame.types import Game

from .base_predictor import Predictor
from .types import Prediction


class EloPredictor(Predictor):
    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}

    def predict_game(self, game: Game) -> Prediction:
        home_rating = self.get_rating(game.home)
        away_rating = self.get_rating(game.away)

        win_prob = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        actual = (
            1.0
            if game.home_score > game.away_score
            else (0.5 if game.home_score == game.away_score else 0.0)
        )
        self._ratings[game.home] = home_rating + 20 * (actual - win_prob)
        self._ratings[game.away] = away_rating + 20 * (win_prob - actual)
        return Prediction(team1_win_prob=win_prob)

    def get_rating(self, team: str) -> float:
        return self._ratings.get(team, 1500)
from endgame.types import Game

from .base_predictor import Predictor
from .types import Prediction


class EloPredictor(Predictor):
    def __init__(self, league: str, home_advantage: float = 105, k: float = 20) -> None:
        super().__init__(league)
        self._ratings: dict[str, float] = {}
        self._home_advantage = home_advantage
        self._k = k

    def predict_game(self, game: Game) -> Prediction:
        home_rating = self.get_rating(game.home)
        adjusted_home_rating = home_rating
        if not game.neutral_site:
            adjusted_home_rating = home_rating + self._home_advantage
        away_rating = self.get_rating(game.away)

        win_prob = 1 / (1 + 10 ** ((away_rating - adjusted_home_rating) / 400))
        actual = (
            1.0
            if game.home_score > game.away_score
            else (0.5 if game.home_score == game.away_score else 0.0)
        )
        self._ratings[game.home] = home_rating + self._k * (actual - win_prob)
        self._ratings[game.away] = away_rating + self._k * (win_prob - actual)
        return Prediction(team1_win_prob=win_prob)

    def get_rating(self, team: str) -> float:
        return self._ratings.get(team, 1500)

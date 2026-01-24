import numpy as np
from endgame.types import Game

from .base_predictor import Predictor
from .types import Prediction


class Elo538Predictor(Predictor):
    """Elo, but based on 538's tweaks.

    https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
    """

    def __init__(self, home_advantage: float = 105, k: float = 20) -> None:
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

        mov = abs(game.home_score - game.away_score)
        winner_elo_diff = abs(home_rating - away_rating)
        mov_multiplier = np.log(1 + mov) / np.log(np.e) * 2.2 / (winner_elo_diff * 0.001 + 2.2)
        actual = (
            1.0
            if game.home_score > game.away_score
            else (0.5 if game.home_score == game.away_score else 0.0)
        )
        self._ratings[game.home] = home_rating + self._k * (actual - win_prob) * mov_multiplier
        self._ratings[game.away] = away_rating + self._k * (win_prob - actual) * mov_multiplier
        return Prediction(team1_win_prob=win_prob)

    def get_rating(self, team: str) -> float:
        return self._ratings.get(team, 1500)
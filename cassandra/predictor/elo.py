import json
from pathlib import Path
from typing import Self

from endgame.types import Game

from .base_predictor import Predictor
from .types import Matchup, Prediction


class EloPredictor(Predictor):
    def __init__(
        self,
        league: str,
        home_advantage: float = 105,
        k: float = 20,
        ratings: dict[str, float] | None = None,
    ) -> None:
        super().__init__(league)
        self._ratings: dict[str, float] = ratings or {}
        self._home_advantage = home_advantage
        self._k = k

    def predict_game(self, matchup: Matchup) -> Prediction:
        home_rating = self.get_rating(matchup.home)
        adjusted_home_rating = home_rating
        if not matchup.neutral_site:
            adjusted_home_rating = home_rating + self._home_advantage
        away_rating = self.get_rating(matchup.away)
        win_prob = 1 / (1 + 10 ** ((away_rating - adjusted_home_rating) / 400))
        return Prediction(team1_win_prob=win_prob)

    def update_game(self, game: Game) -> Prediction:
        prediction = self.predict_game(game)
        home_rating = self.get_rating(game.home)
        away_rating = self.get_rating(game.away)
        actual = (
            1.0
            if game.home_score > game.away_score
            else (0.5 if game.home_score == game.away_score else 0.0)
        )
        self._ratings[game.home] = home_rating + self._k * (
            actual - prediction.team1_win_prob
        )
        self._ratings[game.away] = away_rating + self._k * (
            prediction.team1_win_prob - actual
        )
        return prediction

    def get_rating(self, team: str) -> float:
        return self._ratings.get(team, 1500)

    def save_state(self, path: Path) -> None:
        data = {
            "league": self._league,
            "home_advantage": self._home_advantage,
            "k": self._k,
            "ratings": self._ratings,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load_state(cls, path: Path) -> Self:
        data = json.loads(path.read_text())
        return cls(**data)

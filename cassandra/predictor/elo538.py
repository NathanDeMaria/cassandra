import json
from pathlib import Path
from typing import Self

import numpy as np
from endgame.types import Game

from .base_predictor import Predictor
from .opponent_prior import OpponentPriorManager
from .types import Matchup, Prediction


class Elo538Predictor(Predictor):
    """Elo, but based on 538's tweaks.

    https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 105,
        k: float = 20,
        opponent_prior_manager: OpponentPriorManager | None = None,
        ratings: dict[str, float] | None = None,
    ) -> None:
        super().__init__(league)
        self._home_advantage = home_advantage
        self._k = k

        self._prior_manager = opponent_prior_manager or OpponentPriorManager(league)
        self._ratings = ratings or self._prior_manager.get_ratings()

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

        mov = abs(game.home_score - game.away_score)
        winner_elo_diff = abs(home_rating - away_rating)
        mov_multiplier = (
            np.log(1 + mov) / np.log(np.e) * 2.2 / (winner_elo_diff * 0.001 + 2.2)
        )
        actual = (
            1.0
            if game.home_score > game.away_score
            else (0.5 if game.home_score == game.away_score else 0.0)
        )
        win_prob = prediction.team1_win_prob
        self._ratings[game.home] = (
            home_rating + self._k * (actual - win_prob) * mov_multiplier
        )
        self._ratings[game.away] = (
            away_rating + self._k * (win_prob - actual) * mov_multiplier
        )

        self._prior_manager.add_game(game)
        return prediction

    def get_rating(self, team: str) -> float:
        return self._ratings.get(team, 1500)

    def postrun_callback(self) -> None:
        self._prior_manager.save(self._ratings)

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

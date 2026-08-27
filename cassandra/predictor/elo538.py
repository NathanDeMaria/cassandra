from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from endgame.types import Game

from .base_predictor import (
    Predictor,
    resolved_anchors,
    validated_regression,
)
from .opponent_prior import OpponentPriorManager
from .types import Matchup, Prediction, Rating


class Elo538Predictor(Predictor):
    """Elo, but based on 538's tweaks.

    https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 105,
        k: float = 20,
        season_regression: float = 0.0,
        opponent_prior_manager: OpponentPriorManager | None = None,
        ratings: dict[str, float] | None = None,
        anchors: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(league)
        self._anchors = resolved_anchors(league, anchors)
        self._season_regression = validated_regression(season_regression)
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
        # See EloPredictor.get_rating. The opponent priors already cover most
        # teams by the time the search runs, but they're built by a warm-up
        # replay that comes through here first, so the anchor is what that
        # replay -- and so the priors it saves -- start from.
        return self._ratings.get(team, self.anchor(team))

    def pass_season(self) -> None:
        self._ratings = {
            team: self.regress(team, rating) for team, rating in self._ratings.items()
        }

    def postrun_callback(self) -> None:
        self._prior_manager.save(self._ratings)

    def state_dict(self) -> dict[str, Any]:
        return {
            "league": self._league,
            "home_advantage": self._home_advantage,
            "k": self._k,
            "season_regression": self._season_regression,
            "ratings": self._ratings,
            "anchors": self._anchors,
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

    @property
    def ratings(self) -> dict[str, Rating]:
        return {team: Rating(rating) for team, rating in self._ratings.items()}

    @classmethod
    def from_ratings(
        cls, league: str, ratings: Mapping[str, Rating], **params: Any
    ) -> Self:
        predictor = cls(league, **params)
        # Assigned rather than passed to __init__, which falls back to the
        # locally saved priors for an empty dict. A release's ratings are the
        # ratings, even when it has none.
        predictor._ratings = {team: r.rating for team, r in ratings.items()}
        return predictor

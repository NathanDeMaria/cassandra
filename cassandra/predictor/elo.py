from collections.abc import Mapping
from typing import Any, Self

from endgame.types import Game

from .adjustments import (
    DEFAULT_QB_OUT_PENALTY,
    DEFAULT_TRAVEL_ADVANTAGE,
    MatchupAdjustments,
)
from .base_predictor import (
    Anchor,
    Predictor,
    resolved_anchors,
    validated_regression,
)
from .qb_out import QbOutIndex
from .rest import DEFAULT_REST_ADVANTAGE
from .types import Matchup, Prediction, Rating


class EloPredictor(Predictor):
    def __init__(
        self,
        league: str,
        home_advantage: float = 105,
        k: float = 20,
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        season_regression: float = 0.0,
        ratings: dict[str, float] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
        qb_out: QbOutIndex | None = None,
    ) -> None:
        super().__init__(league)
        self._anchors = resolved_anchors(league, anchors)
        self._season_regression = validated_regression(season_regression)
        self._ratings: dict[str, float] = ratings or {}
        self._home_advantage = home_advantage
        self._k = k
        self._adjustments = MatchupAdjustments(
            rest_advantage=rest_advantage,
            travel_advantage=travel_advantage,
            qb_out_penalty=qb_out_penalty,
            qb_out=qb_out if qb_out is not None else QbOutIndex.for_league(league),
        )

    def predict_game(self, matchup: Matchup) -> Prediction:
        home_rating = self.get_rating(matchup.home)
        adjusted_home_rating = home_rating
        if not matchup.neutral_site:
            adjusted_home_rating = home_rating + self._home_advantage
        # Outside the neutral-site guard: see GlickoPredictor.predict_game.
        adjusted_home_rating += self.matchup_adjustment(matchup)
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
        # After the prediction, so a game never contributes to its own rest.
        self._adjustments.record(game)
        return prediction

    def get_rating(self, team: str) -> float:
        # A team we haven't seen play starts at its anchor, not at the league
        # mean: for ncaafb that's its division's fitted level, which is the
        # only moment the tier gap can enter a rating. Once a D-III team has
        # played, every result it has is against other D-III teams, so nothing
        # downstream can put it on the same scale as an FBS team.
        return self._ratings.get(team, self.anchor(team))

    def _roll_over(self) -> None:
        self._ratings = {
            team: self.regress(team, rating) for team, rating in self._ratings.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "league": self._league,
            "home_advantage": self._home_advantage,
            "k": self._k,
            "rest_advantage": self._adjustments.rest.per_day,
            "travel_advantage": self._adjustments.travel_advantage,
            "qb_out_penalty": self._adjustments.qb_out_penalty,
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
        # No rd here: Elo has nowhere to put one, so a release carrying one
        # loses it rather than having it folded into the rating.
        return cls(
            league,
            ratings={team: r.rating for team, r in ratings.items()},
            **params,
        )

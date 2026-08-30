import math
from collections.abc import Mapping
from typing import Any, NamedTuple, Self

from endgame.types import Game

from ..scoring import get_scoring_function
from .base_predictor import (
    Anchor,
    Predictor,
    resolved_anchors,
    validated_regression,
)
from .opponent_prior import OpponentPriorManager
from .types import Matchup, Prediction, Rating


class _Rating(NamedTuple):
    rating: float
    rating_deviation: float


_Q = math.log(10) / 400


class GlickoPredictor(Predictor):
    """Glicko rating system.

    https://en.wikipedia.org/wiki/Glicko_rating_system
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        k: float = 65,
        # increase is 'c' in the paper
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        scoring_method: str = "binary",
        season_regression: float = 0.0,
        opponent_prior_manager: OpponentPriorManager | None = None,
        ratings: dict[str, _Rating] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
    ) -> None:
        super().__init__(league)
        self._anchors = resolved_anchors(league, anchors)
        self._season_regression = validated_regression(season_regression)
        self._home_advantage = home_advantage
        self._k = k
        self._weekly_rd_increase = weekly_rd_increase
        self._season_rd_increase = season_rd_increase
        self._initial_rd = initial_rd
        self._scoring_method = scoring_method
        self._score = get_scoring_function(scoring_method)

        self._prior_manager = opponent_prior_manager or OpponentPriorManager(
            league, model=self.__class__.__name__
        )
        if ratings is not None:
            self._ratings: dict[str, _Rating] = ratings
        else:
            prior_ratings = self._prior_manager.get_ratings()
            self._ratings = {
                team: _Rating(rating, self._initial_rd)
                for team, rating in prior_ratings.items()
            }

    def predict_game(self, matchup: Matchup) -> Prediction:
        home_rating = self.get_rating(matchup.home)
        adjusted_home_rating = home_rating.rating
        if not matchup.neutral_site:
            adjusted_home_rating += self._home_advantage
        away_rating = self.get_rating(matchup.away)
        win_prob = 1 / (1 + 10 ** ((away_rating.rating - adjusted_home_rating) / 400))
        return Prediction(team1_win_prob=win_prob)

    def update_game(self, game: Game) -> Prediction:
        prediction = self.predict_game(game)
        home_rating = self.get_rating(game.home)
        away_rating = self.get_rating(game.away)
        actual = self._score(game)
        home_adj = 0 if game.neutral_site else self._home_advantage

        self._update_rating(
            game.home, home_rating, away_rating, actual, home_adjustment=home_adj
        )
        self._update_rating(
            game.away, away_rating, home_rating, 1 - actual, home_adjustment=-home_adj
        )

        self._prior_manager.add_game(game)
        return prediction

    def _update_rating(
        self,
        team: str,
        my_rating: _Rating,
        opp_rating: _Rating,
        score: float,
        home_adjustment: float,
    ) -> None:
        g_opp = _g(opp_rating.rating_deviation)
        expected_score = 1 / (
            1
            + 10
            ** (
                g_opp * (opp_rating.rating - (my_rating.rating + home_adjustment)) / 400
            )
        )
        d2 = 1 / (_Q**2 * g_opp**2 * expected_score * (1 - expected_score))
        rd_inv_sq = 1 / my_rating.rating_deviation**2
        rd_inv_plus_d2 = rd_inv_sq + 1 / d2
        rd_new = math.sqrt(1 / rd_inv_plus_d2)
        rating_new = my_rating.rating + (_Q / rd_inv_plus_d2) * g_opp * (
            score - expected_score
        )
        self._ratings[team] = _Rating(rating_new, rd_new)

    def get_rating(self, team: str) -> _Rating:
        # See EloPredictor.get_rating. The rd stays `initial_rd`: knowing which
        # division a team plays in says where its rating starts, not how sure
        # we are of it, and folding the anchor into the deviation would make
        # D-III teams look better-measured than anyone has evidence for.
        return self._ratings.get(team, _Rating(self.anchor(team), self._initial_rd))

    def pass_week(self) -> None:
        self._ratings = {
            team: _Rating(
                rating.rating,
                min(
                    self._initial_rd,
                    math.sqrt(rating.rating_deviation**2 + self._weekly_rd_increase**2),
                ),
            )
            for team, rating in self._ratings.items()
        }

    def _roll_over(self) -> None:
        self._ratings = {
            team: _Rating(
                self.regress(team, rating.rating),
                min(
                    self._initial_rd,
                    math.sqrt(rating.rating_deviation**2 + self._season_rd_increase**2),
                ),
            )
            for team, rating in self._ratings.items()
        }

    def postrun_callback(self) -> None:
        self._prior_manager.save(
            {team: rating.rating for team, rating in self._ratings.items()}
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "league": self._league,
            "home_advantage": self._home_advantage,
            "k": self._k,
            "weekly_rd_increase": self._weekly_rd_increase,
            "season_rd_increase": self._season_rd_increase,
            "initial_rd": self._initial_rd,
            "scoring_method": self._scoring_method,
            "season_regression": self._season_regression,
            "ratings": {
                team: [r.rating, r.rating_deviation]
                for team, r in self._ratings.items()
            },
            "anchors": self._anchors,
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        params = dict(data)
        ratings = params.pop("ratings")
        return cls(
            **params,
            ratings={team: _Rating(r[0], r[1]) for team, r in ratings.items()},
        )

    @property
    def ratings(self) -> dict[str, Rating]:
        return {
            team: Rating(r.rating, r.rating_deviation)
            for team, r in self._ratings.items()
        }

    @classmethod
    def from_ratings(
        cls, league: str, ratings: Mapping[str, Rating], **params: Any
    ) -> Self:
        # Built empty first so `initial_rd` comes from params (or the
        # constructor default) rather than being repeated here.
        predictor = cls(league, ratings={}, **params)
        predictor._ratings = {
            team: _Rating(
                r.rating,
                # A rating with no rd is one we've never seen play, which is
                # exactly what get_rating hands back for an unknown team.
                r.rd if r.rd is not None else predictor._initial_rd,
            )
            for team, r in ratings.items()
        }
        return predictor


def _g(rating_deviation: float) -> float:
    return 1 / math.sqrt(1 + 3 * _Q**2 * rating_deviation**2 / math.pi**2)

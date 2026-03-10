import math
from typing import NamedTuple

from endgame.types import Game

from ..scoring import get_scoring_function
from .base_predictor import Predictor
from .opponent_prior import OpponentPriorManager
from .types import Prediction


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
        opponent_prior_manager: OpponentPriorManager | None = None,
    ) -> None:
        super().__init__(league)
        self._home_advantage = home_advantage
        self._k = k
        self._weekly_rd_increase = weekly_rd_increase
        self._season_rd_increase = season_rd_increase
        self._initial_rd = initial_rd
        self._score = get_scoring_function(scoring_method)

        self._prior_manager = opponent_prior_manager or OpponentPriorManager(
            league, model=self.__class__.__name__
        )
        prior_ratings = self._prior_manager.get_ratings()
        self._ratings: dict[str, _Rating] = {
            team: _Rating(rating, self._initial_rd)
            for team, rating in prior_ratings.items()
        }

    def predict_game(self, game: Game) -> Prediction:
        home_rating = self.get_rating(game.home)
        adjusted_home_rating = home_rating.rating
        if not game.neutral_site:
            adjusted_home_rating += self._home_advantage
        away_rating = self.get_rating(game.away)

        win_prob = 1 / (1 + 10 ** ((away_rating.rating - adjusted_home_rating) / 400))

        actual = self._score(game)

        self._update_rating(
            game.home,
            home_rating,
            away_rating,
            actual,
            home_adjustment=self._home_advantage if not game.neutral_site else 0,
        )

        self._update_rating(
            game.away,
            away_rating,
            home_rating,
            1 - actual,
            home_adjustment=-(self._home_advantage if not game.neutral_site else 0),
        )

        self._prior_manager.add_game(game)

        return Prediction(team1_win_prob=win_prob)

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
        return self._ratings.get(team, _Rating(1500, self._initial_rd))

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

    def pass_season(self):
        self._ratings = {
            team: _Rating(
                rating.rating,
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


def _g(rating_deviation: float) -> float:
    return 1 / math.sqrt(1 + 3 * _Q**2 * rating_deviation**2 / math.pi**2)

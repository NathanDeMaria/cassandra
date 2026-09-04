import math
from collections.abc import Mapping
from typing import Any, Self

from endgame.types import Game

from .base_predictor import (
    Anchor,
    Predictor,
    resolved_anchors,
    validated_regression,
)
from .types import Matchup, Prediction, Rating

# Converting a rating gap to the margin it implies, and back. An Elo gap `d`
# is a statement about a win probability -- p = 1 / (1 + 10 ** (-d / 400)) --
# so logit(p) = d * ln(10) / 400, and a margin of `margin_scale` per unit of
# logit puts the two in the same sentence.
_LOGIT_PER_ELO = math.log(10) / 400


class MarginEloPredictor(Predictor):
    """Elo whose update is driven by the margin, not the result.

    Every other model here learns from whether a team won -- `binary_score`
    outright, `sigmoid_score` and `pythagorean_score` as a softened version
    of the same question. This one learns from *how much* it won by: a team
    that was expected to win by 4 and won by 14 moves up, and a team that was
    expected to win by 14 and won by 4 moves down. That is the question a
    point spread asks, which is why this model exists alongside the others
    rather than instead of them; `brier` still judges the win probability and
    `margin_mae` judges this.

    Ratings stay on the league-wide Elo scale -- 1500-centered, in the same
    units as `EloPredictor` and the fitted division anchors -- rather than
    being kept in points. The model is no less margin-native for it: the
    conversion is one constant, and keeping it here means a release's ratings
    can be read next to any other model's, and that an anchor fit by
    `division_anchors.py` means the same thing to this model as to the ones
    it was fit against. Points are what the model *thinks in*; Elo is the
    scale it reports on.

    `margin_scale` is the exchange rate: points of margin per unit of logit.
    Its job is turning the expected margin into the win probability the rest
    of the pipeline consumes, and the margin arithmetic is deliberately
    arranged so that it does almost nothing else: `home_advantage` is quoted
    in points, and a rating move is the surprise converted back through the
    same constant it was expected through, so the ratings stretch and the
    predicted margins come out the same. Against a league whose teams all
    start level, the margins are *identical* at any `margin_scale` and only
    the probabilities move.

    The exception is per-team anchors, which are quoted on the rating scale:
    the gap between two divisions' anchors is worth more points at a wider
    scale, so a league that has them -- mens, womens, ncaafb -- keeps a weak
    dependence through the ratings teams start at. Weak is measured: over
    mens, `margin_scale` from 6 to 15 moves margin MAE by 0.025 points and
    brier by 0.013, five hundred times more on the probability than on the
    margin.

    So a `margin_mae` search has almost nothing to learn about it and a brier
    run has a great deal: pin it in a margin config at what a brier scan
    says, rather than spending a search dimension the objective can barely
    see. For mens that scan puts it at 7.
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 3.5,
        k: float = 0.06,
        margin_scale: float = 11.0,
        mov_cap: float = 40.0,
        season_regression: float = 0.0,
        ratings: dict[str, float] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
    ) -> None:
        super().__init__(league)
        self._anchors = resolved_anchors(league, anchors)
        self._season_regression = validated_regression(season_regression)
        self._ratings: dict[str, float] = ratings or {}
        self._home_advantage = home_advantage
        self._k = k
        if margin_scale <= 0:
            # Every prediction divides by it, and a search handed a range
            # that reaches 0 would otherwise report a ZeroDivisionError from
            # inside a probe rather than from the config that asked for it.
            raise ValueError(f"margin_scale must be positive, got {margin_scale}")
        self._margin_scale = margin_scale
        if mov_cap <= 0:
            # A cap of 0 makes every game a tie, so every rating stops moving
            # and the model quietly becomes the flat baseline.
            raise ValueError(f"mov_cap must be positive, got {mov_cap}")
        self._mov_cap = mov_cap

    @property
    def _points_per_rating(self) -> float:
        """Points of margin per point of rating gap.

        The rating scale is Elo's -- a 400-point gap is a 10-to-1 favorite --
        so the gap's meaning in points falls out of `margin_scale`.
        """
        return _LOGIT_PER_ELO * self._margin_scale

    def expected_margin(self, matchup: Matchup) -> float:
        """How much the home team is expected to win by. The model's own output.

        Public because it is the number this predictor actually produces --
        `predict_game` is the pipeline-facing wrapper that turns it into a
        probability -- and because a caller holding a fitted model shouldn't
        have to round-trip through a prob->margin calibration to ask the one
        question this model answers directly.
        """
        gap = self.get_rating(matchup.home) - self.get_rating(matchup.away)
        margin = gap * self._points_per_rating
        if not matchup.neutral_site:
            # In points, not rating: what a home floor is worth is a number
            # of baskets, and quoting it that way is also what keeps it
            # independent of `margin_scale`.
            margin += self._home_advantage
        return margin

    def _prediction(self, margin: float) -> Prediction:
        """The expected margin, as the win probability the pipeline reads.

        The inverse of the conversion in `expected_margin`, which makes this
        the same logistic in the rating gap that EloPredictor uses. The
        margin isn't lost on the way out: it's a fixed multiple of the logit,
        so the prob->margin fit downstream recovers it up to the scale it
        fits anyway.
        """
        return Prediction(
            team1_win_prob=1 / (1 + math.exp(-margin / self._margin_scale))
        )

    def predict_game(self, matchup: Matchup) -> Prediction:
        return self._prediction(self.expected_margin(matchup))

    def update_game(self, game: Game) -> Prediction:
        # One `expected_margin` for both the prediction and the update: they
        # are the same number, and a search asks for it a hundred million
        # times.
        expected = self.expected_margin(game)
        prediction = self._prediction(expected)
        # Capped, because the tail of a basketball scoreline is about how
        # long the starters stayed in. A 50-point win says the same thing
        # about a team as a 40-point one and carries several times the
        # rating move, so past the cap the extra points are noise the model
        # would otherwise chase.
        actual = max(
            -self._mov_cap, min(self._mov_cap, game.home_score - game.away_score)
        )
        # The surprise, converted back to the rating scale so `k` is a plain
        # fraction: at k = 0.1 a team that beats its number by 10 gains a
        # point of expected margin.
        delta = self._k * (actual - expected) / self._points_per_rating
        self._ratings[game.home] = self.get_rating(game.home) + delta
        self._ratings[game.away] = self.get_rating(game.away) - delta
        return prediction

    def get_rating(self, team: str) -> float:
        # See EloPredictor.get_rating: a team nobody has seen play starts at
        # its anchor, which is the one moment a division gap can enter.
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
            "margin_scale": self._margin_scale,
            "mov_cap": self._mov_cap,
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
        # No rd, for the same reason EloPredictor has none: there's nowhere
        # to put one, and folding it into the rating would be a lie.
        return cls(
            league,
            ratings={team: r.rating for team, r in ratings.items()},
            **params,
        )

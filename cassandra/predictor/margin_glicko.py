"""Glicko that observes the margin, instead of a squashed copy of it.

`GlickoPredictor` is a Bayesian rating with a Bernoulli observation: the
game's *score* in [0, 1] against the logistic expectation of the rating gap,
and `glicko_step` is a gradient step on that likelihood. `sigmoid` scoring
keeps the Bernoulli algebra and feeds it `sigmoid(margin / scale)` as a soft
label. This model keeps the rating, the deviation, the clocks, the anchors,
the matchup terms and the smoother, and swaps the observation: the margin
itself, in points, with Gaussian noise -- which makes the step a Kalman
update -- or Student-t noise, which makes it a robust one.

Why the observation and not the scoring function
------------------------------------------------

`sigmoid(m / s)` is not the score function of any margin distribution.
Written out it is `sigmoid(logit(E) + (m - mu) / s)`, the expected logit plus
the surprise, squashed back to [0, 1]; the squash saturates *around E*, so a
28-point favorite that wins by 35 more than expected is credited +0.19 and
one that loses by 7 is charged -0.17 for a surprise of the same size, and a
favorite's whole response runs at half a pick'em's. Measured on ncaafb
`glicko_full`, that is the favorite compression -- the actual margin on the
predicted one has slope 0.986 over 21-point favorites -- and the update's
under-credit of blowouts. See `reasoning/residual_sweep_2026-09.md`.

A likelihood on the margin has no such asymmetry. The direction a result
moves the gap is `psi(m - mu) = -f'/f` of the noise density, a function of
the *surprise*: linear for a Gaussian, `tanh` for a logistic, redescending
for a t. And the residuals say which. ncaafb `glicko_full`'s margin
residuals have excess kurtosis 0.38 (FBS-vs-FBS: 0.16), fit a t at nu = 23
(FBS: 37), and a Gaussian beats a logistic in log-likelihood; the 99.9%
quantile is 59 points against a Gaussian's 53. Essentially linear through
+/-20, a gentle tail beyond, never a discount to zero. So the default is the
Gaussian and `nu` is there to let a search say otherwise -- and prototyped,
it said no: `nu` 4 and 8 both lost to the Gaussian on brier and margin.

The step
--------

For a team with rating `r` and deviation `d`, against an opponent `(r', d')`,
with `h` the home edge in rating units and `c` points per rating unit:

    mu    = c * (r + h - r')                    expected margin, points
    S     = sigma^2 + c^2 (d^2 + d'^2)          innovation variance
    w     = (nu + 1) / (nu + (m - mu)^2 / S)    1 for the Gaussian
    R     = sigma^2 / (w c^2) + d'^2            observation noise, rating units
    K     = d^2 / (d^2 + R)
    r    += K (m - mu) / c
    d^2  *= 1 - K

The opponent's deviation is folded into the observation noise, which is what
Glicko's `g(RD)` does for the Bernoulli step: a result against a team nobody
has measured says less. `w` is the one-step IRLS weight of a t likelihood --
a surprise the noise can't explain is trusted less -- and at `nu = None` it
is 1 and the step is the Kalman filter exactly.

Both teams are stepped from their pre-game ratings, like the parent, and the
smoother (`passes`) replays the season with the same step and the opponent
held at its settled mean, like the parent's.

What it predicts
----------------

Under Gaussian noise the win is implied: `P(home) = Phi(mu / sd)`, with
`sd^2 = sigma^2 + c^2 (d^2 + d'^2)` -- the same innovation variance the
update uses, so a game between two unmeasured teams is called closer than
one between two known ones. That is what `prediction_scale = None` does, and
on ncaafb it is the best brier of any prediction tried (0.154282 against
0.154528 for the best constant sd).

The configs search `prediction_scale` instead -- the parent's logistic of
the rating gap -- for a reason that is about the pipeline rather than the
model. A prediction leaves here as one number, a win probability, and every
margin metric downstream (`margin_mae`, the spread record, `residuals`)
recovers a margin from it through one fitted prob->margin curve. A per-game
sd puts the same expected margin at different probabilities, and one curve
cannot undo that: the fitted t model's margin read straight off `mu` is MAE
12.83, and read back through its implied-sd probability 12.95. A constant
logistic scale round-trips exactly, the way the parent's does, and costs
about 0.0001 brier once searched. Until a prediction can carry its margin,
the logistic is the one to publish.

Ratings stay on the league-wide Elo scale, for the reason `MarginEloPredictor`
gives: a release's ratings read next to any other model's, and the anchors
`division_anchors.py` fits mean the same thing here as there. Points are what
the model thinks in; `points_per_rating` is the exchange rate. It is also
the one knob that stretches the anchors: a division ladder fit in rating
units is worth `c` points per rung, and the residual sweep found the
cross-tier gap over-dispersed by about a sixth under `glicko_full`'s
currency, so it is a knob and not a constant. The configs pin it at the
currency `glicko_full`'s fit gave the same anchors, because searched next to
the rating-unit terms it rides a ridge with all of them; open it alone.

What it was measured to be worth
--------------------------------

Prototyped against ncaafb 2002-2026 with `glicko_full`'s matchup terms and
anchors held, ~40 hand probes over `obs_sd` and the three clocks, the
prediction refit to a logistic after the fact:

    glicko_full (1000-probe search)     brier 0.154181   margin MAE 12.756   slope, 21+ favorites 0.986
    Gaussian, sd 13, rd 200 / 60 / 5    brier 0.154292   margin MAE 12.744   0.995
    t nu = 8, same clocks               brier 0.154647   margin MAE 12.768
    t nu = 4                            brier 0.154842   margin MAE 12.783

A tie with the searched model on 4% of the probes, a better margin, and the
favorite compression gone. What it does not do is remove the residual's
autocorrelation (+0.022 per point of previous surprise against 0.027): that
turned out to be a team changing under the model after a bad blowout, which
no observation model fixes. The clocks' meaning changes completely -- under a
Kalman gain `glicko_full`'s `initial_rd` of 663 lets the first game replace
the prior -- so nothing is pinned from that fit but the matchup terms.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Self

from endgame.types import Game

from .adjustments import (
    DEFAULT_QB_OUT_PENALTY,
    DEFAULT_TRAVEL_ADVANTAGE,
    MatchupSources,
)
from .base_predictor import Anchor
from .blend import validated_scale
from .glicko import DEFAULT_PREDICTION_SCALE, GlickoPredictor, _Played, _Rating
from .opponent_prior import OpponentPriorManager
from .rest import DEFAULT_REST_ADVANTAGE
from .types import Matchup, Prediction

#: Standard deviation of a game's margin about the ratings' expectation, in
#: points. 14 is where the ncaafb prototype landed (13 and 14 within a
#: thousandth of brier of each other, 16 and 12 both worse); the residual sd
#: of a fitted model is a little larger, 16, because it carries the ratings'
#: own uncertainty on top of the noise, which the step accounts for
#: separately.
DEFAULT_OBS_SD = 14.0

#: Points of margin per rating unit: the exchange rate between the Elo scale
#: the ratings and anchors are kept on and the points the model thinks in.
#: 0.14 is what the 2026-09 ncaafb `glicko_full` fit came to -- ratings
#: settle at `400 * margin / (sigmoid_scale * ln 10)` under sigmoid scoring,
#: and its scale was 24.4 -- so the same anchors read as the same points
#: here. Not a finding about the right stretch; see the module docstring.
DEFAULT_POINTS_PER_RATING = 0.14


class MarginGlickoPredictor(GlickoPredictor):
    """Glicko with a Gaussian or Student-t observation of the margin.

    The parent's every method does what it did except the two that touch
    the observation: `update_game` steps both ratings on the margin, and
    `_smooth` replays the season with the same step. The parent's
    `scoring_method` and `sigmoid_scale` are not taken: there is no score to
    squash.

    `nu` is `None` for the Gaussian; a number is the degrees of freedom of a
    t. `prediction_scale` is `None` for the implied Gaussian win probability
    and a rating gap for the parent's logistic; see the module docstring for
    why the configs search the second.
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        home_advantage_slope: float = 0.0,
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        obs_sd: float = DEFAULT_OBS_SD,
        nu: float | None = None,
        points_per_rating: float = DEFAULT_POINTS_PER_RATING,
        prediction_scale: float | None = None,
        passes: int = 1,
        rest_advantage: float = DEFAULT_REST_ADVANTAGE,
        travel_advantage: float = DEFAULT_TRAVEL_ADVANTAGE,
        qb_out_penalty: float = DEFAULT_QB_OUT_PENALTY,
        season_regression: float = 0.0,
        opponent_prior_manager: OpponentPriorManager | None = None,
        sources: MatchupSources | None = None,
        ratings: dict[str, _Rating] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
        unanchored_seen: Sequence[float] = (0.0, 0.0, 0),
    ) -> None:
        super().__init__(
            league,
            home_advantage=home_advantage,
            home_advantage_slope=home_advantage_slope,
            weekly_rd_increase=weekly_rd_increase,
            season_rd_increase=season_rd_increase,
            initial_rd=initial_rd,
            # The parent validates and stores it; `None` is this model's
            # "predict from the noise" and is remembered separately.
            prediction_scale=(
                DEFAULT_PREDICTION_SCALE
                if prediction_scale is None
                else prediction_scale
            ),
            passes=passes,
            rest_advantage=rest_advantage,
            travel_advantage=travel_advantage,
            qb_out_penalty=qb_out_penalty,
            season_regression=season_regression,
            opponent_prior_manager=opponent_prior_manager,
            sources=sources,
            ratings=ratings,
            anchors=anchors,
            unanchored_seen=unanchored_seen,
        )
        self._obs_sd = validated_scale("obs_sd", obs_sd)
        self._nu = None if nu is None else validated_scale("nu", nu)
        self._points_per_rating = validated_scale(
            "points_per_rating", points_per_rating
        )
        self._implied_prediction = prediction_scale is None

    @property
    def obs_sd(self) -> float:
        """Noise of a margin about its expectation, in points."""
        return self._obs_sd

    @property
    def nu(self) -> float | None:
        """Degrees of freedom of the t noise; `None` is Gaussian."""
        return self._nu

    @property
    def points_per_rating(self) -> float:
        """Points of margin one rating unit is worth."""
        return self._points_per_rating

    def _step(
        self, my: _Rating, opp: _Rating, margin: float, home_adjustment: float
    ) -> _Rating:
        """One Kalman step on the margin, robustified if `nu` says so.

        `margin` is from `my`'s side and `home_adjustment` is added to `my`'s
        rating before the expectation is taken, the same conventions
        `glicko_step` has. The module docstring has the algebra.
        """
        c = self._points_per_rating
        expected = c * (my.rating + home_adjustment - opp.rating)
        innovation = margin - expected
        noise = self._obs_sd**2
        if self._nu is not None:
            total = noise + c**2 * (my.rating_deviation**2 + opp.rating_deviation**2)
            noise /= (self._nu + 1) / (self._nu + innovation**2 / total)
        observation = noise / c**2 + opp.rating_deviation**2
        gain = my.rating_deviation**2 / (my.rating_deviation**2 + observation)
        return _Rating(
            my.rating + gain * innovation / c,
            math.sqrt(my.rating_deviation**2 * (1 - gain)),
        )

    def _margin_sd(self, home: _Rating, away: _Rating) -> float:
        """The innovation sd of a game between these two, in points."""
        return math.sqrt(
            self._obs_sd**2
            + self._points_per_rating**2
            * (home.rating_deviation**2 + away.rating_deviation**2)
        )

    def predict_game(self, matchup: Matchup) -> Prediction:
        """The parent's logistic at `prediction_scale`, or the Gaussian's own answer."""
        if not self._implied_prediction:
            return super().predict_game(matchup)
        home = self.get_rating(matchup.home)
        away = self.get_rating(matchup.away)
        edge = self.home_edge(matchup) + self.matchup_adjustment(matchup)
        expected = self._points_per_rating * (home.rating + edge - away.rating)
        return Prediction(
            team1_win_prob=_normal_cdf(expected / self._margin_sd(home, away))
        )

    def update_game(self, game: Game) -> Prediction:
        """The parent's update with the margin in place of the score.

        Both sides stepped from their pre-game ratings, the home edge read
        before the rest ledger records the game, and the game kept for the
        smoother with its margin where the parent keeps its score -- `_Played.
        actual` is what the game counted as for the home side, and here that
        is the margin.
        """
        prediction = self.predict_game(game)
        home = self.get_rating(game.home)
        away = self.get_rating(game.away)
        margin = float(game.home_score - game.away_score)
        home_adj = self.home_edge(game) + self.matchup_adjustment(game)
        self._ratings[game.home] = self._step(home, away, margin, home_adj)
        self._ratings[game.away] = self._step(away, home, -margin, -home_adj)
        self._this_week.append(_Played(game.home, game.away, margin, home_adj))
        self._prior_manager.add_game(game)
        self._adjustments.record(game)
        return prediction

    def _smooth(self) -> None:
        """The parent's sweep with the margin step; see `GlickoPredictor._smooth`."""
        settled = {team: rating.rating for team, rating in self._ratings.items()}
        replay = dict(self._preseason)

        def rating_of(team: str) -> _Rating:
            return replay.get(team, _Rating(self.anchor(team), self._initial_rd))

        for week in self._weeks:
            for played in week:
                home, away = rating_of(played.home), rating_of(played.away)
                replay[played.home] = self._step(
                    home,
                    _Rating(
                        settled.get(played.away, away.rating), away.rating_deviation
                    ),
                    played.actual,
                    played.home_adjustment,
                )
                replay[played.away] = self._step(
                    away,
                    _Rating(
                        settled.get(played.home, home.rating), home.rating_deviation
                    ),
                    -played.actual,
                    -played.home_adjustment,
                )
            replay = self._aged(replay, self._weekly_rd_increase)

        self._ratings = {
            team: _Rating(
                replay[team].rating if team in replay else rating.rating,
                rating.rating_deviation,
            )
            for team, rating in self._ratings.items()
        }

    def state_dict(self) -> dict[str, Any]:
        """The parent's state without the scales this model has no use for."""
        state = super().state_dict()
        for unused in ("k", "scoring_method", "sigmoid_scale"):
            state.pop(unused, None)
        return {
            **state,
            "prediction_scale": (
                None if self._implied_prediction else self._prediction_scale
            ),
            "obs_sd": self._obs_sd,
            "nu": self._nu,
            "points_per_rating": self._points_per_rating,
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        params = dict(data)
        ratings = params.pop("ratings")
        return cls(
            **params,
            ratings={team: _Rating(r[0], r[1]) for team, r in ratings.items()},
        )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

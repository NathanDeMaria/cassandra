"""The margin model, with control and EPA as more readings of the same gap.

`MarginGlickoPredictor` observes one number per game: the margin, with
Gaussian noise. Play-by-play offers two more readings of the same quantity --
how much better the home side was -- and they arrive in their own units:
game control as a share of the game, EPA as points the offenses added. Each
is a noisy linear function of the gap the margin is a noisy linear function
of, and the residuals say what kind of noise (`reasoning/residual_sweep_2026-
09.md`): logit(control) is Gaussian about the expected margin at 17 points
per logit (sd 1.13 logits), the EPA margin Gaussian-ish at 0.81 points per
point (sd 14, t at nu 14), and their surprises correlate with the
scoreboard's at 0.78 and 0.84.

So this model hands the Kalman step a **vector** observation:

    y = [ margin,  control_scale * logit(control),  epa_scale * epa_margin ]

all in points, with the noise covariance `R` between them, and the step is
the multivariate one:

    S  = c^2 (d^2 + d'^2) 1 1^T + R          innovation covariance
    K  = d^2 c 1^T S^-1                       gain, one number per reading
    r += K v                                  v = y - mu - b
    d^2 *= 1 - d^2 c^2 1^T S^-1 1

With only the margin present that is the parent's scalar step exactly. `S`
depends on the game only through the scalar `c^2 (d^2 + d'^2)`, so by
Sherman-Morrison `S^-1 1 = R^-1 1 / (1 + c^2 (d^2 + d'^2) 1^T R^-1 1)`: `R`
is solved once each time it changes and the step itself is arithmetic. The
correlation is what makes this the right way to combine them: three readings
whose errors move together are worth less than three independent ones, and
a blend that averaged them with searched weights had to learn that or
overfit it. Here it is measured.

What is measured and what is a knob
------------------------------------

`R`, and the intercepts `b` (control has a home tilt of about 0.3 logits
beyond the home advantage; the EPA clip understates blowouts), are
**estimated as the replay goes** from the innovations of games where all
three readings exist: running mean and covariance, with the ratings' own
uncertainty `c^2 (d^2 + d'^2)` -- common to every reading -- taken back out.
Until `MIN_COVARIANCE_GAMES` such games have been seen the step is the
margin alone. Nothing about the noise is searched, for the reason
`BlendedGlickoPredictor` gives against its own ten-knob search: the weights
of a blend are exactly the quantity the data can measure, and searching them
against brier came back worse than pinning them.

The two knobs are units. `control_scale` is points per logit of control and
`epa_scale` points per point of EPA margin; both have a measured value (17
and 1.24 on ncaafb) and both are searchable, because a per-play expected
points model and a win-probability curve are not the scoreboard and the
exchange rate is the one thing about them a search should be allowed to
doubt. `None` switches a reading off, and both off is the parent.

What it can and can't be
------------------------

Gaussian only: `nu` is refused. The t was measured to be the wrong shape for
the margin, and a multivariate t on correlated readings is a weight the
search would have to learn; the measured covariance is the honest version.

What it was measured to be worth
--------------------------------

Replayed on ncaafb at the 2026-09-19 `glicko_margin` fit, against the same
model with both readings off, on the 25,423 games with plays (and overall):

    reading                 brier, plays   margin MAE, plays   brier, all
    margin only               0.160817        12.613           0.154802
    + control at 17           0.160637        12.602           0.154737
    + control at 13           0.161422        12.652           0.155035
    + control at 21           0.161568        12.664           0.155085
    + control at 26           0.163229        12.772           0.155705
    + EPA at 1.24             0.161823        12.731           0.155260
    + EPA at 1.0              0.163201        12.838           0.155853
    + control 17 + EPA 1.24   0.161809        12.730           0.155256

Control helps, a little, and only at about the measured exchange rate --
13 and 21 are both worse than not reading it -- which is what a biased
reading does to a filter that trusts its covariance: the rate is the one
thing the noise estimate cannot fix. The earlier next-game test had control
carrying +0.34 points per sd of surprise beyond the scoreboard (t 3.0), and
this is what that is worth. EPA hurts at
every setting tried, alone or beside control. The measured noise says why:
its errors correlate with the margin's at 0.83, so the filter's best use of
it is the *difference* between the two readings, and the clip and the
special-teams points that difference is made of are not information about
the gap. That is the "second, slightly noisier copy of the final score"
`glicko_blend` measured, arriving through a filter that knows what to do
with a copy: ignore it, if it were told to. So EPA is off by default and the
control reading is on; the configs search the control exchange rate and a
sibling lets the search say whether EPA at some other rate is any better.
"""

import math
from collections.abc import Iterator, Mapping, Sequence
from itertools import combinations
from typing import Any, NamedTuple, Self

import numpy as np
from endgame.types import Game

from .adjustments import (
    DEFAULT_QB_OUT_PENALTY,
    DEFAULT_TRAVEL_ADVANTAGE,
    MatchupSources,
)
from .base_predictor import Anchor
from .blend import validated_scale
from .epa import EpaIndex
from .game_control import GameControlIndex
from .glicko import _Played, _Rating
from .margin_glicko import (
    DEFAULT_OBS_SD,
    DEFAULT_POINTS_PER_RATING,
    MarginGlickoPredictor,
)
from .opponent_prior import OpponentPriorManager
from .rest import DEFAULT_REST_ADVANTAGE
from .types import Prediction

#: Points of margin per logit of game control. The slope of logit(control) on
#: the expected margin over 25k ncaafb games was 0.059 logits per point.
DEFAULT_CONTROL_SCALE = 17.0

#: Points of margin per point of EPA margin, when it is read at all. EPA's
#: clip understates blowouts, so a point of it is a little more than a point
#: on the scoreboard: the slope of the EPA margin on the expected margin was
#: 0.81. The constructor's default is `None` -- off -- because the reading
#: was measured to hurt (module docstring); the constant is where a config
#: that wants to test it should start.
DEFAULT_EPA_SCALE = 1.24

#: Games with every reading present before the measured covariance is
#: trusted over the margin alone. A 3x3 covariance needs a few hundred
#: draws to be worth more than a guess; ncaafb has 25k such games, so this
#: is over inside the first season the index covers.
MIN_COVARIANCE_GAMES = 200

# Control at exactly 0 or 1 has no logit; the index clips the same way.
_CONTROL_EPS = 1e-3

# Observation indices.
_MARGIN, _CONTROL, _EPA = 0, 1, 2


class _Readings(NamedTuple):
    """One game's readings in points, home side, with `None` where absent."""

    margin: float
    control: float | None
    epa: float | None


class _Solved(NamedTuple):
    """`R^-1 1` over one subset of the readings, which is all the step needs of `R`.

    `kept` is the subset actually read: a present reading whose noise is not
    yet known is dropped, and contributes nothing.
    """

    kept: tuple[int, ...]
    weights: tuple[float, ...]  # R^-1 1
    total: float  # 1^T R^-1 1


class ObservationStats(NamedTuple):
    """Sufficient statistics for the readings' noise, gathered as the replay goes.

    Over games with every reading present: how many, the innovations summed,
    their outer products summed, and the ratings' own variance
    `c^2 (d^2 + d'^2)` summed -- the part of every innovation that is the
    ratings being unsure rather than the reading being noisy.
    """

    count: int
    total: list[float]
    outer: list[list[float]]
    rating_variance: float

    @classmethod
    def empty(cls) -> Self:
        return cls(0, [0.0, 0.0, 0.0], [[0.0] * 3 for _ in range(3)], 0.0)


class VectorMarginGlickoPredictor(MarginGlickoPredictor):
    """`MarginGlickoPredictor` reading control and EPA alongside the margin.

    The parent's update, smoother and state with the observation widened;
    `predict_game` is the parent's, since the readings change what a game
    teaches and not how a gap is read. `control_scale` and `epa_scale` at
    `None` are the parent exactly.
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
        points_per_rating: float = DEFAULT_POINTS_PER_RATING,
        prediction_scale: float | None = None,
        control_scale: float | None = DEFAULT_CONTROL_SCALE,
        epa_scale: float | None = None,
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
        observation_stats: Sequence[Any] | None = None,
        game_control: GameControlIndex | None = None,
        game_epa: EpaIndex | None = None,
    ) -> None:
        super().__init__(
            league,
            home_advantage=home_advantage,
            home_advantage_slope=home_advantage_slope,
            weekly_rd_increase=weekly_rd_increase,
            season_rd_increase=season_rd_increase,
            initial_rd=initial_rd,
            obs_sd=obs_sd,
            nu=None,
            points_per_rating=points_per_rating,
            prediction_scale=prediction_scale,
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
        self._control_scale = (
            None
            if control_scale is None
            else validated_scale("control_scale", control_scale)
        )
        self._epa_scale = (
            None if epa_scale is None else validated_scale("epa_scale", epa_scale)
        )
        self._stats = (
            ObservationStats.empty()
            if observation_stats is None
            else ObservationStats(
                int(observation_stats[0]),
                [float(v) for v in observation_stats[1]],
                [[float(v) for v in row] for row in observation_stats[2]],
                float(observation_stats[3]),
            )
        )
        # Defaulted to the league's saved indexes, like the compound model;
        # a test that wants none passes empty ones. Not carried in the state.
        self._game_control = game_control or GameControlIndex.for_league(league)
        self._game_epa = game_epa or EpaIndex.for_league(league)
        # The smoother needs each played game's readings again, and `_Played`
        # carries the margin only; the game ids ride alongside, one list per
        # week, in step with the parent's `_weeks`.
        self._week_ids: list[list[str]] = []
        self._this_week_ids: list[str] = []
        # `R` and the intercepts change only when a game is noted, and every
        # step in between reads the same solve; `_note` drops both.
        self._intercepts: tuple[float, ...] | None = None
        self._solved: dict[tuple[int, ...], _Solved] | None = None

    @property
    def control_scale(self) -> float | None:
        return self._control_scale

    @property
    def epa_scale(self) -> float | None:
        return self._epa_scale

    @property
    def observation_stats(self) -> ObservationStats:
        return self._stats

    # -- the readings ---------------------------------------------------------

    def _readings(self, game_id: str, margin: float) -> _Readings:
        """Every reading the indexes have for this game, in points, home side."""
        control = None
        if self._control_scale is not None:
            found = self._game_control.get(game_id)
            if found is not None:
                share = min(1 - _CONTROL_EPS, max(_CONTROL_EPS, found.home))
                control = self._control_scale * math.log(share / (1 - share))
        epa = None
        if self._epa_scale is not None:
            found_margin = self._game_epa.margin(game_id)
            if found_margin is not None:
                epa = self._epa_scale * found_margin
        return _Readings(margin, control, epa)

    # -- the measured noise ---------------------------------------------------

    def _covariance_known(self) -> bool:
        return self._stats.count >= MIN_COVARIANCE_GAMES

    def intercepts(self) -> np.ndarray:
        """Mean innovation of each reading; 0 for the margin, which the edge owns."""
        return np.array(self._intercepts_now())

    def _intercepts_now(self) -> tuple[float, ...]:
        if self._intercepts is None:
            n, total, _, _ = self._stats
            self._intercepts = (
                (0.0, 0.0, 0.0)
                if not self._covariance_known()
                else (0.0, *(t / n for t in total[1:]))
            )
        return self._intercepts

    def noise_covariance(self) -> np.ndarray:
        """`R`: the readings' noise about the gap, net of the ratings' own uncertainty.

        The innovations' covariance minus the mean rating variance on every
        entry (it is in every reading alike). The margin's own entry is held
        at `obs_sd^2`, the number the search owns, so the two models agree on
        what a margin is worth when the others are absent; the estimate fills
        in the rest. Kept positive-definite by falling back to the diagonal
        if the subtraction over-reaches, which it can early on.
        """
        n, total, outer, rating_variance = self._stats
        unknown = np.diag([self._obs_sd**2, np.inf, np.inf])
        if not self._covariance_known():
            return unknown
        mean = np.array(total) / n
        cov = np.array(outer) / n - np.outer(mean, mean)
        r = cov - rating_variance / n
        r[_MARGIN, _MARGIN] = self._obs_sd**2
        enabled = self._enabled()
        block = r[np.ix_(enabled, enabled)]
        floor = 1e-6
        diag = np.maximum(np.diag(block), floor)
        np.fill_diagonal(block, diag)
        try:
            np.linalg.cholesky(block)
        except np.linalg.LinAlgError:
            block = np.diag(diag)
        # A reading that is switched off is never present, so its entries
        # are never read; infinite is the honest placeholder.
        out = unknown.copy()
        out[np.ix_(enabled, enabled)] = block
        return out

    def _enabled(self) -> list[int]:
        """Which readings this model takes at all: the margin, and any scale that is on."""
        return [
            i
            for i, on in (
                (_MARGIN, True),
                (_CONTROL, self._control_scale is not None),
                (_EPA, self._epa_scale is not None),
            )
            if on
        ]

    def _note(self, innovations: np.ndarray, rating_variance: float) -> None:
        n, total, outer, rv = self._stats
        self._stats = ObservationStats(
            n + 1,
            [t + v for t, v in zip(total, innovations)],
            [
                [o + a * b for o, b in zip(row, innovations)]
                for row, a in zip(outer, innovations)
            ],
            rv + rating_variance,
        )
        self._intercepts = None
        self._solved = None

    # -- the step -------------------------------------------------------------

    def _present(self) -> Iterator[tuple[int, ...]]:
        """Every set of readings a game can present: the margin, plus any of the rest."""
        others = [i for i in self._enabled() if i != _MARGIN]
        for k in range(len(others) + 1):
            for chosen in combinations(others, k):
                yield (_MARGIN, *chosen)

    def _solved_now(self) -> dict[tuple[int, ...], _Solved]:
        """`R^-1 1` for every set of readings a game can present, at the current `R`."""
        if self._solved is None:
            r = self.noise_covariance()
            self._solved = {}
            for present in self._present():
                kept = tuple(i for i in present if math.isfinite(r[i, i]))
                block = r[list(kept), :][:, list(kept)]
                weights = np.linalg.solve(block, np.ones(len(kept)))
                self._solved[present] = _Solved(
                    kept, tuple(float(w) for w in weights), float(weights.sum())
                )
        return self._solved

    def _vector_step(
        self,
        my: _Rating,
        opp: _Rating,
        readings: _Readings,
        home_adjustment: float,
        sign: float,
    ) -> _Rating:
        """The multivariate Kalman step; `sign` flips the readings to `my`'s side."""
        c = self._points_per_rating
        expected = c * (my.rating + home_adjustment - opp.rating)
        intercepts = self._intercepts_now()
        present = {i: v for i, v in enumerate(readings) if v is not None}
        solved = self._solved_now()[tuple(present)]
        # 1^T R^-1 v: the innovations, weighted by what each reading is worth.
        innovation = sum(
            w * (sign * (present[i] - intercepts[i]) - expected)
            for i, w in zip(solved.kept, solved.weights)
        )
        variance = my.rating_deviation**2
        rating_variance = c**2 * (variance + opp.rating_deviation**2)
        # K = d^2 c 1^T S^-1 = d^2 c R^-1 1 / (1 + c^2 (d^2 + d'^2) 1^T R^-1 1)
        scale = variance * c / (1 + rating_variance * solved.total)
        return _Rating(
            my.rating + scale * innovation,
            math.sqrt(variance * max(0.0, 1 - scale * c * solved.total)),
        )

    def update_game(self, game: Game) -> Prediction:
        prediction = self.predict_game(game)
        home = self.get_rating(game.home)
        away = self.get_rating(game.away)
        margin = float(game.home_score - game.away_score)
        home_adj = self.home_edge(game) + self.matchup_adjustment(game)
        readings = self._readings(game.game_id, margin)
        values = (readings.margin, readings.control, readings.epa)
        if all(values[i] is not None for i in self._enabled()):
            # Fold this game's innovations in before stepping, so the first
            # game past the threshold is measured like the rest. Home side;
            # the away side's innovations are the same numbers negated and
            # would add nothing to the covariance. A reading that is switched
            # off goes in as nothing and is never read back.
            c = self._points_per_rating
            expected = c * (home.rating + home_adj - away.rating)
            self._note(
                np.array([0.0 if v is None else v - expected for v in values]),
                c**2 * (home.rating_deviation**2 + away.rating_deviation**2),
            )
        self._ratings[game.home] = self._vector_step(home, away, readings, home_adj, 1)
        self._ratings[game.away] = self._vector_step(
            away, home, readings, -home_adj, -1
        )
        self._this_week.append(_Played(game.home, game.away, margin, home_adj))
        self._this_week_ids.append(game.game_id)
        self._prior_manager.add_game(game)
        self._adjustments.record(game)
        return prediction

    def pass_week(self) -> None:
        self._week_ids.append(self._this_week_ids)
        self._this_week_ids = []
        super().pass_week()

    def _roll_over(self) -> None:
        super()._roll_over()
        self._week_ids = []
        self._this_week_ids = []

    def _smooth(self) -> None:
        """The parent's sweep with the vector step; see `GlickoPredictor._smooth`."""
        settled = {team: rating.rating for team, rating in self._ratings.items()}
        replay = dict(self._preseason)

        def rating_of(team: str) -> _Rating:
            return replay.get(team, _Rating(self.anchor(team), self._initial_rd))

        for week, ids in zip(self._weeks, self._week_ids):
            for played, game_id in zip(week, ids):
                home, away = rating_of(played.home), rating_of(played.away)
                readings = self._readings(game_id, played.actual)
                replay[played.home] = self._vector_step(
                    home,
                    _Rating(
                        settled.get(played.away, away.rating), away.rating_deviation
                    ),
                    readings,
                    played.home_adjustment,
                    1,
                )
                replay[played.away] = self._vector_step(
                    away,
                    _Rating(
                        settled.get(played.home, home.rating), home.rating_deviation
                    ),
                    readings,
                    -played.home_adjustment,
                    -1,
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
        state = super().state_dict()
        state.pop("nu", None)
        return {
            **state,
            "control_scale": self._control_scale,
            "epa_scale": self._epa_scale,
            "observation_stats": [
                self._stats.count,
                self._stats.total,
                self._stats.outer,
                self._stats.rating_variance,
            ],
        }

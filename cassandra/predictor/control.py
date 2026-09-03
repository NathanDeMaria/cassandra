"""Glicko, learning from a blend of the scoreboard and the play-by-play.

`cassandra.scoring` already reduces a game to one number for the home team
between 0 and 1, and Glicko updates against it. Game control is a second
answer to that same question -- the share of the game the home team spent
winning it, measured from the plays rather than from the final score -- so
`control_weight` interpolates between them. At 0 this is `GlickoPredictor`
exactly; at 1 the scoreboard only decides how many points were scored, not
what the game was worth.

A searched parameter rather than a constant, because how much of a result is
signal and how much is the bounces is a question about a league, not a thing
to assert -- and the search recovering 0 is a real answer, not a failed run.
It recovered 0, in both leagues; see below.

A child of Glicko rather than of Elo or 538 because Glicko is the model whose
result already goes through a scoring function: `_actual` is the seam, and
Elo derives its outcome inline from the sign of the margin. Its own class
rather than a parameter on `GlickoPredictor` so that
`predictor_class(league, **config.params)` stays typed against the
constructors that declare what they take, every release published so far
replays through code this never touches, and a `glicko_control` config could
compete as its own entry in `evaluate` instead of replacing `glicko_full`'s
converged search with one that has an extra dimension.

There is no such config any more. `models/nfl/glicko_control.json` and
`models/ncaafb/glicko_control.json` are deleted and the `game_control` node is
out of the DAG, so nothing scheduled constructs this class or reads the index
it blends. The class, the sweep and `jobs.py game-control` all still work,
because what follows is a map of where not to look next rather than a reason
to throw the seam away.

What this is measured to be worth, so nobody has to find out twice
------------------------------------------------------------------

It does not help. Against ncaafb's 25,307 games with play-by-play, holding
`glicko_full`'s fitted parameters and varying only the blend, the best weight
is ~0.25 and it improves brier by 0.00005. The gap between ncaafb's two best
models is 0.0088, so that is about 1/180th of the resolution at which model
choice matters here. nfl is worse: monotonically negative at every weight.

Searched jointly rather than holding the rest of Glicko fixed, both leagues
put the weight at 0.0 exactly -- 200 probes for nfl, 400 for ncaafb, with the
optimizer reporting the whole top decile crowded against the lower bound in
both. That is the stronger version of the paragraph above: given the freedom
to refit `season_rd_increase` instead, the search would rather have the
parameter off than have any of it. Since `blend` short-circuits on a zero
weight, those two searches were paying for a play-by-play sweep to produce a
number neither of them then read.

The reason it doesn't help is not scale. Standardising control to the scoring
function's exact mean and standard deviation recovers only 15% of the loss,
and rank-matching it onto that distribution exactly -- same marginal
distribution, ordering the only thing left that can differ -- recovers none.

What ties the negative results together is that every summary of the
scoreboard path costs brier in proportion to how far it reorders games away
from the final score, and by nothing else:

    cost = 0.0124 x (1 - rank correlation with the final score)   R^2 = 0.991

That was fit across eleven signals and holds to within 6e-5 for almost all of
them: control, the time-weighted average margin, the margin entering the
fourth quarter, the margin with 5:00 left, the largest lead either side held,
a low quantile of the win probability curve, the share of the clock spent
above 0.75, and the largest drawdown from each side's peak. A signal carrying
information the final score lacks would come in *below* that line. Only one
does -- the average over the game of each side's lowest *remaining* win
probability, by 0.00023 -- and it is negative at every blend weight, because
at 0.940 rank correlation with the final score it is closer to a noisy copy
of the scoreboard than to a second look at the game.

The one thing that does help is off that axis entirely. Success rate
differential -- the share of snaps gaining enough of the distance to stay on
schedule, which is `down`, `distance` and `yards_gained`, no model at all --
is a *worse* replacement for the final score (it sits on the law) and a
*better* supplement: blended at ~0.2 it improves brier by 0.00030, ten times
what control manages, with a smooth unimodal weight curve and a shuffle null
0.0014 away. It is the crude binary form of EPA, which is the signal worth
building next and which would arrive through this exact seam.
"""

from collections.abc import Mapping
from typing import Any

from endgame.types import Game

from .base_predictor import Anchor
from .game_control import GameControlIndex, validated_control_weight
from .glicko import GlickoPredictor, _Rating
from .opponent_prior import OpponentPriorManager

# Where a search starts, and what a hand-built one does with no argument:
# take the controlled line whole wherever there is one. This class exists to
# use the number, so 0 would make it silently identical to the model it
# subclasses -- a default that reads as a bug the first time someone builds
# one and sees Glicko's ratings come out.
DEFAULT_CONTROL_WEIGHT = 1.0


class ControlGlickoPredictor(GlickoPredictor):
    """Glicko, updated on a blend of what the score said and what the plays did.

    Scoring defaults to sigmoid rather than Glicko's binary, and the config
    leaves it out of the search rather than paying a dimension to rediscover
    it: the 2026-08-30 run put `glicko_full` on sigmoid in both football
    leagues -- nfl at 0.2216 brier, ncaafb at 0.1579 -- and in four of the
    other six besides. `scoring_method` is still a constructor argument, so
    trying another one is a keyword, not a subclass.

    One thing to hold while reading a fitted `control_weight`: control is
    pulled toward 0.5 by construction -- every game starts 0-0 at even odds,
    and those early snaps carry full clock weight -- so a wire-to-wire
    blowout tops out nearer 0.9 than 1.0. Against `binary_score`'s 1.0 that
    makes even a heavily weighted blend read as a slightly softer win;
    against `sigmoid_score`, whose own output is compressed by a hardcoded
    divisor of 10, the two are on more similar footing.
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        k: float = 65,
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        # Not Glicko's "binary" -- see the class docstring. The config omits
        # `scoring_method` from its search, so this default is what a probe
        # actually runs with, the way `glicko_binary.json` leans on Glicko's.
        scoring_method: str = "sigmoid",
        season_regression: float = 0.0,
        control_weight: float = DEFAULT_CONTROL_WEIGHT,
        opponent_prior_manager: OpponentPriorManager | None = None,
        ratings: dict[str, _Rating] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
        game_control: GameControlIndex | None = None,
    ) -> None:
        super().__init__(
            league,
            home_advantage=home_advantage,
            k=k,
            weekly_rd_increase=weekly_rd_increase,
            season_rd_increase=season_rd_increase,
            initial_rd=initial_rd,
            scoring_method=scoring_method,
            season_regression=season_regression,
            opponent_prior_manager=opponent_prior_manager,
            ratings=ratings,
            anchors=anchors,
        )
        self._control_weight = validated_control_weight(control_weight)
        # Defaulted rather than required, like `opponent_prior_manager`: every
        # caller wants the league's saved index, and a test that wants none
        # passes `GameControlIndex()`. Not carried in `state_dict` -- see
        # `state_dict` below.
        self._game_control = game_control or GameControlIndex.for_league(league)

    def _actual(self, game: Game) -> float:
        """Glicko's own answer, moved toward what the plays say.

        The whole of the model, and it replaces `_actual` rather than
        `update_game` because that is the only line of the update that is
        about the game rather than about the ratings. Nothing here builds a
        game that wasn't played, and nothing downstream can tell it was
        given anything other than a number between 0 and 1.
        """
        return self._game_control.blend(
            super()._actual(game), game.game_id, self._control_weight
        )

    def state_dict(self) -> dict[str, Any]:
        """The parent's state, plus the one parameter this class adds.

        The index itself stays out. It is tens of thousands of floats against
        a release that is otherwise a page of ratings, and unlike the anchors
        -- which are per-team and small enough that embedding them is what
        makes a release replay exactly -- it would dominate the artifact. The
        cost is that a replay picks up whatever sweep is on disk rather than
        the one the fit ran against.
        """
        return {**super().state_dict(), "control_weight": self._control_weight}

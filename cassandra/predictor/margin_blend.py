"""Margin Elo, rating a team on three answers to "how much better were you?"

`MarginEloPredictor` rates a team on the margin it beat -- expected to win by
4 and won by 14, so it moves up. The scoreboard is one answer to that
question, and the play-by-play has two more:

- **Game control**, the share of the game a team spent winning it, from the
  win probability curve with its fifty-fifty balls split evenly. A share, so
  it converts to points through `control_scale`.
- **EPA per play**, the expected points each offense added per snap. Already
  points, and `EpaIndex.margin` multiplies the two averages back through
  their play counts to get the game's worth of them.

All three are margins, which is what makes this a blend rather than a stack
of unrelated features: the model chases one number in points, and
`play_weight` and `epa_share` decide whose number it is. That is the
difference from `ControlGlickoPredictor`, which does the same trick on the
0-to-1 scale a scoring function produces -- there the second opinion had to be
a share, and EPA isn't one. Margin is the scale all three of these fit on.

Why bother, given control didn't work
-------------------------------------

`cassandra.predictor.control` is a long negative result and it is worth
reading before this class, because most of it still applies. Its finding, in
one line: every summary of the *scoreboard path* costs brier in proportion to
how far it reorders games away from the final score and by nothing else --

    cost = 0.0124 x (1 - rank correlation with the final score)   R^2 = 0.991

-- which held across twelve signals including both readings of game control.
A signal carrying information the final score lacks would come in below that
line, and control does not.

The one thing that came in below it was success rate differential -- the share
of snaps that stayed on schedule, no model at all -- which is a *worse*
replacement for the final score and a *better supplement*: blended at ~0.2 it
improved ncaafb brier by 0.00030, ten times what control managed. That module
says in as many words what to build next: "It is the crude binary form of EPA,
which is the signal worth building next and which would arrive through this
exact seam."

This is that. EPA is the graded version of the thing that worked, not another
reading of the thing that didn't, and it differs from control in the way the
law predicts matters: control is a summary of the same win probability curve
the final score largely determines, while EPA is scored off an expected points
model that never learns who won. Whether that is enough to come in under the
line is a measurement nobody has made yet.

**Nothing here is fitted.** Every default below is a starting point chosen to
be legible, not a number a search found -- unlike the parameters on
`MarginEloPredictor` and `ControlGlickoPredictor`, whose docstrings quote runs
that happened. The measurement this wants is a `margin_mae` search per league
(`models/nfl/margin_blend.json`, `models/ncaafb/margin_blend.json`), against
`margin_elo` on the same objective as the baseline it has to beat, and after a
`jobs.py epa` sweep because the index those configs read does not exist until
one runs. Until then the honest summary of this model is that it is plumbed,
tested and unmeasured.

Two things to expect from that search, so the result is readable. The first
is that `play_weight` may well come back at 0, the way the control searches
did -- which is a real answer and recovers `MarginEloPredictor` exactly, not
approximately, because the blend short-circuits on it. The second is that
coverage is most of what this can be run against: the index covers the NFL
from 2006 and about a third of an NCAAFB schedule, so the great majority of
NCAAFB games fall back to the scoreboard alone and the effective sample behind
a fitted `play_weight` is much smaller than the game count suggests.

Four knobs, three of them searchable
------------------------------------

What actually reaches the target margin is a weight times a scale --
`control_weight * control_scale` for one signal and `epa_weight * epa_scale`
for the other -- so searching all four would spend two dimensions on a product
the objective can only see one number of. The renormalization and the cap
break that degeneracy slightly, which is worse than not at all: a search would
find some ridge of equivalent models and report a corner of it as though the
corner meant something.

So `epa_scale` is pinned and the other three move. It is the one with a
natural value to be pinned at -- 1.0, EPA's points read as points -- while
`control_scale` has none, since a share of a game has no exchange rate into
points except the one a fit finds. That leaves `play_weight`, `epa_share` and
`control_scale`: three parameters for the scoreboard's weight and the two
signals' magnitudes, which is exactly determined rather than one short or one
over.

`margin_scale` is pinned too, for the reason `MarginEloPredictor` gives -- a
margin objective can barely see it and a brier scan is what should set it. The
values in the football configs are *not* from such a scan; mens' 7 is, and
these are the same reasoning applied by eye to leagues whose margins are
wider. A brier scan per league is the small measurement that would replace
them.
"""

from collections.abc import Mapping
from typing import Any

from endgame.types import Game

from .base_predictor import Anchor
from .blend import (
    DEFAULT_EPA_SHARE,
    DEFAULT_PLAY_WEIGHT,
    PlayBlend,
    validated_scale,
)
from .epa import EpaIndex
from .game_control import GameControlIndex
from .margin_elo import MarginEloPredictor

# Points of margin per unit of control, where "a unit" is the whole game: a
# team that controlled all of it beats a team that controlled none of it by
# this much.
#
# A placeholder, and the one number here most in need of a search. Control is
# pulled toward 0.5 by construction -- every game starts 0-0 at even odds, and
# those early snaps carry full clock weight -- so a wire-to-wire blowout tops
# out nearer 0.9 than 1.0, which at this scale reads as a 24-point win. That
# is about what such a game is, which is the whole of the argument for 30.
DEFAULT_CONTROL_SCALE = 30.0

# What a point of net EPA is worth as a point of margin.
#
# 1.0 means taking `EpaIndex.margin` at face value: the expected points a
# team added really are points, so the default is not to rescale them. It is
# still a searched parameter, because the number is systematically small --
# every play is bounded at the clip, so blowouts are understated -- and a
# search is the thing that can say by how much.
DEFAULT_EPA_SCALE = 1.0


class BlendedMarginEloPredictor(MarginEloPredictor):
    """Margin Elo whose target margin is a blend of the three.

    The blend is a convex combination over whichever of the three a game
    actually has, in points:

        (1 - play_weight)                * the final margin
        play_weight * (1 - epa_share)    * what control implies
        play_weight * epa_share          * what EPA implies

    **Two knobs rather than three weights**, because every point of the
    `[0, 1] x [0, 1]` box they span is a valid model. Two independent weights
    would let a search propose `control_weight = 0.8, epa_weight = 0.7`, which
    is not a blend of anything, and every way of handling that -- rejecting
    it, renormalizing it, clipping it -- makes the search's own record of what
    it tried disagree with what it ran. `control_weight` and `epa_weight` are
    properties, so a fitted model still reports the three numbers a reader
    wants.

    **A game contributes only the signals it has**, and the weights
    renormalize over what is left. Most of an NCAAFB schedule has no
    play-by-play at all and none of either league does before 2006, so this
    is the common case rather than the edge one -- a game with neither signal
    is rated on its final margin exactly as `MarginEloPredictor` would rate
    it, with no fallback rule of its own. It is also why `play_weight = 1` is
    a coherent setting: a game with no plays still gets rated, on the only
    number it has.

    **The cap applies twice, and does nothing the second time in the ordinary
    case.** The final margin is capped on the way in, as it is in the parent,
    so at `play_weight = 0` this is byte-identical to `MarginEloPredictor`.
    The blend is capped on the way out, which matters only when a play-derived
    margin is itself past the cap -- a convex combination of three numbers
    inside the cap is inside it too.
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 3.5,
        k: float = 0.06,
        margin_scale: float = 11.0,
        mov_cap: float = 40.0,
        season_regression: float = 0.0,
        play_weight: float = DEFAULT_PLAY_WEIGHT,
        epa_share: float = DEFAULT_EPA_SHARE,
        control_scale: float = DEFAULT_CONTROL_SCALE,
        epa_scale: float = DEFAULT_EPA_SCALE,
        ratings: dict[str, float] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
        game_control: GameControlIndex | None = None,
        game_epa: EpaIndex | None = None,
    ) -> None:
        # The four inherited numbers keep the parent's defaults, which were
        # chosen against basketball -- 3.5 points of home floor, a 40-point
        # cap. This class is football-only in practice, so they are wrong
        # here in the way any unsearched default is wrong; the configs that
        # run it give ranges for all four rather than leaning on these.
        super().__init__(
            league,
            home_advantage=home_advantage,
            k=k,
            margin_scale=margin_scale,
            mov_cap=mov_cap,
            season_regression=season_regression,
            ratings=ratings,
            anchors=anchors,
        )
        self._blend = PlayBlend.validated(play_weight, epa_share)
        self._control_scale = validated_scale("control_scale", control_scale)
        self._epa_scale = validated_scale("epa_scale", epa_scale)
        # Defaulted rather than required, like `opponent_prior_manager` and
        # like `ControlGlickoPredictor`'s index: every caller wants the
        # league's saved indexes, and a test that wants none passes empty
        # ones. Not carried in `state_dict` -- see `state_dict` below.
        self._game_control = game_control or GameControlIndex.for_league(league)
        self._game_epa = game_epa or EpaIndex.for_league(league)

    @property
    def control_weight(self) -> float:
        """How much of the target margin control gets, before renormalizing."""
        return self._blend.control_weight

    @property
    def epa_weight(self) -> float:
        """The same for EPA."""
        return self._blend.epa_weight

    def _control_margin(self, game_id: str) -> float | None:
        """The margin this game's control implies, in points.

        Linear in the share rather than through a logit, because control is
        not a win probability and `cassandra.predictor.game_control` is
        explicit about that: 0.80 doesn't say the home team was ever 80% to
        win, it says that averaged over sixty minutes that's where the model
        had them. A logit would treat the tails as the enormous numbers they
        are for a probability, and control has no tails -- it lives inside
        roughly [0.1, 0.9] by construction.

        The honest alternative is to fit the mapping the way
        `cassandra.prob_to_margin` fits prob->margin, per league, and store
        it in the index. That is a better answer and a bigger one; a scale
        the search can move is the version that can be measured first.
        """
        control = self._game_control.get(game_id)
        if control is None:
            return None
        return (2 * control.home - 1) * self._control_scale

    def _epa_margin(self, game_id: str) -> float | None:
        """The margin this game's EPA implies, in points."""
        margin = self._game_epa.margin(game_id)
        if margin is None:
            return None
        return margin * self._epa_scale

    def _actual_margin(self, game: Game) -> float:
        """The final margin, moved toward what the plays say.

        The whole of the model. It replaces `_actual_margin` rather than
        `update_game` because that is the only line of the update that is
        about the game rather than about the ratings -- the same seam
        `ControlGlickoPredictor` takes in Glicko. Nothing here builds a game
        that wasn't played, and nothing downstream can tell it was given
        anything other than a margin.
        """
        scoreboard = super()._actual_margin(game)
        if not self._blend.uses_plays:
            return scoreboard
        return self._cap(
            self._blend.combine(
                scoreboard,
                self._control_margin(game.game_id),
                self._epa_margin(game.game_id),
            )
        )

    def state_dict(self) -> dict[str, Any]:
        """The parent's state, plus the four parameters this class adds.

        The indexes themselves stay out, for the reason
        `ControlGlickoPredictor.state_dict` gives: they are tens of thousands
        of floats against a release that is otherwise a page of ratings, and
        unlike the anchors -- per-team, and small enough that embedding them
        is what makes a release replay exactly -- they would dominate the
        artifact. The cost is that a replay picks up whatever sweep is on
        disk rather than the one the fit ran against.
        """
        return {
            **super().state_dict(),
            "play_weight": self._blend.play_weight,
            "epa_share": self._blend.epa_share,
            "control_scale": self._control_scale,
            "epa_scale": self._epa_scale,
        }

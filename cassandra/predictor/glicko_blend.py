"""Glicko, learning from the scoreboard, the control and the EPA at once.

`cassandra.scoring` reduces a game to one number for the home team between 0
and 1, and Glicko updates against it. Game control is a second answer to that
same question and EPA per play is a third, so `_actual` returns a blend of all
three and everything around it is untouched Glicko. `ControlGlickoPredictor`
is this with the third one missing.

Why this exists next to `BlendedMarginEloPredictor`
---------------------------------------------------

Both models balance the same three signals. They differ in where they meet,
and the pair is the experiment rather than either one on its own:

- **Here**, on the 0-to-1 scale a scoring function produces. Control needs no
  conversion at all -- it is already a share of the game -- and EPA, which is
  in points, is squashed through the same logistic `sigmoid_score` uses.
- **There**, in points. EPA needs no conversion and control gets an exchange
  rate it has no natural value for.

So neither scale is free of a conversion and the choice isn't made on
tidiness. What decides it is that **this one isolates the signal and that one
doesn't.** `glicko_full` is a converged, measured champion in both football
leagues -- nfl at 0.2216 brier, ncaafb at 0.1579 -- so a result here is
attributable to the blend. `MarginEloPredictor` had no football config at all
before these two landed, so a `margin_blend` result mixes "does EPA help" with
"is margin Elo a good football model", and a disappointing one wouldn't say
which half failed.

The other reason to run this first is that it is the version with evidence
behind it. `cassandra.predictor.control` is a long negative result about
control, and the single positive finding in it was measured through *this*
seam: success rate differential -- which that module calls the crude binary
form of EPA -- blended at ~0.2 improved ncaafb brier by 0.00030, with a smooth
unimodal weight curve and a shuffle null 0.0014 away. EPA is the graded
version of that signal arriving where its crude form already worked.

What the negative result still says
------------------------------------

Control's failure was not about noise, it was about the axis, and that hasn't
changed. Every summary of the scoreboard path costs brier in proportion to how
far it reorders games away from the final score and by nothing else:

    cost = 0.0124 x (1 - rank correlation with the final score)   R^2 = 0.991

Twelve signals sit on that line, both readings of control included. A signal
carrying information the final score lacks comes in *below* it, and EPA is a
candidate for that in a way control never was: control is a summary of the
same win probability curve the final score largely determines, while EPA is
priced by an expected points model that is never told who won.

So the thing to read off a fitted run is not the target alone but where
`epa_share` lands. `play_weight` at 0 is a real answer and recovers
`GlickoPredictor` exactly; `play_weight` up with `epa_share` at 1 is the
signal working and control still not; both up together would be the first
evidence that control helps once it has company.

**Nothing here is fitted.** The defaults are legible starting points, not
numbers a search found, and `models/{nfl,ncaafb}/glicko_blend.json` is the
run that would replace them -- after a `jobs.py epa` sweep, since the index
this reads does not exist until one runs.
"""

import math
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
from .glicko import GlickoPredictor, _Rating
from .opponent_prior import OpponentPriorManager

# Points of margin per unit of logit, for turning EPA's implied margin into
# the share of a game it corresponds to.
#
# 10.0 because that is `sigmoid_score`'s own divisor, which makes this the
# exact function the scoring path already applies to a real margin. That is
# the whole argument for it: EPA's claim is "these teams played like a
# seven-point game", and scoring that claim the way a seven-point game is
# scored means the blend is combining two numbers that mean the same thing.
#
# It stops being free when `scoring_method` isn't sigmoid -- against
# `binary_score`'s hard 0/1 there is nothing for it to agree with -- which is
# one more reason the configs leave the scoring method pinned at sigmoid.
DEFAULT_EPA_MARGIN_SCALE = 10.0


class BlendedGlickoPredictor(GlickoPredictor):
    """Glicko, updated on a blend of the score, the control and the EPA.

    A child of Glicko rather than of `ControlGlickoPredictor`, whose one knob
    this does not have: `control_weight` and the `play_weight`/`epa_share`
    pair are different parameterizations of overlapping models, and a subclass
    that inherited one and ignored it would have two spellings of the same
    blend in one `state_dict`. Every release published so far replays through
    code this never touches.

    Scoring defaults to sigmoid rather than Glicko's binary, and the configs
    leave it out of the search rather than paying a dimension to rediscover
    it: the 2026-08-30 run put `glicko_full` on sigmoid in both football
    leagues, and in four of the other six besides. `scoring_method` is still a
    constructor argument, so trying another one is a keyword, not a subclass.

    One thing to hold while reading a fitted `play_weight`: control is pulled
    toward 0.5 by construction -- every game starts 0-0 at even odds, and
    those early snaps carry full clock weight -- so a wire-to-wire blowout
    tops out nearer 0.9 than 1.0. Against `binary_score`'s 1.0 that makes even
    a heavily weighted blend read as a slightly softer win; against
    `sigmoid_score`, whose output is compressed by the same divisor this
    module puts EPA through, all three are on comparable footing.
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        k: float = 65,
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        # Not Glicko's "binary" -- see the class docstring.
        scoring_method: str = "sigmoid",
        season_regression: float = 0.0,
        play_weight: float = DEFAULT_PLAY_WEIGHT,
        epa_share: float = DEFAULT_EPA_SHARE,
        epa_margin_scale: float = DEFAULT_EPA_MARGIN_SCALE,
        opponent_prior_manager: OpponentPriorManager | None = None,
        ratings: dict[str, _Rating] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
        game_control: GameControlIndex | None = None,
        game_epa: EpaIndex | None = None,
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
        self._blend = PlayBlend.validated(play_weight, epa_share)
        self._epa_margin_scale = validated_scale("epa_margin_scale", epa_margin_scale)
        # Defaulted rather than required, like `opponent_prior_manager`: every
        # caller wants the league's saved indexes, and a test that wants none
        # passes empty ones. Not carried in `state_dict` -- see below.
        self._game_control = game_control or GameControlIndex.for_league(league)
        self._game_epa = game_epa or EpaIndex.for_league(league)

    @property
    def control_weight(self) -> float:
        """How much of the update control gets, before renormalizing."""
        return self._blend.control_weight

    @property
    def epa_weight(self) -> float:
        """The same for EPA."""
        return self._blend.epa_weight

    def _epa_share_of_game(self, game_id: str) -> float | None:
        """What EPA says this game was worth to the home team, in [0, 1].

        The implied margin through the scoring function's own logistic, so a
        game EPA prices as a ten-point home win reads as the same 0.73 a
        ten-point home win reads as. Nothing is refit and no new opinion is
        introduced by the conversion -- it is the scoreboard's own exchange
        rate applied to a different set of points.
        """
        margin = self._game_epa.margin(game_id)
        if margin is None:
            return None
        return 1 / (1 + math.exp(-margin / self._epa_margin_scale))

    def _control_share_of_game(self, game_id: str) -> float | None:
        """What control says, which needs no conversion at all.

        The one signal on either side of this experiment that arrives already
        on the target's scale: `cassandra.scoring` produces the home team's
        share of a game and so does the win probability curve.
        """
        control = self._game_control.get(game_id)
        return None if control is None else control.home

    def _actual(self, game: Game) -> float:
        """Glicko's own answer, moved toward what the plays say.

        The whole of the model, and it replaces `_actual` rather than
        `update_game` because that is the only line of the update that is
        about the game rather than about the ratings. Nothing here builds a
        game that wasn't played, and nothing downstream can tell it was given
        anything other than a number between 0 and 1.

        Convex over three numbers already in [0, 1], so the result is too --
        there is nothing to clamp, which is the one way this side is simpler
        than the margin side.
        """
        scoreboard = super()._actual(game)
        if not self._blend.uses_plays:
            return scoreboard
        return self._blend.combine(
            scoreboard,
            self._control_share_of_game(game.game_id),
            self._epa_share_of_game(game.game_id),
        )

    def state_dict(self) -> dict[str, Any]:
        """The parent's state, plus the three parameters this class adds.

        The indexes stay out, for the reason `ControlGlickoPredictor` gives:
        they are tens of thousands of floats against a release that is
        otherwise a page of ratings, and unlike the anchors -- per-team, and
        small enough that embedding them is what makes a release replay
        exactly -- they would dominate the artifact. The cost is that a replay
        picks up whatever sweep is on disk rather than the one the fit ran
        against.
        """
        return {
            **super().state_dict(),
            "play_weight": self._blend.play_weight,
            "epa_share": self._blend.epa_share,
            "epa_margin_scale": self._epa_margin_scale,
        }

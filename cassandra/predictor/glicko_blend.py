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

What it was measured to be worth: real, and very small
------------------------------------------------------

**Measured before the temperatures existed.** Everything below is the
2026-09-05 run, which searched six parameters with `mov_scale` frozen at
`sigmoid_score`'s 10, `epa_margin_scale` pinned at 10 and control passed
through untransformed. The fitted weights in the table are therefore the
best available under a constraint this class no longer has, and the next run
should replace them. What does not depend on the parameterization is the
shuffle null, which is about whether the index carries per-game information
at all.

Run 2026-09-05, 400 probes per league, against the first full EPA sweep
(4,984 NFL games with play-by-play, 25,315 NCAAFB).

Measured against the same fitted parameters with `play_weight = 0` rather
than against `glicko_full`, because that isolates the blend and nothing else.
Comparing to `glicko_full` instead understates it -- differently fitted
model, and a stored `target` written against whatever seasons existed on the
day of *its* search, which is not comparable across runs. `glicko_full`'s
stored target said 0.15748 and its replay on the day said 0.15800; always
compare through a replay.

    league  signals off  blended     d brier   play_weight  control    epa
    ncaafb     0.157990  0.157890  -0.000101       0.1728   0.1228  0.0500
    nfl        0.221255  0.221140  -0.000116       0.1771   0.0710  0.1061

**The signal is real.** Permute which game each (control, EPA) pair belongs
to -- preserving both marginal distributions exactly, keeping the two paired
with each other, destroying only the correspondence to the game played -- and
the gain does not survive:

    league  shuffles  null mean   vs off   real is
    ncaafb        15   0.158713  +0.00072  20.9 sd better, 0/15 matched it
    nfl            4   0.222109  +0.00085   8.1 sd better, 0/4 matched it

Note the sign on the null. A shuffled index is *worse than no index at all*,
by seven to eight times what the real one is better by. So this is not a flat
direction where any plausibly-shaped number would do: it is a narrow well
that misassignment falls straight out of. That also makes it an operational
guarantee worth having -- an index that ever went misaligned would show up as
degradation rather than as nothing.

**And it is far too small to matter.** 0.0001 against the 0.00906 that
separates ncaafb's two best models -- `glicko_full` to `glicko_binary`, the
resolution at which model choice actually matters here -- is about 1.1%. For
company through this same seam: luck-adjusted control managed 0.00008 and
success rate differential 0.00030. EPA and control together land between the
two signals `cassandra.predictor.control` already measured.

Why it is small, which the distributions above answer
-----------------------------------------------------

EPA is not a second look at the game. It is a second *measurement of the
same thing the scoreboard measures*, and the rank correlation says so --
computed on the axis `cassandra.predictor.control` fits its law on:

        signal        vs final score, rank    law's replacement cost
        nfl    EPA                  0.8502                   0.00186
        nfl    control              0.8020                   0.00245
        ncaafb EPA                  0.9131                   0.00108
        ncaafb control              0.8865                   0.00141

    (control reproduces that module's own 0.7999 and 0.8864, which is what
     says this is the same measurement and not a differently-defined one.)

**EPA is a closer copy of the final score than control is**, in both leagues.
That is the opposite of the property wanted from a signal meant to add
something: the module next door warns that a signal at 0.940 is "closer to a
noisy copy of the scoreboard than to a second look at the game", and ncaafb's
EPA is at 0.913.

Which makes the shape of the result coherent rather than mysterious.
Averaging a second noisy measurement of the same quantity reduces variance,
so the gain is real and the shuffle null finds it. It carries little the
final score lacks, so the gain is tiny. Both halves follow from one number.

If there is more here, it is in the part of EPA the scoreboard does *not*
explain -- the residual after regressing the EPA margin on the final margin,
which is what "this team moved the ball better than it scored" actually
means. That is a different signal from this one and has not been built. EPA
and control correlate 0.72 (nfl) and 0.83 (ncaafb) with each other, so they
are not two independent looks either.

Real and negligible are both true, and neither one is the interesting half
without the other. `cassandra.predictor.control`'s law is not refuted -- EPA
still comes in on the line rather than below it, and the gain is a rounding
error against model choice. What the null adds is that the failure is one of
*magnitude* rather than of content, which is a different finding from
control's: those searches drove their weight to zero exactly, with the top
decile crowded against the bound, because the objective disliked the signal.
This one is kept, at 0.17 in both leagues, because it genuinely helps -- by
an amount nobody should ship a model over.

What is not established: that 0.17 is the optimum. No weight sweep was run,
only the fitted point and the null around it. The split between `control` and
`epa` reverses across the two leagues and no measurement here separates them,
so read the pair as one signal until something does.

`glicko_blend` is nominally ncaafb's best model on brier and on margin MAE
(13.0011 against `glicko_full`'s 13.0077) as of this run.
`cassandra.predictor.margin_blend` has the sibling result on the other scale.
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

# Points of margin per unit of logit, for the two sources that arrive in
# points: the final score and EPA's implied margin.
#
# 10.0 for both because that is `sigmoid_score`'s divisor, so the defaults
# reproduce what this class did before it had temperatures of its own -- the
# scoreboard scored exactly as `scoring_method="sigmoid"` scored it, and EPA
# scored the way a real margin of the same size is. That was the argument for
# pinning them and it is still the argument for defaulting to them. It is not
# an argument for leaving them unsearched: 10 was never fitted to anything.
# `cassandra/scoring/sigmoid.py` carries a `# TODO: fiddle with k` and the
# same 10 is used for hockey goals and college football touchdowns alike.
DEFAULT_MOV_SCALE = 10.0
DEFAULT_EPA_MARGIN_SCALE = 10.0

# Temperature on control, which is the one source that does not arrive in
# points. 1.0 leaves it exactly as the index stores it, which is what this
# class did before.
DEFAULT_CONTROL_TEMP = 1.0

# How far a control share is allowed toward the ends before the logit that
# reshapes it stops being finite. Measured against the indexes that exist, it
# never binds: nfl runs 0.049 to 0.963 and ncaafb 0.012 to 0.99997, with four
# ncaafb games above 0.999 and none at either end exactly. So this is a guard
# on the tail rather than a shaping choice, and a fitted `control_temp` is
# describing the real distribution rather than this constant.
_CONTROL_EPS = 1e-6


class BlendedGlickoPredictor(GlickoPredictor):
    """Glicko, updated on a blend of the score, the control and the EPA.

    A child of Glicko rather than of `ControlGlickoPredictor`, whose one knob
    this does not have: `control_weight` and the `play_weight`/`epa_share`
    pair are different parameterizations of overlapping models, and a subclass
    that inherited one and ignored it would have two spellings of the same
    blend in one `state_dict`. Every release published so far replays through
    code this never touches.

    **Three sources, three temperatures, then the weights.** Each source is
    squashed to [0, 1] with a scale of its own before anything is combined:
    `mov_scale` on the final margin, `epa_margin_scale` on the margin EPA
    implies, `control_temp` on the logit of a share that is already in [0, 1].
    Then `play_weight` and `epa_share` mix them, renormalizing over whichever
    a game actually has.

    Temperature and weight do different jobs and the difference is worth
    holding: the temperature decides how much a 3-point game differs from a
    30-point one, the weight decides how loudly that source speaks. They are
    partly redundant near the middle -- for small margins
    `sigmoid(x/T) ~= 0.5 + x/4T`, so only `weight/T` is visible there -- and
    separate in the tails, where a small temperature saturates and a large one
    does not. A search will feel that ridge; the crowding diagnostics in the
    run report are what would show it.

    `scoring_method` is gone from this class. The parent's scorer hardcodes a
    divisor of 10 and shares it with every Glicko model in every sport, and a
    class whose whole subject is putting three sources on one scale should not
    take the most important of the three on faith. `binary_score` is the limit
    of `mov_scale` going to 0 and `sigmoid_score` is the point at 10, so the
    temperature spans the family rather than dropping a choice.

    One thing to hold while reading a fitted `play_weight`: control is pulled
    toward 0.5 by construction -- every game starts 0-0 at even odds, and
    those early snaps carry full clock weight. Measured, its tenth-to-ninetieth
    range is 0.26-0.84 in nfl and 0.21-0.93 in ncaafb, against a scoreboard
    that reaches 0 and 1 outright. `control_temp` below 1 is the search saying
    it wants that compression stretched back out.

    What the three sources actually look like
    -----------------------------------------

    Measured over the games that have all three, which is what the ranges in
    `models/{nfl,ncaafb}/glicko_blend.json` are set from rather than guessed:

        source                    nfl sd   ncaafb sd   p10..p90 (ncaafb)
        final margin (points)      14.61       22.34   -21 .. +36
        EPA margin (points)        12.53       19.09   -17.4 .. +30.3
        control (share)            0.215       0.267   0.21 .. 0.93
        net EPA *per play*         0.184       0.260   -0.23 .. +0.41

    The row that surprises people is the second against the fourth. EPA per
    play is a tenth of a point and looks nothing like a scoreboard, but
    `EpaIndex.margin` multiplies it back through the play counts, so what
    reaches the squash is a total in points at 0.86 of the final margin's
    spread -- in both leagues, to two digits. That is why `mov_scale` and
    `epa_margin_scale` share a range: they are divisors on two quantities of
    nearly the same size, and a search wanting equal sharpness would put EPA's
    slightly below the scoreboard's. [1, 40] spans a hard step at the tie
    (a divisor well under one standard deviation) to a game that barely
    counts (0.4 to 0.6 of a share, either way).
    """

    def __init__(
        self,
        league: str,
        home_advantage: float = 95,
        k: float = 65,
        weekly_rd_increase: float = 1,
        season_rd_increase: float = 120,
        initial_rd: float = 216,
        season_regression: float = 0.0,
        play_weight: float = DEFAULT_PLAY_WEIGHT,
        epa_share: float = DEFAULT_EPA_SHARE,
        mov_scale: float = DEFAULT_MOV_SCALE,
        control_temp: float = DEFAULT_CONTROL_TEMP,
        epa_margin_scale: float = DEFAULT_EPA_MARGIN_SCALE,
        opponent_prior_manager: OpponentPriorManager | None = None,
        ratings: dict[str, _Rating] | None = None,
        anchors: Mapping[str, Anchor] | None = None,
        game_control: GameControlIndex | None = None,
        game_epa: EpaIndex | None = None,
    ) -> None:
        # `scoring_method` is deliberately not passed through and not accepted:
        # `_actual` never calls the parent's scorer, so the parent keeps its
        # own default and this class's `mov_scale` is the only thing that
        # decides what a final score is worth. See `_mov_share_of_game`.
        super().__init__(
            league,
            home_advantage=home_advantage,
            k=k,
            weekly_rd_increase=weekly_rd_increase,
            season_rd_increase=season_rd_increase,
            initial_rd=initial_rd,
            season_regression=season_regression,
            opponent_prior_manager=opponent_prior_manager,
            ratings=ratings,
            anchors=anchors,
        )
        self._blend = PlayBlend.validated(play_weight, epa_share)
        self._mov_scale = validated_scale("mov_scale", mov_scale)
        self._control_temp = validated_scale("control_temp", control_temp)
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

    @staticmethod
    def _squash(margin: float, scale: float) -> float:
        """A margin in points, as the share of a game it stands for.

        One function for the two sources that arrive in points, because that
        is the claim being made about both: "these teams played like an
        n-point game". Same logistic `sigmoid_score` applies to a real
        margin, with the divisor exposed instead of frozen at 10.
        """
        return 1 / (1 + math.exp(-margin / scale))

    def _mov_share_of_game(self, game: Game) -> float:
        """What the scoreboard says, at this model's own temperature.

        Not `super()._actual`, which is the parent's `scoring_method`. Three
        sources meeting on one scale should get the same treatment, and the
        parent's scorer has no temperature to give: `sigmoid_score` hardcodes
        a divisor of 10 and shares it with every Glicko model in every sport.

        The temperature subsumes the choice this replaces. `binary_score` is
        the limit as `mov_scale` goes to 0 -- a step at the tie -- and a large
        scale is a game that barely counts, so the family spans from "only who
        won" to "the margin, softly" without a categorical to search.
        """
        return self._squash(game.home_score - game.away_score, self._mov_scale)

    def _epa_share_of_game(self, game_id: str) -> float | None:
        """What EPA says this game was worth to the home team, in [0, 1].

        The implied margin through the same logistic the scoreboard gets, so a
        game EPA prices as a ten-point home win reads the way a ten-point home
        win reads -- at `epa_margin_scale == mov_scale`. Letting the two
        differ is the point of searching them separately: EPA's points are
        bounded at the clip and systematically understate a blowout, so the
        margin it claims may deserve a different exchange rate than the one
        the scoreboard actually produced.
        """
        margin = self._game_epa.margin(game_id)
        if margin is None:
            return None
        return self._squash(margin, self._epa_margin_scale)

    def _control_share_of_game(self, game_id: str) -> float | None:
        """What control says, reshaped by its own temperature.

        Control is the one source that does not arrive in points -- it is
        already a share of a game -- so a temperature on it cannot be a
        divisor on a margin. It is a divisor on the *logit* instead, which is
        the same operation the other two get: below 1 sharpens the share away
        from 0.5, above 1 flattens it toward 0.5, and exactly 1 is the number
        the index stores.

        Worth knowing before reading a fitted `control_temp`: this is a
        monotone transform, and `cassandra.predictor.control` measured that
        monotone transforms do not rescue control. Rank-matching it onto the
        scoring function's exact distribution -- the strongest monotone
        reshaping there is -- recovered *none* of the loss when control
        replaced the scoreboard. That was measured at full weight rather than
        in a blend, where sharpening also changes how far the target moves,
        so it is not the same question. But a `control_temp` that fits near
        1.0 is the answer that experiment predicts.
        """
        control = self._game_control.get(game_id)
        if control is None:
            return None
        share = min(1 - _CONTROL_EPS, max(_CONTROL_EPS, control.home))
        if self._control_temp == 1.0:
            return share
        return self._squash(math.log(share / (1 - share)), self._control_temp)

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
        scoreboard = self._mov_share_of_game(game)
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
        state = super().state_dict()
        # The parent's scorer is unused -- `_actual` replaces it with
        # `_mov_share_of_game` -- so recording which one it would have been
        # would put a parameter in a release that nothing reads and that
        # `from_state_dict` would hand back to a constructor that no longer
        # takes it.
        state.pop("scoring_method", None)
        return {
            **state,
            "play_weight": self._blend.play_weight,
            "epa_share": self._blend.epa_share,
            "mov_scale": self._mov_scale,
            "control_temp": self._control_temp,
            "epa_margin_scale": self._epa_margin_scale,
        }

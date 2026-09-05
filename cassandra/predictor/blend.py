"""Balancing the scoreboard against the two things the play-by-play says.

Two models blend the same three opinions about a game -- what the final score
said, how much of it a team controlled, and how well its offense actually
played -- and they differ only in what scale they meet on.
`BlendedGlickoPredictor` meets them on the 0-to-1 share a scoring function
produces; `BlendedMarginEloPredictor` meets them in points. The weights, the
knobs a search moves, and the rule for a game that only has some of the three
are identical, so they live here rather than in whichever one was written
first.

Why two knobs rather than three weights
---------------------------------------

The blend is a convex combination:

    (1 - play_weight)              on the scoreboard
    play_weight * (1 - epa_share)  on control
    play_weight * epa_share        on EPA

so every point of the `[0, 1] x [0, 1]` box the two knobs span is a valid
model, and a Bayesian search over box ranges cannot propose one that isn't.
Two independent weights could reach `control_weight = 0.8, epa_weight = 0.7`,
which is not a blend of anything, and every way of handling that -- rejecting
it, renormalizing it, clipping it -- makes the search's record of what it
tried disagree with what it ran.

`play_weight` reads as "how far from the scoreboard", `epa_share` as "of that,
how much is EPA". `control_weight` and `epa_weight` are properties, so a
fitted model still reports the three numbers a reader wants.

Why a game contributes only what it has
----------------------------------------

Most of an NCAAFB schedule has no play-by-play at all and neither league has
any before 2006, so a game missing one or both signals is the common case
rather than the edge one. The weights renormalize over what is present, which
means a game with nothing is rated on its scoreboard alone -- exactly as the
model each of these subclasses would rate it, with no fallback rule of its
own. It is also what makes `play_weight = 1` a coherent setting rather than a
model that has no target for most of its games.
"""

from typing import NamedTuple

# How much of the target comes from the plays rather than the scoreboard, and
# how that half splits between the two play signals.
#
# An even three-way-ish split -- half the scoreboard, a quarter each -- which
# asserts nothing about which is right and is visibly a blend. Not 1.0, the
# way `DEFAULT_CONTROL_WEIGHT` is: there the scoreboard still decided the
# score line being blended, while here `play_weight = 1` means a rating that
# never sees the final result at all, which is a strong claim for a default.
# Not 0 either, which would make a hand-built one silently identical to the
# model it subclasses.
DEFAULT_PLAY_WEIGHT = 0.5
DEFAULT_EPA_SHARE = 0.5


def validated_fraction(name: str, value: float) -> float:
    """Check one of the two blend knobs on its way into a predictor.

    A free function rather than validation inside `Predictor.__init__`, for
    the reason `validated_regression` spells out: every dynamic construction
    site does `predictor_class(league, **config.params)` against a
    `type[Predictor]`, so anything in the base signature is checked against a
    config's `float | str` values.

    Outside [0, 1] is not a blend. Above 1 overshoots -- the target lands past
    the signal it was moving toward -- and below 0 reflects the game through
    its own result, which is a model that learns backwards from the games it
    has the most information about. Either would search for an hour and report
    a plausible score.
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def validated_scale(name: str, value: float) -> float:
    """Check an exchange rate between a signal's units and the target's.

    Positive, and zero is excluded along with the negatives even though it
    would run: a scale of 0 is a second way of spelling a weight of 0, and two
    spellings of the same model make a fitted config ambiguous about which one
    the search actually found. Negative reads the signal backwards.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class PlayBlend(NamedTuple):
    """The two knobs, and the combination they describe.

    A value rather than a mixin, so both predictors hold one and neither
    inherits from it -- they already have a base class each, and what they
    share is arithmetic rather than behaviour.
    """

    play_weight: float
    epa_share: float

    @classmethod
    def validated(cls, play_weight: float, epa_share: float) -> "PlayBlend":
        """The pair, checked. What a constructor calls."""
        return cls(
            validated_fraction("play_weight", play_weight),
            validated_fraction("epa_share", epa_share),
        )

    @property
    def uses_plays(self) -> bool:
        """Whether this blend reads the indexes at all.

        Checked by callers *before* they look a game up, not just inside
        `combine`: the two signals are arguments, so they are computed before
        `combine` can decline to use them. A search that finds the plays
        worthless replays a hundred million games, and at `play_weight = 0`
        none of them should cost an index lookup.
        """
        return bool(self.play_weight)

    @property
    def control_weight(self) -> float:
        """How much of the target control gets, before renormalizing.

        The number a reader of a fitted config wants, which is neither of the
        two the search stores. Renormalizing is per game and happens in
        `combine`; this is the weight a game with everything would use.
        """
        return self.play_weight * (1 - self.epa_share)

    @property
    def epa_weight(self) -> float:
        """The same for EPA. `control_weight + epa_weight == play_weight`."""
        return self.play_weight * self.epa_share

    def combine(
        self, scoreboard: float, control: float | None, epa: float | None
    ) -> float:
        """The three, weighted over whichever of them this game has.

        `scoreboard` is always there -- every game cassandra replays has a
        final score -- so it is the one that isn't optional and the one a game
        with neither signal falls back to whole.

        All three arguments have to already be on one scale. That is the
        caller's job and it is the only real difference between the two
        models: control arrives as a share and EPA as points, so whichever of
        them doesn't match the target gets converted first.
        """
        if not self.play_weight:
            # Recovers the plain model exactly rather than approximately.
            # Callers check `uses_plays` before looking anything up, so this
            # is the correctness half of that short-circuit rather than the
            # cost half -- `combine` still has to be right on its own.
            return scoreboard

        total = 1 - self.play_weight
        blended = total * scoreboard
        for weight, value in ((self.control_weight, control), (self.epa_weight, epa)):
            if value is None:
                continue
            total += weight
            blended += weight * value
        if not total:
            # `play_weight` is 1 and this game has neither signal, so there is
            # nothing to renormalize over.
            return scoreboard
        return blended / total

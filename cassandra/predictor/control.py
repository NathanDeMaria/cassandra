"""Glicko, fed a blend of the game that happened and the game that was played.

`control_weight` interpolates between the two: at 0 this is `GlickoPredictor`
exactly, at 1 it learns from the score control says the game looked like, and
in between it takes both. It is a searched parameter rather than a constant,
because how much of a result is signal and how much is the bounces is a
question about a league, not a thing to assert -- and the search recovering 0
is a real answer, not a failed run.

A child of Glicko rather than of Elo or 538, and rather than a parameter on
all three, for two reasons. The narrow one is that plain Elo reads only the
sign of the margin, so control would move a rating there only when it flips
the winner outright -- `scoring_method` is what lets the magnitude through,
and Glicko is the model that has it. The broader one is that
`predictor_class(league, **config.params)` stays typed against the
constructors that declare what they take, every release published so far
replays through code this never touches, and `models/<league>/glicko_control.
json` competes as its own entry in `evaluate` instead of replacing
`glicko_full`'s converged search with one that has an extra dimension.

The substitution is one line: hand `super().update_game` the blended score
line instead of the real one. Everything below it -- Glicko's binary,
pythagorean and sigmoid scoring -- reads the result out of `home_score` and
`away_score` and so needs no changes at all. See
`game_control.GameControlIndex.alternate` for what the line is, and for why
the real game still reaches the replay that scores it.
"""

from collections.abc import Mapping
from typing import Any

from endgame.types import Game

from .base_predictor import Anchor
from .game_control import GameControlIndex, validated_control_weight
from .glicko import GlickoPredictor, _Rating
from .opponent_prior import OpponentPriorManager
from .types import Prediction

# Where a search starts, and what a hand-built one does with no argument:
# take the controlled line whole wherever there is one. This class exists to
# use the number, so 0 would make it silently identical to the model it
# subclasses -- a default that reads as a bug the first time someone builds
# one and sees Glicko's ratings come out.
DEFAULT_CONTROL_WEIGHT = 1.0


class ControlGlickoPredictor(GlickoPredictor):
    """Glicko, updated on a blend of the real score and the controlled one.

    Scoring defaults to sigmoid rather than Glicko's binary, and the config
    leaves it out of the search rather than paying a dimension to rediscover
    it: the 2026-08-30 run put `glicko_full` on sigmoid in both football
    leagues -- nfl at 0.2216 brier, ncaafb at 0.1579 -- and in four of the
    other six besides. `scoring_method` is still a constructor argument, so
    trying another one is a keyword, not a subclass.

    It also happens to be the choice that can hear this model at all.
    `binary_score` reads only the sign, so two control numbers on the same
    side of even give the identical update and the blend reaches a binary
    Glicko only when it hands the game to the other team. Sigmoid and
    pythagorean both read the margin.

    One thing to hold while reading a fitted `control_weight`: control is
    pulled toward 0.5 by construction -- every game starts 0-0 at even odds,
    and those early snaps carry full clock weight -- so a controlled line's
    margins are compressed against real ones. `k` and `control_weight` both
    absorb some of that, but `sigmoid_score`'s divisor is a hardcoded 10 that
    no search touches, so a controlled blowout reads as closer to a coin flip
    than a real one of the same margin would.
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

    def update_game(self, game: Game) -> Prediction:
        return super().update_game(
            self._game_control.alternate(game, self._control_weight)
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

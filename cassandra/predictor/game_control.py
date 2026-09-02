"""Game control: how much of a game a team spent winning it.

The number comes from `the-lucky-ones`, which fits an in-game win probability
model on play-by-play and averages the curve over the game, weighted by how
long each snap's situation stood. Read it as a share of the game controlled
rather than as a win probability: 0.80 doesn't say the home team was ever 80%
to win, it says that averaged over sixty minutes that's where the model had
them.

What cassandra does with it is turn a game into the score it *looked* like:
the team that led wire to wire and lost on a last-second field goal did not
play a one-point game, and a rating system that only sees 20-17 can't know
that. `GameControlIndex.alternate` re-splits the game's real total by
control, and every rating model here picks that up without changing, because
all of them read the result through `home_score` / `away_score` -- Elo's
sign, 538's margin-of-victory multiplier, and all three of Glicko's scoring
functions.

Two things this module deliberately does not do.

It doesn't read the plays. Deriving control means pyarrow and ~190MB of
parquet per league, and the output is one float per game -- so the sweep is a
build step that writes `{league}_game_control.json`, and everything on the
replay path reads that. `cassandra.predictor` stays installable without the
fitting stack.

And it doesn't touch the game the replay records. `generate_predictions`
yields the real `Game` alongside the prediction, and `_build_prediction`
scores against its real score. The alternate line exists for the predictor's
own state update and nowhere else -- put it in the replay instead and the
brier score, the margin fit and the against-spread metrics are all computed
against a game nobody played, which the optimizer would happily maximize.
"""

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Self

from endgame.types import Game
from pydantic import BaseModel

from cassandra.constants import CASSANDRA_HOME

from .types import GameControl

# The same directory the division anchors and the opponent priors write to,
# which is what makes `artifacts.py` carry this between Batch stages: it
# ships the whole `predictor/data/` prefix, so a file that lands here needs no
# plumbing of its own.
_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"

# The leagues there can be a control number for at all. Football only -- it's
# what has play-by-play in the bucket and what `lucky_ones` ships fits for --
# and within football, only the games ESPN has plays for: NFL coverage is
# complete back to 2006, while an NCAAFB week is more like a third of its
# games, the FBS end of it.
#
# Here rather than in the build module for the same reason `ANCHOR_LEAGUES`
# sits in `base_predictor`: the batch launcher sizes an array job from it and
# the array child turns an index back into a league, and neither of those
# should have to import the half that reads parquet.
CONTROL_LEAGUES = ("nfl", "ncaafb")


def game_control_path(league: str) -> Path:
    """Where a league's game control lands, whether or not it exists yet.

    One definition because both sides of the handoff need it: the build
    writes this path, `load_game_control` reads it, and the batch stages
    upload and download it by name.
    """
    return _PREDICTOR_DATA_DIR / f"{league}_game_control.json"


class ControlFit(BaseModel):
    """Which win probability model produced an index.

    Two fields because two different things move the numbers, and only one of
    them is visible in the other. `run_id` is the training run behind the
    league's shipped fit, so it changes when the coefficients are refit.
    `lucky_ones` is the pinned commit of the package, so it changes when the
    code that turns plays into a curve changes -- new features, a different
    clock weighting -- which moves every control number without touching a
    coefficient.

    Together they are what makes the sweep idempotent: a stage that finds its
    own fit already stored has nothing to do, and one that finds a different
    one has to rebuild rather than merge into numbers from another model.
    """

    lucky_ones: str
    run_id: str


class GameControlFile(BaseModel):
    """The `{league}_game_control.json` artifact.

    A schema rather than a bare map because the header is the point: without
    it "rebuild this" is a human decision made from memory about which fit
    was current when the file was written.
    """

    league: str
    fit: ControlFit
    games: dict[str, GameControl]


def read_game_control_file(league: str) -> GameControlFile | None:
    """A league's artifact as it was written, or None if there isn't one.

    The whole document, header included -- what the build stage reads to
    decide whether it has anything to do. Predictors want `load_game_control`
    instead, which is cached and hands back only the games.
    """
    path = game_control_path(league)
    if not path.exists():
        return None
    return GameControlFile.model_validate_json(path.read_text())


@cache
def load_game_control(league: str) -> Mapping[str, GameControl]:
    """A league's saved control numbers, empty if it has none.

    Empty is an ordinary answer, not a failure: four of the six leagues
    cassandra rates aren't football, and a football league whose sweep hasn't
    run yet has nothing either. A predictor with an empty index leaves every
    game exactly as it was played.

    Cached for the same reason `load_anchors` is -- an optimization run builds
    a predictor per probe, hundreds of them, and they would all read the same
    file. Callers copy it rather than holding the shared mapping.
    """
    stored = read_game_control_file(league)
    return {} if stored is None else stored.games


def validated_control_weight(control_weight: float) -> float:
    """Check a `control_weight` on its way into a predictor.

    A free function rather than a `Predictor.__init__` parameter, for the
    reason `validated_regression` spells out: every dynamic construction site
    does `predictor_class(league, **config.params)` against a
    `type[Predictor]`, so anything in the base signature is checked against a
    config's `float | str` values.

    Outside [0, 1] is not a blend. Above 1 overshoots the control line --
    a game controlled 0.6 comes out scored as though it were controlled 0.9 --
    and below 0 reflects the game through its real result, which is a model
    that learns backwards from the games it has the most information about.
    Either would search for an hour and report a plausible brier score.
    """
    if not 0 <= control_weight <= 1:
        raise ValueError(f"control_weight must be in [0, 1], got {control_weight}")
    return control_weight


class GameControlIndex:
    """Control numbers by game id, and the alternate score line they imply.

    Managed rather than a bare dict because the same game is asked about
    repeatedly -- an optimization run replays every season once per probe --
    and because the lookup and the arithmetic that consumes it belong
    together: a caller that got the float and did its own blend is a caller
    that can get the weighting or the orientation subtly wrong.
    """

    def __init__(self, control: Mapping[str, GameControl] | None = None) -> None:
        # Copied, not held: `load_game_control` is cached, so every predictor
        # in an optimization run is handed the same mapping, and a later
        # top-up of one index would otherwise appear in all of them.
        self._control = dict(control or {})

    @classmethod
    def for_league(cls, league: str) -> Self:
        """The league's saved index. Empty for a league with no sweep."""
        return cls(load_game_control(league))

    def __len__(self) -> int:
        return len(self._control)

    def get(self, game_id: str) -> GameControl | None:
        """This game's control, or None if there's nothing for it.

        None covers both halves of "no play-by-play": a game ESPN never had
        plays for, and a game in a league or a season the sweep hasn't
        reached. Nothing downstream needs to tell those apart -- both mean
        the game is taken at its real score.
        """
        return self._control.get(game_id)

    def alternate(self, game: Game, weight: float) -> Game:
        """`game` as control says it looked, blended `weight` of the way there.

        The real total is preserved and only re-split, which is what keeps
        every consumer of the result on its usual scale: a 6-3 defensive
        game stays a low-scoring game, and `pythagorean_score` stays in the
        domain it was written for. So the whole substitution is one number,
        the home score, with the away score following from the total.

        `weight` interpolates between the game as played and the game as
        controlled, and it earns its place twice over. At 0 this returns the
        game untouched, so a search that finds control worthless recovers the
        plain model exactly rather than approximately. And a game with no
        control is simply a game at weight 0 -- which is most of an NCAAFB
        schedule and all of it before 2006 -- so the missing half needs no
        fallback rule of its own.

        Returns `game` itself, not a copy, whenever there is nothing to
        change.
        """
        if not weight:
            return game
        control = self.get(game.game_id)
        if control is None:
            return game
        # Rounded because `Game` scores are ints and a synthetic line that
        # isn't one would be a lie about the type rather than a convenience.
        # Against a football total the rounding is under a point.
        total = game.home_score + game.away_score
        home = round((1 - weight) * game.home_score + weight * total * control.home)
        return game._replace(home_score=home, away_score=total - home)

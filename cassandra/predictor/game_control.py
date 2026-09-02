"""Game control: how much of a game a team spent winning it.

The number comes from `the-lucky-ones`, which fits an in-game win probability
model on play-by-play and averages the curve over the game, weighted by how
long each snap's situation stood. Read it as a share of the game controlled
rather than as a win probability: 0.80 doesn't say the home team was ever 80%
to win, it says that averaged over sixty minutes that's where the model had
them.

Which is the same shape as the thing a rating model already asks of a game:
`cassandra.scoring` turns a final score into a number between 0 and 1 for the
home team, and Glicko updates against it. So control doesn't need converting
into anything -- it is another answer to that question, from the play-by-play
instead of the scoreboard, and `GameControlIndex.blend` is how far to move
from one to the other.

That is the whole of what this module does with it. It doesn't read the plays:
deriving control means pyarrow and ~190MB of parquet per league, and the
output is one float per game, so the sweep is a build step
(`cassandra.game_control_build`) that writes `{league}_game_control.json` and
everything on the replay path reads that. `cassandra.predictor` stays
installable without the fitting stack.

And it never touches the game itself. The replay records the game that was
played -- `_build_prediction` takes the score, the winner and the margin every
metric is computed from off it -- so control enters where the *rating* is
decided and nowhere near where the model is scored. A model that learned from
control and was then graded against control would look excellent and mean
nothing.
"""

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Self

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
    """Control numbers by game id, and the blend that folds them into a result.

    Managed rather than a bare dict because the same game is asked about
    repeatedly -- an optimization run replays every season once per probe --
    and because the lookup and the arithmetic that consumes it belong
    together: a caller that got the float and blended it itself is a caller
    that can put a `1 -` in the wrong place.
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

    def blend(self, result: float, game_id: str, weight: float) -> float:
        """`result` moved `weight` of the way toward what control says.

        Both numbers are already the same measurement -- the share of the
        game that belongs to the home team, between 0 and 1 -- so this is a
        plain convex combination with nothing to rescale. `result` is what
        the scoreboard says through `cassandra.scoring`; control is what the
        play-by-play says.

        `weight` earns its place twice over. At 0 this returns `result`
        untouched, so a search that finds control worthless recovers the
        plain model exactly rather than approximately. And a game with no
        control is a game at weight 0 -- which is most of an NCAAFB schedule
        and all of it before 2006 -- so the missing half needs no fallback
        rule of its own.
        """
        if not weight:
            return result
        control = self.get(game_id)
        if control is None:
            return result
        return (1 - weight) * result + weight * control.home

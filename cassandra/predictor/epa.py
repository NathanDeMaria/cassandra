"""EPA per play: who actually played better, in points.

The number comes from `the-lucky-ones`, which prices every situation with an
expected points model and calls the difference across a snap that snap's
expected points added. Averaged over an offense's snaps it is the closest
thing play-by-play offers to "how well did this team play", and the thing
that makes it worth carrying next to game control is that *it doesn't care
who won*. Control says who was ahead of the game; this says who moved the
ball. The two disagree, and the disagreement is most of what there is to say
about a team that keeps winning close ones.

Two adjustments come baked into the numbers the sweep stores, both measured
upstream and neither of them cassandra's to make: each play's contribution is
bounded at +/- 3 expected points, because a game is only ~130 snaps and an
unbounded pick-six gets to be five plays; and the reading taken is the
unweighted one, every snap counted once. `EpaFit` records both, so an index
built under different settings is found stale rather than merged into.

Which is a different shape from what a rating model usually asks of a game.
`cassandra.scoring` turns a final score into a number between 0 and 1, and
`GameControlIndex` blends against that because control is an answer to the
same question. EPA is not -- it is in points, and the thing in this package
that already thinks in points is `MarginEloPredictor`, whose update is driven
by the margin a team beat rather than by whether it won. So this index
converts to a margin (`EpaIndex.margin`) and the blending happens there. See
`cassandra.predictor.margin_blend`.

That is the whole of what this module does with it. It doesn't read the
plays: deriving EPA means pyarrow and ~190MB of parquet per league, and the
output is four small numbers per game, so the sweep is a build step
(`cassandra.epa_build`) that writes `{league}_epa.json` and everything on the
replay path reads that. `cassandra.predictor` stays installable without the
fitting stack, and nothing here imports `lucky_ones`.

And it never touches the game itself. The replay records the game that was
played -- `_build_prediction` takes the score, the winner and the margin
every metric is computed from off it -- so EPA enters where the *rating* is
decided and nowhere near where the model is scored. A model that learned from
EPA and was then graded against EPA would look excellent and mean nothing.
"""

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from cassandra.constants import CASSANDRA_HOME

from .types import GameEpa

# The same directory the division anchors, the opponent priors and the game
# control index write to, which is what makes `artifacts.py` carry this
# between Batch stages: it ships the whole `predictor/data/` prefix, so a file
# that lands here needs no plumbing of its own.
_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"

# The leagues there can be an EPA number for at all.
#
# The same two as `CONTROL_LEAGUES` and for most of the same reasons --
# football only, and within football only the games ESPN has plays for -- but
# spelled separately rather than aliased, because the condition is a
# different one and could come apart. Control needs a league's win
# probability fit; EPA needs that *and* an expected points fit, which is a
# second release `lucky_ones` ships and retrains separately. A league with
# one and not the other belongs in exactly one of these tuples.
EPA_LEAGUES = ("nfl", "ncaafb")


def epa_path(league: str) -> Path:
    """Where a league's EPA index lands, whether or not it exists yet.

    One definition because both sides of the handoff need it: the build
    writes this path, `load_epa` reads it, and the batch stages upload and
    download it by name.
    """
    return _PREDICTOR_DATA_DIR / f"{league}_epa.json"


# Which of the two averages `lucky_ones` reports the sweep keeps. A constant
# rather than a literal in the build, because it is half of a comparison:
# `EpaFit.reading` is what a stored index was built with, and this is what one
# built now would be. See `GameEpa` for why it is this one.
EPA_READING = "unweighted"


class EpaFit(BaseModel):
    """Which models produced an index, and what was read off them.

    Five fields because five different things move the numbers, and none of
    them is visible in the others.

    `run_id` and `ep_run_id` are the two training runs behind the league's
    shipped fits -- win probability and expected points. Both, because
    `epa_per_play` reads both: expected points prices the snaps, and win
    probability decides how much each snap's game was still in doubt. They
    retrain separately, so one can move without the other.

    `lucky_ones` is the pinned commit of the package, so it changes when the
    code that turns plays into numbers changes -- a different notion of what
    counts as a snap, say -- which moves every entry without touching a
    coefficient.

    `clip` is the bound on one play's contribution, and `reading` is which of
    the two averages cassandra takes. Both are choices made here rather than
    there: the package defaults them and takes them as keywords, and changing
    either rewrites every number while the models behind them are identical.

    `weight_power` is deliberately *not* here, which is worth saying because
    it is the one knob of `epa_per_play` that's missing. The unweighted
    reading is the flat mean over the same snaps, so the competitiveness
    weighting cannot move it -- recording the power would make a sweep
    rebuild an entire league over a parameter that provably changed nothing.
    An index that ever holds `reading == "weighted"` needs it, and adding it
    then is what makes every existing file correctly stale.

    Together these are what makes the sweep idempotent: a stage that finds
    its own fit already stored has nothing to do, and one that finds a
    different one has to rebuild rather than merge into numbers from another
    model.

    Nothing is defaulted, unlike `ControlFit.reading`. There are no EPA files
    written before this schema existed, so there is no older shape to stay
    honest about -- and a required field is what stops the next one being
    added with a default that makes old files claim to be something they
    aren't.
    """

    lucky_ones: str
    run_id: str
    ep_run_id: str
    clip: float
    reading: str


class EpaFile(BaseModel):
    """The `{league}_epa.json` artifact.

    A schema rather than a bare map because the header is the point: without
    it "rebuild this" is a human decision made from memory about which fits
    were current when the file was written.
    """

    league: str
    fit: EpaFit
    games: dict[str, GameEpa]


def read_epa_file(league: str) -> EpaFile | None:
    """A league's artifact as it was written, or None if there isn't one.

    The whole document, header included -- what the build stage reads to
    decide whether it has anything to do. Predictors want `load_epa`
    instead, which is cached and hands back only the games.
    """
    path = epa_path(league)
    if not path.exists():
        return None
    return EpaFile.model_validate_json(path.read_text())


@cache
def load_epa(league: str) -> Mapping[str, GameEpa]:
    """A league's saved EPA numbers, empty if it has none.

    Empty is an ordinary answer, not a failure: four of the six leagues
    cassandra rates aren't football, and a football league whose sweep hasn't
    run yet has nothing either. A predictor with an empty index leaves every
    game exactly as it was played.

    Cached for the same reason `load_anchors` and `load_game_control` are --
    an optimization run builds a predictor per probe, hundreds of them, and
    they would all read the same file. Callers copy it rather than holding
    the shared mapping.
    """
    stored = read_epa_file(league)
    return {} if stored is None else stored.games


class EpaIndex:
    """EPA by game id, and the conversion that puts it on the scoreboard.

    Managed rather than a bare dict for the reason `GameControlIndex` is: the
    same game is asked about repeatedly -- an optimization run replays every
    season once per probe -- and the lookup and the arithmetic that consumes
    it belong together.
    """

    def __init__(self, epa: Mapping[str, GameEpa] | None = None) -> None:
        # Copied, not held: `load_epa` is cached, so every predictor in an
        # optimization run is handed the same mapping, and a later top-up of
        # one index would otherwise appear in all of them.
        self._epa = dict(epa or {})

    @classmethod
    def for_league(cls, league: str) -> Self:
        """The league's saved index. Empty for a league with no sweep."""
        return cls(load_epa(league))

    def __len__(self) -> int:
        return len(self._epa)

    def get(self, game_id: str) -> GameEpa | None:
        """This game's EPA, or None if there's nothing for it.

        None covers both halves of "no play-by-play": a game ESPN never had
        plays for, and a game in a league or a season the sweep hasn't
        reached. Nothing downstream needs to tell those apart -- both mean
        the game is taken at its real score.
        """
        return self._epa.get(game_id)

    def margin(self, game_id: str) -> float | None:
        """The margin this game's play-by-play implies, in points.

        Per-play averages back out to the totals they came from -- `home` over
        `home_plays` snaps, `away` over `away_plays` -- and the difference is
        the expected points the home team's offense added over the away
        team's, across the game. That is a margin, in the same units and on
        the same scale as the one on the scoreboard, which is what lets a
        margin-native model treat the two as answers to one question.

        Multiplying back through the play counts rather than differencing the
        two averages is the whole point. A team that averages +0.10 over
        eighty snaps did more than one that averages +0.12 over forty, and it
        is the difference in *points* the scoreboard is denominated in.

        It does not come out equal to the real margin and isn't meant to.
        Every play is bounded at the clip, so blowouts are systematically
        understated; special teams and the points that come off a defense's
        own scoring plays land where the model puts them rather than where
        the scoreboard does. `epa_scale` in `BlendedMarginEloPredictor` is
        the one knob for that, and 1.0 -- taking this at face value as the
        points it claims to be -- is where a search should start rather than
        where it should stay.

        None for a game with nothing stored, which is most of an NCAAFB
        schedule and all of it before 2006.
        """
        epa = self.get(game_id)
        if epa is None:
            return None
        return epa.home * epa.home_plays - epa.away * epa.away_plays

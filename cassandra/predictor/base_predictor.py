import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Self

from endgame.types import Game

from cassandra.constants import CASSANDRA_HOME

from .types import Matchup, Prediction, Rating

_ANCHOR_DIR = CASSANDRA_HOME / "predictor" / "data"

# Leagues whose teams don't all play each other, and so need a per-team anchor
# rather than one league-wide MEAN_RATING to start from and regress toward.
#
# ncaafb is the obvious one -- it spans FBS through D-III, and a D-III team's
# schedule never touches FBS. mens and womens are all D-I, but division_anchors
# tiers by conference within a division, which is what separates the ACC from
# the MEAC. nfl is deliberately absent: 32 teams who all play each other have
# nothing for a tier fit to find.
#
# Here rather than in `division_anchors.py` because three callers need the list
# without fitting anything: `run_models.sh`, the batch launcher that sizes the
# anchor array job, and the array child that turns an index back into a league.
ANCHOR_LEAGUES = ("ncaafb", "mens", "womens")


def anchor_path(league: str) -> Path:
    """Where a league's fitted anchors live, whether or not they exist yet.

    One definition because both sides of the handoff need it: the fit writes
    this path, `load_anchors` reads it, and the batch stages upload and
    download it by name.
    """
    return _ANCHOR_DIR / f"{league}_division_anchors.json"


class RatingsUnsupported(NotImplementedError):
    """A predictor with no per-team ratings was asked for them.

    FlatPredictor is the case that matters: it predicts 0.5 for everyone and
    has no state to normalize. Raising is better than handing back an empty
    dict, which reads as "this model rates nobody" and is indistinguishable
    from a release that lost its ratings.
    """


MEAN_RATING = 1500.0


def validated_regression(season_regression: float) -> float:
    """Check a `season_regression` on its way into a predictor.

    A free function rather than a `Predictor.__init__` parameter: every
    dynamic construction site does `predictor_class(league, **config.params)`
    against a `type[Predictor]`, so anything in the base signature is checked
    against a config's `float | str` values and a `str` there -- Glicko's
    `scoring_method` -- becomes a type error at every one of those sites.
    The subclasses that have the parameter declare it themselves and run it
    through here.
    """
    if not 0 <= season_regression <= 1:
        # Above 1 reflects a team through its anchor -- last year's best team
        # becomes this year's worst -- which is nonsense that a search over a
        # mis-typed parameter range would otherwise explore for an hour
        # before reporting a plausible-looking brier score.
        raise ValueError(
            f"season_regression must be in [0, 1], got {season_regression}"
        )
    return season_regression


@cache
def load_anchors(league: str) -> Mapping[str, float]:
    """A league's saved per-team regression targets, empty if it has none.

    Empty is the normal case: only ncaafb spans divisions, and even there
    the file has to be built by `division_anchors.py` first. A league
    without one regresses everybody toward MEAN_RATING, which is what a
    league whose teams all play each other wants anyway.

    Cached because an optimization run builds a predictor per probe --
    hundreds of them -- and they'd all read the same file. Callers copy it
    rather than holding the shared mapping.
    """
    path = anchor_path(league)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def resolved_anchors(
    league: str, anchors: Mapping[str, float] | None
) -> dict[str, float]:
    """The anchors a predictor should use, given what it was handed.

    A free function for the same reason `validated_regression` is: this
    can't go in `Predictor.__init__`'s signature without every
    `predictor_class(league, **config.params)` site type-checking a config's
    `float | str` values against it.

    `None` means "use the league's saved anchors", which is what a fresh
    replay wants. An explicit `{}` means "no anchors" and is left alone: a
    release that shipped without them has to replay without them, or its
    ratings stop matching what it was fit against.
    """
    return dict(load_anchors(league) if anchors is None else anchors)


class Predictor(ABC):
    def __init__(self, league: str) -> None:
        self._league = league
        # Overwritten by the subclasses that expose the parameter; 0.0 here
        # means `regress` is a no-op for the ones that don't.
        self._season_regression = 0.0
        # Per-team regression targets, set by the subclasses that take them
        # (through `resolved_anchors`). Empty means every team regresses
        # toward MEAN_RATING, which is right for a league whose teams all
        # play each other. It is *not* right for one like ncaafb, where D-III
        # teams are effectively a separate closed pool: pulling them toward
        # the same 1500 as an SEC team is what the anchor exists to fix.
        self._anchors: dict[str, float] = {}

    @property
    def league(self) -> str:
        """Which league this predictor was built for.

        Exposed because the replay needs it to pick up the league's team
        alias map, and the predictor is the only thing it's handed that
        knows.
        """
        return self._league

    @abstractmethod
    def predict_game(self, matchup: Matchup) -> Prediction:
        # A Matchup is only the pre-game half of a Game, so implementations
        # can't peek at the results.
        ...

    def update_game(self, game: Game) -> Prediction:
        """Predict and update internal state. Override in stateful subclasses."""
        return self.predict_game(game)

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this predictor, as plain JSON types.

        Keys are the constructor's keyword arguments, so `from_state_dict` is
        usually just `cls(**data)`.
        """

    @classmethod
    @abstractmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a predictor from what `state_dict` emitted."""

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps(self.state_dict()))

    @classmethod
    def load_state(cls, path: Path) -> Self:
        """Read back a `save_state` file.

        The file is one serialization of `state_dict`, not a second format:
        callers that already hold the data -- a web service reading a
        ModelRelease, say -- should use `from_state_dict` and skip the round
        trip through a temp file.
        """
        return cls.from_state_dict(json.loads(path.read_text()))

    @property
    def ratings(self) -> dict[str, Rating]:
        """Per-team ratings, normalized. Raises for a predictor without any."""
        raise RatingsUnsupported(f"{type(self).__name__} has no team ratings")

    @classmethod
    def from_ratings(
        cls, league: str, ratings: Mapping[str, Rating], **params: Any
    ) -> Self:
        """Rebuild a predictor from normalized ratings and its params.

        The inverse of the `ratings` property, and the seam a consumer that
        holds a release -- ratings and params, no state file -- comes in
        through. `params` are the constructor's tuned keyword arguments
        (home_advantage, k, ...).
        """
        raise RatingsUnsupported(f"{cls.__name__} cannot be built from ratings")

    def anchor(self, team: str) -> float:
        """Where this team's rating sits before any of its games are seen.

        Both the rating a team enters the replay at and the one `regress`
        pulls it back toward. Those have to be the same number: a team that
        starts at its division's level and reverts to the league mean would
        have the anchor slowly undone every offseason, and one that starts at
        the mean never gets to the anchor at all -- with `season_regression`
        tuned to 0, as it is in every ncaafb model, `regress` is a no-op and
        the starting value is the *only* thing the anchor gets to say.

        Falls back to MEAN_RATING for a team with no anchor, which is every
        team in a league whose divisions all play each other.
        """
        return self._anchors.get(team, MEAN_RATING)

    def regress(self, team: str, rating: float) -> float:
        """One season's worth of reversion toward the team's anchor.

        Without this a rating only ever moves by beating people, so a program
        that dominates its own schedule accumulates forever: nothing in a
        replay of twenty seasons ever pulls it back. 538 reverts a third of
        the way each offseason for the same reason.

        Shared rather than written three times because the Elo family and
        Glicko want the identical formula, and a version of it that differs
        between models by a sign or a factor is the kind of bug that shows up
        as a slightly worse brier score and nothing else.
        """
        anchor = self.anchor(team)
        return anchor + (1 - self._season_regression) * (rating - anchor)

    def pass_week(self) -> None:
        pass

    def pass_season(self) -> None:
        pass

    def postrun_callback(self) -> None:
        """Called after all seasons have been processed.

        This is the place for things like saving off final ratings."""

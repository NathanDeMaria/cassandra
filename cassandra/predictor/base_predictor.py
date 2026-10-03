import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Self

from endgame.types import Game

from cassandra.constants import CASSANDRA_HOME

from .adjustments import MatchupAdjustments
from .types import Matchup, Prediction, Rating

_ANCHOR_DIR = CASSANDRA_HOME / "predictor" / "data"

# Leagues whose teams don't all play each other, and so need a per-team anchor
# rather than one league-wide MEAN_RATING to start from and regress toward.
#
# ncaafb is the obvious one -- it spans FBS through D-III, and a D-III team's
# schedule never touches FBS. mens and womens are all D-I, but division_anchors
# tiers by conference within a division, which is what separates the ACC from
# the MEAC. nfl, nhl and wnba are deliberately absent: a closed pro league whose
# teams all play each other has nothing for a tier fit to find.
#
# Here rather than in `division_anchors.py` because two callers need the list
# without fitting anything: the batch launcher that sizes the anchor array job,
# and the array child that turns an index back into a league.
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

#: How many unfiled teams' seasons the running estimate of `unanchored_prior`
#: needs before it is used over `MEAN_RATING`. One team's rating is that team;
#: a handful is a population.
MIN_UNANCHORED_SEEN = 5


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


# What one team's anchor looks like. A bare number is a team that stayed put
# and is anchored the same way in every season it played. A list of
# [year, rating] steps is a program that moved between divisions: the rating
# is the one in effect from that year on, so the seasons it spent in D-II are
# anchored at D-II and the ones after the move are anchored where it moved to.
#
# Both shapes rather than only the second because most teams never move, and
# writing every one of them a 24-entry history would bury the handful that
# did. The bare number is also the shape every release published before this
# existed carries, and those have to keep replaying unchanged.
Anchor = float | Sequence[Sequence[float]]


def anchor_in(anchor: Anchor, season: int | None) -> float:
    """The rating `anchor` calls for in `season`.

    Clamped at both ends. Before the first step is the same answer as the
    first step -- a team can't be anchored against a division it hadn't
    reached yet, and its earliest recorded tier is the best guess for
    whatever came before. After the last is the last, which is what an
    offseason rollover into a season nobody has played needs: the anchors
    are fit from played seasons, so there is nothing later to find.

    `season` is None before the replay has entered any season, which reads
    as "the earliest" for the same reason.
    """
    if isinstance(anchor, (int, float)):
        return float(anchor)
    rating = float(anchor[0][1])
    if season is None:
        return rating
    for year, value in anchor:
        if year > season:
            break
        rating = float(value)
    return rating


@cache
def load_anchors(league: str) -> Mapping[str, Anchor]:
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
    league: str, anchors: Mapping[str, Anchor] | None
) -> dict[str, Anchor]:
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
        self._anchors: dict[str, Anchor] = {}
        # Rest, travel and quarterback availability. Inert here for the same
        # reason `_season_regression` is 0: a predictor that exposes none of
        # the weights gets a bundle whose `points` is always 0, rather than a
        # None to guard against at every call site.
        self._adjustments = MatchupAdjustments()
        # Which season the replay is in, set by `pass_season`. None until it
        # enters one, which is every team's earliest anchor -- see
        # `anchor_in`. Only anchors with a history read it at all.
        self._season: int | None = None
        # What teams with no anchor have turned out to be: the sum, sum of
        # squares and count of their end-of-season ratings, gathered as the
        # replay goes. State rather than a knob, and overwritten by the
        # subclasses that carry it through a state dict. See
        # `unanchored_prior`.
        self._unanchored_seen: tuple[float, float, int] = (0.0, 0.0, 0)
        # Who has played since the last rollover, which is the population
        # `_note_unanchored` folds in. Recorded here rather than by each
        # subclass for the reason `_roll_over` is a hook rather than an
        # override of `pass_season`: a subclass that forgot would silently
        # stop learning what an unfiled team is.
        self._played_this_season: set[str] = set()
        # `anchor` for the season in hand, filled as teams are asked for and
        # emptied at every rollover -- see `anchor`.
        self._anchor_cache: dict[str, float] = {}

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
        """Predict a game and fold its result in. Not for overriding.

        Records who played -- which is what `_note_unanchored` needs at the
        rollover -- and hands the game to `_update_game`, which is the one a
        rating model implements. Split for the same reason `pass_season`
        wraps `_roll_over`: the bookkeeping every model owes is in one place
        rather than in each model's good intentions.
        """
        self._played_this_season.update((game.home, game.away))
        return self._update_game(game)

    def _update_game(self, game: Game) -> Prediction:
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
        the mean never gets to the anchor at all -- where `season_regression`
        tunes to 0, as it does for most models in most leagues, `regress` is a
        no-op and the starting value is the *only* thing the anchor says.

        That is the usual outcome, not a corner case: of the twelve configs
        across ncaafb, mens and womens, nine tune it to 0 and the other three
        land under 0.07. Regressing toward a *tier* pulls every team in one
        toward the same number, which costs more than the reversion buys.

        Read at the season the replay is in, so a program that moved up is
        anchored where it moved to for the seasons after the move. Anchoring
        it at the division it started in instead costs it twice: it enters
        the replay far below the teams it now plays, and every offseason
        after the promotion pulls it back down there again.

        Falls back to `unanchored_prior` for a team with no anchor: the
        league mean here, which is right for every team in a league whose
        divisions all play each other, and the wrong number in one whose
        registry leaves a tier of programs unfiled. `GlickoPredictor`
        overrides it with what such teams have turned out to be.

        Memoized for the season the replay is in, because the smoother asks
        for the same few hundred answers a great many times: `passes=2` on
        ncaafb walks the season again every week, which came to 1.8 million
        calls a probe, a third of them stepping through a moved program's
        `[year, rating]` history. The cache is cleared in `pass_season`,
        the only thing that moves either input -- the clock, and (for an
        unfiled team) the running estimate `_note_unanchored` folds the
        season's teams into. Both move there, and `_note_unanchored` runs
        before the regression that reads them, so nothing can read a value
        cached from before the rollover.
        """
        hit = self._anchor_cache.get(team)
        if hit is not None:
            return hit
        found = self._anchors.get(team)
        value = (
            self.unanchored_prior() if found is None else anchor_in(found, self._season)
        )
        self._anchor_cache[team] = value
        return value

    def unanchored_prior(self) -> float:
        """Where a team with no anchor enters, and regresses toward.

        Not the middle of the league: where the last unfiled teams ended up.

        The registry classifies every FBS and FCS program, so a team it has
        no tier for is almost always a D-II or D-III program it hasn't merged
        yet, or an exhibition opponent -- and either way not an average team.
        On ncaafb 133 such teams entered at the league mean of 1500 and lost
        their first game by 33 points more than predicted, their second by 10,
        and took eight games to be rated where they belonged. Entering them at
        1100 was worth 0.0004 brier and 0.036 points of margin over the whole
        league, on 3% of its team-games.

        Rather than a number to search, this is measured as the replay goes:
        at every rollover the end-of-season rating of each unanchored team
        that played is folded into a running mean, and the next unfiled team
        enters there. The first such teams of a replay enter at the league
        mean, which is the honest answer with nothing seen; the estimate
        settles within a few seasons. `regress` pulls toward the same number,
        so an unfiled team's offseason takes it back to what unfiled teams
        are, not to the middle of the league.

        Here rather than on one rating system because it is a fact about the
        league's registry, not about Glicko: it lived on `GlickoPredictor`
        for its first year, and the four Elo configs on the three anchored
        leagues entered their unfiled teams at 1500 that whole time.
        """
        total, _, count = self._unanchored_seen
        if count < MIN_UNANCHORED_SEEN:
            return MEAN_RATING
        return total / count

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

    def matchup_adjustment(self, matchup: Matchup) -> float:
        """Rating points the home side gets from rest, travel and availability.

        0 for a predictor that exposes none of the weights, and 0 for a term
        whose input is missing -- see `MatchupAdjustments`. Shared here
        rather than written per subclass for the reason `regress` is: the
        rating models want the identical arithmetic, and a version that
        differs between them by a sign is the kind of bug that shows up as a
        slightly worse brier score and nothing else.
        """
        return self._adjustments.points(matchup)

    def pass_season(self, year: int | None = None) -> None:
        """Cross into a new season. `year` is the one being entered.

        The clock moves before the rollover, not after, so a team that
        changed division regresses toward where it is about to play rather
        than where it just was -- that ordering is the whole point of
        knowing the year, so it's fixed here instead of left to each
        subclass's `_roll_over`.

        `year` is optional because the callers that roll over into a season
        nobody has played can't name one, and because a flat anchor doesn't
        care. Omitting it leaves the clock where it was rather than resetting
        it, so those callers keep each team's most recent anchor instead of
        silently falling back to its first.
        """
        if year is not None:
            self._season = year
        # Whatever `anchor` answered last season was answered for last
        # season's clock, and for an unfiled team against an estimate that
        # `_note_unanchored` is about to fold this season's teams into. Both
        # move below, and both move before anything reads an anchor again.
        self._anchor_cache.clear()
        # A team's last game of one season says nothing about how rested it
        # is for the next, and the gap between them is an offseason rather
        # than a bye. Cleared here rather than clamped, because clamping
        # would still hand the season opener a differential built out of
        # which team played a bowl.
        self._adjustments.pass_season()
        # Before the rollover, so an unfiled team is counted at the rating it
        # earned rather than at the one regression is about to pull back.
        self._note_unanchored()
        self._roll_over()
        self._played_this_season = set()

    def _note_unanchored(self) -> None:
        """Fold this season's unfiled teams into the running estimate.

        Only teams that played since the last rollover, so a team that
        appeared once in 2007 isn't counted again every offseason after.

        A model with no team ratings has nothing to fold in and nothing that
        reads the estimate either -- `FlatPredictor` is the one -- so the
        refusal is the answer rather than an error.
        """
        try:
            rated = self.ratings
        except RatingsUnsupported:
            return
        total, squares, count = self._unanchored_seen
        for team in self._played_this_season:
            if team in self._anchors:
                continue
            found = rated.get(team)
            if found is None:
                continue
            total += found.rating
            squares += found.rating**2
            count += 1
        self._unanchored_seen = (total, squares, count)

    def _roll_over(self) -> None:
        """The offseason itself: regression, and whatever else a model does.

        Overridden instead of `pass_season` so a subclass cannot forget to
        advance the clock first.
        """

    def postrun_callback(self) -> None:
        """Called after all seasons have been processed.

        This is the place for things like saving off final ratings."""

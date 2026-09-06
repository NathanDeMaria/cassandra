"""Fit the rating each team should regress toward, from its division.

    python division_anchors.py --league ncaafb

`ncaafb` covers every division, and a D-III program's schedule is almost
entirely other D-III programs. So its ratings form a closed pool: nothing in
the results connects that pool's scale to FBS's, and Brier score can't see
the problem either, because it's dominated by intra-division games and is
unchanged by sliding a whole division up or down. Left alone, every team
starts and regresses toward the same 1500, and Mount Union rates alongside
Georgia.

The fix is a per-team anchor -- the rating `Predictor.regress` pulls toward
between seasons, and the rating a team enters the replay at -- set from the
tier the team plays in rather than from one number for the whole league.
Two pieces feed it:

*Which tier a team is in* comes from `call_it_what_you_want`, which records
the division and conference ESPN filed each team under, per season. A team
is anchored season by season, so a program that moves up isn't judged
against its later tier for its early seasons -- nor, once it has moved,
held to the one it left.

*How far apart the tiers are* is fit here, from the games that actually
cross between them -- the FBS/FCS games in September, the playoff games that
span divisions. Those are the only results that constrain the gap, which is
why this is fit directly rather than handed to the Bayesian optimizer: the
optimizer scores on overall Brier, and overall Brier is exactly what's blind
to it.

The fit is two-level: a rating is its division's level plus its conference's
offset within that division, and the offsets are held centred inside each
division. Rating conferences flat against each other instead looks like it
should work -- it's the same model with more tiers -- but it puts each
conference's whole position in the hands of the few games it plays outside
itself, and some conferences play none at all. Splitting the two lets the
division ladder rest on the thousands of games that cross a division while
a conference offset only has to explain the games inside one.

Nothing here fetches anything. It reads the stored seasons and the recorded
classifications and writes one file, which the predictors pick up on their
next run.
"""

import asyncio
import json
import math
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from typing import NamedTuple

import fire
from call_it_what_you_want import (
    TeamClassification,
    TeamNamer,
    default_classifications,
    registry_league,
)
from endgame.types import Game, Season, iter_weeks
from endgame_aws import Config

from cassandra.predictor.base_predictor import MEAN_RATING, Anchor, anchor_path
from cassandra.save_predictions import read_all_seasons

# The Elo scale: a 400-point gap is 10:1 odds, so a point of rating is this
# many logits. Written out rather than imported for the same reason the
# predictors each write it inline -- it is the definition of the scale, not a
# tunable -- and `DEFAULT_MARGIN_SCALE` turns it into points.
_LOGIT_PER_ELO = math.log(10) / 400

# A conference needs this many games *against teams outside it* before its
# own offset is worth fitting; below it, its teams fall back to the bare
# division. Counting the conference's whole schedule instead is the trap
# this file exists to avoid, one level down: a round-robin conference plays
# hundreds of games a season and every one of them cancels out of its own
# gradient, so the count sails past any threshold while the thing being
# thresholded -- evidence about where the conference sits -- stays at zero.
# The NESCAC is the clean case: 1200 games in the pool, none of them against
# anybody else, and an offset fit on nothing at all.
_MIN_CROSS_TIER_GAMES = 200

# A conference also needs this many teams before its offset means anything.
# Cross-tier games and team count catch different failures: a single team can
# play a full out-of-conference schedule and clear the games threshold on its
# own, at which point the "conference" anchor is that one team's fitted level
# wearing a conference label, with nothing to average it against.
#
# No league triggers this today -- it is a guard against a shape that has not
# turned up yet, not a fix for one that has. Don't reach for it to explain a
# suspicious anchor without checking the count first: the run report lists a
# tier's teams *at entry* (see `counts` in `_build`), so a conference can show
# "(1 teams)" there while holding a dozen across the seasons the fit pools.
# ncaafb's apparent one-team tiers are all of that kind, 8 to 20 teams each.
_MIN_TIER_TEAMS = 3

# A division needs this much crossing out of it before it can be placed on
# the ladder at all. Lower than the conference threshold on purpose: there
# are only a handful of divisions and a real one -- even the lumped label
# once most of its teams are resolved away -- clears this easily, while the
# thing it's here to catch clears nothing, having never played anyone.
_MIN_CROSS_DIVISION_GAMES = 30

# ESPN filed everything below FCS under one "Division II/III" label through
# 2008 and split it in two from 2011. So the lumped label isn't a division a
# team played in, it's a season where nobody recorded which of the two it
# was -- `_resolved_divisions` fills it in from the team's own later seasons.
_LUMPED_DIVISION = "Division II/III"
_SPANNED_DIVISIONS = frozenset({"NCAA Division II", "NCAA Division III"})

# How many of the moved teams `_build` lists. Enough to see the promotions
# that matter; the rest are conference shuffles nobody reads a log for.
_REPORTED_MOVES = 15

# Points of margin per point of rating gap, through `MarginEloPredictor`'s
# exchange rate: a 400-point gap is 10:1 odds, which is `margin_scale` points
# per unit of logit. The ladder is quoted in rating units either way, but
# *which* gap reproduces a given margin depends on this, so it is a knob and
# not a constant -- and the default is the scale ncaafb's margin models fit,
# since ncaafb is the only league whose ladder has more than one rung.
DEFAULT_MARGIN_SCALE = 13.0

# Margins are capped before they are fit on, the same way the margin models
# cap theirs. A 70-0 guarantee game is evidence that the gap is large and
# very weak evidence about how large; without a cap a handful of them set the
# rung. ncaafb's `margin_blend` fits `mov_cap` at 56, and the ladder moved by
# 3 rating points between a cap of 56 and one of 100, so this is a guard
# rather than a parameter worth tuning.
DEFAULT_MARGIN_CAP = 56.0

# Divisions that span tiers rather than naming one. ESPN filed everything
# below FCS under "Division II/III" and `resolve_lumped` fills that in from a
# team's own later seasons -- but 54 teams were classified once, under the
# spanning label, and never again, so the label survives in every season
# through 2026 rather than ending in 2010.
#
# Those teams still need an anchor, so the tier is still rated. What it is
# not is a rung: a level fit between D-II and D-III, on games played by a
# mixture of both, is a constraint that pulls the two together, and it pulled
# them to within 11 rating points of each other while their own games said
# 114 to 151. So these divisions read the ladder rather than setting it --
# see `fit_tiers`.
_MIXTURE_DIVISIONS = frozenset({_LUMPED_DIVISION})

# Gradient descent on squared error, stopped on the gradient rather than at a
# fixed iteration count. A fixed count is what hides a fit that quietly
# stopped early: the numbers it prints look like ratings either way, and the
# only symptom is a scale that's too narrow -- which is the exact failure
# this file is here to prevent.
_TOLERANCE = 1e-4
_MAX_ITERATIONS = 500_000

# The least-squares Hessian for one rating is `n * points_per_rating**2`, so
# dividing the gradient by that is a Newton step and this is the damping on
# it. Not a rate to tune: at 1.0 the step is the exact minimiser of the
# quadratic and overshoots on the coupled parameters.
_DAMPING = 0.5


class Tier(NamedTuple):
    """A division, and the conference inside it a team plays in.

    `conference` is None for a team whose conference isn't worth its own
    offset -- an independent, or one that never plays outside itself -- and
    that tier is rated at its division's level.
    """

    division: str
    conference: str | None = None

    def __str__(self) -> str:
        return (
            self.division
            if self.conference is None
            else f"{self.division} / {self.conference}"
        )


class TierGame(NamedTuple):
    """One game, reduced to the two tiers that played it.

    `home_margin` rather than the win it used to carry. Win/loss is what
    saturates: once the favourite always wins, the likelihood stops caring
    how much it wins by, and the rungs at the bottom of the ladder are
    exactly the ones where every game is a blowout. Measured on ncaafb, the
    two estimators agree on FBS/FCS (315 vs 307) and disagree by half the
    rung below it -- FCS over D-III reads as 238 from the win rate and 347
    from the margin.

    It is also what the consumer wants. The football models are scored on
    `margin_mae` now, and a ladder fit to reproduce win *frequency* is not
    one that reproduces win *margin*.
    """

    home_tier: Tier
    away_tier: Tier
    home_margin: float
    neutral_site: bool


class Fit(NamedTuple):
    """Fitted ratings, the division levels under them, and home advantage."""

    ratings: dict[Tier, float]
    divisions: dict[str, float]
    home_advantage: float

    def rating(self, tier: Tier) -> float | None:
        """This tier's rating, falling back to its division's level.

        The fallback is what a team in a conference too thin to fit gets,
        and it's a real answer rather than a guess: the division term is
        fit on every game that crossed a division, and the offset that
        isn't being applied is one the data never pinned down anyway.
        """
        if tier in self.ratings:
            return self.ratings[tier]
        return self.divisions.get(tier.division)


class _Aggregated(NamedTuple):
    """Games collapsed to counts per distinct matchup.

    The gradient only ever reads a game through its (home tier, away tier,
    neutral) triple, so a hundred identical matchups contribute a hundred
    times one term. Summing them first is the same arithmetic against a few
    hundred rows instead of seventy thousand, which is what makes running
    to convergence affordable at all.
    """

    played: Counter[tuple[Tier, Tier, bool]]
    home_margin: Counter[tuple[Tier, Tier, bool]]


def _aggregate(
    games: Iterable[TierGame], cap: float = DEFAULT_MARGIN_CAP
) -> _Aggregated:
    """Games collapsed to a count and a summed margin per matchup.

    The sum is all a least-squares gradient reads -- the derivative of
    `sum((y - yhat)**2)` in a rating is `sum(y - yhat)`, and `yhat` is the
    same for every game in a matchup -- so the collapse that made the win
    fit affordable survives the change of loss unchanged.
    """
    played: Counter[tuple[Tier, Tier, bool]] = Counter()
    margin: Counter[tuple[Tier, Tier, bool]] = Counter()
    for home, away, home_margin, neutral in games:
        played[(home, away, neutral)] += 1
        margin[(home, away, neutral)] += max(-cap, min(cap, home_margin))
    return _Aggregated(played, margin)


def fit_tiers(
    games: Iterable[TierGame],
    mean: float = MEAN_RATING,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    margin_cap: float = DEFAULT_MARGIN_CAP,
    mixture_divisions: Collection[str] = _MIXTURE_DIVISIONS,
) -> Fit:
    """Rate each tier from the margins of the games played between them.

    Least squares on the margin a rating gap implies, through
    `MarginEloPredictor`'s own exchange rate, rather than the win/loss
    likelihood this used to maximize. `TierGame` has the measurement that
    motivated the change; what it buys is the bottom of the ladder, where
    every crossing game is a blowout and a win is the same win at any gap.

    A tier's rating is its division's level plus its conference's offset,
    and the offsets are recentred inside each division on every step. The
    split is otherwise unidentified -- adding ten to a division and taking
    ten off each of its conferences predicts identically -- and leaving it
    that way is what lets a conference drift away from its own division
    until D-III conferences are interleaved with FCS ones.

    Home advantage is fit at the same time rather than assumed, because the
    cross-division games are overwhelmingly played at the bigger school's
    stadium -- FBS teams buy home games against FCS ones. Holding it at zero
    would charge the whole home-field effect to the smaller division and
    exaggerate every gap.

    A division in `mixture_divisions` is rated but does not rate anybody
    else: its games move its own level and leave its opponents' alone. Such
    a division is a label spanning two real tiers rather than a tier, so a
    level fit between them is a rung that isn't there, and letting it pull
    on both is how D-II and D-III ended up 11 rating points apart. It still
    gets a level, because the teams under the label still need an anchor.

    Ratings come back centred so the games-weighted average is `mean`. Any
    constant added to every tier predicts identically, so a centre has to be
    chosen; keeping the existing one leaves teams sitting where they always
    have and only spreads the tiers apart around it.

    `home_advantage` comes back in **points of margin**, not rating units,
    which is how `MarginEloPredictor` quotes its own. On ncaafb it fits 2.9
    against that model's independently fitted 2.78.
    """
    played, home_margin = _aggregate(games, margin_cap)
    if not played:
        return Fit({}, {}, 0.0)
    points_per_rating = _LOGIT_PER_ELO * margin_scale
    mixture = frozenset(mixture_divisions)

    tiers = {tier for home, away, _ in played for tier in (home, away)}
    divisions = dict.fromkeys({tier.division for tier in tiers}, 0.0)
    offsets = dict.fromkeys(
        (tier for tier in tiers if tier.conference is not None), 0.0
    )
    by_division: dict[str, list[Tier]] = defaultdict(list)
    for tier in offsets:
        by_division[tier.division].append(tier)

    # Each parameter is stepped against the games that can actually move it.
    # Dividing by the whole pool instead would scale the division ladder by
    # the tens of thousands of games played *inside* a division, every one
    # of which cancels out of its gradient -- the step ends up ~70x too
    # small and 2000 iterations stop a long way short of the answer.
    total = sum(played.values())
    cross_division = sum(
        n for (home, away, _), n in played.items() if home.division != away.division
    )
    cross_tier = sum(n for (home, away, _), n in played.items() if home != away)
    # Newton, damped: the second derivative of the squared error in a rating
    # is `n * points_per_rating**2`, so dividing by it puts the step in
    # rating units whatever the margin scale is.
    division_step = _DAMPING / max(1, cross_division) / points_per_rating
    offset_step = _DAMPING / max(1, cross_tier) / points_per_rating
    home_step = _DAMPING / total

    appearances: Counter[Tier] = Counter()
    for (home, away, _), n in played.items():
        appearances[home] += n
        appearances[away] += n

    home_advantage = 0.0
    for _ in range(_MAX_ITERATIONS):
        division_gradient = dict.fromkeys(divisions, 0.0)
        offset_gradient = dict.fromkeys(offsets, 0.0)
        home_gradient = 0.0
        for (home, away, neutral), n in played.items():
            edge = 0.0 if neutral else home_advantage
            expected = (
                _rating(home, divisions, offsets) - _rating(away, divisions, offsets)
            ) * points_per_rating + edge
            error = home_margin[(home, away, neutral)] - n * expected
            home_mixture = home.division in mixture
            away_mixture = away.division in mixture
            if home_mixture or away_mixture:
                # A spanning label reads the ladder rather than setting a
                # rung of it, so this game places the mixture and leaves the
                # real tier it played where the tiers it *is* put it.
                if home_mixture:
                    division_gradient[home.division] += error
                if away_mixture:
                    division_gradient[away.division] -= error
            else:
                division_gradient[home.division] += error
                division_gradient[away.division] -= error
            if home.conference is not None:
                offset_gradient[home] += error
            if away.conference is not None:
                offset_gradient[away] -= error
            if not neutral:
                home_gradient += error

        # Hold the offsets centred by taking the un-centred direction out of
        # their gradient, rather than by correcting after the step. A game
        # that crosses a division moves both the division term and the two
        # offsets, so recentring afterwards hands the division that second
        # copy and the whole ladder drifts upward forever -- every gap
        # already converged, `moved` pinned at the size of the drift, and
        # nothing to show for the next half million iterations. Projected
        # here, an offset can only ever say where a conference sits inside
        # its division, and the division term moves on its own gradient.
        for members in by_division.values():
            scale = sum(appearances[tier] ** 2 for tier in members)
            if not scale:
                continue
            share = (
                sum(offset_gradient[tier] * appearances[tier] for tier in members)
                / scale
            )
            for tier in members:
                offset_gradient[tier] -= share * appearances[tier]

        moved = 0.0
        for division, value in division_gradient.items():
            divisions[division] += division_step * value
            moved = max(moved, abs(division_step * value))
        for tier, value in offset_gradient.items():
            offsets[tier] += offset_step * value
            moved = max(moved, abs(offset_step * value))
        home_advantage += home_step * home_gradient

        if moved < _TOLERANCE:
            break

    ratings = {tier: _rating(tier, divisions, offsets) for tier in tiers}
    weighted = sum(ratings[tier] * n for tier, n in appearances.items()) / sum(
        appearances.values()
    )
    shift = mean - weighted
    return Fit(
        {tier: rating + shift for tier, rating in ratings.items()},
        {division: level + shift for division, level in divisions.items()},
        home_advantage,
    )


def _rating(
    tier: Tier, divisions: Mapping[str, float], offsets: Mapping[Tier, float]
) -> float:
    return divisions[tier.division] + (
        0.0 if tier.conference is None else offsets[tier]
    )


def _played_games(seasons: Iterable[Season]) -> Iterator[tuple[Season, Game]]:
    """Every game that has actually been played, with the season it's in.

    Season pickles carry the fixtures ahead of them as well as the results
    behind them, and neither thing here wants a fixture: a scheduled game
    has no result to fit a tier gap on, and anchoring a team by the first
    season it *appears* in would let a game it hasn't played yet decide
    which tier it's judged against for good.
    """
    for season in seasons:
        for week in iter_weeks(season, validate=False):
            for game in week.games:
                if not game.completed:
                    continue
                yield season, game


def _seasons_played(
    seasons: Iterable[Season], namer: TeamNamer
) -> dict[str, list[int]]:
    """Every season each canonical team name shows up in, in order."""
    played: dict[str, set[int]] = defaultdict(set)
    for season, game in _played_games(seasons):
        for name in (game.home, game.away):
            played[namer.canonical(name)].add(season.year)
    return {team: sorted(years) for team, years in played.items()}


def _first_years(seasons: Iterable[Season], namer: TeamNamer) -> dict[str, int]:
    """The earliest season each canonical team name shows up in."""
    return {team: years[0] for team, years in _seasons_played(seasons, namer).items()}


class _Classifier:
    """Each team-season's tier, with ESPN's lumped label filled in.

    Built once and asked per team-season, because the gap between tiers has
    to be fit on which tiers actually played -- a 2015 game between a team
    that was FBS in 2015 and one that was FCS in 2015 is evidence about the
    FBS/FCS gap no matter where either program started out. Reading every
    game through its teams' *first* classification instead mixes the tiers
    together: a fifth of the FCS bucket is FBS by its last season, and 88%
    of the lumped bucket has a specific division recorded later, so the
    buckets end up describing each other and the fitted gaps collapse.
    """

    def __init__(self, namer: TeamNamer, league: str) -> None:
        self._namer = namer
        self._classifications = default_classifications()
        self._registry = registry_league(league)
        self._resolved: dict[str, str] = {}

    def resolve_lumped(self, teams: Mapping[str, list[int]]) -> None:
        """Decide, per team, which division the lumped label meant.

        The label spans D-II and D-III, so it's only ever filled in from a
        season where ESPN recorded one of those two. A program that appears
        lumped and is next seen in FCS moved up; backfilling FCS onto its
        earlier seasons would be inventing a promotion that hadn't happened
        yet, so those seasons keep the lumped tier and are rated as their
        own thing.

        The seasons it played are asked first, then the record past them.
        ESPN re-surveyed everything below FCS in 2011, and a program that
        had already stopped playing by then is never asked about a year it
        has no game in -- which is how Colorado College, filed as D-III in
        2011, kept the spanning label for every season it did play. Five of
        the fifty-four teams still carrying it are that case.

        A team that moved between its last game and the survey is filled in
        from where it ended rather than where it was, which is the same
        trade `_anchors` already makes inside the span and is worth five
        teams' worth of anchors.
        """
        last_recorded = max(
            (year for years in teams.values() for year in years), default=0
        )
        for team, years in teams.items():
            for year in (*years, last_recorded):
                found = self._classification(team, year)
                if found is not None and found.division in _SPANNED_DIVISIONS:
                    self._resolved[team] = found.division
                    break

    def tier(self, team: str, year: int) -> Tier | None:
        """Where `team` sat in `year`, or None if nobody classified it.

        A team the registry can't place, or one nobody has classified, gets
        no tier at all: it keeps the default anchor, which is the same
        behaviour it has today. Guessing a division from the company it
        keeps would be a rating decision dressed up as a data one.
        """
        found = self._classification(team, year)
        if found is None:
            return None
        division = found.division
        if division == _LUMPED_DIVISION:
            division = self._resolved.get(team, _LUMPED_DIVISION)
        return Tier(division, found.conference)

    def _classification(self, team: str, year: int) -> TeamClassification | None:
        espn_id = self._namer.espn_id(team)
        if espn_id is None or self._registry is None:
            return None
        return self._classifications.classification_in(espn_id, year, self._registry)


def _tier_games(
    seasons: Iterable[Season], namer: TeamNamer, classifier: _Classifier
) -> list[TierGame]:
    games = []
    for season, game in _played_games(seasons):
        # A drawn game used to be dropped, because it was neither the win nor
        # the loss the likelihood took. A margin of zero is an ordinary
        # observation, so it stays.
        home = classifier.tier(namer.canonical(game.home), season.year)
        away = classifier.tier(namer.canonical(game.away), season.year)
        if home is None or away is None:
            continue
        games.append(
            TierGame(home, away, game.home_score - game.away_score, game.neutral_site)
        )
    return games


class _Resolved(NamedTuple):
    """Games after thin conferences fold in, and the folding that happened.

    Both halves are needed downstream: the fit reads the games, and the
    anchor for a team in a folded conference has to look its tier up under
    the name the fit actually rated it by.
    """

    games: list[TierGame]
    folded: dict[Tier, Tier]


def _drop_thin_conferences(
    games: Iterable[TierGame], team_counts: Mapping[Tier, int] | None = None
) -> _Resolved:
    """Fold conferences with too little to fit on back into their division.

    A conference whose games are nearly all against itself would otherwise
    get an offset driven by whichever few outside results it happened to
    have -- or, where it has none, by nothing at all -- and a confident
    wrong anchor is worse than the shared one, because it's the rating every
    team in it starts from.

    `team_counts` applies the same reasoning to how many teams the offset is
    averaged over. Omitting it checks games only, which is what the fit tests
    that build games directly want.
    """
    games = list(games)
    team_counts = team_counts or {}
    crossing: Counter[Tier] = Counter()
    for home, away, _, _ in games:
        if home == away:
            continue
        crossing[home] += 1
        crossing[away] += 1
    tiers = {tier for game in games for tier in (game.home_tier, game.away_tier)}
    thin_games = {
        tier
        for tier in tiers
        if tier.conference is not None and crossing[tier] < _MIN_CROSS_TIER_GAMES
    }
    thin_teams = {
        tier
        for tier in tiers
        if tier.conference is not None
        and tier not in thin_games
        and team_counts.get(tier, _MIN_TIER_TEAMS) < _MIN_TIER_TEAMS
    }
    thin = thin_games | thin_teams
    if thin_games:
        print(
            f"  {len(thin_games)} conference(s) below {_MIN_CROSS_TIER_GAMES} games "
            "outside themselves, using division:"
        )
        for tier in sorted(thin_games, key=str):
            print(f"    {tier} ({crossing[tier]} of {_played_by(games, tier)})")
    if thin_teams:
        print(
            f"  {len(thin_teams)} conference(s) below {_MIN_TIER_TEAMS} teams, "
            "using division:"
        )
        for tier in sorted(thin_teams, key=str):
            print(f"    {tier} ({team_counts.get(tier, 0)} team(s))")

    folded = {tier: Tier(tier.division) for tier in thin}

    def resolve(tier: Tier) -> Tier:
        return folded.get(tier, tier)

    return _Resolved(
        [
            TierGame(
                resolve(g.home_tier),
                resolve(g.away_tier),
                g.home_margin,
                g.neutral_site,
            )
            for g in games
        ],
        folded,
    )


def _tier_team_counts(
    played: Mapping[str, Sequence[int]], classifier: _Classifier
) -> Counter[Tier]:
    """How many distinct teams each tier ever held.

    Counted over every season rather than at entry, to match what the fit is
    averaging over: a conference that had three teams across twenty seasons
    but never two at once is as thin as the fit's view of it.
    """
    counts: Counter[Tier] = Counter()
    for team, years in played.items():
        for tier in {t for year in years if (t := classifier.tier(team, year))}:
            counts[tier] += 1
    return counts


def _played_by(games: Iterable[TierGame], tier: Tier) -> int:
    return sum(
        1 for g in games for played in (g.home_tier, g.away_tier) if played == tier
    )


def _unplaceable_divisions(games: Iterable[TierGame]) -> set[str]:
    """Divisions with too little crossing out of them to sit anywhere.

    A thin conference has somewhere to fall back to; a thin division does
    not, and its level is whatever the fit's centring happens to leave it
    at. ESPN files the all-star bowls as their own division -- six squads
    that exist for two Januaries and play only each other -- and rating
    them is how a team called "East" ends up with an anchor. Teams in one
    of these get no anchor at all, which is the same treatment as a team
    nobody classified.
    """
    crossing: Counter[str] = Counter()
    for home, away, _, _ in games:
        if home.division == away.division:
            continue
        crossing[home.division] += 1
        crossing[away.division] += 1
    divisions = {
        tier.division for game in games for tier in (game.home_tier, game.away_tier)
    }
    if len(divisions) < 2:
        # A league with one division has no ladder to place anything on, so
        # "never plays another division" is the normal state rather than the
        # symptom -- mens and womens are entirely D-I, and every game in them
        # trips the count below. Without this the only division a league has
        # is declared unplaceable and every team in it loses its anchor.
        return set()
    unplaceable = {d for d in divisions if crossing[d] < _MIN_CROSS_DIVISION_GAMES}
    if unplaceable:
        print(
            f"  {len(unplaceable)} division(s) below {_MIN_CROSS_DIVISION_GAMES} "
            "games outside themselves, left unanchored:"
        )
        for division in sorted(unplaceable):
            print(f"    {division} ({crossing[division]} games)")
    return unplaceable


def _anchors(
    seasons: Iterable[Season],
    namer: TeamNamer,
    classifier: _Classifier,
    fit: Fit,
    thin: Mapping[Tier, Tier],
    unplaceable: frozenset[str] = frozenset(),
) -> dict[str, Anchor]:
    """Each team's anchor, per season, as the steps where it changes.

    A program that moved up gets one anchor for the seasons before the move
    and another after. Pinning it to its first season instead charged it
    twice over: North Dakota State, D-II in 2002 and the best team in FCS
    ever since, entered the replay 366 points below the teams it plays and
    then got pulled back down there every offseason.

    Teams that never moved -- almost all of them -- come out as a bare
    number rather than a one-entry history, which keeps the file readable
    and is the shape the predictors already understood.
    """
    anchors: dict[str, Anchor] = {}
    for team, years in _seasons_played(seasons, namer).items():
        steps: list[list[float]] = []
        for year in years:
            tier = classifier.tier(team, year)
            if tier is None or tier.division in unplaceable:
                continue
            rating = fit.rating(thin.get(tier, tier))
            # A season the fit couldn't place doesn't end the previous step:
            # the team kept playing, and the tier it was last placed in is a
            # better answer than dropping back to the league mean for a year.
            if rating is None or (steps and steps[-1][1] == rating):
                continue
            steps.append([year, rating])
        if steps:
            anchors[team] = steps[0][1] if len(steps) == 1 else steps
    return anchors


def _steps(anchor: Anchor) -> Sequence[Sequence[float]] | None:
    """The history behind an anchor, or None for a team that never moved."""
    return None if isinstance(anchor, (int, float)) else anchor


def _first_step(anchor: Anchor) -> float:
    """The rating a team enters the replay at."""
    if isinstance(anchor, (int, float)):
        return float(anchor)
    return float(anchor[0][1])


def _last_step(anchor: Anchor) -> float:
    """The rating a team ends up anchored at."""
    if isinstance(anchor, (int, float)):
        return float(anchor)
    return float(anchor[-1][1])


async def _build(league: str, write: bool) -> None:
    bucket = Config.init_from_file().bucket
    print(f"Loading {league} seasons from s3://{bucket}")
    seasons = [s async for s in read_all_seasons(league, bucket)]
    if not seasons:
        raise ValueError(f"No seasons for {league!r} in s3://{bucket}/seasons/")
    namer = TeamNamer.for_league(league)
    print(f"  {len(seasons)} seasons; {namer.report()}")

    played = _seasons_played(seasons, namer)
    classifier = _Classifier(namer, league)
    classifier.resolve_lumped(played)

    raw = _tier_games(seasons, namer, classifier)
    classified = {
        team for team, years in played.items() if classifier.tier(team, years[0])
    }
    print(f"{len(played)} teams -> {len(classified)} classified")
    if not classified:
        raise ValueError(
            f"No team in {league!r} has a recorded division. Run "
            f"`ciwyw classifications fetch {league} <first_year> <last_year>` "
            "first -- without it there's nothing to anchor teams to."
        )

    games, folded = _drop_thin_conferences(raw, _tier_team_counts(played, classifier))
    unplaceable = frozenset(_unplaceable_divisions(games))
    divisions = {tier.division for g in games for tier in (g.home_tier, g.away_tier)}
    print(
        f"{len(games)} games across {len(divisions)} division(s), "
        f"{len(set(t for g in games for t in (g.home_tier, g.away_tier)))} tier(s)"
    )

    fit = fit_tiers(games)
    print(f"\nhome advantage {fit.home_advantage:.0f}\n")
    print("  division ladder:")
    for division, level in sorted(fit.divisions.items(), key=lambda kv: -kv[1]):
        print(f"  {level:7.0f}  {division}")

    anchors = _anchors(seasons, namer, classifier, fit, folded, unplaceable)
    if not anchors:
        # Nothing downstream would notice: the file would be written empty,
        # every team would silently fall back to MEAN_RATING, and the run
        # would report success. There is no league where "classified teams
        # but no anchors" is a real answer, so it's a bug in the fit above
        # rather than something to publish.
        raise ValueError(
            f"Fit {len(classified)} classified {league!r} team(s) but produced "
            "no anchors. Every division was dropped as unplaceable, or no tier "
            "the teams are in came back rated."
        )
    # Counted by where each team *enters*, since that's the tier it's being
    # listed under. A team that moved is counted once, at the division it
    # started in, and shows up again in the move list below.
    counts = Counter(_first_step(anchor) for anchor in anchors.values())
    print("\n  tiers, by the anchor each gives:")
    for tier, rating in sorted(fit.ratings.items(), key=lambda kv: -kv[1]):
        if counts[rating]:
            print(f"  {rating:7.0f}  {tier}  ({counts[rating]} teams)")

    moved = {
        team: steps
        for team, anchor in anchors.items()
        if (steps := _steps(anchor)) is not None
    }
    climbed = sorted(
        moved.items(), key=lambda kv: _first_step(kv[1]) - _last_step(kv[1])
    )
    print(
        f"\n  {len(moved)} of {len(anchors)} team(s) changed tier at some point "
        "and carry a history. Biggest climbs:"
    )
    # Just the top of the list: every run prints this, and the point of it is
    # to make the promotions visible at a glance, not to list four hundred
    # conference shuffles.
    for team, steps in climbed[:_REPORTED_MOVES]:
        path = " -> ".join(f"{int(year)}:{rating:.0f}" for year, rating in steps)
        print(f"    {_first_step(steps) - _last_step(steps):+6.0f}  {team}: {path}")

    path = anchor_path(league)
    if not write:
        print(f"\n--write not passed; {len(anchors)} anchor(s) not saved to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(anchors, indent=2, sort_keys=True))
    print(f"\nWrote {len(anchors)} anchor(s) to {path}")


def main(league: str = "ncaafb", write: bool = False, if_missing: bool = False) -> None:
    """
    Fit per-team regression anchors from each team's division.

    Prints the fitted tiers and writes nothing unless `--write` is passed,
    since the file it replaces changes every rating the next run produces.

    `--if-missing` turns this into a no-op when the league already has an
    anchor file. That's what the anchors job calls, so a first run builds the
    anchors and every run after it leaves them alone -- refitting on each run
    would silently re-rate every model against a slightly different scale.
    Rebuilding on purpose means deleting the file, which is the same shape as
    every other decision here that moves published ratings.
    """
    path = anchor_path(league)
    if if_missing and path.exists():
        # Checked before `_build`, which reads every season out of s3 --
        # minutes of work to reach a file we already know is there.
        print(f"{league}: anchors already at {path}")
        return
    asyncio.run(_build(league, write))


if __name__ == "__main__":
    fire.Fire(main)

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
between seasons -- set from the tier the team plays in rather than from one
number for the whole league. Two pieces feed it:

*Which tier a team is in* comes from `call_it_what_you_want`, which records
the division and conference ESPN filed each team under, per season. A team
is anchored by where it was the first year it appears, so a program that
moves up doesn't have its early seasons judged against its later tier.

*How far apart the tiers are* is fit here, from the games that actually
cross between them -- the FBS/FCS games in September, the playoff games that
span divisions. Those are the only results that constrain the gap, which is
why this is fit directly rather than handed to the Bayesian optimizer: the
optimizer scores on overall Brier, and overall Brier is exactly what's blind
to it.

Nothing here fetches anything. It reads the stored seasons and the recorded
classifications and writes one file, which the predictors pick up on their
next run.
"""

import asyncio
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from typing import NamedTuple

import fire
from call_it_what_you_want import TeamNamer, default_classifications, registry_league
from endgame.types import Game, Season, iter_weeks
from endgame_aws import Config

from cassandra.predictor.base_predictor import MEAN_RATING, anchor_path
from cassandra.save_predictions import read_all_seasons

# The Elo scale: a 400-point gap is 10:1 odds. Shared with the predictors by
# construction rather than by import -- they each write the forward direction
# inline, and this is the only place that inverts it.
_SCALE = 400 / math.log(10)

# A tier needs this many games before its own conference-level rating is
# worth fitting; below it, its teams fall back to the division's. A single
# conference plays a few hundred games a season, so this only catches the
# genuinely thin ones -- an independent, or a conference that existed for
# two years.
_MIN_TIER_GAMES = 200

# Gradient ascent on the log-likelihood. There are a few dozen tiers against
# tens of thousands of games, so this is a tiny, very over-determined fit;
# these are set to converge it well past the point the numbers stop moving.
_ITERATIONS = 2000
_LEARNING_RATE = 400.0


class TierGame(NamedTuple):
    """One game, reduced to the two tiers that played it."""

    home_tier: str
    away_tier: str
    home_won: float
    neutral_site: bool


class Fit(NamedTuple):
    """Fitted tier ratings, and the home advantage they were fit alongside."""

    ratings: dict[str, float]
    home_advantage: float


def fit_tiers(games: Iterable[TierGame], mean: float = MEAN_RATING) -> Fit:
    """Rate each tier from the games played between them.

    Home advantage is fit at the same time rather than assumed, because the
    cross-division games are overwhelmingly played at the bigger school's
    stadium -- FBS teams buy home games against FCS ones. Holding it at zero
    would charge the whole home-field effect to the smaller division and
    exaggerate every gap.

    Ratings come back centred so the games-weighted average is `mean`. Any
    constant added to every tier predicts identically, so a centre has to be
    chosen; keeping the existing one leaves teams sitting where they always
    have and only spreads the tiers apart around it.
    """
    games = list(games)
    ratings = {tier: 0.0 for game in games for tier in (game.home_tier, game.away_tier)}
    if not games:
        return Fit({}, 0.0)

    home_advantage = 0.0
    step = _LEARNING_RATE / len(games)
    for _ in range(_ITERATIONS):
        gradient = dict.fromkeys(ratings, 0.0)
        home_gradient = 0.0
        for home_tier, away_tier, home_won, neutral in games:
            edge = 0.0 if neutral else home_advantage
            expected = _win_probability(
                ratings[home_tier] + edge - ratings[away_tier],
            )
            error = home_won - expected
            gradient[home_tier] += error
            gradient[away_tier] -= error
            if not neutral:
                home_gradient += error
        for tier, value in gradient.items():
            ratings[tier] += step * value
        home_advantage += step * home_gradient

    appearances = Counter(
        tier for game in games for tier in (game.home_tier, game.away_tier)
    )
    total = sum(appearances.values())
    weighted = sum(ratings[tier] * n for tier, n in appearances.items()) / total
    return Fit(
        {tier: mean + rating - weighted for tier, rating in ratings.items()},
        home_advantage,
    )


def _win_probability(rating_difference: float) -> float:
    return 1 / (1 + math.exp(-rating_difference / _SCALE))


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


def _first_years(seasons: Iterable[Season], namer: TeamNamer) -> dict[str, int]:
    """The earliest season each canonical team name shows up in."""
    first: dict[str, int] = {}
    for season, game in _played_games(seasons):
        for name in (game.home, game.away):
            canonical = namer.canonical(name)
            if season.year < first.get(canonical, season.year + 1):
                first[canonical] = season.year
    return first


class _Tiers(NamedTuple):
    """Each team's tier, and the coarser one it falls back to."""

    full: dict[str, str]
    division: dict[str, str]


def _team_tiers(
    first_years: Mapping[str, int], namer: TeamNamer, league: str
) -> _Tiers:
    """Where each team sat the first season it appears.

    A team the registry can't place, or one nobody has classified, gets no
    tier at all: it keeps the default anchor, which is the same behaviour it
    has today. Guessing a division from the company it keeps would be a
    rating decision dressed up as a data one.
    """
    classifications = default_classifications()
    registry = registry_league(league)
    full: dict[str, str] = {}
    division: dict[str, str] = {}
    for team, year in first_years.items():
        espn_id = namer.espn_id(team)
        if espn_id is None or registry is None:
            continue
        found = classifications.classification_in(espn_id, year, registry)
        if found is None:
            continue
        division[team] = found.division
        full[team] = (
            found.division
            if found.conference is None
            else f"{found.division} / {found.conference}"
        )
    return _Tiers(full, division)


def _tier_games(
    seasons: Iterable[Season], namer: TeamNamer, tiers: Mapping[str, str]
) -> list[TierGame]:
    games = []
    for _, game in _played_games(seasons):
        home = tiers.get(namer.canonical(game.home))
        away = tiers.get(namer.canonical(game.away))
        if home is None or away is None:
            continue
        if game.home_score == game.away_score:
            continue
        games.append(
            TierGame(
                home,
                away,
                1.0 if game.home_score > game.away_score else 0.0,
                game.neutral_site,
            )
        )
    return games


def _merge_thin_tiers(tiers: _Tiers, games: Iterable[TierGame]) -> dict[str, str]:
    """Drop back to the division for tiers with too little to fit on.

    A conference with a handful of games would otherwise get a rating driven
    by whichever few results it happened to have, and a confident wrong
    anchor is worse than the shared one -- it's applied every offseason to
    every team in it.
    """
    played = Counter(
        tier for game in games for tier in (game.home_tier, game.away_tier)
    )
    thin = {tier for tier, n in played.items() if n < _MIN_TIER_GAMES}
    thin |= {tier for tier in set(tiers.full.values()) if tier not in played}
    if thin:
        print(f"  {len(thin)} tier(s) below {_MIN_TIER_GAMES} games, using division:")
        for tier in sorted(thin):
            print(f"    {tier} ({played.get(tier, 0)} games)")
    return {
        team: tiers.division[team] if tier in thin else tier
        for team, tier in tiers.full.items()
    }


async def _build(league: str, write: bool) -> None:
    bucket = Config.init_from_file().bucket
    print(f"Loading {league} seasons from s3://{bucket}")
    seasons = [s async for s in read_all_seasons(league, bucket)]
    if not seasons:
        raise ValueError(f"No seasons for {league!r} in s3://{bucket}/seasons/")
    namer = TeamNamer.for_league(league)
    print(f"  {len(seasons)} seasons; {namer.report()}")

    first_years = _first_years(seasons, namer)
    tiers = _team_tiers(first_years, namer, league)
    print(f"{len(first_years)} teams -> {len(tiers.full)} classified")
    if not tiers.full:
        raise ValueError(
            f"No team in {league!r} has a recorded division. Run "
            f"`ciwyw classifications fetch {league} <first_year> <last_year>` "
            "first -- without it there's nothing to anchor teams to."
        )

    resolved = _merge_thin_tiers(tiers, _tier_games(seasons, namer, tiers.full))
    games = _tier_games(seasons, namer, resolved)
    print(f"{len(games)} games across {len(set(resolved.values()))} tier(s)")

    fit = fit_tiers(games)
    print(f"\nhome advantage {fit.home_advantage:.0f}\n")
    counts = Counter(resolved.values())
    for tier, rating in sorted(fit.ratings.items(), key=lambda kv: -kv[1]):
        print(f"  {rating:7.0f}  {tier}  ({counts[tier]} teams)")

    anchors = {
        team: fit.ratings[tier]
        for team, tier in resolved.items()
        if tier in fit.ratings
    }
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

"""What a game's players say its margin should be, from a model of players.

Every rating model here rates teams from their results. ten-minute-version
rates *players*, from who was on the floor for every possession (lineup
RAPM, chained season to season and seeded by recruiting ranks), and files
each team as of each week of a season: its players at their expected shares
of its floor time, its possessions per game. `{league}_player_ratings.csv`
under the predictor data prefix, mens and womens:

    league, season, as_of, team_id, team, offense, defense, net, pace,
    games, home_edge

`net` is offense plus defense per 100 possessions and `pace` possessions per
game; a row is fitted on the games before `as_of` and stands for games on
or after it, until the team's next row.

**What a model reads** is the neutral-site margin those say: the two teams'
paces averaged, times the gap in their nets. The home edge is left to the
model, which has its own. `GlickoPredictor` blends it into the prediction
alone -- its own rating gap at `glicko_share`, plus this margin at
`player_rating_weight` rating points per point -- and its ratings still
learn from the results alone.

Why the prediction alone: priced in the update as well, as the matchup
terms are, the ratings learn what the players' margin misses, but the
prediction still reads them at full weight on top of it -- and at no
weight did that come near the blend. Against `glicko_full`'s 2026-10-19
fits on every game the file covers (2014-2025, the whole season):

    Brier            glicko_full   in the update too   prediction only
    mens                 0.17313    0.17225 (w 0.3)     0.17102 (t -14)
    womens               0.15489    0.15463 (w 0.2)     0.15285 (t -12)

the blend a logistic fit with each season held out, at about 0.8 of
glicko's logit and 0.57 per 10 points of the players' margin, both
leagues. It helps in every stretch of the season but mens' first games.
Replayed here at that fit (`glicko_share` 0.82 and `player_rating_weight`
10 mens, 0.76 and 9.7 womens), every game the league has: mens 0.16987 ->
0.16836, womens 0.15454 -> 0.15310 -- the games the file doesn't cover
unchanged.

Measured before this existed (each season's first ten weeks, a normal on
the margin), the players' margin alone beat an older `glicko_full` on the
same games -- 0.1573 -> 0.1526 mens, 0.1395 -> 0.1349 womens.

**A ranking reads the blend too.** A team's published rating is the
blend's view of it at a neutral site (`GlickoPredictor._with_players`), the
rating it learned from results alone beside it as `results`; a team the
file doesn't reach ranks on that alone.

**A row is stale after `MAX_AGE`.** The file is weekly; a game more than
two weeks after a team's last row -- the offseason, or a season the file
doesn't reach -- has no players' margin, and the model predicts it
unblended rather than from last March's team.
"""

import csv
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import NamedTuple, Self

from call_it_what_you_want import TeamNamer

from cassandra.constants import CASSANDRA_HOME

from .types import Matchup

_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"

#: The leagues ten-minute-version rates.
PLAYER_RATING_LEAGUES = ("mens", "womens")

#: Rating points per point of the players' margin. 0 is off.
DEFAULT_PLAYER_RATING_WEIGHT = 0.0

#: How long a team's row stands for its games.
MAX_AGE = timedelta(days=14)


def player_ratings_path(league: str) -> Path:
    """Where a league's file lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_player_ratings.csv"


def validated_player_rating_weight(weight: float) -> float:
    """Non-negative: a players' margin that says the home side is better is
    not evidence it's worse. 0 is already "no effect"."""
    if weight < 0:
        raise ValueError(f"player_rating_weight must be non-negative, got {weight}")
    return weight


def season_for(day: date) -> int:
    """The basketball season `day` falls in, named for the year it starts,
    as ten-minute-version names them."""
    return day.year if day.month >= 8 else day.year - 1


class TeamAsOf(NamedTuple):
    """A team as of a date: net per 100 possessions, possessions per game."""

    as_of: date
    net: float
    pace: float
    # ten-minute-version's, the year the season starts; None where a caller
    # building rows by hand has no use for it
    season: int | None = None


@cache
def load_player_ratings(league: str) -> Mapping[str, Sequence[TeamAsOf]]:
    """A league's teams by name, each's rows in date order; empty if it has
    no file.

    Names go through the league's `TeamNamer`, as the games' do, so a team
    ESPN renamed is still one team. Cached for the reason `load_qb_out` is.
    """
    path = player_ratings_path(league)
    if not path.exists():
        return {}
    namer = TeamNamer.for_league(league)
    found: dict[str, list[TeamAsOf]] = {}
    for row in csv.DictReader(path.read_text().splitlines()):
        if not row["team"]:
            continue
        found.setdefault(namer.canonical(row["team"]), []).append(
            TeamAsOf(
                date.fromisoformat(row["as_of"]),
                float(row["net"]),
                float(row["pace"]),
                int(row["season"]),
            )
        )
    return {team: sorted(rows) for team, rows in found.items()}


class PlayerRatingIndex:
    """Teams as their players rate them, by date.

    Constructible from a mapping for tests and for a caller who knows
    better, and loadable per league for a replay, as `QbOutIndex` is.
    """

    def __init__(self, teams: Mapping[str, Sequence[TeamAsOf]] | None = None):
        self._teams = {team: list(rows) for team, rows in (teams or {}).items()}
        self._dates = {
            team: [row.as_of for row in rows] for team, rows in self._teams.items()
        }

    @classmethod
    def for_league(cls, league: str) -> Self:
        return cls(load_player_ratings(league))

    def __len__(self) -> int:
        return len(self._teams)

    def team(self, team: str, day: date) -> TeamAsOf | None:
        """The team's latest row from before `day`, if one is recent enough.

        Strictly before: a row's `as_of` is the first day its fit left out,
        and a game's date here is UTC, a day after the local date of a late
        tip -- so a row dated the game's own day might have seen it.
        """
        dates = self._dates.get(team)
        if not dates:
            return None
        at = bisect_right(dates, day - timedelta(days=1))
        if not at:
            return None
        row = self._teams[team][at - 1]
        return row if day - row.as_of <= MAX_AGE else None

    def newest(self, team: str, season: int) -> TeamAsOf | None:
        """The team's latest row of `season`, however old: what a season
        nobody has played yet says of it, read before its first game."""
        for row in reversed(self._teams.get(team, ())):
            if row.season == season:
                return row
        return None

    def margin(self, matchup: Matchup) -> float | None:
        """The players' neutral-site margin for the home side; None if either
        team has no recent row."""
        day = matchup.date.date()
        home, away = self.team(matchup.home, day), self.team(matchup.away, day)
        if home is None or away is None:
            return None
        return (home.pace + away.pace) / 2 * (home.net - away.net) / 100

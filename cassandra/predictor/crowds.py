"""How big a crowd each basketball team plays its home games in front of.

A home side's edge grows with its crowd: about half a point for each
doubling, in both leagues. ten-minute-version files each team's reading
(`ten_minute_version.crowds`) -- a running average of the log attendance at
its home games, over the league's typical crowd, so 0 is an ordinary
building -- as it stood each day, in `{league}_crowds.csv` under the
predictor data prefix, mens and womens:

    league, season, team_id, team, as_of, crowd, total, weight, games

`crowd` is the reading and is empty until a team has enough home games with
an attendance; the last three columns are the running average's own, for
ten-minute-version to carry on from. 2020-21 is read from that season's
crowds alone and kept out of what a team carries past it.

`GlickoPredictor` adds `home_crowd_slope` rating points per unit of it to
the home side's edge (`home_edge`), which the prediction and the update
share: a visitor is given less of a chance in a big building, and credited
with more for winning in one.

`glicko_players` at its 2026-11-12 fits, the slope at 15 and nothing else
moved but the men's `home_advantage_slope` to 0 -- the slope on a team's
tier, which was standing in for this -- Brier over every game 2010-2026
against the fit as published, and in how many of 2014-2025's twelve seasons
it's better:

    mens     -0.00012 (11)
    womens   -0.00010 (11)

With the men's tier slope at 10 instead it's -0.00013, in nine. What didn't
add to it, in ten-minute-version's notes: the crowd against the building's
biggest, the visitor's own, and any shape but a line in the log.
"""

import csv
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from functools import cache
from pathlib import Path
from typing import NamedTuple, Self

from call_it_what_you_want import TeamNamer
from pydantic import BaseModel

from cassandra.constants import CASSANDRA_HOME

from .types import Matchup

_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"


def crowds_path(league: str) -> Path:
    """Where a league's file lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_crowds.csv"


def validated_home_crowd_slope(slope: float) -> float:
    """At least 0: a bigger crowd is no help to the visitor."""
    if slope < 0:
        raise ValueError(f"home_crowd_slope must be at least 0, got {slope}")
    return slope


class CrowdAsOf(NamedTuple):
    """A team's reading from `as_of` on; None while it has too few games."""

    as_of: date
    crowd: float | None


@cache
def load_crowds(league: str) -> Mapping[str, Sequence[CrowdAsOf]]:
    """A league's teams by canonical name, each's rows in date order; empty
    if it has no file. Cached for the reason `load_qb_out` is."""
    path = crowds_path(league)
    if not path.exists():
        return {}
    namer = TeamNamer.for_league(league)
    found: defaultdict[str, list[CrowdAsOf]] = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if not row["team"]:
                continue
            found[namer.canonical(row["team"])].append(
                CrowdAsOf(
                    date.fromisoformat(row["as_of"]),
                    float(row["crowd"]) if row["crowd"] else None,
                )
            )
    # By day alone, and stably: two rows of one day keep the file's order.
    return {
        team: sorted(rows, key=lambda row: row.as_of) for team, rows in found.items()
    }


class CrowdsFile(BaseModel):
    """What a consumer predicting the games ahead needs of the file: each
    team's reading today, by canonical name. Published beside the releases
    (`publish.crowds_artifact`) for the reason `PlayerRatingsFile` is."""

    league: str
    as_of: date
    teams: dict[str, float]


def latest_crowds(league: str) -> CrowdsFile | None:
    """Each team's newest reading, or None if the league has no file."""
    teams = load_crowds(league)
    if not teams:
        return None
    return CrowdsFile(
        league=league,
        as_of=max(rows[-1].as_of for rows in teams.values()),
        teams={
            team: rows[-1].crowd
            for team, rows in teams.items()
            if rows[-1].crowd is not None
        },
    )


class CrowdIndex:
    """Each team's home crowd reading, by date.

    Constructible from a mapping for tests and loadable per league for a
    replay, as `PlayerRatingIndex` is.
    """

    def __init__(self, teams: Mapping[str, Sequence[CrowdAsOf]] | None = None):
        self._teams = {team: list(rows) for team, rows in (teams or {}).items()}
        self._dates = {
            team: [row.as_of for row in rows] for team, rows in self._teams.items()
        }

    @classmethod
    def for_league(cls, league: str) -> Self:
        return cls(load_crowds(league))

    @classmethod
    def from_file(cls, stored: CrowdsFile) -> Self:
        """The index a published file says: one reading a team, for any day
        -- a consumer of it is predicting the games ahead."""
        return cls(
            {team: [CrowdAsOf(date.min, crowd)] for team, crowd in stored.teams.items()}
        )

    def __len__(self) -> int:
        return len(self._teams)

    def home_crowd(self, matchup: Matchup) -> float | None:
        """The home side's reading on the day of `matchup`; None if it has
        none yet, or the game is at a neutral site.

        A row is dated the day after the game that made it, so the latest
        one on or before the day has seen nothing of this game.
        """
        if matchup.neutral_site:
            return None
        dates = self._dates.get(matchup.home)
        if not dates:
            return None
        at = bisect_right(dates, matchup.date.date())
        return self._teams[matchup.home][at - 1].crowd if at else None

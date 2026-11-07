"""How much of each basketball team's last season came back.

A team that lost most of its minutes is a worse team than its rating and its
players' projections say, and a less known one. The turnover of a season is
the share of the team's minutes the season before played by players who aren't
on its table now (`cassandra.turnover_build` makes the file from
ten-minute-version's players' tables; it covers what they cover, mens from 2014
and womens from 2020):

    league, season, team_id, team, minutes, returning

`GlickoPredictor` reads it at the rollover, against the season's average:
returning minutes fell from about 55% to under 30% with the transfer portal,
and a team is only worse off for losing more than its peers do.
"""

import csv
from collections import defaultdict
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Self

from call_it_what_you_want import TeamNamer

from cassandra.constants import CASSANDRA_HOME

_PREDICTOR_DATA_DIR = CASSANDRA_HOME / "predictor" / "data"


def turnover_path(league: str) -> Path:
    """Where a league's file lands, whether or not it exists yet."""
    return _PREDICTOR_DATA_DIR / f"{league}_turnover.csv"


@cache
def load_turnover(league: str) -> Mapping[int, Mapping[str, float]]:
    """Each season's teams' turnover, by canonical name; empty if the league
    has no file. A row with no minutes has no share to lose and is left out.

    Cached for the reason `load_qb_out` is.
    """
    path = turnover_path(league)
    if not path.exists():
        return {}
    namer = TeamNamer.for_league(league)
    seasons: defaultdict[int, dict[str, float]] = defaultdict(dict)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            minutes = float(row["minutes"])
            if minutes > 0:
                seasons[int(row["season"])][namer.canonical(row["team"])] = (
                    1 - float(row["returning"]) / minutes
                )
    return dict(seasons)


class TurnoverIndex:
    """Each team's turnover, measured against its season's average."""

    def __init__(self, seasons: Mapping[int, Mapping[str, float]] | None = None):
        self._above_average = {
            season: {
                team: share - sum(teams.values()) / len(teams)
                for team, share in teams.items()
            }
            for season, teams in (seasons or {}).items()
        }

    @classmethod
    def for_league(cls, league: str) -> Self:
        return cls(load_turnover(league))

    def above_average(self, team: str, season: int) -> float | None:
        """How much more of its minutes the team lost than the season's average
        team, as a share; None if the file doesn't have it."""
        return self._above_average.get(season, {}).get(team)

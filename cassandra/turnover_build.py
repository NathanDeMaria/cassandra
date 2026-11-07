"""Each basketball team's returning minutes, from the players' tables.

`{league}_turnover.csv` (`cassandra.predictor.turnover`) is rebuilt whole from
`{league}_players_{season}.csv` (`cassandra.players`), because those aren't in
`predictor/data/` and every optimize child needs this: a morning's table moves
the season in progress, finished seasons come out the same, and the tables are
a few MB a league. It's the same call that makes the file the first time:

    python -m cassandra.turnover_build BUCKET mens womens

A player came back if the team's table this season has him: he's played, or he
hasn't but is on the roster (`projected`). A finished season's table only has
the players who played, so a returner who sat out all year counts as gone in
the backtest where a live morning sees him on the roster. Those are 1-2% of
minutes, and calling them back leaves what turnover predicts nearly where it is.
"""

import asyncio
import csv
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

from cassandra.batch import artifacts
from cassandra.players import PLAYERS_PREFIX, PlayersFile, read_players_file
from cassandra.predictor.turnover import load_turnover, turnover_path

logger = logging.getLogger(__name__)

_COLUMNS = ("league", "season", "team_id", "team", "minutes", "returning")


class TeamReturning(NamedTuple):
    """A team's minutes last season, and those played by players still on it."""

    league: str
    season: int
    team_id: str
    team: str
    minutes: float
    returning: float


def returning(before: PlayersFile, now: PlayersFile) -> list[TeamReturning]:
    """`now`'s teams' returning minutes out of `before`.

    A team with nobody on `now`'s table is left out rather than called 100%
    turnover: the file not having it says nothing about its roster.
    """
    still_on = {
        (row.team_id, row.athlete_id)
        for row in now.players
        if row.projected or (row.minutes or 0) > 0
    }
    teams_now = {team_id for team_id, _ in still_on}
    minutes: defaultdict[str, float] = defaultdict(float)
    back: defaultdict[str, float] = defaultdict(float)
    names: dict[str, str] = {}
    for row in before.players:
        if row.projected or not row.minutes or row.team_id not in teams_now:
            continue
        minutes[row.team_id] += row.minutes
        if row.team:
            names[row.team_id] = row.team
        if (row.team_id, row.athlete_id) in still_on:
            back[row.team_id] += row.minutes
    return [
        TeamReturning(
            now.league, now.season, team_id, names[team_id], total, back[team_id]
        )
        for team_id, total in sorted(minutes.items())
        if team_id in names
    ]


def every_season(tables: Sequence[PlayersFile]) -> list[TeamReturning]:
    """Every season with a table the season before it."""
    by_season = {table.season: table for table in tables}
    return [
        row
        for season, table in sorted(by_season.items())
        if season - 1 in by_season
        for row in returning(by_season[season - 1], table)
    ]


def write(path: Path, rows: Iterable[TeamReturning]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(_COLUMNS)
        writer.writerows(rows)


async def rebuild(bucket: str, league: str) -> list[str]:
    """Make the league's file from every season's table, in s3."""
    prefix = f"{PLAYERS_PREFIX}{league}_players_"
    tables = [
        read_players_file(path) for path in await artifacts.download(bucket, prefix)
    ]
    rows = every_season(tables)
    path = turnover_path(league)
    write(path, rows)
    load_turnover.cache_clear()
    keys = await artifacts.upload(bucket, [path])
    logger.info("%s: wrote %s, %d team-seasons", league, keys, len(rows))
    return keys


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for name in sys.argv[2:]:
        asyncio.run(rebuild(sys.argv[1], name))

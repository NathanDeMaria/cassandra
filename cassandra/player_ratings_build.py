"""The players' team ratings, refreshed each morning: the `player-ratings` sweep.

`glicko_players` reads `{league}_player_ratings.csv` (`cassandra.predictor.
player_ratings`) -- every mens and womens team as ten-minute-version's
player ratings say it is, as of each day. The backfill is a one-off; in
season the file needs today's row before the daily publish reads it, so
this runs as a sweep in the same DAG, ahead of publish the way the football
sweeps are. It also runs on its own at 5am, after EndGame's 4:30 basketball
pulls have landed the night's games, so the picks job can bet a team's next
game hours before the publish (jobs/main.tf, `player_ratings_early`). The
publish's run then refits on whatever landed since.

A morning is ten-minute-version's `morning` (`ten_minute_version.build`):
the season's stint tables from the processed plays and summaries, ESPN's
rosters for the teams that haven't played yet, and one fit of the season's
games before today on the priors its kit froze before the season. What it
needs is copied down first, each where ten-minute-version reads it:

    s3                                                  local, under the work dir
    processed/basketball_plays/league=L/season=S/       plays/league=L/season=S/
    summaries/ncaabb/L/S/                               summaries/L/S/
    so-high-school/ncaabb/L/recruits.csv                recruits/L/recruits.csv
    ten-minute-version/live/L-S.pkl                     live/L-S.pkl

and the file itself, which `morning` adds today's rows to, from and back to
the predictor data prefix -- with `{league}_luck.csv` (`cassandra.predictor.
luck`), which it adds the games before today to, and `{league}_crowds.csv`
(`cassandra.predictor.crowds`), whose season it rebuilds. About 250MB a league; the fit is seconds, the
rosters a few minutes until every team has played.

A morning also rewrites the season's players' table (`cassandra.players`),
for reading only. It and each team's last roster, which `morning` keeps for
players who haven't played yet, live under their own prefix. The returning
minutes a model does read come from those tables (`cassandra.turnover_build`),
last, so a failure there has already kept the morning's ratings.

A season with no kit -- the summer, before one is built for the next -- is
skipped, not failed: the file keeps what it has, and the model blends
nothing into a game the file doesn't reach.
"""

import asyncio
import logging
import shutil
import tempfile
from datetime import date
from pathlib import Path

from aiobotocore.session import get_session

from cassandra.batch import artifacts
from cassandra.players import PLAYERS_PREFIX, players_csv_path, roster_path
from cassandra.predictor.crowds import crowds_path, load_crowds
from cassandra.predictor.luck import load_luck, luck_path
from cassandra.predictor.player_ratings import (
    load_player_ratings,
    player_ratings_path,
    season_for,
)
from cassandra.turnover_build import rebuild as rebuild_turnover

logger = logging.getLogger(__name__)

_MAX_CONCURRENCY = 16


def inputs(league: str, season: int) -> dict[str, str]:
    """Each s3 prefix a morning reads, and where under the work dir it goes."""
    return {
        f"processed/basketball_plays/league={league}/season={season}/": (
            f"plays/league={league}/season={season}/"
        ),
        f"summaries/ncaabb/{league}/{season}/": f"summaries/{league}/{season}/",
        f"so-high-school/ncaabb/{league}/recruits.csv": f"recruits/{league}/recruits.csv",
        f"ten-minute-version/live/{league}-{season}.pkl": f"live/{league}-{season}.pkl",
    }


async def _copy_down(bucket: str, prefix: str, dest: Path) -> int:
    """Every object under `prefix`, to `dest` plus its key past the prefix."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    async with get_session().create_client("s3") as client:
        keys = []
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys += [
                i["Key"] for i in page.get("Contents", []) if not i["Key"].endswith("/")
            ]

        async def get(key: str) -> None:
            async with semaphore:
                response = await client.get_object(Bucket=bucket, Key=key)
                async with response["Body"] as stream:
                    body = await stream.read()
            path = dest / key.removeprefix(prefix) if prefix.endswith("/") else dest
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)

        await asyncio.gather(*(get(key) for key in keys))
    return len(keys)


async def refresh(bucket: str, league: str, today: date | None = None) -> bool:
    """Add today's rows to the league's file, in s3. False if there was
    nothing to do: no kit for the season."""
    # Imported here so a serving install, which has no ten-minute-version,
    # can import this module's neighbours.
    from ten_minute_version.build import morning

    day = today or date.today()
    season = season_for(day)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for prefix, local in inputs(league, season).items():
            found = await _copy_down(bucket, prefix, work / local)
            logger.info("%s: %d objects from %s", league, found, prefix)
        if not (work / f"live/{league}-{season}.pkl").exists():
            logger.warning("%s: no kit for %d, nothing to refresh", league, season)
            return False
        await artifacts.download(bucket, f"predictor/data/{league}_player_ratings.csv")
        await artifacts.download(bucket, f"predictor/data/{league}_luck.csv")
        await artifacts.download(bucket, f"predictor/data/{league}_crowds.csv")
        roster = roster_path(league, season)
        tables = work / "stints" / f"league={league}" / f"season={season}"
        if await artifacts.download(bucket, f"{PLAYERS_PREFIX}{roster.name}"):
            tables.mkdir(parents=True, exist_ok=True)
            shutil.copy(roster, tables / "roster.parquet")
        path = player_ratings_path(league)
        path.parent.mkdir(parents=True, exist_ok=True)
        morning(
            league,
            season,
            day.isoformat(),
            kit=str(work / "live"),
            root=str(work / "plays"),
            recruits=str(work / "recruits"),
            summaries=str(work / "summaries"),
            tables=str(work / "stints"),
            roster_root=str(work / "rosters"),
            out=str(path.parent),
        )
        table = players_csv_path(league, season)
        written = path.parent / table.name
        reading = []
        if written.exists():
            table.parent.mkdir(parents=True, exist_ok=True)
            written.replace(table)
            reading.append(table)
        if (tables / "roster.parquet").exists():
            roster.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(tables / "roster.parquet", roster)
            reading.append(roster)
    load_player_ratings.cache_clear()
    load_luck.cache_clear()
    load_crowds.cache_clear()
    beside = [p for p in (luck_path(league), crowds_path(league)) if p.exists()]
    keys = await artifacts.upload(bucket, [path, *beside])
    logger.info("%s: wrote %s", league, keys)
    if reading:
        logger.info("%s: wrote %s", league, await artifacts.upload(bucket, reading))
    await rebuild_turnover(bucket, league)
    return True

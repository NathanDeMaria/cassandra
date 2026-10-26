"""The players' team ratings, refreshed each morning: the `player-ratings` sweep.

`glicko_players` reads `{league}_player_ratings.csv` (`cassandra.predictor.
player_ratings`) -- every mens and womens team as ten-minute-version's
player ratings say it is, as of each day. The backfill is a one-off; in
season the file needs today's row before the daily publish reads it, so
this runs as a sweep in the same DAG, ahead of publish the way the football
sweeps are, after EndGame's 8am jobs have landed the night's games.

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
the predictor data prefix. About 250MB a league; the fit is seconds, the
rosters a few minutes until every team has played.

A season with no kit -- the summer, before one is built for the next -- is
skipped, not failed: the file keeps what it has, and the model blends
nothing into a game the file doesn't reach.
"""

import asyncio
import logging
import tempfile
from datetime import date
from pathlib import Path

from aiobotocore.session import get_session

from cassandra.batch import artifacts
from cassandra.predictor.player_ratings import (
    load_player_ratings,
    player_ratings_path,
    season_for,
)

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
    load_player_ratings.cache_clear()
    keys = await artifacts.upload(bucket, [path])
    logger.info("%s: wrote %s", league, keys)
    return True

"""Sweep a league with the production sweep, dropbacks and runs split, into a private index.

`../units/sweep_index.py` with its own output, so the harness's index is
left as it was, and a check that every reading the two share agrees.
Never writes the shared `{league}_epa.json`.

    python sweep_split.py LEAGUE
"""

import asyncio
import pickle
import sys
from pathlib import Path

from endgame_aws import Config
from endgame_aws.pbp_parquet import get_processed_plays_store
from lucky_ones.arrow import StorePlaySource

import cassandra
from cassandra.epa_build import SINCE, sweep
from cassandra.save_predictions import read_all_seasons

HOME = Path.home() / ".cassandra/success_rate"


async def main(league: str) -> None:
    print("cassandra from", cassandra.__file__, flush=True)
    bucket = Config.init_from_file().bucket
    seasons = [s async for s in read_all_seasons(league, bucket) if s.year >= SINCE]
    games, stats = await sweep(league, seasons, StorePlaySource(get_processed_plays_store()))
    print(stats, flush=True)
    (HOME / f"mixed/{league}_index_split.pkl").write_bytes(pickle.dumps(games))
    before = pickle.loads((HOME / f"units/{league}_index.pkl").read_bytes())
    shared = [g for g in games if g in before]
    same = sum(games[g][: len(before[g])] == before[g] for g in shared)
    split = [e for e in games.values() if e.home_dropback_plays + e.home_run_plays]
    adds_up = sum(
        e.home_dropback_plays + e.home_run_plays == e.home_success_plays
        and e.away_dropback_plays + e.away_run_plays == e.away_success_plays
        for e in games.values()
    )
    downs = sum(e.home_success_plays + e.away_success_plays for e in games.values())
    dropbacks = sum(e.home_dropback_plays + e.away_dropback_plays for e in games.values())
    print(f"{league}: {len(games)} games, {len(shared)} in the harness index, {same} with every "
          f"earlier reading unchanged; {len(split)} split, {adds_up} adding back up; dropbacks "
          f"{dropbacks / downs:.3f} of {downs} downs", flush=True)


asyncio.run(main(sys.argv[1]))

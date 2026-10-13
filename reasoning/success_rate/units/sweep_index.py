"""Sweep a league's EPA index with the new scorer, in memory, into a scratch pickle.

Never writes the shared `{league}_epa.json`: other sessions read it, and an
older checkout can't parse the new header.

    python sweep_index.py LEAGUE [FIRST_SEASON]
"""

import asyncio
import pickle
import sys
from pathlib import Path

import numpy as np
from endgame_aws import Config
from endgame_aws.pbp_parquet import get_processed_plays_store
from lucky_ones.arrow import StorePlaySource

import cassandra
from cassandra.epa_build import SINCE, sweep
from cassandra.save_predictions import read_all_seasons

OUT = Path.home() / ".cassandra/success_rate/units"


async def main(league: str, first: int) -> None:
    print("cassandra from", cassandra.__file__, flush=True)
    bucket = Config.init_from_file().bucket
    seasons = [s async for s in read_all_seasons(league, bucket) if s.year >= max(first, SINCE)]
    source = StorePlaySource(get_processed_plays_store())
    games, stats = await sweep(league, seasons, source)
    print(stats, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{league}_index.pkl").write_bytes(pickle.dumps(games))
    results = {g.game_id: g.home_score - g.away_score for s in seasons for w in s.weeks for g in w.games}
    rows = [(e.home, e.away, e.home_success, e.away_success, results[gid])
            for gid, e in games.items() if e.home_success is not None and e.away_success is not None]
    eh, ea, sh, sa, margin = map(np.array, zip(*rows))
    s_epa = np.polyfit(eh - ea, margin, 1)[0]
    s_sr = np.polyfit(sh - sa, margin, 1)[0]
    scale = s_sr / s_epa
    level = np.r_[eh, ea].mean() - scale * np.r_[sh, sa].mean()
    print(f"{league}: {len(rows)} games; margin per EPA/play {s_epa:.1f}, per unit success {s_sr:.1f}; "
          f"scale {scale:.3f}, level {level:.3f}; mean success {np.r_[sh, sa].mean():.3f}", flush=True)


asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0))

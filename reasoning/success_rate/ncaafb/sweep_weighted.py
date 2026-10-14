"""A league's index again, with each snap's success weighted by how competitive the game was.

A blowout's second half is backups against backups, and a success rate
counts every one of those snaps like a snap in a one-score game. The sweep's
own competitiveness weight -- `PlayEPA.weight`, the home win probability's
distance from certain at `WEIGHT_POWER` 0.5, the same weight behind
`home_weighted` -- turns the count into a weighted share. Only the success
fields change; EPA and the snap counts are the sweep's own, so the noise a
reading is priced at is the same.

Patches `epa_build._success` in this process only and writes a scratch
pickle, never the shared file.

    python sweep_weighted.py [LEAGUE]     ncaafb by default
"""

import asyncio
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

from endgame_aws import Config
from endgame_aws.pbp_parquet import get_processed_plays_store
from lucky_ones import GamePlays
from lucky_ones.arrow import StorePlaySource
from lucky_ones.epa import PlayEPA

from cassandra import epa_build
from cassandra.save_predictions import read_all_seasons

DATA = Path.home() / ".cassandra/success_rate/units"


def _weighted_success(
    game: GamePlays, plays: Sequence[PlayEPA]
) -> tuple[float | None, float | None, int, int]:
    by_id = {play.play_id: play for play in game.plays}
    won = {True: 0.0, False: 0.0}
    weight = {True: 0.0, False: 0.0}
    downs = {True: 0, False: 0}
    for snap in plays:
        play = by_id.get(snap.play_id)
        if play is None or not epa_build.is_down(play.play_type, play.text):
            continue
        downs[snap.offense_is_home] += 1
        weight[snap.offense_is_home] += snap.weight
        won[snap.offense_is_home] += snap.weight * (snap.epa > 0)
    return (
        won[True] / weight[True] if weight[True] else None,
        won[False] / weight[False] if weight[False] else None,
        downs[True],
        downs[False],
    )


async def main(league: str) -> None:
    epa_build._success = _weighted_success
    bucket = Config.init_from_file().bucket
    seasons = [s async for s in read_all_seasons(league, bucket) if s.year >= epa_build.SINCE]
    games, stats = await epa_build.sweep(league, seasons, StorePlaySource(get_processed_plays_store()))
    print(stats, flush=True)
    (DATA / f"{league}_index_weighted.pkl").write_bytes(pickle.dumps(games))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ncaafb"))

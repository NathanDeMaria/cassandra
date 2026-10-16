"""Per game and side: the starting QB, who took snaps, and each passer's attempts and EPA.

ncaafb 2014 on, through cassandra's play sweep (same coverage rules as the EPA
index), EPA from lucky_ones at the default clip, unweighted.
"""
import asyncio
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from endgame_aws.pbp_parquet import get_processed_plays_store
from lucky_ones import MODELS, GamePlays
from lucky_ones.arrow import StorePlaySource
from lucky_ones.epa import DEFAULT_CLIP
from say_youll_remember_me.qb import name_key, passer, team_games

import cassandra
from cassandra.pbp_sweep import sweep

OUT = Path.home() / ".cassandra/qb_strength/ncaafb_qb_games.json"
SEASONS = Path.home() / ".cassandra/success_rate/units/ncaafb_seasons.pkl"


def score(game: GamePlays):
    epa = MODELS.NCAAFB.epa_per_play(game, clip=DEFAULT_CLIP, weight_power=0.0)
    by_id = {play.play_id: play for play in game.plays}
    passers = {"home": defaultdict(lambda: [0, 0.0]), "away": defaultdict(lambda: [0, 0.0])}
    for p in epa.plays:
        name = passer(by_id[p.play_id].text)
        if not name:
            continue
        tally = passers["home" if p.offense_is_home else "away"][name_key(name)]
        tally[0] += 1
        tally[1] += p.epa
    plays = list(game.plays)
    qbs = team_games([game.game_id] * len(plays), [p.offense_team_id for p in plays],
                     [p.text for p in plays])
    out = {}
    for side, team_id in (("home", game.home_team_id), ("away", game.away_team_id)):
        qb = qbs.get((game.game_id, team_id))
        out[side] = {
            "starter_key": qb.starter_key if qb else None,
            "starter": qb.starter if qb else None,
            "share": qb.share if qb else None,
            "snap_keys": sorted(qb.snap_keys) if qb else [],
            "passers": {k: v for k, v in passers[side].items()},
        }
    return out


async def main() -> None:
    print("cassandra from", cassandra.__file__, flush=True)
    seasons = [s for s in pickle.loads(SEASONS.read_bytes()) if s.year >= int(sys.argv[1])]
    print("seasons", sorted(s.year for s in seasons), flush=True)
    source = StorePlaySource(get_processed_plays_store())
    scored, stats = await sweep("ncaafb", seasons, source, score)
    print(stats, flush=True)
    OUT.write_text(json.dumps(scored))
    print("wrote", len(scored), "games to", OUT, flush=True)


asyncio.run(main())

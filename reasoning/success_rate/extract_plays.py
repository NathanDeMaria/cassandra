"""Every NFL regulation snap with its raw EPA, cached to one parquet.

Reuses cassandra's play sweep so the coverage rules (plays reach the final
score, the winner agrees with the schedule) are the same ones the EPA index
uses. EPA is unclipped and unweighted: success is a sign, so neither knob
matters to it, and the raw number stays on the row for anything else.
"""

import asyncio
import math
import sys
from pathlib import Path

import pandas as pd
from endgame_aws import Config
from endgame_aws.pbp_parquet import get_processed_plays_store
from lucky_ones import MODELS, GamePlays
from lucky_ones.arrow import StorePlaySource

import cassandra
from cassandra.pbp_sweep import completed_games, sweep
from cassandra.save_predictions import read_all_seasons

OUT = Path.home() / ".cassandra" / "success_rate" / "nfl_plays.parquet"
FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 2006


def score(game: GamePlays) -> list[dict] | None:
    by_id = {play.play_id: play for play in game.plays}
    epa = MODELS.NFL.epa_per_play(game, clip=math.inf, weight_power=0.0)
    rows = []
    for p in epa.plays:
        play = by_id[p.play_id]
        rows.append(
            dict(
                play_id=p.play_id,
                play_number=p.play_number,
                offense_is_home=p.offense_is_home,
                period=play.period,
                clock_seconds=play.clock_seconds,
                down=play.down,
                distance=play.distance,
                yardline=play.yardline,
                play_type=play.play_type,
                is_penalty=play.is_penalty,
                text=play.text,
                expected_points=p.expected_points,
                epa=p.epa,
                home_wp=p.win_probability,
            )
        )
    return rows or None


async def main() -> None:
    print("cassandra from", cassandra.__file__)
    bucket = Config.init_from_file().bucket
    seasons = [s async for s in read_all_seasons("nfl", bucket) if s.year >= FIRST]
    print("seasons", sorted(s.year for s in seasons))
    played = completed_games(seasons)
    year_of = {
        g.game_id: s.year for s in seasons for w in s.weeks for g in w.games
    }
    week_of = {
        g.game_id: w.number for s in seasons for w in s.weeks for g in w.games
    }
    source = StorePlaySource(get_processed_plays_store())
    scored, stats = await sweep("nfl", seasons, source, score)
    print(stats)

    frames = []
    for game_id, rows in scored.items():
        game = played[game_id]
        frame = pd.DataFrame(rows)
        frame["game_id"] = game_id
        frame["season"] = year_of[game_id]
        frame["week"] = week_of[game_id]
        frame["date"] = game.date
        frame["neutral_site"] = game.neutral_site
        frame["offense"] = frame.offense_is_home.map({True: game.home, False: game.away})
        frame["defense"] = frame.offense_is_home.map({True: game.away, False: game.home})
        frame["home_score"] = game.home_score
        frame["away_score"] = game.away_score
        frames.append(frame)
    plays = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plays.to_parquet(OUT)
    print(f"{len(plays)} snaps over {plays.game_id.nunique()} games -> {OUT}")


asyncio.run(main())

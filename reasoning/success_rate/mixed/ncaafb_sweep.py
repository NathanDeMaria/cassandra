"""Sweep ncaafb's play-by-play into each side's downs and successes by kind, and a slim play table.

The production sweep's coverage rules and EPA call (`cassandra.epa_build`:
the same clip and competitiveness weight), so a game here is a game in the
index and a success is the same success. Per game and side: dropbacks and
runs, and their successes, flat and weighted. Per play, with no text: what
the situational checks need. Plus how many runs the text calls scrambles,
since the college feed may not say so.

    python ncaafb_sweep.py

Writes ~/.cassandra/success_rate/mixed/ncaafb_counts.parquet and
ncaafb_plays.parquet.
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd
from endgame_aws import Config
from endgame_aws.pbp_parquet import get_processed_plays_store
from lucky_ones import MODELS, GamePlays
from lucky_ones.arrow import StorePlaySource

sys.path.insert(0, str(Path(__file__).parent))
from kinds import _SCRAMBLE, PASS, kind_of  # noqa: E402

import cassandra  # noqa: E402
from cassandra.epa_build import CLIP, SINCE, WEIGHT_POWER, is_down  # noqa: E402
from cassandra.pbp_sweep import sweep  # noqa: E402
from cassandra.save_predictions import read_all_seasons  # noqa: E402

OUT = Path.home() / ".cassandra/success_rate/mixed"


def score(game: GamePlays) -> list[tuple] | None:
    by_id = {play.play_id: play for play in game.plays}
    scored = MODELS.NCAAFB.epa_per_play(game, clip=CLIP, weight_power=WEIGHT_POWER)
    rows = []
    for snap in scored.plays:
        play = by_id.get(snap.play_id)
        if play is None or not is_down(play.play_type, play.text):
            continue
        kind = kind_of(play.play_type, play.text)
        rush_type = "rush" in (play.play_type or "").lower()
        rows.append((
            snap.offense_is_home, kind == PASS, snap.epa > 0, snap.weight, snap.epa,
            play.down, play.distance, play.yardline, play.period, play.clock_seconds,
            snap.win_probability, rush_type and bool(_SCRAMBLE.search((play.text or "").lower())),
            rush_type,
        ))
    return rows or None


async def main() -> None:
    print("cassandra from", cassandra.__file__, flush=True)
    bucket = Config.init_from_file().bucket
    seasons = [s async for s in read_all_seasons("ncaafb", bucket) if s.year >= SINCE]
    year_of = {g.game_id: s.year for s in seasons for w in s.weeks for g in w.games}
    week_of = {g.game_id: w.number for s in seasons for w in s.weeks for g in w.games}
    source = StorePlaySource(get_processed_plays_store())
    scored, stats = await sweep("ncaafb", seasons, source, score)
    print(stats, flush=True)
    columns = ["offense_is_home", "dropback", "win", "weight", "epa", "down", "distance", "yardline",
               "period", "clock_seconds", "home_wp", "scramble_text", "rush_type"]
    frames = []
    for game_id, rows in scored.items():
        frame = pd.DataFrame(rows, columns=columns)
        frame["game_id"] = game_id
        frames.append(frame)
    plays = pd.concat(frames, ignore_index=True)
    plays["season"] = plays.game_id.map(year_of)
    plays["week"] = plays.game_id.map(week_of)
    OUT.mkdir(parents=True, exist_ok=True)
    plays.to_parquet(OUT / "ncaafb_plays.parquet")
    plays["w_win"] = plays.weight * plays.win
    plays["pass_win"] = plays.dropback & plays.win
    plays["w_pass"] = plays.weight * plays.dropback
    plays["w_pass_win"] = plays.weight * plays.pass_win
    counts = plays.groupby(["game_id", "season", "offense_is_home"]).agg(
        downs=("win", "size"), wins=("win", "sum"), pass_downs=("dropback", "sum"),
        pass_wins=("pass_win", "sum"), weight=("weight", "sum"), w_wins=("w_win", "sum"),
        w_pass=("w_pass", "sum"), w_pass_wins=("w_pass_win", "sum"),
    ).reset_index()
    counts.to_parquet(OUT / "ncaafb_counts.parquet")
    rush = plays[plays.rush_type]
    print(f"{len(plays)} downs in {plays.game_id.nunique()} games; dropback share {plays.dropback.mean():.3f}; "
          f"runs the text calls scrambles {rush.scramble_text.mean():.3%}", flush=True)
    print(plays.groupby("season").agg(downs=("win", "size"), dropback=("dropback", "mean"),
                                      scramble=("scramble_text", "mean")).round(4).to_string(), flush=True)


asyncio.run(main())

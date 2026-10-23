"""Every NFL down with its kind (`kinds.py`) and situation, by down, distance and win probability.

The situation never looks at the call: clock (fourth quarter, the offense's
win probability over 0.8), hurry (fourth quarter, under 0.2), two_min (the
first half's last two minutes), short (3rd/4th and 2 or less, 2nd and 1),
long (3rd/4th and 7+, 2nd and 12+), neutral (the rest).

    python nfl_situations.py      each situation's share, dropbacks and success
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kinds import kind_of  # noqa: E402

from cassandra.epa_build import is_down  # noqa: E402

PLAYS = "/home/dev/.cassandra/success_rate/nfl_plays.parquet"
COLS = [
    "game_id", "season", "week", "date", "offense", "defense", "offense_is_home", "neutral_site",
    "home_score", "away_score", "play_type", "text", "epa", "down", "distance", "yardline",
    "period", "clock_seconds", "home_wp",
]


def load(last=2025):
    df = pd.read_parquet(PLAYS, columns=COLS)
    df = df[(df.season <= last) & (df.down >= 1)]
    df = df[[is_down(t, x) for t, x in zip(df.play_type, df.text)]].copy()
    df["kind"] = [kind_of(t, x) for t, x in zip(df.play_type, df.text)]
    df["win"] = (df.epa > 0).astype(float)
    owp = np.where(df.offense_is_home, df.home_wp, 1 - df.home_wp)
    late = df.period == 4
    short = ((df.down >= 3) & (df.distance <= 2)) | ((df.down == 2) & (df.distance <= 1))
    long = ((df.down >= 3) & (df.distance >= 7)) | ((df.down == 2) & (df.distance >= 12))
    two_min = (df.period == 2) & (df.clock_seconds <= 120)
    df["situation"] = np.select(
        [late & (owp > 0.8), late & (owp < 0.2), two_min, short, long],
        ["clock", "hurry", "two_min", "short", "long"],
        "neutral",
    )
    return df


if __name__ == "__main__":
    df = load()
    t = df.groupby("situation").agg(
        share=("win", lambda s: len(s) / len(df)),
        pass_share=("kind", lambda k: (k == "pass").mean()),
    )
    sr = df.groupby(["situation", "kind"]).win.mean().unstack()
    print(t.join(sr.add_prefix("SR_")).round(3).to_string())

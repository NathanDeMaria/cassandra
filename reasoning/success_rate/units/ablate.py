"""Paired significance of the winners, and what the offense's weekly forgetting is worth.

For each comparison: the brier difference per game, with a standard error
from the paired per-game differences and one from season blocks (13 seasons
2014-2026), on 2014+ and on the held-out 2022+.

Ablations swap one knob in a fitted run and replay: the success-rate fits
with the offense weekly increase set to 0, and the EPA refit given the
success-rate fit's weekly increase -- whether forgetting is the gain, or
only works on a reading quiet enough to forget with.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from replay import replay  # noqa: E402

OUT = Path.home() / ".cassandra/success_rate/units/out"


def knobs(name: str) -> tuple[float, int, dict]:
    meta = json.loads((OUT / f"{name}.json").read_text())
    return meta["beta"], meta["since"], meta["knobs"]


def loss(df: pd.DataFrame) -> pd.Series:
    return (df.team1_win_prob - df.team1_win) ** 2


def paired(a: str, b: str) -> None:
    da, db = pd.read_parquet(OUT / f"{a}.parquet"), pd.read_parquet(OUT / f"{b}.parquet")
    d = loss(da) - loss(db)
    for label, mask in (("2014+", da.year >= 2014), ("2022+", da.year >= 2022)):
        x = d[mask]
        by_season = x.groupby(da.year[mask]).mean()
        se_game = x.std() / np.sqrt(len(x))
        se_season = by_season.std() / np.sqrt(len(by_season)) if len(by_season) > 2 else np.nan
        print(f"  {a} - {b} {label}: {1e4 * x.mean():+.2f}e-4  se(game) {1e4 * se_game:.2f}  "
              f"se(season) {1e4 * se_season:.2f}  t {x.mean() / se_game:+.2f}  "
              f"seasons better {int((by_season < 0).sum())}/{len(by_season)}")


if __name__ == "__main__":
    for name in ("fit_sr", "fit_b75", "fit_b50"):
        beta, since, k = knobs(name)
        replay(f"{name}_noweekly", beta, since, **{**k, "offense_weekly_sd_increase": 0.0})
    beta, since, k = knobs("fit_epa")
    for weekly in (0.01, 0.02):
        replay(f"fit_epa_weekly{weekly}", beta, since, **{**k, "offense_weekly_sd_increase": weekly})
    print("paired, negative is the first better:")
    for a, b in (("fit_sr", "fit_epa"), ("fit_b75", "fit_epa"), ("fit_b50", "fit_epa"),
                 ("fit_sr", "base"), ("fit_b75", "fit_sr"),
                 ("fit_sr", "fit_sr_noweekly"), ("fit_b75", "fit_b75_noweekly"),
                 ("fit_epa_weekly0.02", "fit_epa")):
        paired(a, b)

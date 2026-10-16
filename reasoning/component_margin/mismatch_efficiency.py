"""Do lopsided games move a team's rating too far? The innovation test.

    python mismatch_efficiency.py [replay]

A filter that weighs each game right leaves its surprises unpredictable: a
team's surprise next game is uncorrelated with its surprise this game, at
every size of expected margin. If a lopsided game moves a rating too far,
the next game says so -- a favorite that won by less than a linear
expectation was marked down, then beats the next one by more than expected
-- and the regression of the next surprise on this one is negative in the
lopsided bins.

So, per team, consecutive games within a season, surprise = actual margin
less the model's expected margin (10.18 points per logit of its win
probability, the published model's own conversion), regressed:

    e_{t+1} = a + beta_bin * e_t

with game t binned by |expected margin|. Standard errors are clustered by
team-season. `beta` against what the filter actually moved -- the change in
the team's expected margin per point of surprise, `K`, measured the same way
on the following game -- gives the weight the game should have had:
`lambda = 1 + beta / K`, where 1 means it was weighed right and 0.5 means
it should have moved the rating half as far.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.home() / ".cassandra/success_rate/units/ncaafb_out"
PTS = 0.1403 * 167.04790817058102 / np.log(10)
BINS = [0, 7, 14, 21, 28, 35, 45, 200]


def team_games(df: pd.DataFrame) -> pd.DataFrame:
    p = df.team1_win_prob.clip(1e-9, 1 - 1e-9)
    df = df.assign(expected=PTS * np.log(p / (1 - p)), margin=df.home_score - df.away_score)
    sides = []
    for team, opp, sign in (("home_team", "away_team", 1.0), ("away_team", "home_team", -1.0)):
        sides.append(pd.DataFrame({
            "game_id": df.game_id, "year": df.year, "date": df.date, "fbs": df.fbs,
            "team": df[team], "opp": df[opp],
            "expected": sign * df.expected, "surprise": sign * (df.margin - df.expected),
        }))
    tg = pd.concat(sides).sort_values(["team", "year", "date", "game_id"])
    group = tg.groupby(["team", "year"])
    tg["g"] = group.cumcount() + 1
    tg["next_surprise"] = group.surprise.shift(-1)
    tg["next_expected"] = group.expected.shift(-1)
    tg["next_opp"] = group.opp.shift(-1)
    return tg.dropna(subset=["next_surprise"])


def clustered(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for c in np.unique(clusters):
        rows = clusters == c
        s = X[rows].T @ e[rows]
        meat += np.outer(s, s)
    return b, np.sqrt(np.diag(bread @ meat @ bread))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "batch_nc_success_cap_p"
    df = pd.read_parquet(OUT / f"{name}.parquet").drop_duplicates("game_id")
    tg = team_games(df)
    tg["bin"] = pd.cut(tg.expected.abs(), BINS, right=False)
    tg["cluster"] = tg.team + "_" + tg.year.astype(str)
    print(f"{name}: next game's surprise on this game's, by this game's |expected margin|")
    for label, rows in (("every game 2002-21", tg.year <= 2021),
                        ("every game 2022-26", tg.year >= 2022),
                        ("a team's games 1-4, 2002-21", (tg.year <= 2021) & (tg.g <= 4)),
                        ("a team's games 1-4, 2022-26", (tg.year >= 2022) & (tg.g <= 4))):
        print(f" {label}")
        sub = tg[rows]
        for b_, s in sub.groupby("bin", observed=True):
            if len(s) < 200:
                continue
            (a, beta), (_, se) = clustered(s.surprise.to_numpy(), s.next_surprise.to_numpy(),
                                           s.cluster.to_numpy())
            fav = s.expected > 0
            print(f"   |exp| {str(b_):10s} n {len(s):6d}  beta {beta:+.3f} ± {se:.3f} (t {beta / se:+5.1f})"
                  f"   favorites' mean surprise {s.surprise[fav].mean():+5.2f}")


if __name__ == "__main__":
    main()

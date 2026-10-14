"""The readings a units replay can rate on: EPA per play, success rate, or a blend.

Success rate goes onto EPA's scale before it is blended, so the units' knobs
(points_per_epa, the deviations, play_sd) keep their meaning: `sr_eq = a + b
sr`, with `b` the ratio of the two same-game margin slopes -- how many points
of margin a unit of success-rate differential is worth, over how many a unit
of EPA differential is. The intercept is irrelevant to the units (their center
is the running mean of whatever they read) and only keeps the two means equal.
"""

import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from success import load_plays, scrimmage  # noqa: E402

from cassandra.predictor.epa import EpaIndex  # noqa: E402
from cassandra.predictor.types import GameEpa  # noqa: E402


@cache
def success_by_game() -> pd.DataFrame:
    """Per game: each offense's success rate, its pass + rush plays, and the margin."""
    plays = load_plays()
    margin = plays.groupby("game_id").apply(
        lambda g: g.home_score.iloc[0] - g.away_score.iloc[0], include_groups=False)
    p = scrimmage(plays)
    g = (p.groupby(["game_id", "season", "offense_is_home"]).success
         .agg(n="size", k="sum").reset_index())
    wide = g.pivot_table(index=["game_id", "season"], columns="offense_is_home", values=["n", "k"])
    wide.columns = [f"{v}_{'home' if h else 'away'}" for v, h in wide.columns]
    wide = wide.dropna().reset_index()
    wide["sr_home"] = wide.k_home / wide.n_home
    wide["sr_away"] = wide.k_away / wide.n_away
    wide["margin"] = wide.game_id.map(margin)
    return wide


@cache
def epa_games() -> dict[str, GameEpa]:
    return dict(EpaIndex.for_league("nfl")._epa)


@cache
def mapping() -> tuple[float, float]:
    """`(a, b)` for sr_eq = a + b sr, off the games that have both readings."""
    sr = success_by_game()
    epa = epa_games()
    both = sr[sr.game_id.isin(list(epa))]
    e_home = np.array([epa[g].home for g in both.game_id])
    e_away = np.array([epa[g].away for g in both.game_id])
    s_epa = np.polyfit(e_home - e_away, both.margin, 1)[0]
    s_sr = np.polyfit(both.sr_home - both.sr_away, both.margin, 1)[0]
    b = s_sr / s_epa
    a = np.concatenate([e_home, e_away]).mean() - b * np.concatenate([both.sr_home, both.sr_away]).mean()
    print(f"margin slopes: {s_epa:.1f} pts per EPA/play, {s_sr:.1f} per unit SR -> b={b:.3f}, a={a:.3f}; "
          f"same-game r: epa {np.corrcoef(e_home - e_away, both.margin)[0, 1]:.3f}, "
          f"sr {np.corrcoef(both.sr_home - both.sr_away, both.margin)[0, 1]:.3f}; "
          f"epa~sr r {np.corrcoef(np.r_[e_home, e_away], np.r_[both.sr_home, both.sr_away])[0, 1]:.3f}")
    return float(a), float(b)


def build(beta: float, since: int = 2014) -> EpaIndex:
    """An index for the units with weight `beta` on success rate, 0 being EPA alone.

    Where both readings exist the two are blended over EPA's play count; at
    `beta` 1 it is success rate over its own pass + rush plays. Seasons from
    `since` to 2013, which EPA's index doesn't cover, read success rate alone.
    """
    epa = epa_games()
    if beta == 0 and since >= 2014:
        return EpaIndex(epa)
    a, b = mapping()
    sr = success_by_game()
    out: dict[str, GameEpa] = {}
    for row in sr[sr.season >= since].itertuples(index=False):
        home_eq, away_eq = a + b * row.sr_home, a + b * row.sr_away
        e = epa.get(row.game_id)
        if e is None or beta == 1:
            if e is None and row.season >= 2014:
                continue  # EPA's sweep dropped it; keep the coverage the same
            out[row.game_id] = GameEpa(home_eq, away_eq, int(row.n_home), int(row.n_away))
        else:
            out[row.game_id] = GameEpa(
                (1 - beta) * e.home + beta * home_eq,
                (1 - beta) * e.away + beta * away_eq,
                e.home_plays,
                e.away_plays,
            )
    if beta == 0:
        out.update(epa)
    return EpaIndex(out)

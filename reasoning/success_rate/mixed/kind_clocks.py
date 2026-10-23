"""Clocks for the displayed dropback and run success rates, measured as the pooled rate's were.

    python kind_clocks.py nfl|ncaafb

`SUCCESS_RATE_CLOCKS` came from `../combine/persistence.py`: each side's
opponent-adjusted rate within a season (`rate = mu + home + O - D` by
downs), its true spread from split halves of a season, and its carry from
one season's fit to the next as a share of the true variance. The same fit
here on three rates -- every down, dropbacks, runs -- each weighted by its
own downs. The every-down row should land on `SUCCESS_RATE_CLOCKS`.

The NFL's downs come from `nfl_plays.parquet`, ncaafb's from
`ncaafb_counts.parquet` (`ncaafb_sweep.py`), teams from the harness's seasons.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "combine"))
sys.path.insert(0, str(Path(__file__).parent))
from kinds import PASS, kind_of  # noqa: E402
from persistence import IDENTIFY, MIN_HALF, MIN_SEASON, SPLITS, fit  # noqa: E402

from cassandra.epa_build import is_down  # noqa: E402

HOME = Path.home() / ".cassandra/success_rate"
KINDS = ("all", "dropback", "run")


def sides_nfl() -> pd.DataFrame:
    plays = pd.read_parquet(
        HOME / "nfl_plays.parquet",
        columns=["game_id", "season", "offense", "defense", "offense_is_home", "neutral_site",
                 "play_type", "text", "epa"],
    )
    plays = plays[[is_down(t, x) for t, x in zip(plays.play_type, plays.text)]].copy()
    plays["dropback"] = [kind_of(t, x) == PASS for t, x in zip(plays.play_type, plays.text)]
    plays["win"] = plays.epa > 0
    plays["dropback_win"] = plays.dropback & plays.win
    g = plays.groupby(["game_id", "season", "offense", "defense", "offense_is_home", "neutral_site"]).agg(
        downs=("win", "size"), wins=("win", "sum"),
        dropbacks=("dropback", "sum"), dropback_wins=("dropback_win", "sum"),
    ).reset_index()
    g["home"] = np.where(g.neutral_site, 0.0, np.where(g.offense_is_home, 1.0, -1.0))
    return g.rename(columns={"season": "year"})


def sides_ncaafb() -> pd.DataFrame:
    g = pd.read_parquet(HOME / "mixed/ncaafb_counts.parquet")
    seasons = pickle.loads((HOME / "units/ncaafb_seasons.pkl").read_bytes())
    teams = {game.game_id: (game.home, game.away, bool(game.neutral_site))
             for s in seasons for w in s.weeks for game in w.games}
    g = g[g.game_id.isin(teams)].copy()
    home = g.game_id.map(lambda gid: teams[gid][0])
    away = g.game_id.map(lambda gid: teams[gid][1])
    neutral = g.game_id.map(lambda gid: teams[gid][2])
    g["offense"] = np.where(g.offense_is_home, home, away)
    g["defense"] = np.where(g.offense_is_home, away, home)
    g["home"] = np.where(neutral, 0.0, np.where(g.offense_is_home, 1.0, -1.0))
    return g.rename(columns={"season": "year", "pass_downs": "dropbacks", "pass_wins": "dropback_wins"})


def with_rates(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["run_downs"] = g.downs - g.dropbacks
    g["run_wins"] = g.wins - g.dropback_wins
    for kind, (wins, downs) in {"all": ("wins", "downs"), "dropback": ("dropback_wins", "dropbacks"),
                                "run": ("run_wins", "run_downs")}.items():
        g[f"n_{kind}"] = g[downs].astype(float)
        g[f"r_{kind}"] = np.where(g[downs] > 0, g[wins] / g[downs].where(g[downs] > 0, 1), np.nan)
    return g


def measure(g: pd.DataFrame, rng: np.random.Generator) -> None:
    halves, seasons = [], []
    fits = {}
    for year, season in g.groupby("year"):
        ids = season.game_id.unique()
        for kind in KINDS:
            part = season[season[f"n_{kind}"] > 0]
            fits[year, kind] = fit(part, f"r_{kind}", f"n_{kind}", IDENTIFY)
        for _ in range(SPLITS):
            half = set(rng.choice(ids, len(ids) // 2, replace=False))
            for kind in KINDS:
                part = season[season[f"n_{kind}"] > 0]
                a, b = part[part.game_id.isin(half)], part[~part.game_id.isin(half)]
                oa, da, na = fit(a, f"r_{kind}", f"n_{kind}", IDENTIFY)
                ob, db, nb = fit(b, f"r_{kind}", f"n_{kind}", IDENTIFY)
                for t in oa:
                    if na.get(t, 0) >= MIN_HALF and nb.get(t, 0) >= MIN_HALF and t in ob:
                        halves.append((year, kind, "offense", oa[t], ob[t]))
                        halves.append((year, kind, "defense", da[t], db[t]))
    years = sorted(g.year.unique())
    for year, nxt in zip(years, years[1:]):
        for kind in KINDS:
            o0, d0, n0 = fits[year, kind]
            o1, d1, n1 = fits[nxt, kind]
            for t in o0:
                if n0.get(t, 0) >= MIN_SEASON and n1.get(t, 0) >= MIN_SEASON and t in o1:
                    seasons.append((year, kind, "offense", o0[t], o1[t]))
                    seasons.append((year, kind, "defense", d0[t], d1[t]))
    halves = pd.DataFrame(halves, columns=["year", "kind", "side", "a", "b"])
    seasons = pd.DataFrame(seasons, columns=["year", "kind", "side", "a", "b"])
    # The clock's carry is the season-to-season correlation over the season's
    # reliability, as `SUCCESS_RATE_CLOCKS` has it; the covariance over the
    # true variance runs past 1 on ncaafb, whose tiers keep their spread.
    print(f"{'kind':9s} {'side':8s} {'season rel':>10s} {'true sd':>8s} {'r(t,t+1)':>9s} {'carry':>6s}")
    for (kind, side), h in halves.groupby(["kind", "side"], sort=False):
        a = h.a - h.groupby("year").a.transform("mean")
        b = h.b - h.groupby("year").b.transform("mean")
        r = float(np.corrcoef(a, b)[0, 1])
        rel = 2 * r / (1 + r)
        true_var = float(np.mean(a * b))
        s = seasons[(seasons.kind == kind) & (seasons.side == side)]
        sa = s.a - s.groupby("year").a.transform("mean")
        sb = s.b - s.groupby("year").b.transform("mean")
        r_next = float(np.corrcoef(sa, sb)[0, 1])
        print(f"{kind:9s} {side:8s} {rel:10.3f} {np.sqrt(true_var):8.4f} {r_next:9.3f} {r_next / rel:6.3f}",
              flush=True)


if __name__ == "__main__":
    league = sys.argv[1]
    sides = with_rates(sides_nfl() if league == "nfl" else sides_ncaafb())
    if league == "nfl":
        sides = sides[sides.year >= 2014]  # the span the index, and the pooled clocks, cover
    print(f"{league}: {len(sides)} offense-games, {sides.year.min()}-{sides.year.max()}")
    measure(sides, np.random.default_rng(0))

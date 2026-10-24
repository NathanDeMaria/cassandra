"""Clocks for the displayed success rates and EPA per down, by kind, measured as the pooled rate's were.

    python kind_clocks.py nfl|ncaafb

`SUCCESS_RATE_CLOCKS` came from `../combine/persistence.py`: each side's
opponent-adjusted reading within a season (`reading = mu + home + O - D` by
downs), its true spread from split halves of a season, and its carry -- the
correlation with next season over the season's reliability. The same fit
here on six readings -- success rate and EPA per down (each play bounded at
`CLIP`), over every down, dropbacks and runs -- each weighted by its own
downs; the every-down success rows should land on `SUCCESS_RATE_CLOCKS`.
And each reading's noise per down: the split halves' noise variance times
the downs behind a half, which is what a displayed EPA side prices a game's
reading at.

The NFL's downs come from `nfl_plays.parquet` (2014 on, the span the index
covers), ncaafb's from `ncaafb_plays.parquet` (`ncaafb_sweep.py`), teams
from the harness's seasons.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "combine"))
from persistence import IDENTIFY, MIN_HALF, MIN_SEASON, SPLITS, fit  # noqa: E402

from cassandra.epa_build import CLIP, is_down, is_pass  # noqa: E402

HOME = Path.home() / ".cassandra/success_rate"
KINDS = ("all", "dropback", "run")
PARTS = [(value, kind) for value in ("sr", "epa") for kind in KINDS]


def _sides(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per game and offense, each reading over each kind's downs, with teams and home."""
    plays = plays.assign(bounded=plays.epa.clip(-CLIP, CLIP), win=plays.win.astype(float))
    parts = []
    for kind, mask in (("all", slice(None)), ("dropback", plays.dropback), ("run", ~plays.dropback)):
        g = plays[mask].groupby(["game_id", "offense_is_home"]).agg(
            **{f"n_{kind}": ("win", "size"), f"sr_{kind}": ("win", "mean"), f"epa_{kind}": ("bounded", "mean")})
        parts.append(g)
    sides = pd.concat(parts, axis=1).reset_index()
    for kind in KINDS:
        sides[f"n_{kind}"] = sides[f"n_{kind}"].fillna(0.0)
    sides = sides.merge(games, left_on="game_id", right_index=True)
    sides["offense"] = np.where(sides.offense_is_home, sides.home_team, sides.away_team)
    sides["defense"] = np.where(sides.offense_is_home, sides.away_team, sides.home_team)
    sides["home"] = np.where(sides.neutral, 0.0, np.where(sides.offense_is_home, 1.0, -1.0))
    return sides


def sides_nfl() -> pd.DataFrame:
    plays = pd.read_parquet(HOME / "nfl_plays.parquet", columns=[
        "game_id", "season", "offense", "defense", "offense_is_home", "neutral_site", "play_type", "text", "epa"])
    plays = plays[plays.season >= 2014]
    plays = plays[[is_down(t, x) for t, x in zip(plays.play_type, plays.text)]].copy()
    plays["dropback"] = [is_pass(t, x) for t, x in zip(plays.play_type, plays.text)]
    plays["win"] = plays.epa > 0
    first = plays.drop_duplicates("game_id").set_index("game_id")
    games = pd.DataFrame({
        "year": first.season, "neutral": first.neutral_site,
        "home_team": np.where(first.offense_is_home, first.offense, first.defense),
        "away_team": np.where(first.offense_is_home, first.defense, first.offense),
    })
    return _sides(plays, games)


def sides_ncaafb() -> pd.DataFrame:
    plays = pd.read_parquet(HOME / "mixed/ncaafb_plays.parquet",
                            columns=["game_id", "offense_is_home", "dropback", "win", "epa"])
    seasons = pickle.loads((HOME / "units/ncaafb_seasons.pkl").read_bytes())
    games = pd.DataFrame([(g.game_id, s.year, bool(g.neutral_site), g.home, g.away)
                          for s in seasons for w in s.weeks for g in w.games],
                         columns=["game_id", "year", "neutral", "home_team", "away_team"])
    return _sides(plays, games.drop_duplicates("game_id").set_index("game_id"))


def measure(g: pd.DataFrame, rng: np.random.Generator) -> None:
    halves, seasons = [], []
    fits = {}
    for year, season in g.groupby("year"):
        ids = season.game_id.unique()
        for value, kind in PARTS:
            part = season[season[f"n_{kind}"] > 0]
            fits[year, value, kind] = fit(part, f"{value}_{kind}", f"n_{kind}", IDENTIFY)
        for _ in range(SPLITS):
            half = set(rng.choice(ids, len(ids) // 2, replace=False))
            for value, kind in PARTS:
                part = season[season[f"n_{kind}"] > 0]
                a, b = part[part.game_id.isin(half)], part[~part.game_id.isin(half)]
                oa, da, na = fit(a, f"{value}_{kind}", f"n_{kind}", IDENTIFY)
                ob, db, nb = fit(b, f"{value}_{kind}", f"n_{kind}", IDENTIFY)
                for t in oa:
                    if na.get(t, 0) >= MIN_HALF and nb.get(t, 0) >= MIN_HALF and t in ob:
                        halves.append((year, value, kind, "offense", oa[t], ob[t]))
                        halves.append((year, value, kind, "defense", da[t], db[t]))
    years = sorted(g.year.unique())
    for year, nxt in zip(years, years[1:]):
        for value, kind in PARTS:
            o0, d0, n0 = fits[year, value, kind]
            o1, d1, n1 = fits[nxt, value, kind]
            for t in o0:
                if n0.get(t, 0) >= MIN_SEASON and n1.get(t, 0) >= MIN_SEASON and t in o1:
                    seasons.append((year, value, kind, "offense", o0[t], o1[t]))
                    seasons.append((year, value, kind, "defense", d0[t], d1[t]))
    halves = pd.DataFrame(halves, columns=["year", "value", "kind", "side", "a", "b"])
    seasons = pd.DataFrame(seasons, columns=["year", "value", "kind", "side", "a", "b"])
    # Downs behind half a season's estimate of a side, for the noise per down.
    downs_per_half = {kind: g.groupby(["year", "offense"])[f"n_{kind}"].sum().median() / 2 for kind in KINDS}
    # The clock's carry is the season-to-season correlation over the season's
    # reliability, as `SUCCESS_RATE_CLOCKS` has it; the covariance over the
    # true variance runs past 1 on ncaafb, whose tiers keep their spread.
    print(f"{'reading':13s} {'side':8s} {'season rel':>10s} {'true sd':>8s} {'r(t,t+1)':>9s}"
          f" {'carry':>6s} {'noise/down':>10s}")
    for (value, kind, side), h in halves.groupby(["value", "kind", "side"], sort=False):
        a = h.a - h.groupby("year").a.transform("mean")
        b = h.b - h.groupby("year").b.transform("mean")
        r = float(np.corrcoef(a, b)[0, 1])
        rel = 2 * r / (1 + r)
        true_var = float(np.mean(a * b))
        noise_var = float((np.mean(a * a) + np.mean(b * b)) / 2) - true_var
        s = seasons[(seasons.value == value) & (seasons.kind == kind) & (seasons.side == side)]
        sa = s.a - s.groupby("year").a.transform("mean")
        sb = s.b - s.groupby("year").b.transform("mean")
        r_next = float(np.corrcoef(sa, sb)[0, 1])
        print(f"{value + ' ' + kind:13s} {side:8s} {rel:10.3f} {np.sqrt(true_var):8.4f} {r_next:9.3f}"
              f" {r_next / rel:6.3f} {np.sqrt(noise_var * downs_per_half[kind]):10.3f}", flush=True)


if __name__ == "__main__":
    league = sys.argv[1]
    sides = sides_nfl() if league == "nfl" else sides_ncaafb()
    print(f"{league}: {len(sides)} offense-games, {sides.year.min()}-{sides.year.max()}")
    measure(sides, np.random.default_rng(0))

"""How each split of an offense or defense persists, and what moves it.

    python split_persistence.py nfl|ncaafb

Twelve readings of a side: success rate and EPA per down (clipped at
`CLIP`), each over every down, over dropbacks and over runs, for offenses
and defenses. Each is opponent-adjusted within a window by weighted least
squares, `reading = mu + home + O[offense] - D[defense]` by the reading's
downs, as `../combine/persistence.py` does, and asked:

1. Season to season: the season's reliability (split halves, Spearman-
   Brown), the true spread, and the carry -- the correlation with next
   season over the reliability, as `SUCCESS_RATE_CLOCKS` reads it.
2. Within a season: the first half of the season's weeks against the
   second, the correlation over the root of the two halves' reliabilities.
   1 is a side that stays what it was; under 1 is a side that moved.
3. Season to season again, split by what changed over the summer (a new
   week-one quarterback, a head coach leaving), from `cassandra.offseason`.
4. In season: the offense's reading in games its starting quarterback
   missed (`qb_out`), as a term in each season's fit, in the reading's
   units and in its true spreads.

The NFL from 2006 (`nfl_plays.parquet`), ncaafb from 2014
(`ncaafb_plays.parquet`), through 2025. Reads the pinned predictor data.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pin import pin  # noqa: E402

pin()

from cassandra.epa_build import CLIP, is_down, is_pass  # noqa: E402
from cassandra.offseason import OffseasonFacts  # noqa: E402
from cassandra.predictor.qb_out import load_qb_out  # noqa: E402

HOME = Path.home() / ".cassandra/success_rate"
LAST = 2025
KINDS = ("all", "dropback", "run")
PARTS = [(value, kind) for value in ("sr", "epa") for kind in KINDS]
#: Downs of prior weight toward 0: enough to identify a side seen once.
IDENTIFY = 20.0
MIN_HALF, MIN_QUARTER, MIN_SEASON = 4, 2, 8
SPLITS = 8


def _per_side(plays: pd.DataFrame) -> pd.DataFrame:
    """Per game and offense: downs, successes and clipped EPA, by kind."""
    plays = plays.assign(epa=plays.epa.clip(-CLIP, CLIP), win=plays.win.astype(float))
    out = []
    for kind, mask in (("all", slice(None)), ("dropback", plays.dropback), ("run", ~plays.dropback)):
        g = plays[mask].groupby(["game_id", "offense_is_home"]).agg(
            **{f"n_{kind}": ("win", "size"), f"w_{kind}": ("win", "sum"), f"e_{kind}": ("epa", "sum")})
        out.append(g)
    sides = pd.concat(out, axis=1).fillna(0.0).reset_index()
    for kind in KINDS:
        n = sides[f"n_{kind}"].where(sides[f"n_{kind}"] > 0)
        sides[f"sr_{kind}"] = sides[f"w_{kind}"] / n
        sides[f"epa_{kind}"] = sides[f"e_{kind}"] / n
    return sides


def _with_games(sides: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    sides = sides.merge(games, left_on="game_id", right_index=True)
    sides["offense"] = np.where(sides.offense_is_home, sides.home_team, sides.away_team)
    sides["defense"] = np.where(sides.offense_is_home, sides.away_team, sides.home_team)
    sides["home"] = np.where(sides.neutral, 0.0, np.where(sides.offense_is_home, 1.0, -1.0))
    start = sides.groupby("year").date.transform("min")
    sides["week"] = (sides.date - start).dt.days // 7
    return sides


def load_nfl() -> pd.DataFrame:
    plays = pd.read_parquet(HOME / "nfl_plays.parquet", columns=[
        "game_id", "season", "date", "offense", "defense", "offense_is_home", "neutral_site",
        "play_type", "text", "epa"])
    plays = plays[(plays.season <= LAST)]
    plays = plays[[is_down(t, x) for t, x in zip(plays.play_type, plays.text)]].copy()
    plays["dropback"] = [is_pass(t, x) for t, x in zip(plays.play_type, plays.text)]
    plays["win"] = plays.epa > 0
    first = plays.drop_duplicates("game_id").set_index("game_id")
    games = pd.DataFrame({
        "year": first.season, "date": first.date, "neutral": first.neutral_site,
        "home_team": np.where(first.offense_is_home, first.offense, first.defense),
        "away_team": np.where(first.offense_is_home, first.defense, first.offense),
    })
    return _with_games(_per_side(plays), games)


def load_ncaafb() -> pd.DataFrame:
    plays = pd.read_parquet(HOME / "mixed/ncaafb_plays.parquet",
                            columns=["game_id", "offense_is_home", "dropback", "win", "epa", "season"])
    plays = plays[plays.season <= LAST]
    seasons = pickle.loads((HOME / "units/ncaafb_seasons.pkl").read_bytes())
    games = pd.DataFrame([
        (g.game_id, s.year, g.date, bool(g.neutral_site), g.home, g.away)
        for s in seasons for w in s.weeks for g in w.games
    ], columns=["game_id", "year", "date", "neutral", "home_team", "away_team"]).drop_duplicates("game_id")
    return _with_games(_per_side(plays), games.set_index("game_id"))


def fit(df: pd.DataFrame, value: str, weight: str, extra: tuple[str, ...] = ()) -> tuple[dict, dict, dict, np.ndarray]:
    """O and D per team, `IDENTIFY` downs toward 0, and the unpenalized `extra` terms."""
    df = df[df[weight] > 0]
    teams = sorted(set(df.offense) | set(df.defense))
    at = {t: i for i, t in enumerate(teams)}
    k, n, e = len(teams), len(df), len(extra)
    design = np.zeros((n, 2 + e + 2 * k))
    design[:, 0] = 1.0
    design[:, 1] = df.home.to_numpy()
    for j, col in enumerate(extra):
        design[:, 2 + j] = df[col].to_numpy()
    rows = np.arange(n)
    design[rows, 2 + e + df.offense.map(at).to_numpy()] = 1.0
    design[rows, 2 + e + k + df.defense.map(at).to_numpy()] = -1.0
    weighted = design.T * df[weight].to_numpy()
    normal = weighted @ design
    normal[2 + e:, 2 + e:] += IDENTIFY * np.eye(2 * k)
    normal[:2 + e, :2 + e] += 1e-9 * np.eye(2 + e)
    solved = np.linalg.solve(normal, weighted @ df[value].to_numpy())
    games = df.offense.value_counts().to_dict()
    return (dict(zip(teams, solved[2 + e:2 + e + k])), dict(zip(teams, solved[2 + e + k:])),
            games, solved[2:2 + e])


def _split_pairs(frame: pd.DataFrame, minimum: int, rng: np.random.Generator) -> list[tuple]:
    """(part, side, a, b) for every team seen `minimum` times in both random halves of `frame`."""
    pairs = []
    ids = frame.game_id.unique()
    for _ in range(SPLITS):
        half = set(rng.choice(ids, len(ids) // 2, replace=False))
        a, b = frame[frame.game_id.isin(half)], frame[~frame.game_id.isin(half)]
        for value, kind in PARTS:
            col, w = f"{value}_{kind}", f"n_{kind}"
            oa, da, na, _ = fit(a, col, w)
            ob, db, nb, _ = fit(b, col, w)
            for t in oa:
                if na.get(t, 0) >= minimum and nb.get(t, 0) >= minimum and t in ob:
                    pairs.append((f"{value} {kind}", "offense", oa[t], ob[t]))
                    pairs.append((f"{value} {kind}", "defense", da[t], db[t]))
    return pairs


def _centered_r(df: pd.DataFrame) -> float:
    a = df.a - df.groupby("year").a.transform("mean")
    b = df.b - df.groupby("year").b.transform("mean")
    return float(np.corrcoef(a, b)[0, 1])


def _true_var(df: pd.DataFrame) -> float:
    a = df.a - df.groupby("year").a.transform("mean")
    b = df.b - df.groupby("year").b.transform("mean")
    return float(np.mean(a * b))


def main(league: str) -> None:
    sides = load_nfl() if league == "nfl" else load_ncaafb()
    facts = OffseasonFacts.for_league(league)
    out = load_qb_out(league)
    sides["qb_out"] = [float(team in out.get(gid, ())) for gid, team in zip(sides.game_id, sides.offense)]
    print(f"{league}: {len(sides)} offense-games {sides.year.min()}-{sides.year.max()}, "
          f"dropbacks {sides.n_dropback.sum() / sides.n_all.sum():.3f} of downs, "
          f"starter out in {int(sides.qb_out.sum())}", flush=True)
    rng = np.random.default_rng(0)

    halves, within, seasons, qb_terms = [], [], {}, []
    for year, season in sides.groupby("year"):
        halves += [(year, *p) for p in _split_pairs(season, MIN_HALF, rng)]
        cut = season.week.median()
        early, late = season[season.week <= cut], season[season.week > cut]
        for label, frame in (("early", early), ("late", late)):
            within += [(year, label, *p) for p in _split_pairs(frame, MIN_QUARTER, rng)]
        for value, kind in PARTS:
            col, w = f"{value}_{kind}", f"n_{kind}"
            o, d, n, _ = fit(season, col, w)
            seasons[year, value, kind] = (o, d, n)
            oe, de, ne, _ = fit(early, col, w)
            ol, dl, nl, _ = fit(late, col, w)
            for t in oe:
                if ne.get(t, 0) >= MIN_HALF and nl.get(t, 0) >= MIN_HALF and t in ol:
                    within.append((year, "across", f"{value} {kind}", "offense", oe[t], ol[t]))
                    within.append((year, "across", f"{value} {kind}", "defense", de[t], dl[t]))
            if season.qb_out.sum() >= 5:
                *_, coef = fit(season, col, w, extra=("qb_out",))
                qb_terms.append((year, f"{value} {kind}", float(coef[0])))
        print(f"  {year} done", flush=True)

    halves = pd.DataFrame(halves, columns=["year", "part", "side", "a", "b"])
    within = pd.DataFrame(within, columns=["year", "window", "part", "side", "a", "b"])
    qb_terms = pd.DataFrame(qb_terms, columns=["year", "part", "coef"])

    pairs = []
    years = sorted(sides.year.unique())
    for year, nxt in zip(years, years[1:]):
        for value, kind in PARTS:
            o0, d0, n0 = seasons[year, value, kind]
            o1, d1, n1 = seasons[nxt, value, kind]
            for t in o0:
                if n0.get(t, 0) >= MIN_SEASON and n1.get(t, 0) >= MIN_SEASON and t in o1:
                    fact = facts.get(t, nxt)
                    new_qb = None if fact is None else fact.new_quarterback
                    coach = None if fact is None else fact.coach_departure is not None
                    for side, a, b in (("offense", o0[t], o1[t]), ("defense", d0[t], d1[t])):
                        pairs.append((year, f"{value} {kind}", side, a, b, new_qb, coach))
    pairs = pd.DataFrame(pairs, columns=["year", "part", "side", "a", "b", "new_qb", "coach"])
    for name, frame in (("halves", halves), ("within", within), ("pairs", pairs), ("qb_terms", qb_terms)):
        frame.to_parquet(HOME / f"mixed/persistence_{league}_{name}.parquet")

    def rel_of(frame: pd.DataFrame) -> float:
        r = _centered_r(frame)
        return 2 * r / (1 + r)

    rows = []
    for (part, side), h in halves.groupby(["part", "side"], sort=False):
        rel = rel_of(h)
        s = pairs[(pairs.part == part) & (pairs.side == side)]
        w = within[(within.part == part) & (within.side == side)]
        rel_e = rel_of(w[w.window == "early"])
        rel_l = rel_of(w[w.window == "late"])
        r_within = _centered_r(w[w.window == "across"])
        row = {"part": part, "side": side, "season rel": rel, "true sd": np.sqrt(_true_var(h)),
               "carry": _centered_r(s) / rel, "within": r_within / np.sqrt(rel_e * rel_l)}
        for flag, label in (("new_qb", "new QB"), ("coach", "new coach")):
            for value, name in ((True, label), (False, f"no {label}")):
                g = s[s[flag] == value]
                # Centered on the whole season's mean, as the carry is.
                a = g.a - s.groupby("year").a.transform("mean").loc[g.index]
                b = g.b - s.groupby("year").b.transform("mean").loc[g.index]
                row[f"carry, {name}"] = float(np.corrcoef(a, b)[0, 1]) / rel if len(g) > 10 else np.nan
                row[f"n, {name}"] = len(g)
        if side == "offense":
            q = qb_terms[qb_terms.part == part].coef
            row["qb out"] = q.mean()
            row["qb out se"] = q.std() / np.sqrt(len(q))
            row["qb out / sd"] = q.mean() / np.sqrt(_true_var(h))
        rows.append(row)
    table = pd.DataFrame(rows).set_index(["part", "side"])
    pd.set_option("display.width", 250)
    print("\nseason to season, and within a season (1 = stays what it was)")
    print(table[["season rel", "true sd", "carry", "within"]].round(3).to_string())
    print("\ncarry by what changed over the summer (pairs in each group)")
    cols = ["carry, new QB", "carry, no new QB", "carry, new coach", "carry, no new coach",
            "n, new QB", "n, no new QB", "n, new coach", "n, no new coach"]
    print(table[cols].round(3).to_string())
    print("\noffense's reading with its starting quarterback out (per down, and in true spreads)")
    print(table.xs("offense", level="side")[["qb out", "qb out se", "qb out / sd"]].round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])

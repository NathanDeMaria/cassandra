"""Replay a league's published success model on an index, keeping its predictions and its displayed rates.

    PYTHONHASHSEED=0 python split_replay.py LEAGUE INDEX NAME
    python split_replay.py compare LEAGUE NAME_A NAME_B

The first writes `out/NAME.parquet` (every game's prediction) and
`out/NAME.json` (each team's displayed success rates at the end); INDEX is
a pickled index (`../units/{league}_index.pkl`, `sweep_split.py`'s). The
second says whether two runs predicted alike and showed the same every-down
rates, and summarizes the split the second shows.
"""

import json
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = Path.home() / ".cassandra/success_rate/mixed/out"


def _nfl(index_path: Path):
    sys.path.insert(0, str(HERE))
    from nfl_replay import BASE, NAMED, NAMER, PRIORS

    from cassandra.odds import OddsDatabase
    from cassandra.predictor.epa import EpaIndex
    from cassandra.predictor.opponent_prior import OpponentPriorManager
    from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor
    from cassandra.save_predictions import join_with_odds

    predictor = UnitMarginGlickoPredictor(
        "nfl", **BASE,
        opponent_prior_manager=OpponentPriorManager("nfl", path=PRIORS),
        game_epa=EpaIndex(pickle.loads(index_path.read_bytes())),
    )
    df = pd.DataFrame([asdict(p) for p in join_with_odds(
        predictor, NAMED, OddsDatabase({}), post_callbacks=False, namer=NAMER)])
    return predictor, df


def _ncaafb(index_path: Path):
    os.environ["NCAAFB_INDEX"] = str(index_path)
    sys.path.insert(0, str(HERE.parent / "ncaafb"))
    import replay

    from cassandra.predictor.opponent_prior import OpponentPriorManager
    from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor

    knobs = json.loads((replay.OUT / "batch_nc_success_cap_p.json").read_text())["knobs"]
    named, _, index = replay._inputs()
    warm, early = replay._warm()
    predictor = UnitMarginGlickoPredictor.from_state_dict({
        **warm.state_dict(), **knobs,
        "opponent_prior_manager": OpponentPriorManager("ncaafb", path=replay.PRIORS),
        "game_epa": index,
    })
    predictor._season = warm._season
    predictor._qb_pending = set(warm._qb_pending)
    late = replay._replay(predictor, [s for s in named if s.year >= replay.FIRST_READING])
    return predictor, pd.concat([early, late], ignore_index=True)


def run(league: str, index_path: Path, name: str) -> None:
    predictor, df = (_nfl if league == "nfl" else _ncaafb)(index_path)
    for kind, rates in (("every down", predictor._success_rates), *predictor._kind_rates.items()):
        if rates.enabled and rates.center is not None:
            print(f"  {kind}: center {rates.center:.3f}, gap slope {rates.gap_slope:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{name}.parquet")
    shown = {
        team: {"success": r.success._asdict() if r.success else None,
               "epa": r.epa._asdict() if r.epa else None}
        for team, r in predictor.ratings.items()
    }
    (OUT / f"{name}.json").write_text(json.dumps(shown, default=lambda t: t._asdict()))
    print(name, f"{len(df)} games, {sum(v['success'] is not None for v in shown.values())} teams with "
          f"rates, {sum(v['epa'] is not None for v in shown.values())} with EPA", flush=True)


def _block(entry: dict | None, which: str) -> dict | None:
    """One team's success or EPA block; files from before the EPA hold the success block alone."""
    if entry is None:
        return None
    if "success" not in entry and "epa" not in entry:
        return entry if which == "success" else None
    return entry.get(which)


def _table(shown: dict, which: str) -> pd.DataFrame:
    # JSON keeps the outer tuple's names and writes the inner ones as lists:
    # a side is [value, sd], a kind is [offense, defense].
    rows = []
    for team, entry in shown.items():
        s = _block(entry, which)
        if not s or not s.get("dropbacks") or not s.get("runs"):
            continue
        (db_off, db_def), (run_off, run_def) = s["dropbacks"], s["runs"]
        rows.append({
            "team": team,
            "off": s["offense"][0], "def": s["defense"][0],
            "off_db": db_off[0], "off_run": run_off[0], "def_db": db_def[0], "def_run": run_def[0],
            "off_db_sd": db_off[1], "off_run_sd": run_off[1],
        })
    return pd.DataFrame(rows).set_index("team")


def compare(league: str, a: str, b: str) -> None:
    da, db = pd.read_parquet(OUT / f"{a}.parquet"), pd.read_parquet(OUT / f"{b}.parquet")
    diff = np.abs(da.team1_win_prob.to_numpy() - db.team1_win_prob.to_numpy())
    print(f"{league}: {len(da)} games; predictions identical: {bool((diff == 0).all())} (max diff {diff.max():.2e})")
    ra, rb = json.loads((OUT / f"{a}.json").read_text()), json.loads((OUT / f"{b}.json").read_text())
    for which in ("success", "epa"):
        pairs = [(_block(ra[t], which), _block(rb[t], which)) for t in ra]
        both = [(x, y) for x, y in pairs if x is not None and y is not None]
        same = all(x["offense"] == y["offense"] and x["defense"] == y["defense"] for x, y in both)
        kinds = all(x.get("dropbacks") == y.get("dropbacks") and x.get("runs") == y.get("runs")
                    for x, y in both)
        print(f"{which}: {len(both)} teams in both; every-down identical: {same}; split identical: {kinds}")
    year = int(db.year.max())
    for which, scale, unit in (("success", 100, "%"), ("epa", 1, " EPA/down")):
        t = _table(rb, which)
        if t.empty:
            continue
        if league == "ncaafb":
            from cassandra.objective import _in_fbs

            t = t[[_in_fbs(team, year) for team in t.index]]
        print(f"\n{which}{' (FBS ' + str(year) + ')' if league == 'ncaafb' else ''}, {len(t)} teams, in{unit}:")
        print("  means:", (t[["off", "def", "off_db", "off_run", "def_db", "def_run"]].mean() * scale).round(3).to_dict())
        print("  sd across teams:", (t[["off_db", "off_run", "def_db", "def_run"]].std() * scale).round(3).to_dict())
        print("  typical sd of a side:", (t[["off_db_sd", "off_run_sd"]].median() * scale).round(3).to_dict())
        print("  corr dropback vs run, offense:", round(t.off_db.corr(t.off_run), 2),
              "defense:", round(t.def_db.corr(t.def_run), 2))
        cols = ["off", "off_db", "off_run", "def", "def_db", "def_run"]
        for col, asc, label in (("off_db", False, "best dropback offenses"), ("off_run", False, "best run offenses"),
                                ("def_db", True, "best dropback defenses"), ("def_run", True, "best run defenses")):
            print(f"  {label}:\n{(t.sort_values(col, ascending=asc).head(5)[cols] * scale).round(3).to_string()}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run(sys.argv[1], Path(sys.argv[2]), sys.argv[3])

"""Score every replay in out/ on the slices that matter, against the EPA refit.

Slices: the search's own window (through 2021), the held-out seasons (2022+),
the EPA era (2014+), each team's first four games of a season (2014+), and
the injury slices -- a game where either quarterback is flagged out by the
qb_out index, and the subset where he was also out the game before (the
games a faster-forgetting unit should get right sooner).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cassandra.predictor.qb_out import load_qb_out

OUT = Path.home() / ".cassandra/success_rate/units/out"
CONTROL = sys.argv[1] if len(sys.argv) > 1 else "fit_epa"


def flags(df: pd.DataFrame) -> pd.DataFrame:
    out = load_qb_out("nfl")
    long = pd.concat([
        df[["game_id", "date", "home_team"]].rename(columns={"home_team": "team"}),
        df[["game_id", "date", "away_team"]].rename(columns={"away_team": "team"}),
    ]).sort_values("date")
    long["out"] = [t in out.get(g, ()) for g, t in zip(long.game_id, long.team)]
    long["prev_out"] = long.groupby("team").out.shift(1, fill_value=False)
    long["year"] = long.date.dt.year - (long.date.dt.month <= 3)
    long["n"] = long.groupby(["team", "year"]).cumcount() + 1
    g = long.groupby("game_id").agg(qb_out=("out", "any"),
                                    still_out=("out", lambda s: False), early=("n", "min"))
    g["still_out"] = long.assign(both=long.out & long.prev_out).groupby("game_id").both.any()
    return g


def slices(df: pd.DataFrame, f: pd.DataFrame) -> dict[str, np.ndarray]:
    d = df.join(f, on="game_id")
    return {
        "≤2021": (d.year <= 2021).to_numpy(),
        "2022+": (d.year >= 2022).to_numpy(),
        "2014+": (d.year >= 2014).to_numpy(),
        "2014+ g1-4": ((d.year >= 2014) & (d.early <= 4)).to_numpy(),
        "2014+ qb out": ((d.year >= 2014) & d.qb_out).to_numpy(),
        "  still out": ((d.year >= 2014) & d.still_out).to_numpy(),
    }


def main() -> None:
    runs = {p.stem: pd.read_parquet(p) for p in sorted(OUT.glob("*.parquet"))}
    control = runs[CONTROL]
    f = flags(control)
    cuts = slices(control, f)
    sq = {name: ((r.team1_win_prob - r.team1_win) ** 2).to_numpy() for name, r in runs.items()}
    rows = []
    for name, loss in sq.items():
        row = {"run": name}
        for cut, mask in cuts.items():
            row[cut] = 1e4 * (loss[mask].mean() - sq[CONTROL][mask].mean())
        years = control.year.to_numpy()
        per = [loss[years == y].mean() - sq[CONTROL][years == y].mean() for y in range(2014, 2027)]
        row["seasons better"] = f"{sum(p < 0 for p in per)}/{len(per)}"
        meta = OUT / f"{name}.json"
        row["knobs"] = json.loads(meta.read_text())["knobs"] if meta.exists() else {}
        rows.append(row)
    table = pd.DataFrame(rows).set_index("run")
    print(f"brier vs {CONTROL}, in 1e-4 (negative is better); games per slice:",
          {k: int(v.sum()) for k, v in cuts.items()})
    print(f"{CONTROL} brier:", {k: round(float(sq[CONTROL][v].mean()), 5) for k, v in cuts.items()})
    print(table.drop(columns="knobs").round(2).to_string())
    for name, knobs in table.knobs.items():
        if knobs:
            print(name, {k.replace("_sd_increase", "_inc"): round(v, 4) for k, v in knobs.items()})


if __name__ == "__main__":
    main()

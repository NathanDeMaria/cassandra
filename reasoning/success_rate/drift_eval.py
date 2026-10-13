"""Does a weekly drift let an injury show up faster? Season-constant vs drifting betas.

Scores both fits game by game on the slices an injury touches, using the
qb_out index (the usual starter didn't start): the offense's quarterback out,
out for the second game running or more -- where a beta that forgets should
already have caught the backup's level -- and the first game back.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from collide import Params, replay  # noqa: E402
from fit import OUT as DATA  # noqa: E402
from fit import load_games

from cassandra.predictor.qb_out import load_qb_out  # noqa: E402


def with_flags(history: pd.DataFrame) -> pd.DataFrame:
    out = load_qb_out("nfl")
    h = history.sort_values(["date", "game_id"]).copy()
    h["qb_out"] = [o in out.get(g, ()) for g, o in zip(h.game_id, h.offense)]
    prev = h.groupby("offense").qb_out.shift(1, fill_value=False)
    h["still_out"] = h.qb_out & prev
    h["back"] = ~h.qb_out & prev
    h["game_no"] = h.groupby(["season", "offense"]).cumcount() + 1
    return h


def main() -> None:
    games = load_games()
    fits = {name: Params(**json.loads((DATA / f).read_text()))
            for name, f in (("constant", "collide_params.json"), ("drift", "collide_params_drift.json"))}
    for name, p in fits.items():
        print(name, {k: round(float(v), 4) for k, v in p.__dict__.items()})
    runs = {name: with_flags(replay(games, p, size=48)) for name, p in fits.items()}
    base, drift = runs["constant"], runs["drift"]
    assert (base.game_id.to_numpy() == drift.game_id.to_numpy()).all()
    fit_years = base.season.between(2007, 2025)
    cuts = {
        "2007-25": fit_years,
        "2026": base.season == 2026,
        "games 1-4": fit_years & (base.game_no <= 4),
        "games 9+": fit_years & (base.game_no >= 9),
        "QB out": fit_years & base.qb_out,
        "  2nd+ game out": fit_years & base.still_out,
        "  first game back": fit_years & base.back,
    }
    rows = []
    for cut, mask in cuts.items():
        b, d = base[mask], drift[mask]
        rate = b.k / b.n
        rows.append({
            "slice": cut, "meetings": int(mask.sum()),
            "Δ loglik/meeting": (d.loglik - b.loglik).mean(),
            "rmse const (pts)": 100 * np.sqrt(np.average((rate - b.p) ** 2, weights=b.n)),
            "rmse drift (pts)": 100 * np.sqrt(np.average((rate - d.p) ** 2, weights=b.n)),
            "bias const (pts)": 100 * np.average(rate - b.p, weights=b.n),
            "bias drift (pts)": 100 * np.average(rate - d.p, weights=b.n),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

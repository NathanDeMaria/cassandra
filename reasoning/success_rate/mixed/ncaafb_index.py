"""ncaafb's swept index with each side's success rate at a fixed mix.

`nfl_index.py` for ncaafb: the counts from `ncaafb_sweep.py`, the index the
ncaafb harness replays (`units/ncaafb_index.pkl`), margins from the
harness's seasons. Writes `ncaafb_index_w{W}.pkl` and its constants, for
`../ncaafb/replay.py` through `NCAAFB_INDEX` and `NCAAFB_SUCCESS_SCALE`.

    python ncaafb_index.py [--weighted] W [W ...]

`--weighted` mixes the competitiveness-weighted kind rates instead, into
`ncaafb_index_w{W}_weighted.pkl` -- in the flat rate's slot, with its own
constants, so the harness reads it the same way.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kinds import Counts, mixed_rate  # noqa: E402
from nfl_index import PRIOR, gaps, scale_of  # noqa: E402

HOME = Path.home() / ".cassandra/success_rate"
OUT = HOME / "mixed"
WEIGHTED = "--weighted" in sys.argv
SUFFIX = "_weighted" if WEIGHTED else ""


def main(weights: list[float]) -> None:
    g = pd.read_parquet(OUT / "ncaafb_counts.parquet")
    if WEIGHTED:
        # Each snap counted by its competitiveness weight, as the weighted EPA does.
        g = g.assign(downs=g.weight, wins=g.w_wins, pass_downs=g.w_pass, pass_wins=g.w_pass_wins)
    g["run_downs"] = g.downs - g.pass_downs
    g["run_wins"] = g.wins - g.pass_wins
    gap = gaps(g)
    print("dropback share by season:", gap.share.round(3).to_dict())
    index = pickle.loads((HOME / "units/ncaafb_index.pkl").read_bytes())
    seasons = pickle.loads((HOME / "units/ncaafb_seasons.pkl").read_bytes())
    margins = {game.game_id: game.home_score - game.away_score
               for s in seasons for w in s.weeks for game in w.games
               if game.home_score is not None and game.away_score is not None}
    sides = {(r.game_id, r.offense_is_home): r for r in g.itertuples(index=False)}

    def side(gid, home):
        r = sides.get((gid, home))
        return None if r is None else Counts(r.pass_downs, r.pass_wins, r.run_downs, r.run_wins)

    have = [gid for gid in index if side(gid, True) and side(gid, False)]
    print(f"index games {len(index)}, with counts on both sides {len(have)}")
    stored = (lambda e: e.home_success_weighted) if WEIGHTED else (lambda e: e.home_success)
    agree = np.mean([abs(side(gid, True).pooled - stored(index[gid])) < 1e-9
                     for gid in have if stored(index[gid]) is not None])
    print(f"pooled rate from the counts equals the index's on {agree:.2%} of games")
    season_of = {gid: sides[(gid, True)].season for gid in have}
    for w in ["pooled", *weights]:
        rates = {}
        for gid in have:
            if w == "pooled":
                rates[gid] = (side(gid, True).pooled, side(gid, False).pooled)
                continue
            gp, gr = gap.loc[season_of[gid], ["pass_gap", "run_gap"]]
            rates[gid] = tuple(mixed_rate(side(gid, h), w, gp, gr, PRIOR) for h in (True, False))
        print(f"{w}:" if w == "pooled" else f"mixed w={w}:")
        scale, level = scale_of(rates, index, margins)
        if w == "pooled" and not WEIGHTED:
            continue  # the harness's own index
        out = {
            gid: (e._replace(home_success=rates[gid][0], away_success=rates[gid][1]) if gid in rates
                  else e._replace(home_success=None, away_success=None))
            for gid, e in index.items()
        }
        (OUT / f"ncaafb_index_w{w}{SUFFIX}.pkl").write_bytes(pickle.dumps(out))
        (OUT / f"ncaafb_index_w{w}{SUFFIX}.json").write_text(json.dumps({"scale": scale, "level": level, "prior": PRIOR}))


if __name__ == "__main__":
    main([float(w) for w in sys.argv[1:] if w != "--weighted"])

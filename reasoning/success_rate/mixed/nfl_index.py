"""The NFL's swept index with each side's success rate at a fixed mix.

Reads every snap from `nfl_plays.parquet` (`../extract_plays.py`) and the
index the units harness replays (`units/nfl_index.pkl`, `units/
sweep_index.py`), and writes a copy of the index whose `home_success` and
`away_success` are the mixed rate (`kinds.py`) -- one per mix weight -- with
the reading's own EPA-scale constants beside it, measured as `SUCCESS_SCALE`
was: the ratio of the two same-game margin slopes.

    python nfl_index.py W [W ...]      e.g. 0.6 0.65 0.7
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kinds import PASS, Counts, kind_of, mixed_rate  # noqa: E402

from cassandra.epa_build import is_down  # noqa: E402

HOME = Path.home() / ".cassandra/success_rate"
OUT = HOME / "mixed"
PRIOR = 10.0


def counts() -> pd.DataFrame:
    """Per game and side: downs and successes by kind, and the final margin."""
    plays = pd.read_parquet(
        HOME / "nfl_plays.parquet",
        columns=["game_id", "season", "offense_is_home", "play_type", "text", "epa",
                 "home_score", "away_score"],
    )
    plays = plays[[is_down(t, x) for t, x in zip(plays.play_type, plays.text)]].copy()
    plays["dropback"] = [kind_of(t, x) == PASS for t, x in zip(plays.play_type, plays.text)]
    plays["win"] = plays.epa > 0
    g = plays.groupby(["game_id", "season", "offense_is_home"]).agg(
        pass_downs=("dropback", "sum"),
        pass_wins=("win", lambda s: (s & plays.loc[s.index, "dropback"]).sum()),
        downs=("win", "size"),
        wins=("win", "sum"),
        margin=("home_score", "first"),
        away_score=("away_score", "first"),
    ).reset_index()
    g["margin"] -= g.pop("away_score")
    g["run_downs"] = g.downs - g.pass_downs
    g["run_wins"] = g.wins - g.pass_wins
    return g


def gaps(g: pd.DataFrame) -> pd.DataFrame:
    """Each season's pass and run rates less its pooled rate."""
    s = g.groupby("season")[["pass_downs", "pass_wins", "run_downs", "run_wins", "downs", "wins"]].sum()
    pooled = s.wins / s.downs
    return pd.DataFrame({
        "pass_gap": s.pass_wins / s.pass_downs - pooled,
        "run_gap": s.run_wins / s.run_downs - pooled,
        "share": s.pass_downs / s.downs,
    })


def scale_of(rates: dict, epa: dict, margins: dict) -> tuple[float, float]:
    """`(scale, level)` putting a success rate on EPA's scale, as `SUCCESS_SCALE` is measured."""
    ids = [gid for gid in rates if gid in epa and gid in margins and None not in rates[gid]]
    eh = np.array([epa[g].home for g in ids])
    ea = np.array([epa[g].away for g in ids])
    sh = np.array([rates[g][0] for g in ids])
    sa = np.array([rates[g][1] for g in ids])
    m = np.array([margins[g] for g in ids])
    s_epa = np.polyfit(eh - ea, m, 1)[0]
    s_sr = np.polyfit(sh - sa, m, 1)[0]
    scale = s_sr / s_epa
    level = np.r_[eh, ea].mean() - scale * np.r_[sh, sa].mean()
    print(f"  {len(ids)} games; margin per EPA/play {s_epa:.1f}, per unit rate {s_sr:.1f}; "
          f"scale {scale:.3f}, level {level:.3f}; same-game r with margin "
          f"{np.corrcoef(sh - sa, m)[0, 1]:.3f}", flush=True)
    return float(scale), float(level)


def main(weights: list[float]) -> None:
    g = counts()
    gap = gaps(g)
    print("dropback share by season:", gap.share.round(3).to_dict())
    index = pickle.loads((HOME / "units/nfl_index.pkl").read_bytes())
    margins = g.groupby("game_id").margin.first().to_dict()
    sides = {(r.game_id, r.offense_is_home): r for r in g.itertuples(index=False)}

    def side(gid, home):
        r = sides.get((gid, home))
        return None if r is None else Counts(r.pass_downs, r.pass_wins, r.run_downs, r.run_wins)

    have = [gid for gid in index if side(gid, True) and side(gid, False)]
    print(f"index games {len(index)}, with plays on both sides {len(have)}")
    agree = np.mean([
        abs(side(gid, True).pooled - index[gid].home_success) < 1e-12 for gid in have
        if index[gid].home_success is not None
    ])
    print(f"pooled rate from the plays equals the index's on {agree:.1%} of games")
    print("pooled:")
    pooled = {gid: (side(gid, True).pooled, side(gid, False).pooled) for gid in have}
    scale_of(pooled, index, margins)

    for w in weights:
        season_of = {gid: sides[(gid, True)].season for gid in have}
        rates = {}
        for gid in have:
            gp, gr = gap.loc[season_of[gid], ["pass_gap", "run_gap"]]
            rates[gid] = tuple(mixed_rate(side(gid, h), w, gp, gr, PRIOR) for h in (True, False))
        print(f"mixed w={w}:")
        scale, level = scale_of(rates, index, margins)
        out = {
            gid: (e._replace(home_success=rates[gid][0], away_success=rates[gid][1]) if gid in rates
                  else e._replace(home_success=None, away_success=None))
            for gid, e in index.items()
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"nfl_index_w{w}.pkl").write_bytes(pickle.dumps(out))
        (OUT / f"nfl_index_w{w}.json").write_text(json.dumps({"scale": scale, "level": level, "prior": PRIOR}))


if __name__ == "__main__":
    main([float(w) for w in sys.argv[1:]])

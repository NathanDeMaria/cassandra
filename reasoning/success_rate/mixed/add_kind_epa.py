"""Put each kind's bounded EPA per down into `sweep_split.py`'s index, off the local play tables.

    python add_kind_epa.py nfl|ncaafb

For a check that needs no s3: the plays behind `nfl_plays.parquet` and
`ncaafb_plays.parquet` are the sweep's, with each play's raw EPA, so each
side's EPA per pass and per run -- bounded at `CLIP` as the sweep bounds
it -- comes off them. Every game's kind counts must match the index's.
Writes `{league}_index_split_same_epa.pkl` beside the index it reads.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cassandra.epa_build import CLIP, is_down, is_pass

HOME = Path.home() / ".cassandra/success_rate"


def plays(league: str) -> pd.DataFrame:
    if league == "nfl":
        p = pd.read_parquet(HOME / "nfl_plays.parquet",
                            columns=["game_id", "offense_is_home", "play_type", "text", "epa"])
        p = p[[is_down(t, x) for t, x in zip(p.play_type, p.text)]].copy()
        p["dropback"] = [is_pass(t, x) for t, x in zip(p.play_type, p.text)]
        return p
    return pd.read_parquet(HOME / "mixed/ncaafb_plays.parquet",
                           columns=["game_id", "offense_is_home", "dropback", "epa"])


def main(league: str) -> None:
    p = plays(league)
    p["bounded"] = p.epa.clip(-CLIP, CLIP)
    g = p.groupby(["game_id", "offense_is_home", "dropback"]).bounded.agg(["mean", "size"])
    index = pickle.loads((HOME / f"mixed/{league}_index_split_same.pkl").read_bytes())
    out, mismatched = {}, 0
    for gid, e in index.items():
        values = {}
        for home, side in ((True, "home"), (False, "away")):
            for passing, kind in ((True, "pass"), (False, "run")):
                key = (gid, home, passing)
                n = int(g["size"].get(key, 0))
                mismatched += n != getattr(e, f"{side}_{kind}_plays")
                values[f"{side}_{kind}_epa"] = float(g["mean"][key]) if n else None
        out[gid] = e._replace(**values)
    (HOME / f"mixed/{league}_index_split_same_epa.pkl").write_bytes(pickle.dumps(out))
    with_epa = sum(e.home_pass_epa is not None for e in out.values())
    print(f"{league}: {len(out)} games, {with_epa} with a pass EPA; kind counts off the index's "
          f"in {mismatched} side-kinds of {4 * len(out)}")
    # Where the flat EPA is over downs alone, the kinds together are it.
    sample = [e for e in out.values() if e.home_pass_epa is not None and e.home_run_epa is not None]
    together = np.array([(e.home_pass_plays * e.home_pass_epa + e.home_run_plays * e.home_run_epa)
                         / (e.home_pass_plays + e.home_run_plays) for e in sample])
    flat = np.array([e.home for e in sample])
    print(f"  kinds together vs the flat EPA per snap: r {np.corrcoef(together, flat)[0, 1]:.4f}, "
          f"mean {together.mean():+.3f} vs {flat.mean():+.3f}")


if __name__ == "__main__":
    main(sys.argv[1])

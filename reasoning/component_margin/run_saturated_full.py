"""The saturating record step over the whole history, 2002 on, by window."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "success_rate" / "ncaafb"))
sys.path.insert(0, str(HERE))

from replay import OUT, PRIORS, _inputs, _replay  # noqa: E402  (pins the data)
from cassandra.predictor.opponent_prior import OpponentPriorManager  # noqa: E402
from saturated import SaturatedUnitMarginPredictor  # noqa: E402

PUBLISHED = json.loads((OUT / "batch_nc_success_cap_p.json").read_text())["knobs"]

if __name__ == "__main__":
    name = sys.argv[1]
    knobs = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:])}
    named, _, index = _inputs()
    predictor = SaturatedUnitMarginPredictor(
        "ncaafb", **{**PUBLISHED, **knobs},
        opponent_prior_manager=OpponentPriorManager("ncaafb", path=PRIORS), game_epa=index)
    df = _replay(predictor, named)
    df.to_parquet(OUT / f"cm_{name}.parquet")
    loss = (df.team1_win_prob - df.team1_win) ** 2
    out = {f"{lo}-{hi}": float(loss[df.fbs & df.year.between(lo, hi)].mean())
           for lo, hi in ((2002, 2013), (2014, 2021), (2022, 2026), (2002, 2026))}
    print(name, knobs, {k: round(v, 6) for k, v in out.items()}, flush=True)

"""Replay ncaafb with `ComponentMarginPredictor` and score it beside the published model.

    python run.py NAME [knob=value ...]    one replay, to ncaafb_out/cm_NAME.parquet

Resumes from the success-rate harness's 2013 snapshot of the published
model (`../success_rate/ncaafb/replay.py`), on its pinned predictor data:
nothing before 2014 has a reading, so the record through 2013 is the level
every part starts from. Scored on FBS v FBS games as the search scores them,
and on the margin: each game's mean and sd are kept beside its probability.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "success_rate" / "ncaafb"))
sys.path.insert(0, str(HERE))

from replay import (  # noqa: E402  (pins the predictor data on import)
    FIRST_READING,
    PRIORS,
    TRAIN_THROUGH,
    _inputs,
    _replay,
    _warm,
    save,
    scores,
)

from cassandra.predictor.opponent_prior import OpponentPriorManager  # noqa: E402
from component import ComponentMarginPredictor  # noqa: E402

#: The published success model's reading knobs, where the parts start.
START = {
    "success_weight": 0.5891,
    "defense_success_weight": 0.5891,
    "points_per_epa": 86.9467,
    "play_sd": 1.6686,
}


def run(knobs: dict, through: int = 2026) -> pd.DataFrame:
    named, _, index = _inputs()
    warm, early = _warm()
    cls = ComponentMarginPredictor
    if "qb_gap_scale" in knobs:
        from component_qb import ComponentQbPredictor as cls
    predictor = cls.from_state_dict({
        **warm.state_dict(),
        **START,
        **knobs,
        "opponent_prior_manager": OpponentPriorManager("ncaafb", path=PRIORS),
        "game_epa": index,
    })
    predictor._season = warm._season
    predictor._qb_pending = set(warm._qb_pending)
    late = _replay(predictor, [s for s in named if FIRST_READING <= s.year <= through])
    recorded = predictor.recorded
    late["mu"] = [recorded.get(g, (math.nan, math.nan))[0] for g in late.game_id]
    late["sd"] = [recorded.get(g, (math.nan, math.nan))[1] for g in late.game_id]
    return pd.concat([early, late], ignore_index=True)


def margin_scores(df: pd.DataFrame) -> dict[str, float]:
    """Native margin log loss (discretized Normal) on FBS v FBS, train and held out."""
    from scipy.stats import norm

    rows = df[df.fbs & df.mu.notna()]
    m = (rows.home_score - rows.away_score).to_numpy(float)
    mu, sd = rows.mu.to_numpy(), rows.sd.to_numpy()
    p = norm.cdf((m + 0.5 - mu) / sd) - norm.cdf((m - 0.5 - mu) / sd)
    loss = -np.log(np.maximum(p, 1e-300))
    train = (rows.year <= TRAIN_THROUGH).to_numpy()
    slope = np.polyfit(mu, m, 1)[0]
    return {
        "margin_ll_train": float(loss[train].mean()),
        "margin_ll_held_out": float(loss[~train].mean()),
        "margin_slope": float(slope),
        "resid_sd": float(np.std(m - mu)),
    }


if __name__ == "__main__":
    name = sys.argv[1]
    knobs = {k: float(v) for k, v in (arg.split("=") for arg in sys.argv[2:])}
    through = int(knobs.pop("through", 2026))
    start = time.time()
    df = run(knobs, through=through)
    extra = margin_scores(df)
    save(f"cm_{name}", df, {**START, **knobs}, **extra)
    print(name, knobs, {k: round(v, 6) for k, v in {**scores(df), **extra}.items()},
          f"({time.time() - start:.0f}s)", flush=True)

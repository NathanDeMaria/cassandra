"""The shipped mismatch_scale knob (PR #112) over the whole history, by window.

Run with PYTHONPATH on the mismatch-scale worktree: the pinned published knobs
plus overrides, e.g. ``fit mismatch_scale=267.03 obs_sd=20.036 prediction_scale=169.82``.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "success_rate" / "ncaafb"))

from replay import OUT, PRIORS, _inputs, _replay  # noqa: E402  (pins the data)
from cassandra.predictor.opponent_prior import OpponentPriorManager  # noqa: E402
from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor  # noqa: E402

PUBLISHED = json.loads((OUT / "batch_nc_success_cap_p.json").read_text())["knobs"]

if __name__ == "__main__":
    name = sys.argv[1]
    knobs = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:])}
    named, _, index = _inputs()
    predictor = UnitMarginGlickoPredictor(
        "ncaafb", **{**PUBLISHED, **knobs},
        opponent_prior_manager=OpponentPriorManager("ncaafb", path=PRIORS), game_epa=index)
    df = _replay(predictor, named)
    df.to_parquet(OUT / f"cm_{name}.parquet")
    loss = (df.team1_win_prob - df.team1_win) ** 2
    out = {f"{lo}-{hi}": float(loss[df.fbs & df.year.between(lo, hi)].mean())
           for lo, hi in ((2002, 2013), (2014, 2021), (2022, 2026), (2002, 2026))}
    print(name, knobs, {k: round(v, 6) for k, v in out.items()}, flush=True)

"""Replay the published success model with a saturating record step.

    python run_saturated.py NAME [mismatch_scale=M] [knob=value ...]   -> ncaafb_out/cm_NAME
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "success_rate" / "ncaafb"))
sys.path.insert(0, str(HERE))

from replay import OUT, run_from_2014, save, scores  # noqa: E402  (pins the data)
from saturated import SaturatedUnitMarginPredictor  # noqa: E402

PUBLISHED = json.loads((OUT / "batch_nc_success_cap_p.json").read_text())["knobs"]

if __name__ == "__main__":
    name = sys.argv[1]
    knobs = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:])}
    start = time.time()
    df = run_from_2014({**PUBLISHED, **knobs}, predictor_class=SaturatedUnitMarginPredictor)
    save(f"cm_{name}", df, {**PUBLISHED, **knobs})
    print(name, knobs, {k: round(v, 6) for k, v in scores(df).items()}, f"({time.time() - start:.0f}s)", flush=True)

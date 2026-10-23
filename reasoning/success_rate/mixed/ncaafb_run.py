"""One ncaafb replay from 2014 at a saved run's knobs, through `../ncaafb/replay.py`.

    NCAAFB_INDEX=... NCAAFB_SUCCESS_SCALE=s,l python ncaafb_run.py NAME FROM [knob=value ...]

FROM is a run in `units/ncaafb_out` (`batch_nc_success_cap_p` is the
published success model on the pinned data). The index and its constants
come in through the harness's environment variables; see `mixed.sh`.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ncaafb"))
from replay import INDEX, OUT, run_from_2014, save, scores  # noqa: E402

name, start = sys.argv[1], sys.argv[2]
knobs = json.loads((OUT / f"{start}.json").read_text())["knobs"]
knobs.update({k: float(v) for k, v in (pair.split("=") for pair in sys.argv[3:])})
began = time.time()
df = run_from_2014(knobs)
save(name, df, knobs, index=str(INDEX), source=start)
print(name, INDEX.name, {k: round(v, 6) for k, v in scores(df).items()}, f"({time.time() - began:.0f}s)", flush=True)

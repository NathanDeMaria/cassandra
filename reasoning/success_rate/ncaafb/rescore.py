"""Score knob sets again on the pinned data, as NAME_p.

    python rescore.py NAME [SOURCE] [knob=value ...]

SOURCE is the run whose knobs to take (NAME by default); the reading's
environment (`NCAAFB_INDEX`, `NCAAFB_EPA`, `NCAAFB_SUCCESS_SCALE`) is the
caller's, as for `replay.py`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay import OUT, run_from_2014, save, scores  # noqa: E402

name = sys.argv[1]
rest = sys.argv[2:]
source = rest.pop(0) if rest and "=" not in rest[0] else name
overrides = {k: float(v) for k, v in (a.split("=") for a in rest)}
knobs = {**json.loads((OUT / f"{source}.json").read_text())["knobs"], **overrides}
df = run_from_2014(knobs)
save(f"{name}_p", df, knobs, source=source)
print(f"{name}_p", {k: round(v, 6) for k, v in scores(df).items()}, flush=True)

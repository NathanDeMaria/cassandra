"""Replay the config's seeds through production code: train/held-out brier."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prod import run  # noqa: E402

config = json.loads(Path(sys.argv[1]).read_text())
for i, seed in enumerate(config["seeds"]):
    df = run({**config["fixed"], **seed})
    loss = (df.team1_win_prob - df.team1_win) ** 2
    print(i, f"train {loss[df.year <= 2021].mean():.6f}  held out {loss[df.year >= 2022].mean():.6f}  "
          f"2014+ {loss[df.year >= 2014].mean():.6f}", flush=True)

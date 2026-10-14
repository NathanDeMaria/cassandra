"""Replay against a private copy of `~/.cassandra/predictor/data`, not the shared one.

Other sessions refresh the shared directory from s3 -- the betting tools do
it first thing -- and a replay reads the anchors, the quarterback-out index,
the roster files and the rest from wherever the directory stands when the
process first asks. On 2026-10-13 a refresh landed between two refits started
14 minutes apart, which moved 2014-on ncaafb win probabilities by up to 0.027
and made the two refits incomparable. Every replay that compares to another
reads the same pinned copy.

    python pin.py      snapshot the shared directory, once; refuses to overwrite

`pin()` points every loader at the copy. Call it before anything is loaded.
"""

import importlib
import shutil
import sys
from pathlib import Path

SHARED = Path.home() / ".cassandra/predictor/data"
PINNED = Path.home() / ".cassandra/success_rate/pinned_data"

#: Each module that reads the directory, and the name it holds it under.
_LOADERS = {
    "cassandra.roster": "_PREDICTOR_DATA_DIR",
    "cassandra.predictor.qb_out": "_PREDICTOR_DATA_DIR",
    "cassandra.predictor.base_predictor": "_ANCHOR_DIR",
    "cassandra.predictor.opponent_prior": "_PREDICTOR_DATA_DIR",
    "cassandra.predictor.game_control": "_PREDICTOR_DATA_DIR",
    "cassandra.predictor.epa": "_PREDICTOR_DATA_DIR",
}


def snapshot() -> None:
    if PINNED.exists():
        raise SystemExit(f"{PINNED} exists; delete it to take a new snapshot")
    shutil.copytree(SHARED, PINNED)
    print(f"pinned {len(list(PINNED.iterdir()))} files from {SHARED}")


def pin() -> None:
    if not PINNED.exists():
        raise SystemExit(f"no snapshot at {PINNED}; run pin.py first")
    for name, attribute in _LOADERS.items():
        setattr(importlib.import_module(name), attribute, PINNED)


if __name__ == "__main__" and len(sys.argv) == 1:
    snapshot()

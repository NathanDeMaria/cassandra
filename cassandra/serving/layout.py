"""Where a model's artifacts live under the bucket.

One definition, because a release, its rating history and its stored
predictions have to land in the same directory and be found there by four
different callers -- `publish` writing them, the refresh job upserting two
of them, the API reading all three, and a human doing `cp runs/<old>.json
latest.json` to roll back.

The layout, rooted at a bucket or at a local `--out` directory:

    models/{league}/{model}/latest.json
    models/{league}/{model}/runs/{run_id}.json
    models/{league}/{model}/history.parquet
    models/{league}/{model}/predictions.parquet

The parquet files have no per-run copy on purpose. They are the whole
table rather than a snapshot of it, and `run_id` is a column: which run
wrote a row is a question you ask of the row, not of the filename.
"""

from pathlib import Path


def model_dir(root: Path, league: str, model: str) -> Path:
    """The directory holding everything published for one model."""
    return root / "models" / league / model

"""Where a model's artifacts live under the bucket.

One definition, because a release, its rating history and its stored
predictions have to land in the same directory and be found there by four
different callers -- `publish` writing them, the refresh job upserting two
of them, the API reading all three, and a human doing `cp runs/<old>.json
latest.json` to roll back.

The layout, rooted at a bucket or at a local `--out` directory:

    models/{league}/qb_out.json
    models/{league}/{model}/latest.json
    models/{league}/{model}/runs/{run_id}.json
    models/{league}/{model}/history.parquet
    models/{league}/{model}/predictions.parquet

The parquet files have no per-run copy on purpose. They are the whole
table rather than a snapshot of it, and `run_id` is a column: which run
wrote a row is a question you ask of the row, not of the filename.

`qb_out.json` sits at the league level because it is a fact about games,
not a model's output: which side was missing its expected starter, by
ESPN game id, the same `QbOutFile` the football models replay with. It
is here so a consumer can say of a played game what the model priced,
from the bucket it already reads, rather than from a file that only
exists on a machine that ran the sweep -- `QbOutIndex.for_league` reads
`~/.cassandra`, and an app that called it read an empty index for a
month and told every page both quarterbacks had started.
"""

from pathlib import Path


def league_dir(root: Path, league: str) -> Path:
    """The directory holding everything published for one league."""
    return root / "models" / league


def model_dir(root: Path, league: str, model: str) -> Path:
    """The directory holding everything published for one model."""
    return league_dir(root, league) / model


def qb_out_artifact_path(root: Path, league: str) -> Path:
    """Where a league's published quarterback index lives."""
    return league_dir(root, league) / "qb_out.json"

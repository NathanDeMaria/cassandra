import json
import os
from pathlib import Path

CASSANDRA_HOME = Path.home() / ".cassandra"

# How a Batch container is told which buckets it has; terraform sets both in
# every job definition (`jobs/main.tf`, `job_environment`). A laptop has
# neither and reads the shared outputs file instead.
_TEMP_BUCKET_VAR = "CASSANDRA_TEMP_BUCKET"
_OUTPUTS_FILE = Path.home() / ".aws-batch" / "config.json"


def temp_bucket() -> str:
    """The bucket for anything s3-shaped that nothing needs to keep.

    `aws-batch-optimization` provisions it beside the batch bucket and
    expires what lands in it, so a file put here cannot outlive its
    usefulness by much whatever happens to the job that wrote it. Search
    checkpoints go here (`cassandra.checkpoint`); so should any future
    intermediate that only has to survive a retry or a hand-off. The batch
    bucket is for what a consumer reads -- results, releases, the indexes
    the predictors replay against -- and nothing temporary belongs in it.

    Resolved the way `endgame_aws.Config` resolves the batch bucket: the
    environment inside a container, the shared outputs file on a laptop.
    """
    from_env = os.environ.get(_TEMP_BUCKET_VAR)
    if from_env:
        return from_env
    if _OUTPUTS_FILE.is_file():
        return json.loads(_OUTPUTS_FILE.read_text())["temp_bucket"]["value"]
    raise FileNotFoundError(
        f"No temp bucket: set {_TEMP_BUCKET_VAR}, or put the shared outputs at "
        f"{_OUTPUTS_FILE} (`make outputs` in aws-batch-optimization)."
    )

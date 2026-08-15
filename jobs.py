"""The container entrypoint: one subcommand per node in the Batch DAG.

    python jobs.py manifest                       # what would run, in order
    python jobs.py optimize --index 3             # one array child
    python jobs.py optimize --league mens --model elo
    python jobs.py evaluate --league mens
    python jobs.py publish --league mens
    python jobs.py submit --league mens           # launch the whole DAG

Every stage is independently runnable, which is the point: a job definition
per stage means "re-publish mens without re-optimizing anything" is a job
submission rather than a code change. The stages hand off through s3 (see
`cassandra.batch.artifacts`) because they don't share a disk.

The local `make run-all` path is unchanged and still writes only to
`~/.cassandra` -- `run_models.sh` remains the way to run everything on one
machine, and nothing here uploads unless it was asked to.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import fire
from endgame_aws import Config

from cassandra.batch import artifacts, dag, manifest
from cassandra.constants import CASSANDRA_HOME


def _bucket() -> str:
    """The batch bucket, from the config terraform writes.

    Same resolution the rest of the repo uses, so a job reads the same bucket
    a laptop does.
    """
    return Config.init_from_file().bucket


def manifest_command() -> None:
    """Print the work list, in the order the array job indexes it."""
    work = manifest.load_manifest()
    width = max((len(item.name) for item in work), default=0)
    for index, item in enumerate(work):
        print(
            f"{index:>3}  {item.name:<{width}}  {item.predictor_class}  "
            f"n_iter={item.n_iter}"
        )
    print(f"\n{len(work)} configs")


async def _optimize(
    index: int | None, league: str | None, model: str | None, upload: bool
) -> None:
    # Import here rather than at module scope: `optimize.py` pulls in
    # bayes_opt and sklearn, and `jobs.py submit` runs in the same image but
    # has no reason to pay for loading them.
    from optimize import _run_optimization

    if league and model:
        work = manifest.find(league, model)
    elif league or model:
        raise ValueError("--league and --model have to be given together")
    else:
        work = manifest.resolve_index(index)

    print(f"=== {work.name} ({work.predictor_class}, n_iter={work.n_iter}) ===")

    # `optimize.py` does a warm-up pass with post_callbacks=True to build the
    # opponent priors the search starts from, and OpponentPriorManager refuses
    # to overwrite an existing file. A fresh container has none, but a rerun
    # in a warm one (or a local invocation) does -- same clearing
    # run_models.sh does, for the same reason.
    if work.prior_path is not None and work.prior_path.exists():
        print(f"  clearing stale priors: {work.prior_path}")
        work.prior_path.unlink()

    await _run_optimization(str(work.config_path))

    if upload:
        keys = await artifacts.upload(
            _bucket(), artifacts.results_for(work.league, work.model)
        )
        for key in keys:
            print(f"  uploaded s3://{_bucket()}/{key}")


async def _evaluate(leagues: list[str] | None, download: bool, upload: bool) -> None:
    from evaluate_models import _main as evaluate_main

    bucket = _bucket()
    if download:
        paths = await artifacts.download(bucket, "models/")
        print(f"Pulled {len(paths)} result files from s3://{bucket}")
        if not paths:
            raise FileNotFoundError(
                f"Nothing under s3://{bucket}/{artifacts.ARTIFACT_PREFIX}/models/; "
                "optimize has to run before evaluate has anything to score."
            )

    await evaluate_main(leagues)

    if upload:
        # Just what evaluate produces: the timestamped metrics csv, and a
        # calibration per model/fitter. Deliberately not everything under
        # models/, which at this point is mostly the results this job just
        # downloaded -- echoing those back would rewrite optimize's output
        # with a copy of itself on every run.
        produced = [
            *(CASSANDRA_HOME / "evaluations").glob("*.csv"),
            *(CASSANDRA_HOME / "models").rglob("*_calibration.json"),
        ]
        keys = await artifacts.upload(bucket, produced)
        print(f"Uploaded {len(keys)} files to s3://{bucket}")


async def _publish(
    leagues: list[str] | None, index: int | None, download: bool, upload: bool
) -> None:
    from publish import _publish as publish_main

    # An array child gets its league from the list the launcher pinned; a
    # hand-run job gets it from --league.
    if leagues is None and index is not None:
        leagues = [dag.publish_league(index)]

    bucket = _bucket()
    if download:
        paths = await artifacts.download(bucket, "models/")
        print(f"Pulled {len(paths)} result files from s3://{bucket}")
        if not paths:
            raise FileNotFoundError(
                f"Nothing under s3://{bucket}/{artifacts.ARTIFACT_PREFIX}/models/; "
                "optimize has to run before there is anything to publish."
            )

    failures = await publish_main(None, leagues or [], Path("releases"), upload=upload)
    if failures:
        # Publish keeps going past a bad model so the other nineteen still get
        # releases; the job still has to fail, or a broken model looks green.
        print(f"\n{len(failures)} failed:")
        print("\n".join(f"  {failure}" for failure in failures))
        sys.exit(1)


class Jobs:
    """Subcommands, one per node in the DAG."""

    def manifest(self) -> None:
        """Print the work list, in array-index order."""
        manifest_command()

    def optimize(
        self,
        index: int | None = None,
        league: str | None = None,
        model: str | None = None,
        upload: bool = True,
    ) -> None:
        """Optimize one config. Defaults to this array child's index.

        `--upload` is on by default because the only reason to run this in a
        container is to get the result somewhere the next stage can read it;
        pass `--no-upload` for a local dry run.
        """
        asyncio.run(_optimize(index, league, model, upload))

    def evaluate(
        self,
        league: list[str] | str | None = None,
        download: bool = True,
        upload: bool = True,
    ) -> None:
        """Score every model in scope and write the metrics csv."""
        asyncio.run(_evaluate(_as_list(league), download, upload))

    def publish(
        self,
        league: list[str] | str | None = None,
        index: int | None = None,
        download: bool = True,
        upload: bool = True,
    ) -> None:
        """Build releases for a league and push them to the artifacts bucket."""
        asyncio.run(_publish(_as_list(league), index, download, upload))

    def submit(
        self,
        league: list[str] | str | None = None,
        model: list[str] | str | None = None,
        skip_optimize: bool = False,
        skip_evaluate: bool = False,
        skip_publish: bool = False,
        dry_run: bool = False,
        wait: bool = False,
        job_queue: str | None = None,
        optimize_job_definition: str | None = None,
        evaluate_job_definition: str | None = None,
        publish_job_definition: str | None = None,
    ) -> None:
        """Submit the DAG to Batch and print what went out."""
        queue = job_queue or _default_queue()
        submitted = asyncio.run(
            dag.submit(
                optimize_job_definition=_job_definition(
                    "OPTIMIZE", optimize_job_definition
                ),
                evaluate_job_definition=_job_definition(
                    "EVALUATE", evaluate_job_definition
                ),
                publish_job_definition=_job_definition(
                    "PUBLISH", publish_job_definition
                ),
                job_queue=queue,
                leagues=_as_list(league),
                models=_as_list(model),
                skip_optimize=skip_optimize,
                skip_evaluate=skip_evaluate,
                skip_publish=skip_publish,
                dry_run=dry_run,
            )
        )
        for job in submitted:
            suffix = f" (array of {job.size})" if job.size > 1 else ""
            print(f"{job.job_id}  {job.name}{suffix}")
        if dry_run:
            print("\nDry run: nothing was submitted.")
            return
        if wait:
            statuses = asyncio.run(dag.wait([j.job_id for j in submitted]))
            failed = [job_id for job_id, s in statuses.items() if s != "SUCCEEDED"]
            for job in submitted:
                print(f"{statuses[job.job_id]:<10} {job.name}")
            if failed:
                sys.exit(1)


def _as_list(value: list[str] | str | None) -> list[str] | None:
    """fire hands back a str for one flag and a tuple for repeats."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _job_definition(stage: str, override: str | None) -> str:
    """Which definition a stage submits to.

    Terraform is the source of truth and puts these in the launcher's
    environment, so renaming a definition doesn't need a matching change here.
    The literal fallback is for running the launcher from a laptop against the
    conventional names.
    """
    if override is not None:
        return override
    return os.environ.get(
        f"CASSANDRA_{stage}_JOB_DEFINITION", f"cassandra-{stage.lower()}"
    )


def _default_queue() -> str:
    """The queue name, from terraform's environment or its config file."""
    from_env = os.environ.get("CASSANDRA_JOB_QUEUE")
    if from_env:
        return from_env

    for candidate in (
        os.environ.get("BATCH_CONFIG_FILE_PATH"),
        str(Path.home() / ".aws-batch" / "config.json"),
    ):
        if candidate and Path(candidate).exists():
            raw = json.loads(Path(candidate).read_text())
            if "job_queue_name" in raw:
                return raw["job_queue_name"]["value"]
    raise FileNotFoundError(
        "No job queue found; pass --job-queue, set CASSANDRA_JOB_QUEUE, or put "
        "the terraform output at ~/.aws-batch/config.json (`make outputs` in "
        "aws-batch-optimization)."
    )


if __name__ == "__main__":
    fire.Fire(Jobs)

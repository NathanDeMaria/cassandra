"""The container entrypoint: one subcommand per node in the Batch DAG.

    python jobs.py manifest                       # what would run, in order
    python jobs.py anchors --league ncaafb        # fit division anchors
    python jobs.py game-control --league nfl      # sweep play-by-play
    python jobs.py optimize --index 3             # one array child
    python jobs.py optimize --league mens --model elo
    python jobs.py evaluate --league mens
    python jobs.py publish --league mens
    python jobs.py submit --league mens           # launch the whole DAG

Every stage is independently runnable, which is the point: a job definition
per stage means "re-publish mens without re-optimizing anything" is a job
submission rather than a code change. The stages hand off through s3 (see
`cassandra.batch.artifacts`) because they don't share a disk.

Every stage also runs locally, which is how a change gets tested before it
costs a queue slot: `--download=False --upload=False` leaves s3 alone entirely
and writes only to `~/.cassandra`. There is deliberately no local "run
everything" driver -- `submit` is the only thing that knows the whole DAG, and
a second copy of that ordering is a second thing to keep in step.
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
from cassandra.predictor import (
    ANCHOR_LEAGUES,
    CONTROL_LEAGUES,
    anchor_path,
)


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


async def _anchors(
    leagues: list[str] | None, index: int | None, if_missing: bool, upload: bool
) -> None:
    # Same late import as the optimize path below: the fit pulls in
    # call_it_what_you_want and every stored season, and `jobs.py submit` has
    # no reason to pay for either.
    from division_anchors import _build

    # An array child gets its league from the list the launcher pinned, by the
    # index Batch set on it. A hand-run job gets it from --league, and one
    # given neither does every anchored league.
    if leagues is None:
        child = manifest.array_index(index)
        leagues = (
            [dag.anchor_league(child)] if child is not None else list(ANCHOR_LEAGUES)
        )

    bucket = _bucket()
    # `--if-missing` has to mean "missing in the bucket", not "missing on this
    # disk": a fresh container never has an anchor file, so the local check
    # `division_anchors.main` does would refit every league on every run --
    # and refitting moves every rating the downstream models produce.
    if if_missing:
        await artifacts.download_predictor_data(bucket)

    for league in leagues:
        path = anchor_path(league)
        if if_missing and path.exists():
            print(f"{league}: anchors already in s3://{bucket}, leaving them alone")
            continue

        print(f"=== {league} / division anchors ===")
        await _build(league, write=True)

        if upload:
            keys = await artifacts.upload(bucket, artifacts.anchors_for(league))
            for key in keys:
                print(f"  uploaded s3://{bucket}/{key}")


async def _game_control(
    leagues: list[str] | None,
    index: int | None,
    rebuild: bool,
    upload: bool,
) -> None:
    # Late, like the anchors fit above: this pulls pyarrow and `lucky_ones`,
    # and `jobs.py submit` -- which runs on a laptop to launch the DAG -- has
    # no reason to import a fitting stack to send a SubmitJob.
    from endgame_aws.pbp_parquet import get_processed_plays_store
    from lucky_ones.arrow import StorePlaySource

    from cassandra.game_control_build import build
    from cassandra.save_predictions import read_all_seasons

    if leagues is None:
        child = manifest.array_index(index)
        leagues = (
            [dag.control_league(child)] if child is not None else list(CONTROL_LEAGUES)
        )

    bucket = _bucket()
    # Whether there is anything to do is a question about the *stored* index,
    # and a fresh container has no files at all -- so the decision can only be
    # made after this. It also drops the cached reads that resolving anything
    # above may have populated.
    await artifacts.download_predictor_data(bucket)

    source = StorePlaySource(get_processed_plays_store())
    for league in leagues:
        print(f"=== {league} / game control ===")
        seasons = [s async for s in read_all_seasons(league, bucket)]
        await build(league, seasons, source, rebuild=rebuild)

        if upload:
            keys = await artifacts.upload(bucket, artifacts.game_control_for(league))
            for key in keys:
                print(f"  uploaded s3://{bucket}/{key}")


async def _optimize(
    index: int | None,
    league: str | None,
    model: str | None,
    download: bool,
    upload: bool,
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

    # The anchors decide what a rating means, so they have to be on disk
    # before the search starts: without them every team regresses toward
    # MEAN_RATING, and the parameters that come out are fit to a different
    # scale than the ones the last run produced. The same download brings the
    # game control `glicko_control` blends in, and drops the cached reads that
    # resolving the manifest just populated.
    if download:
        pulled = await artifacts.download_predictor_data(_bucket())
        print(f"  {len(pulled)} predictor data file(s) from s3://{_bucket()}")

    # `optimize.py` does a warm-up pass with post_callbacks=True to build the
    # opponent priors the search starts from, and OpponentPriorManager refuses
    # to overwrite an existing file. A fresh container has none, but a rerun
    # in a warm one (or a local invocation) does, and an uncleared file is an
    # instant ValueError rather than a search.
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
        # A result file carries the fitted parameters but not the anchors --
        # `PredictorConfig.params` is float|str, and a per-team mapping isn't
        # one. So replaying a model here rebuilds its anchors from the file,
        # and a container without the file scores every model against a scale
        # the model was never fit on.
        await artifacts.download_predictor_data(bucket)

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

    # An array child gets its league from the list the launcher pinned, by the
    # index Batch set on it -- read from the environment, because the command
    # the job definition carries is just `publish` with no index in it. A
    # hand-run job gets it from --league, and one given neither publishes
    # every league.
    if leagues is None:
        child = manifest.array_index(index)
        if child is not None:
            leagues = [dag.publish_league(child)]

    bucket = _bucket()
    if download:
        paths = await artifacts.download(bucket, "models/")
        print(f"Pulled {len(paths)} result files from s3://{bucket}")
        if not paths:
            raise FileNotFoundError(
                f"Nothing under s3://{bucket}/{artifacts.ARTIFACT_PREFIX}/models/; "
                "optimize has to run before there is anything to publish."
            )
        # Same reason as evaluate: the replay behind a release reads the
        # anchors off disk, and publishing without them ships ratings on a
        # different scale than the ones the models were fit on.
        await artifacts.download_predictor_data(bucket)

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

    def anchors(
        self,
        league: list[str] | str | None = None,
        index: int | None = None,
        if_missing: bool = True,
        upload: bool = True,
    ) -> None:
        """Fit a league's division anchors. Defaults to this child's index.

        `--if-missing` is on by default, and checks the bucket rather than
        this container's disk. Refitting changes the rating every downstream
        model produces, so it is something you ask for -- with
        `--if-missing=False`, or by deleting the file from s3 -- not something
        a scheduled run does to itself weekly.
        """
        asyncio.run(_anchors(_as_list(league), index, if_missing, upload))

    def game_control(
        self,
        league: list[str] | str | None = None,
        index: int | None = None,
        rebuild: bool = False,
        upload: bool = True,
    ) -> None:
        """Sweep play-by-play into a league's game control index.

        Idempotent on the win probability fit rather than on the file: a
        league whose stored index was built by the `lucky_ones` this image
        has re-sweeps only the season still being played, which is the
        weekly case and about twenty weeks of reads. A league whose index
        came from a different fit -- a retrain, or a bumped pin -- is rebuilt
        from scratch, because an index holding two models' numbers is one
        nobody can reproduce.

        `--rebuild` forces the full sweep for a league that is already
        current. Unlike the anchors, that is not a decision that re-rates
        anything on its own: the same fit over the same plays produces the
        same numbers, so this costs queue time rather than changing results.
        """
        asyncio.run(_game_control(_as_list(league), index, rebuild, upload))

    def optimize(
        self,
        index: int | None = None,
        league: str | None = None,
        model: str | None = None,
        download: bool = True,
        upload: bool = True,
    ) -> None:
        """Optimize one config. Defaults to this array child's index.

        `--upload` is on by default because the only reason to run this in a
        container is to get the result somewhere the next stage can read it;
        pass `--upload=False` for a local dry run. `--download=False` likewise
        leaves the anchors already on this disk alone.
        """
        asyncio.run(_optimize(index, league, model, download, upload))

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
        skip_anchors: bool = False,
        rebuild_anchors: bool = False,
        skip_game_control: bool = False,
        rebuild_game_control: bool = False,
        skip_optimize: bool = False,
        skip_evaluate: bool = False,
        skip_publish: bool = False,
        dry_run: bool = False,
        wait: bool = False,
        job_queue: str | None = None,
        anchors_job_definition: str | None = None,
        game_control_job_definition: str | None = None,
        optimize_job_definition: str | None = None,
        evaluate_job_definition: str | None = None,
        publish_job_definition: str | None = None,
    ) -> None:
        """Submit the DAG to Batch and print what went out.

        `--rebuild-anchors` refits the division anchors even though s3
        already has some, and overwrites them. That re-rates every model in
        the league against a new scale, so it is deliberately not what a
        scheduled run does -- see `Jobs.anchors`.

        `--skip-game-control` is the one skip a scheduled republish should
        not pass: unlike the anchors, the control index gains entries every
        week, and publishing without sweeping them replays the newest games
        as though nobody had play-by-play for them. It is there for a run
        that knows the sweep has already happened.
        """
        queue = job_queue or _default_queue()
        submitted = asyncio.run(
            dag.submit(
                anchors_job_definition=_job_definition(
                    "ANCHORS", anchors_job_definition
                ),
                game_control_job_definition=_job_definition(
                    "GAME_CONTROL", game_control_job_definition
                ),
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
                skip_anchors=skip_anchors,
                rebuild_anchors=rebuild_anchors,
                skip_game_control=skip_game_control,
                rebuild_game_control=rebuild_game_control,
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
    conventional names. Underscores become hyphens on the way: the stages are
    named `GAME_CONTROL` here because the environment variable is, and
    `cassandra-game-control` in terraform because every other job definition
    is hyphenated.
    """
    if override is not None:
        return override
    default = f"cassandra-{stage.lower().replace('_', '-')}"
    return os.environ.get(f"CASSANDRA_{stage}_JOB_DEFINITION", default)


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

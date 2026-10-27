"""The container entrypoint: one subcommand per node in the Batch DAG.

    python jobs.py manifest                       # what would run, in order
    python jobs.py anchors --league ncaafb        # fit division anchors
    python jobs.py optimize --index 3             # one array child
    python jobs.py optimize --league mens --model elo
    python jobs.py evaluate --league mens
    python jobs.py publish --league mens
    python jobs.py submit --league mens           # launch the whole DAG
    python jobs.py picks                          # email new Kalshi picks

Every stage is independently runnable, which is the point: a job definition
per stage means "re-publish mens without re-optimizing anything" is a job
submission rather than a code change. The stages hand off through s3 (see
`cassandra.batch.artifacts`) because they don't share a disk.

Every stage also runs locally, which is how a change gets tested before it
costs a queue slot: `--download=False --upload=False` leaves s3 alone entirely
and writes only to `~/.cassandra`. There is deliberately no local "run
everything" driver -- `submit` is the only thing that knows the whole DAG, and
a second copy of that ordering is a second thing to keep in step.

`game_control` and `epa` are the exceptions to the first line: subcommands
with no node behind them. They are kept runnable -- `python jobs.py
game-control --league nfl`, `python jobs.py epa --league nfl` -- because the
signals they feed are ones worth measuring, and a stage nobody can run is a
stage nobody will revive. Control has been measured and didn't help (see
`cassandra.predictor.control`); EPA is the sweep that hasn't been run yet, and
`models/*/margin_blend.json` is what would read it.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import fire
from endgame_aws import Config

from cassandra.batch import artifacts, dag, manifest
from cassandra.constants import CASSANDRA_HOME
from cassandra.predictor import (
    ANCHOR_LEAGUES,
    CONTROL_LEAGUES,
    EPA_LEAGUES,
    anchor_path,
)
from cassandra.predictor.player_ratings import PLAYER_RATING_LEAGUES
from cassandra.predictor.qb_out import QB_LEAGUES

# When this process started, which is as close as it can see to when Batch
# started the attempt's clock.
_STARTED = time.monotonic()

# The job definition's per-attempt timeout, set beside it in terraform. Unset
# outside Batch, where nothing kills a search and it runs to the end.
ATTEMPT_TIMEOUT_ENV = "CASSANDRA_ATTEMPT_TIMEOUT_SECONDS"

# What a search leaves itself between stopping and the kill: the result
# written and uploaded, which is seconds, and the container's start before
# this process, which is seconds more. Ten minutes is for being wrong about
# both.
DEADLINE_MARGIN_SECONDS = 600


def _search_deadline() -> float | None:
    """The `time.monotonic()` reading a search has to be done by, if any."""
    timeout = os.environ.get(ATTEMPT_TIMEOUT_ENV)
    if not timeout:
        return None
    return _STARTED + float(timeout) - DEADLINE_MARGIN_SECONDS


def _host_line() -> str:
    """What this search is running on, for comparing probe times across runs.

    Per-probe time moved 0.67x-2.5x between two runs a day apart with no
    config change, and Batch forgets an instance's type once it's gone. The
    CPU model says which generation and vendor; siblings against cores says
    whether the two vCPUs a .large search shares are one core's
    hyperthreads.
    """
    fields: dict[str, str] = {}
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                name, _, value = line.partition(":")
                fields.setdefault(name.strip(), value.strip())
    except OSError:
        return "[host] cpuinfo unavailable"
    return (
        f"[host] {fields.get('model name', '?')}, "
        f"{fields.get('siblings', '?')} threads on {fields.get('cpu cores', '?')} "
        f"cores per socket, {os.cpu_count()} vCPU visible"
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

    # An array child gets its league from the list the launcher pinned, by the
    # index Batch set on it. A hand-run job gets it from --league, and one
    # given neither sweeps every football league.
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


async def _qb_out(
    leagues: list[str] | None,
    index: int | None,
    upload: bool,
    latest: bool = False,
) -> None:
    # Late, like the sweeps above: this pulls pyarrow, and `jobs.py submit`
    # runs on a laptop to send a SubmitJob and has no reason to import a
    # fitting stack.
    from endgame_aws.pbp_parquet import get_processed_plays_store

    from cassandra.predictor.qb_out import load_qb_out, read_qb_out_file
    from cassandra.qb_out_build import build, with_season, write
    from cassandra.save_predictions import read_all_seasons

    if leagues is None:
        child = manifest.array_index(index)
        leagues = [dag.qb_out_league(child)] if child is not None else list(QB_LEAGUES)

    bucket = _bucket()
    await artifacts.download_predictor_data(bucket)

    store = get_processed_plays_store()

    class _Weeks:
        async def load_week(self, league: str, season: int, week: int):
            return await store.load_week_or_empty(league, season, week)

    for league in leagues:
        print(f"=== {league} / quarterback availability ===")
        seasons = [s async for s in read_all_seasons(league, bucket)]
        # With nothing stored to keep (a new league, a deleted file) the
        # other seasons would be lost, so that rebuilds whole.
        if latest and read_qb_out_file(league) is not None:
            season = max(seasons, key=lambda s: s.year)
            fresh = await build(league, [season], _Weeks())
            missing = with_season(load_qb_out(league), season, fresh)
        else:
            missing = await build(league, seasons, _Weeks())
        write(league, missing)
        flagged = sum(len(teams) for teams in missing.values())
        print(f"  {len(missing):,} games with somebody out, {flagged:,} team-games")

        if upload:
            keys = await artifacts.upload(bucket, artifacts.qb_out_for(league))
            for key in keys:
                print(f"  uploaded s3://{bucket}/{key}")


async def _player_ratings(
    leagues: list[str] | None, index: int | None, today: str | None
) -> None:
    # Late, like the sweeps above: ten-minute-version brings scipy and
    # pyarrow, and `jobs.py submit` has no reason to import them.
    from datetime import date

    from cassandra.player_ratings_build import refresh

    if leagues is None:
        child = manifest.array_index(index)
        leagues = (
            [dag.player_ratings_league(child)]
            if child is not None
            else list(PLAYER_RATING_LEAGUES)
        )
    bucket = _bucket()
    day = date.fromisoformat(today) if today else None
    for league in leagues:
        print(f"=== {league} / player ratings ===")
        done = await refresh(bucket, league, day)
        print("  refreshed" if done else "  no kit for the season; left as it was")


async def _epa(
    leagues: list[str] | None,
    index: int | None,
    rebuild: bool,
    upload: bool,
) -> None:
    # Late, for the reason `_game_control` says: this pulls pyarrow and
    # `lucky_ones`, and `jobs.py submit` has no reason to import a fitting
    # stack to send a SubmitJob.
    from endgame_aws.pbp_parquet import get_processed_plays_store
    from lucky_ones.arrow import StorePlaySource

    from cassandra.epa_build import build
    from cassandra.save_predictions import read_all_seasons

    # Same as `_game_control` above: the launcher pins the league list it
    # sized the array against, and this child reads its own out by index.
    if leagues is None:
        child = manifest.array_index(index)
        leagues = [dag.epa_league(child)] if child is not None else list(EPA_LEAGUES)

    bucket = _bucket()
    # Whether there is anything to do is a question about the *stored* index,
    # and a fresh container has no files at all -- so the decision can only be
    # made after this. It also drops the cached reads that resolving anything
    # above may have populated.
    await artifacts.download_predictor_data(bucket)

    source = StorePlaySource(get_processed_plays_store())
    for league in leagues:
        print(f"=== {league} / epa ===")
        seasons = [s async for s in read_all_seasons(league, bucket)]
        await build(league, seasons, source, rebuild=rebuild)

        if upload:
            keys = await artifacts.upload(bucket, artifacts.epa_for(league))
            for key in keys:
                print(f"  uploaded s3://{bucket}/{key}")


async def _optimize(
    index: int | None,
    league: str | None,
    model: str | None,
    download: bool,
    upload: bool,
    rebuild: bool = False,
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
    print(_host_line())

    # The anchors decide what a rating means, so they have to be on disk
    # before the search starts: without them every team regresses toward
    # MEAN_RATING, and the parameters that come out are fit to a different
    # scale than the ones the last run produced. The same download brings the
    # game control `glicko_control` blends in, and drops the cached reads that
    # resolving the manifest just populated.
    if download:
        pulled = await artifacts.download_predictor_data(_bucket())
        print(f"  {len(pulled)} predictor data file(s) from s3://{_bucket()}")
        # And the last fit of this config, which `optimize.py` puts in front
        # of the search so it can't end up below where the last one did. Just
        # this model's result: the rest of `models/` is what evaluate reads,
        # and a search has no use for it.
        previous = await artifacts.download(
            _bucket(), f"models/{work.league}/{work.model}_result.json"
        )
        if previous:
            print(f"  previous fit: {previous[0]}")
        else:
            print(f"  no previous fit for {work.name} in s3://{_bucket()}")

    # `optimize.py` does a warm-up pass with post_callbacks=True to build the
    # opponent priors the search starts from, and OpponentPriorManager refuses
    # to overwrite an existing file. A fresh container has none, but a rerun
    # in a warm one (or a local invocation) does, and an uncleared file is an
    # instant ValueError rather than a search.
    if work.prior_path is not None and work.prior_path.exists():
        print(f"  clearing stale priors: {work.prior_path}")
        work.prior_path.unlink()

    await _run_optimization(
        str(work.config_path), rebuild=rebuild, deadline=_search_deadline()
    )

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


class _Tee:
    """stdout that's also kept, so a run's printout can be emailed."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.kept: list[str] = []

    def write(self, text: str) -> int:
        self.kept.append(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def text(self) -> str:
        return "".join(self.kept)


def _kalshi_key(secret_id: str | None) -> bool:
    """Put the Kalshi key where `KalshiClient.from_env` reads it, from Secrets
    Manager. False when there's no secret or it has no value yet: the run
    then sizes on `--budget` and can't place."""
    if not secret_id:
        return False
    from botocore.exceptions import ClientError

    from cassandra.picks_job import client

    try:
        value = client("secretsmanager").get_secret_value(SecretId=secret_id)
    except ClientError as error:
        print(f"no Kalshi key ({error.response['Error']['Code']}): emailing only")
        return False
    key = json.loads(value["SecretString"])
    pem = CASSANDRA_HOME / "kalshi" / "private_key.pem"
    pem.parent.mkdir(parents=True, exist_ok=True)
    pem.touch(mode=0o600)
    pem.write_text(key["private_key"])
    os.environ["KALSHI_KEY_ID"] = key["key_id"]
    os.environ["KALSHI_KEY_FILE"] = str(pem)
    return True


def _picks(kelly_fraction: float, max_bet: float, budget: float, espn: bool) -> None:
    import contextlib
    import logging
    import uuid

    import picks as picks_script
    from cassandra import picks_job
    from cassandra.constants import temp_bucket
    from cassandra.placing import CENTRAL

    logging.getLogger("endgame_aws.pbp_transform").setLevel(logging.ERROR)
    s3 = picks_job.client("s3")
    bucket, scratch = _bucket(), temp_bucket()
    now = picks_job.now_utc()
    if not picks_job.acquire(
        s3, scratch, now, os.environ.get("AWS_BATCH_JOB_ID", str(uuid.uuid4()))
    ):
        print("another picks run holds the lock; this one has nothing to add")
        return
    try:
        try:
            found = s3.get_object(Bucket=bucket, Key=picks_job.OVERRIDES_KEY)
            picks_script.OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
            picks_script.OVERRIDES.write_bytes(found["Body"].read())
        except s3.exceptions.NoSuchKey:
            print(f"no s3://{bucket}/{picks_job.OVERRIDES_KEY}: no lasting overrides")

        # A rehearsal of bedtime: the real key reads the account, the orders
        # are printed rather than sent, and the night's budget isn't charged.
        rehearsal = bool(os.environ.get("CASSANDRA_PICKS_DRY_RUN"))
        keyed = rehearsal or _kalshi_key(os.environ.get("CASSANDRA_KALSHI_SECRET"))
        night = picks_job.bedtime(s3, bucket)
        asleep = (
            keyed and night is not None and night.asleep(now) and night.remaining >= 1
        )
        if asleep:
            argv = [
                "--place",
                "--yes",
                "--kelly-fraction",
                str(night.kelly_fraction),
                "--max-bet",
                str(night.max_bet),
                "--budget",
                f"{night.remaining:.2f}",
                "--rest",
                f"{night.rest_hours(now):.2f}",
            ]
            if rehearsal:
                argv.append("--dry-run")
            print(
                f"bedtime until {night.until.astimezone(CENTRAL):%a %H:%M} CT, "
                f"${night.remaining:,.2f} of ${night.budget:,.2f} left: placing"
            )
        elif keyed:
            # Sized on the real account, and what's held or resting is known,
            # so a game already bet isn't emailed as a pick.
            argv = [
                "--notify",
                "--place",
                "--dry-run",
                "--kelly-fraction",
                str(kelly_fraction),
                "--max-bet",
                str(max_bet),
            ]
            # Nothing is placed awake, but the night's rungs rest until
            # kickoff, and one the model has stopped backing comes off.
            if not rehearsal:
                argv.append("--cancel-rungs")
        else:
            argv = ["--notify", "--budget", str(budget), "--max-bet", str(max_bet)]

        if not espn:
            argv.append("--no-espn")

        tee = _Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):
            args = picks_script.parse(argv)
            outcome = picks_script.run(args)

        # Every run's prices go up for `make ledger`, so a bet is read against
        # the run that made it; the orders only when they were really sent.
        if not rehearsal:
            for key in picks_job.keep(s3, bucket, now, orders=asleep):
                print(f"kept s3://{bucket}/{key}")

        sent, _ = picks_job.read_json(s3, bucket, picks_job.SENT_KEY)
        sent = sent or {}
        at = now.astimezone(CENTRAL)
        if asleep:
            placed = outcome.placed
            if placed is None or placed.empty:
                print("bedtime: nothing placed")
                return
            if not rehearsal:
                night = picks_job.charge(s3, bucket, picks_job.committed(placed))
            shown = outcome.picks[
                outcome.picks["game_id"].isin(set(placed.get("game_id", [])))
            ]
            fresh = shown
            subject = picks_job.subject(shown, fresh, placed=True, at=at)
        else:
            shown = outcome.picks
            fresh = picks_job.unsent(shown, sent)
            if fresh.empty and not outcome.cancelled:
                print(f"{len(shown)} picks, none new: no email")
                return
            subject = picks_job.subject(
                shown, fresh, placed=False, at=at, cancelled=len(outcome.cancelled)
            )
            placed = None
        text = picks_job.body(
            shown,
            fresh,
            tee.text(),
            placed,
            night if asleep else None,
            args.slippage,
            outcome.cancelled,
        )
        topic = os.environ.get("CASSANDRA_PICKS_TOPIC_ARN")
        if topic:
            picks_job.client("sns").publish(
                TopicArn=topic, Subject=subject, Message=text[:250_000]
            )
            print(f"emailed: {subject}")
        else:
            print(
                f"no CASSANDRA_PICKS_TOPIC_ARN; would have emailed: {subject}\n{text}"
            )
        if not rehearsal:
            picks_job.write_json(
                s3,
                bucket,
                picks_job.SENT_KEY,
                picks_job.record(sent, fresh, now, subject),
            )
    finally:
        picks_job.release(s3, scratch)


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
        """Sweep play-by-play into a league's game control index, by hand.

        No longer a stage of the submitted DAG -- nothing in the run reads
        what it writes. It is here for the next look at the play-by-play; see
        `cassandra.predictor.control` for why there isn't one today.

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

    def epa(
        self,
        league: list[str] | str | None = None,
        index: int | None = None,
        rebuild: bool = False,
        upload: bool = True,
    ) -> None:
        """Sweep play-by-play into a league's EPA index, by hand.

        Not a stage of the submitted DAG, for the same reason `game_control`
        isn't: a sweep is an hour of reads, and it earns a node once a
        scheduled model reads what it writes. `models/*/margin_blend.json` is
        the config that would, and it needs this to have run at least once --
        until then `BlendedMarginEloPredictor` holds an empty index and rates
        every game on its final margin alone.

        Idempotent on the two fits rather than on the file: a league whose
        stored index was built by the win probability and expected points
        models this image has re-sweeps only the season still being played,
        which is the weekly case. A league whose index came from a different
        fit -- a retrain of either, or a bumped pin -- is rebuilt from
        scratch, because an index holding two models' numbers is one nobody
        can reproduce.

        `--rebuild` forces the full sweep for a league that is already
        current. Like `game_control`, that is not a decision that re-rates
        anything on its own: the same fits over the same plays produce the
        same numbers, so it costs queue time rather than changing results.
        """
        asyncio.run(_epa(_as_list(league), index, rebuild, upload))

    def qb_out(
        self,
        league: list[str] | str | None = None,
        index: int | None = None,
        upload: bool = True,
        latest: bool = False,
    ) -> None:
        """Sweep play-by-play into a league's quarterback availability index.

        Rebuilt whole every time rather than topped up. Unlike the two sweeps
        above there is no model fit behind this to key idempotency on -- the
        input is a regex over play text -- and the parse is cheap next to the
        read, so "what does the current parser say about every game" is the
        only question worth answering. Merging two parsers' opinions into one
        file is the thing to avoid, and that has already bitten once: a
        format the parser could not read left the 2026 index empty and half
        of 2025 missing, and only a full rebuild fixed it.

        `models/{ncaafb,nfl}/glicko_full.json` and their siblings search
        `qb_out_penalty`, so this has to have run at least once or that
        parameter is a dimension over an empty index -- the search would
        spend probes on a term that is always zero.

        `--latest` is the exception, for the run after a final: it rebuilds
        the season being played and keeps the stored copy's others
        (`qb_out_build.with_season`). Thirteen seconds against minutes --
        the whole rebuild reads 750 weeks of plays to change one season.
        """
        asyncio.run(_qb_out(_as_list(league), index, upload, latest))

    def player_ratings(
        self,
        league: list[str] | str | None = None,
        index: int | None = None,
        today: str | None = None,
    ) -> None:
        """Add today's rows to a basketball league's players' team ratings
        (`cassandra.player_ratings_build`), which `glicko_players` blends in.

        Every morning, before publish: a row stands for the games on or after
        its day, so one that missed a morning leaves the model reading the day
        before's. `--today` refits as of another day (ISO), replacing that
        day's rows.
        """
        asyncio.run(_player_ratings(_as_list(league), index, today))

    def sweeps(self, league: str) -> None:
        """Both play-by-play sweeps a bet reads, EPA then quarterbacks, in one
        container: what a final submits.

        The quarterbacks rebuild only the season being played (`qb_out
        --latest`); the daily publish still rebuilds every season.

        One job rather than two, because the picks run that follows has to
        wait for both, and EventBridge can only start it off one. Two jobs
        also meant two queue waits, and on a busy queue Batch ran them one
        after the other anyway -- on 2026-10-20 the EPA sweep finished seven
        seconds after the picks run had started off the quarterbacks'.
        """
        asyncio.run(_epa([league], None, rebuild=False, upload=True))
        asyncio.run(_qb_out([league], None, upload=True, latest=True))

    def picks(
        self,
        kelly_fraction: float = 0.25,
        max_bet: float = 10.0,
        budget: float = 20.0,
        espn: bool = True,
    ) -> None:
        """Price the Kalshi ncaafb books and email the new picks, or place them
        at bedtime. See `cassandra.picks_job`.

        The sizing flags are the awake run's, and match `make bet`'s defaults:
        with the Kalshi key it sizes `--kelly-fraction` on the real account,
        without it it splits `--budget`. Bedtime brings its own.

        `--espn=False` skips the season's top-up from ESPN (`picks.py
        --no-espn`). The run after a final passes it: the chain that wrote
        that final to s3 finished minutes ago, and the top-up is 54
        scoreboard reads that would only say it again. The hourly runs keep
        it, since they're the ones that catch a final no chain has written.
        """
        _picks(kelly_fraction, max_bet, budget, espn)

    def optimize(
        self,
        index: int | None = None,
        league: str | None = None,
        model: str | None = None,
        download: bool = True,
        upload: bool = True,
        rebuild: bool = False,
    ) -> None:
        """Optimize one config. Defaults to this array child's index.

        `--upload` is on by default because the only reason to run this in a
        container is to get the result somewhere the next stage can read it;
        pass `--upload=False` for a local dry run. `--download=False` likewise
        leaves the anchors already on this disk alone.

        `--rebuild` searches even when nothing the last fit was found against
        has changed, which is otherwise a skip -- see `cassandra.fingerprint`.
        The escape hatch for when the fingerprint is wrong, or for proving it
        isn't.
        """
        asyncio.run(_optimize(index, league, model, download, upload, rebuild))

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
        skip_sweeps: bool = False,
        rebuild_sweeps: bool = False,
        skip_optimize: bool = False,
        rebuild_optimize: bool = False,
        skip_evaluate: bool = False,
        skip_publish: bool = False,
        dry_run: bool = False,
        wait: bool = False,
        job_queue: str | None = None,
        anchors_job_definition: str | None = None,
        game_control_job_definition: str | None = None,
        epa_job_definition: str | None = None,
        qb_out_job_definition: str | None = None,
        player_ratings_job_definition: str | None = None,
        optimize_job_definition: str | None = None,
        evaluate_job_definition: str | None = None,
        publish_job_definition: str | None = None,
    ) -> None:
        """Submit the DAG to Batch and print what went out.

        `--rebuild-anchors` refits the division anchors even though s3
        already has some, and overwrites them. That re-rates every model in
        the league against a new scale, so it is deliberately not what a
        scheduled run does -- see `Jobs.anchors`.

        `--skip-sweeps` drops the two play-by-play jobs for a run whose
        indexes are already current, and `--rebuild-sweeps` forces the full
        historical sweep rather than a refresh of the season in progress.
        Neither changes what a rating means on its own; see `dag.submit`.
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
                epa_job_definition=_job_definition("EPA", epa_job_definition),
                qb_out_job_definition=_job_definition("QB_OUT", qb_out_job_definition),
                # No conventional fallback, unlike the others: until terraform
                # puts this in the launcher's environment there is no such
                # definition, and the run goes ahead without the sweep.
                player_ratings_job_definition=player_ratings_job_definition
                or os.environ.get("CASSANDRA_PLAYER_RATINGS_JOB_DEFINITION"),
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
                skip_sweeps=skip_sweeps,
                rebuild_sweeps=rebuild_sweeps,
                skip_optimize=skip_optimize,
                rebuild_optimize=rebuild_optimize,
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
    """fire hands back a str for one value and a tuple for `--league a,b`.

    Comma-separated, not repeated: `--league a --league b` keeps only `b`,
    silently, which is how a two-league run once came out nfl-only.
    """
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
    conventional names. Underscores become hyphens on the way because the two
    conventions differ: an environment variable is
    `CASSANDRA_GAME_CONTROL_JOB_DEFINITION` and a job definition is
    `cassandra-game-control`. `game_control` is back in the DAG, so the
    conversion is load-bearing again rather than waiting for the next stage
    with a two-word name.
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

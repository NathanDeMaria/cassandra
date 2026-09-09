"""Turn model runs into ModelRelease artifacts the webapp can serve.

    python publish.py                                  # every model, every league
    python publish.py --league mens --league nfl       # some leagues
    python publish.py --config <one>_result.json       # one model
    python publish.py --upload                         # ... and push to s3

Building the artifact and publishing it are different risk levels, so they're
split: this writes a release directory locally by default and only touches s3
behind `--upload`. That means you can read the JSON before anything serves it.

A `--config` is a *result* config -- a `PredictorConfig`, with `target` and
`params` -- which is what `optimize.py` writes to
`~/.cassandra/models/<league>/<name>_result.json`. The checked-in
`models/<league>/*.json` files are optimization configs (parameter ranges and
n_iter) and are not publishable; `_load_config` says so rather than letting
pydantic report two missing fields. With no `--config`, the models are
discovered the same way `evaluate_models.py` discovers them, so a release and
a row in the evaluations csv mean the same thing by the same name.

Nothing here is new capability. `join_with_odds` runs the model,
`score_predictions` fits the prob->margin mapping, and `cassandra.serving`
owns the schema; this is the assembly.

A publish produces four objects, under `models/{league}/{model}/`:
`latest.json` and its copy under `runs/`, plus `history.parquet` (what every
team was rated at the end of every week) and `predictions.parquet` (what the
model said before each game, from before that game updated it). All four
carry one `run_id` and are written -- and uploaded -- together or not at
all, because the webapp reads the stored prediction for a completed game
and predicts live for an unplayed one, and a release without its
predictions silently turns the first case back into hindsight. See
`build_artifacts` and `cassandra.serving.predictions`.

The one judgment call it does make is *when* the offseason rollover lands in
a release, since publishing is the only caller that reads a predictor's
ratings after the replay rather than its predictions during one. See
`rollover_due`.
"""

import asyncio
import json
import sys
from collections.abc import Collection, Iterator, Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from itertools import groupby
from pathlib import Path
from typing import NamedTuple

import fire
import pandas as pd
from endgame.types import Season
from endgame_aws import Config, save_data_to_s3
from pydantic import ValidationError

from cassandra.constants import CASSANDRA_HOME
from cassandra.model_eval import (
    DEFAULT_FITTERS,
    ScoredPredictions,
    score_predictions,
)
from cassandra.odds import OddsDatabase
from cassandra.predictor import (
    Predictor,
    PredictorConfig,
    RatingsUnsupported,
    load_predictor_class,
)
from cassandra.save_predictions import join_with_odds, read_all_seasons
from cassandra.serving import (
    ModelRelease,
    RatingHistory,
    TeamRating,
    TrainedThrough,
    WeekObserver,
    calibration_from_predictor,
    history_bytes,
    history_path,
    metrics_from_scored,
    model_dir,
    predictions_bytes,
    predictions_frame,
    predictions_path,
    ratings_from_predictor,
    tally,
    write_artifact_bytes,
)
from evaluate_models import _models


def _load_config(config_path: Path) -> PredictorConfig:
    try:
        return PredictorConfig.model_validate_json(config_path.read_text())
    except ValidationError as e:
        raise ValueError(
            f"{config_path} is not a predictor config. Publish the *result* of "
            "an optimization run -- the file with `target` and `params`, under "
            f"{CASSANDRA_HOME / 'models'} -- not the optimization config that "
            "produced it."
        ) from e


def _model_name(config_path: Path) -> str:
    """The model's name in the release, from its config's filename.

    `_result` comes off so a release is labeled the same way
    `evaluate_models.py` labels its rows -- `glicko_full`, not
    `glicko_full_result` -- and the two can be lined up by name.
    """
    return config_path.stem.removesuffix("_result")


def _best_fit(df: pd.DataFrame) -> ScoredPredictions:
    """Score every default fitter on the same predictions and keep the best.

    `score_predictions` is cheap relative to the replay that produced `df` --
    its docstring says as much -- so comparing fitters costs a fit apiece
    rather than another walk through the schedule.

    `min`, not `max`: `margin_mae` is a mean absolute error, so lower is
    better. Getting this backwards publishes the worse fit and looks entirely
    plausible doing it, which is why there's a test holding the sign.
    """
    scored = {name: score_predictions(df, f) for name, f in DEFAULT_FITTERS.items()}
    for name, candidate in scored.items():
        print(f"  {name}: margin_mae={candidate.metrics['margin_mae']:.4f}")
    return min(scored.values(), key=lambda s: s.metrics["margin_mae"])


def _with_records(
    ratings: dict[str, TeamRating], df: pd.DataFrame
) -> dict[str, TeamRating]:
    """Fill in win/loss records, which the predictor doesn't have.

    `ratings_from_predictor` leaves these at 0 on purpose -- records come from
    the games, not the model -- and the games are right here.

    Current season only, matching `trained_through.season_year`: a rating sits
    next to its record on the front page, and a cumulative record over every
    season the model ever replayed isn't what that table means. A team the
    predictor rates but that hasn't played this season keeps its 0-0.

    `serving.history.tally` does the counting, and it is the same function
    the week observer runs inside the replay -- including its rule that a
    tie counts for neither side, since a TeamRating has nowhere to put one
    and `team1_win` calling it a home loss is fine for scoring a probability
    and wrong on a standings table. Shared rather than written twice because
    a release's record and the last row of that team's history are the same
    claim in two places, and there is a test holding them equal.
    """
    season = df[df["year"] == df["year"].max()]
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    margins = (season["home_score"] - season["away_score"]).tolist()
    homes = season["home_team"].astype(str).tolist()
    aways = season["away_team"].astype(str).tolist()
    for home, away, margin in zip(homes, aways, margins, strict=True):
        tally(wins, losses, home, away, margin)
    return {
        team: rating.model_copy(
            update={"wins": wins.get(team, 0), "losses": losses.get(team, 0)},
        )
        for team, rating in ratings.items()
    }


def _trained_through(df: pd.DataFrame) -> TrainedThrough:
    """The watermark the incremental refresh resumes from.

    Current season only, per `TrainedThrough` -- it's a resume marker, not an
    audit log. `processed_game_ids` is what makes a re-run exact, since games
    get re-fetched and scores corrected, so a timestamp alone would either
    reprocess or skip the corrections.
    """
    season_year = int(df["year"].max())
    season = df[df["year"] == season_year]
    return TrainedThrough(
        season_year=season_year,
        last_game_date=pd.Timestamp(season["date"].max()).to_pydatetime(),
        processed_game_ids=sorted(season["game_id"].astype(str)),
    )


def build_release(
    config: PredictorConfig,
    model: str,
    predictor: Predictor,
    df: pd.DataFrame,
) -> ModelRelease:
    """Assemble one release from a finished run.

    Takes the predictions and the trained predictor rather than fetching them,
    so the assembly can be tested without replaying a season.
    """
    scored = _best_fit(df)
    # One `now`, two fields: run_id is derived from created_at rather than
    # sampled again, so an artifact can't say it was created at one instant
    # and named for another.
    created_at = datetime.now(UTC)
    return ModelRelease(
        run_id=created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        league=config.league,
        model=model,
        predictor_class=config.predictor_class,
        # Params off the config, not off the predictor: these are exactly what
        # `ModelRelease.rating_predictor` feeds back into the constructor on
        # the other side, and keeping it the same object is what makes the
        # round trip exact.
        params=config.params,
        ratings=_with_records(ratings_from_predictor(predictor), df),
        margin_calibration=calibration_from_predictor(scored.margin_predictor),
        metrics=metrics_from_scored(scored.metrics),
        trained_through=_trained_through(df),
        created_at=created_at,
        created_by="publish",
    )


def release_json(release: ModelRelease) -> str:
    """The exact bytes of the artifact.

    `json.dumps(..., allow_nan=False)` rather than pydantic's
    `model_dump_json`, which writes a nan as the bare token `NaN` -- not valid
    JSON, and rejected by browsers. `metrics_from_scored` already maps the
    nans that are expected (a league the odds database doesn't cover) to null,
    so a nan surviving to here is a bug and should stop the publish rather
    than ship an artifact the API will refuse.
    """
    return json.dumps(release.model_dump(mode="json"), allow_nan=False, indent=2)


def write_release(release: ModelRelease, out: Path) -> tuple[Path, Path]:
    """Write the release into the bucket's layout, rooted at `out`."""
    payload = release_json(release)
    directory = model_dir(out, release.league, release.model)
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    run_path = directory / "runs" / f"{release.run_id}.json"
    latest_path = directory / "latest.json"
    run_path.write_text(payload)
    # A copy of the run file, not a pointer to it: the API serves latest.json
    # with one GET, and rolling back is `cp runs/<old>.json latest.json`.
    latest_path.write_text(payload)
    return run_path, latest_path


class Artifacts(NamedTuple):
    """One run's three artifacts, serialized, before any of them is stored.

    Bytes rather than frames because the same object goes to two places --
    a local directory and an s3 key -- and serializing twice would leave
    nothing guaranteeing they match. The release is carried whole because
    `write_release` and `upload_release` want the id and the layout off it.

    Built in one step (`build_artifacts`) and stored in another, which is
    what makes "all three or none" checkable: a schema problem raises
    before the first byte lands anywhere.
    """

    release: ModelRelease
    history: bytes
    predictions: bytes


def build_artifacts(
    release: ModelRelease, history: RatingHistory, df: pd.DataFrame
) -> Artifacts:
    """Serialize a finished run's three artifacts, or raise having written none.

    The webapp's honesty rule depends on the three being one set: a
    completed game shows the stored pre-game prediction and an unplayed one
    is predicted live from the release, so a release published without its
    predictions leaves the consumer re-predicting games the model has
    already trained on. See `cassandra.serving.predictions`.
    """
    margin_predictor = release.margin_predictor()
    if margin_predictor is None:
        # `build_release` always fits one, so this is a can't-happen guard
        # rather than a case. Raising keeps it that way: predictions.parquet
        # stores `predicted_margin` from the run's own calibration, and a
        # release with no calibration has no honest value to put there.
        raise ValueError(
            f"{release.league}/{release.model} has no margin calibration; "
            "refusing to publish a release without its predictions"
        )
    return Artifacts(
        release=release,
        history=history_bytes(history.frame(release.run_id)),
        predictions=predictions_bytes(
            predictions_frame(df, release.run_id, margin_predictor)
        ),
    )


# The order the four objects are stored in, on disk and in the bucket
# alike, and the one thing that makes a half-finished publish harmless.
#
# There is no transaction to be had over four objects, so the ordering
# stands in for one. The parquet artifacts go first and `latest.json` goes
# last, because `latest.json` is the pointer the API reads: stopping
# part-way leaves a history and a set of predictions that nothing is
# pointing at yet, which is inert. The reverse order leaves a live release
# whose completed games have no stored forecast, and a consumer following
# the "completed game -> stored prediction" rule would quietly fall back to
# re-predicting games the model has memorized. That is the failure this
# whole pairing exists to prevent.
#
# `runs/<id>.json` sits between them for the reason `upload_release`
# already gives: the archive copy before the served one, so there is never
# a release being handed out with nothing to roll back to.


def write_artifacts(artifacts: Artifacts, out: Path) -> tuple[Path, Path, Path, Path]:
    """Write a run's artifacts into the bucket's layout, rooted at `out`.

    Each parquet is written whole, replacing what was there: a full replay
    walks every season and every game, so it is the only writer that can
    rewrite them. `upsert_history` / `upsert_predictions` are what the
    incremental refresh uses instead.
    """
    release = artifacts.release
    history_file = history_path(out, release.league, release.model)
    predictions_file = predictions_path(out, release.league, release.model)
    write_artifact_bytes(artifacts.history, history_file)
    write_artifact_bytes(artifacts.predictions, predictions_file)
    print(f"  wrote {history_file}\n  wrote {predictions_file}")

    run_path, latest_path = write_release(release, out)
    print(f"  wrote {run_path}\n  wrote {latest_path}")
    return history_file, predictions_file, run_path, latest_path


async def upload_artifacts(artifacts: Artifacts, bucket: str) -> tuple[str, ...]:
    """Put the same four objects in s3, in the same order, under the same layout.

    The same `bytes` that went to disk, not a second serialization of the
    same frames -- so reading the local artifact really is reading what got
    served.
    """
    release = artifacts.release
    prefix = f"models/{release.league}/{release.model}"
    history_key = f"{prefix}/history.parquet"
    predictions_key = f"{prefix}/predictions.parquet"
    await save_data_to_s3(bucket, history_key, artifacts.history)
    await save_data_to_s3(bucket, predictions_key, artifacts.predictions)
    run_key, latest_key = await upload_release(release, bucket)
    return history_key, predictions_key, run_key, latest_key


async def upload_release(release: ModelRelease, bucket: str) -> tuple[str, str]:
    """Put the same two objects in s3, under the same layout."""
    payload = release_json(release).encode()
    prefix = f"models/{release.league}/{release.model}"
    run_key = f"{prefix}/runs/{release.run_id}.json"
    latest_key = f"{prefix}/latest.json"
    # Archive first, then serve. The other order leaves a window where the
    # release the API is handing out has no copy under runs/ to roll back to.
    await save_data_to_s3(bucket, run_key, payload)
    await save_data_to_s3(bucket, latest_key, payload)
    return run_key, latest_key


class _Job(NamedTuple):
    """One model to publish, with its config already parsed.

    Configs are read up front so a typo in one of twenty fails before the
    first minute of s3 reads rather than after it.
    """

    league: str
    model: str
    config: PredictorConfig


def _job(config_path: Path, league: str | None = None) -> _Job:
    config = _load_config(config_path)
    if league is not None and config.league != league:
        # evaluate_models.py labels a model by its directory, so a config that
        # disagrees would be published under one league and scored under
        # another. `manifest._all_work` makes the same check for the configs
        # the optimize array is built from.
        raise ValueError(
            f"{config_path}: league {config.league!r} does not match directory "
            f"{league!r}"
        )
    return _Job(config.league, _model_name(config_path), config)


def _jobs(config: str | None, leagues: Collection[str]) -> list[_Job]:
    """What to publish: one named config, or everything discovered."""
    if config is not None:
        return [_job(Path(config))]
    return [_job(path, league) for league, _, path in _models(leagues)]


def _as_leagues(league: str | Sequence[str] | None) -> list[str]:
    """`--league mens`, `--league mens,nfl`, or neither.

    fire hands over a bare string for one and a tuple for a comma-separated
    list, and `_models` wants a collection either way.
    """
    if league is None:
        return []
    return [league] if isinstance(league, str) else list(league)


# How long after the last game the offseason rollover waits. A month, so the
# release people read for the weeks right after a title game still says where
# the season left the teams.
ROLLOVER_DELAY = timedelta(days=30)


def rollover_due(df: pd.DataFrame, now: datetime) -> bool:
    """Whether this release should carry the offseason regression yet.

    The rollover -- `pass_season`, regression toward each team's anchor plus
    Glicko's season rd bump -- is a prior on a season nobody has played. It
    belongs in the ratings that predict that season's first game, not in the
    ratings someone reads the week after the last one, where it silently
    pulls the champion back toward the field.

    "The season ended" is read off the games rather than a calendar: the last
    game in the data is the last game there was. In season that date is days
    old and this is False, which is what keeps a mid-season publish from
    regressing everybody every week -- so the same rule covers both without a
    separate "is the season over" flag to keep in sync.
    """
    last_game = pd.Timestamp(df["date"].max())
    # Season pickles carry naive datetimes in some leagues and aware ones in
    # others; comparing the wrong pair raises rather than misjudging by a day.
    if last_game.tzinfo is None:
        last_game = last_game.tz_localize(UTC)
    return now - last_game.to_pydatetime() >= ROLLOVER_DELAY


def _predictions(
    job: _Job,
    seasons: Sequence[Season],
    odds_db: OddsDatabase,
    now: datetime | None = None,
    week_observer: WeekObserver | None = None,
) -> tuple[Predictor, pd.DataFrame]:
    """Replay one model over already-loaded seasons.

    `join_with_odds` rather than `get_predictions`, which reads the seasons
    and the whole odds database itself on every call -- about a minute of s3
    per model, and the point of publishing every model in one process is to
    pay that once. It also hands back the trained predictor directly, so the
    ratings don't have to come back off the state file.

    The replay stops short of the final `pass_season` and this decides
    whether to apply it, because publishing is the one caller that reads the
    ratings afterward. See `rollover_due`.
    """
    predictor_class = load_predictor_class(job.config.predictor_class)
    predictor = predictor_class(job.league, **job.config.params)  # type: ignore[call-arg]
    predictions = join_with_odds(
        predictor,
        seasons,
        odds_db,
        post_callbacks=False,
        roll_over_final_season=False,
        week_observer=week_observer,
    )
    df = pd.DataFrame([asdict(p) for p in predictions])
    # An empty frame has no `date` to ask about. Nothing to roll over either,
    # and `score_predictions` is about to say "No games to score" -- a better
    # message than whatever a max() over nothing would produce here.
    if not df.empty:
        rolled = rollover_due(df, now or datetime.now(UTC))
        if rolled:
            predictor.pass_season()
        # Said out loud either way: two runs a day apart can publish
        # different ratings off identical games, and this is the only thing
        # that explains it.
        print(f"  offseason rollover: {'applied' if rolled else 'not yet'}")
    return predictor, df


async def _publish_one(
    job: _Job,
    seasons: Sequence[Season],
    odds_db: OddsDatabase,
    out: Path,
    upload: bool,
    bucket: str,
) -> None:
    print(f"=== {job.league}/{job.model} ({job.config.predictor_class}) ===")
    history = RatingHistory()
    predictor, df = _predictions(job, seasons, odds_db, week_observer=history)
    # Written for parity with evaluate_models.py, which leaves the same file
    # for the same config. Nothing here reads it back -- the predictor is
    # already in hand -- so it's an artifact of the run, not a step in it.
    state_path = CASSANDRA_HOME / "models" / job.league / f"{job.model}_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_state(state_path)

    release = build_release(job.config, job.model, predictor, df)
    kind = release.margin_calibration.kind if release.margin_calibration else "none"
    print(
        f"  chose {kind}, {len(release.ratings)} teams, "
        f"through {release.trained_through.season_year} "
        f"({len(release.trained_through.processed_game_ids)} games this season)"
    )
    artifacts = build_artifacts(release, history, df)
    write_artifacts(artifacts, out)

    if not upload:
        return
    for key in await upload_artifacts(artifacts, bucket):
        print(f"  uploaded s3://{bucket}/{key}")


def _by_league(jobs: Sequence[_Job]) -> Iterator[tuple[str, list[_Job]]]:
    ordered = sorted(jobs, key=lambda j: (j.league, j.model))
    for league, group in groupby(ordered, key=lambda j: j.league):
        yield league, list(group)


async def _publish(
    config: str | None,
    leagues: Collection[str],
    out: Path,
    upload: bool,
    # TODO: different!!
    upload_bucket: str = "invisible-string-artifacts-080353813015",
) -> list[str]:
    jobs = _jobs(config, leagues)
    if not jobs:
        raise ValueError(
            f"No models to publish. Optimized results land under "
            f"{CASSANDRA_HOME / 'models'}; run `make submit` first, or "
            "`jobs.py publish` to pull them from s3."
        )

    bucket = Config.init_from_file().bucket
    # Read once for the whole run: the odds database covers every league, and
    # it's the single most expensive read here.
    print(f"Loading odds from s3://{bucket}")
    odds_db = await OddsDatabase.from_s3(bucket)

    failures = []
    for league, league_jobs in _by_league(jobs):
        print(f"Loading {league} seasons from s3://{bucket}")
        seasons = [s async for s in read_all_seasons(league, bucket)]
        if not seasons:
            # Same failure optimize.py guards: every model in the league would
            # otherwise die deep inside scoring on an empty set of games.
            print(f"  FAILED {league}: no seasons in s3://{bucket}/seasons/")
            failures.extend(f"{league}/{job.model}" for job in league_jobs)
            continue
        for job in league_jobs:
            try:
                await _publish_one(job, seasons, odds_db, out, upload, upload_bucket)
            except RatingsUnsupported:
                # Not a failure: a release is a table of team ratings, and
                # FlatPredictor deliberately has none. It's a checked-in
                # baseline in every league, so counting it would leave
                # `jobs.py publish` exiting 1 on every league on every run --
                # a red stage that can never go green, with nothing to fix.
                print(f"  skipped {job.league}/{job.model}: no team ratings")
            except Exception as e:
                # One bad model shouldn't cost the other nineteen their
                # releases, the same way one unscoreable model doesn't cost
                # evaluate_models.py its table.
                print(f"  FAILED {job.league}/{job.model}: {e!r}")
                failures.append(f"{job.league}/{job.model}")
    return failures


def main(
    config: str | None = None,
    league: str | Sequence[str] | None = None,
    out: str = "./releases",
    upload: bool = False,
) -> None:
    failures = asyncio.run(
        _publish(config, _as_leagues(league), Path(out), upload=upload)
    )
    if not upload:
        print("\nNothing uploaded; pass --upload to push to s3.")
    if failures:
        print(f"\n{len(failures)} failed:")
        print("\n".join(f"  {failure}" for failure in failures))
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)

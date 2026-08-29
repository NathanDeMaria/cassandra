"""Tests for publish.py.

Seasons and odds are hand-written here rather than read from s3, which is the
one expensive part of a real run. Everything downstream of the read is real:
the replay, the fit, the assembly, and the bytes on disk.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from endgame.types import Game, Season, Week

import publish
from cassandra import score_predictions
from cassandra.odds import OddsDatabase
from cassandra.predictor import (
    Predictor,
    PredictorConfig,
    predict_matchup,
)
from cassandra.predictor.base_predictor import MEAN_RATING
from cassandra.prob_to_margin import (
    BaseProbToMarginFitter,
    BaseProbToMarginPredictor,
    LogisticProbToMarginPredictor,
)
from cassandra.serving import LogisticMarginCalibration, Metrics, ModelRelease
from publish import build_release, release_json, write_release

# A league nobody has run, so no opponent priors on disk get picked up.
_LEAGUE = "test_league"
_MODEL = "glicko_tuned"

# Two seasons, so anything that should be scoped to the current one can be
# caught scooping up the older games too. In 2026 Team A goes 2-1, Team B and
# Team C go 1-1 and Team D goes 0-1; across both seasons Team A would be 3-2.
_SCHEDULE: dict[int, list[list[tuple[str, str, int, int]]]] = {
    2025: [
        [("Team A", "Team B", 80, 60)],
        [("Team C", "Team A", 70, 60)],
    ],
    2026: [
        [("Team A", "Team B", 80, 61), ("Team C", "Team D", 55, 50)],
        [("Team B", "Team A", 70, 65)],
        [("Team A", "Team C", 90, 51)],
    ],
}


def _seasons() -> list[Season]:
    """`_SCHEDULE` as Seasons, one week per list.

    Each week gets its own day because `iter_weeks` refuses a season whose
    weeks overlap in time.
    """
    seasons = []
    for year, weeks in _SCHEDULE.items():
        season_weeks = []
        for number, games in enumerate(weeks, start=1):
            date = datetime(year, 1, number, tzinfo=timezone.utc)
            season_weeks.append(
                Week(
                    number=number,
                    games=[
                        Game(
                            home=home,
                            away=away,
                            home_score=home_score,
                            away_score=away_score,
                            neutral_site=False,
                            completed=True,
                            date=date,
                            game_id=f"{year}-{number}-{away}@{home}",
                        )
                        for home, away, home_score, away_score in games
                    ],
                )
            )
        seasons.append(Season(year=year, weeks=season_weeks))
    return seasons


def _config(
    predictor_class: str = "EloPredictor",
    league: str = _LEAGUE,
    **params: float | str,
) -> PredictorConfig:
    return PredictorConfig(
        predictor_class=predictor_class,
        league=league,
        target=-0.25,
        params=params or {"home_advantage": 88.0, "k": 31.0},
    )


# The last game in `_SCHEDULE`, which is what publish measures the offseason
# from.
_LAST_GAME = datetime(2026, 1, 3, tzinfo=timezone.utc)


def _run(
    config: PredictorConfig, now: datetime = _LAST_GAME + timedelta(days=365)
) -> tuple[Predictor, pd.DataFrame]:
    """Replay the schedule the way publishing does, through `_predictions`.

    The real function rather than a copy of it: everything expensive about a
    publish is the s3 read that produced `seasons`, and passing hand-written
    ones covers the rest of the path -- including the offseason rollover,
    which `_predictions` and nothing else decides.

    `now` defaults to long after the schedule ends, so a test that isn't
    about the rollover gets the settled ratings and doesn't have to say so.
    """
    return publish._predictions(
        publish._Job(config.league, _MODEL, config),
        _seasons(),
        OddsDatabase({}),
        now=now,
    )


def _published(config: PredictorConfig, out: Path) -> ModelRelease:
    """A release that has been through the filesystem, as a consumer sees it."""
    predictor, df = _run(config)
    release = build_release(config, _MODEL, predictor, df)
    _, latest_path = write_release(release, out)
    return ModelRelease.model_validate(json.loads(latest_path.read_text()))


_ROUND_TRIP_CONFIGS = [
    _config("EloPredictor", home_advantage=88.0, k=31.0),
    _config("Elo538Predictor", home_advantage=64.0, k=12.0),
    _config(
        "GlickoPredictor",
        home_advantage=88.0,
        k=61.0,
        initial_rd=205.0,
        scoring_method="binary",
    ),
]


@pytest.mark.parametrize("config", _ROUND_TRIP_CONFIGS, ids=lambda c: c.predictor_class)
def test_a_published_release_predicts_what_the_model_predicted(
    config: PredictorConfig, tmp_path: Path
) -> None:
    """The whole point of the artifact: same matchup, same number.

    Asserted on probabilities rather than on fields, because the fields can
    match while the answer is wrong -- a home advantage dropped between the
    config and the release doesn't show up in the ratings at all.
    """
    original, _ = _run(config)

    rebuilt = _published(config, tmp_path).rating_predictor()

    for home, away in (("Team A", "Team B"), ("Team C", "Team A")):
        for neutral_site in (False, True):
            assert predict_matchup(
                rebuilt, home, away, neutral_site=neutral_site
            ).team1_win_prob == pytest.approx(
                predict_matchup(
                    original, home, away, neutral_site=neutral_site
                ).team1_win_prob
            )


class _FixedScaleFitter(BaseProbToMarginFitter):
    """Ignores the games and returns the mapping it was handed."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def fit(
        self, win_probs: np.ndarray, margins: np.ndarray
    ) -> BaseProbToMarginPredictor:
        return LogisticProbToMarginPredictor(self.scale)


def test_the_fitter_with_the_lower_margin_mae_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The min/max guard.

    `margin_mae` is an error measure, so the better fit is the smaller number.
    A `max` here would publish the worse mapping and look entirely reasonable
    doing it: the artifact would validate, serve, and quietly give worse
    margins. So the sign gets its own test, with two fitters whose errors are
    known -- one calling every game level, one roughly the size a real fit
    lands on.
    """
    config = _config()
    _, df = _run(config)

    fitters = {"flat": _FixedScaleFitter(0.0), "sloped": _FixedScaleFitter(11.0)}
    errors = {
        name: score_predictions(df, fitter).metrics["margin_mae"]
        for name, fitter in fitters.items()
    }
    best, worst = sorted(errors, key=lambda name: errors[name])
    assert errors[best] < errors[worst], "the two fitters have to be distinguishable"

    monkeypatch.setattr(publish, "DEFAULT_FITTERS", fitters)
    release = _published(config, tmp_path)

    calibration = release.margin_calibration
    assert isinstance(calibration, LogisticMarginCalibration)
    assert calibration.scale == pytest.approx(fitters[best].scale)
    assert release.metrics.margin_mae == pytest.approx(errors[best])


def test_records_come_from_this_season_only(tmp_path: Path) -> None:
    """The hole `ratings_from_predictor` deliberately leaves, filled in.

    Records belong to the games, not the model, so they arrive at 0-0 and
    publish fills them. Scoped to the current season: across both seasons Team
    A is 3-2, and a release saying so has quietly put a career record next to
    a current rating.
    """
    release = _published(_config(), tmp_path)

    assert {
        team: (rating.wins, rating.losses) for team, rating in release.ratings.items()
    } == {
        "Team A": (2, 1),
        "Team B": (1, 1),
        "Team C": (1, 1),
        "Team D": (0, 1),
    }


def test_the_offseason_regression_waits_a_month_after_the_last_game() -> None:
    """A release published the week after a title game says where teams ended.

    `pass_season` regresses every rating toward its anchor -- a prior on a
    season nobody has played. Applying it the moment the replay runs out of
    games quietly pulls the champion back toward the field in the release
    people read all offseason, so publish holds it for a month.
    """
    config = _config("EloPredictor", home_advantage=88.0, k=31.0, season_regression=0.5)

    fresh, _ = _run(config, now=_LAST_GAME + timedelta(days=29))
    rolled, _ = _run(config, now=_LAST_GAME + timedelta(days=30))

    # Otherwise a model whose ratings all sat at the mean would pass this.
    assert any(r.rating != MEAN_RATING for r in fresh.ratings.values())
    assert {t: r.rating for t, r in rolled.ratings.items()} == pytest.approx(
        {
            team: MEAN_RATING + 0.5 * (rating.rating - MEAN_RATING)
            for team, rating in fresh.ratings.items()
        }
    )


def test_the_offseason_clock_starts_at_the_last_game() -> None:
    """Measured off the games, not a calendar, and read as UTC when naive.

    Season pickles carry both. A naive date compared against an aware `now`
    raises rather than being off by a day, so this is the guard that keeps a
    league's publish from failing on the timezone its games were stored in.
    """
    naive = pd.DataFrame([{"date": _LAST_GAME.replace(tzinfo=None)}])
    aware = pd.DataFrame([{"date": _LAST_GAME}])

    for df in (naive, aware):
        assert not publish.rollover_due(df, _LAST_GAME + timedelta(days=29))
        assert publish.rollover_due(df, _LAST_GAME + timedelta(days=31))


def test_latest_is_a_copy_of_the_run_it_names(tmp_path: Path) -> None:
    """The layout the API reads, and the one that makes rollback a copy.

    latest.json holds the run file's bytes rather than pointing at it, so
    serving a release is one GET and going back to an older one is
    `aws s3 cp runs/<old>.json latest.json`.
    """
    config = _config()
    predictor, df = _run(config)
    release = build_release(config, _MODEL, predictor, df)

    run_path, latest_path = write_release(release, tmp_path)

    model_dir = tmp_path / "models" / _LEAGUE / _MODEL
    assert run_path == model_dir / "runs" / f"{release.run_id}.json"
    assert latest_path == model_dir / "latest.json"
    assert latest_path.read_bytes() == run_path.read_bytes()


def test_the_artifact_is_valid_json_and_still_a_release(tmp_path: Path) -> None:
    """What gets written parses back into the schema it came from.

    This league has no odds at all, so `score_predictions` returns nan for
    every betting metric -- the case `metrics_from_scored` maps to null, and
    the path where a nan would otherwise reach the file.
    """
    config = _config()
    predictor, df = _run(config)
    release = build_release(config, _MODEL, predictor, df)

    _, latest_path = write_release(release, tmp_path)
    written = json.loads(latest_path.read_text())

    assert ModelRelease.model_validate(written) == release
    assert written["metrics"]["against_spread_accuracy"] is None
    assert written["trained_through"]["season_year"] == 2026
    # Current season only: the 2025 games are trained on but not watermarked,
    # because this is a resume marker for the refresh job, not an audit log.
    assert written["trained_through"]["processed_game_ids"] == [
        "2026-1-Team B@Team A",
        "2026-1-Team D@Team C",
        "2026-2-Team A@Team B",
        "2026-3-Team C@Team A",
    ]
    assert written["trained_through"]["last_game_date"].startswith("2026-01-03")


def test_a_nan_stops_the_publish_instead_of_being_written() -> None:
    """A nan in the artifact is invalid JSON, and browsers reject it.

    `json.dumps` writes one as the bare token `NaN` unless told not to, which
    is the failure that would reach a bucket before anything complained.
    Everything expected to be nan is already null by this point, so raising is
    the right answer rather than a repair.
    """
    config = _config()
    predictor, df = _run(config)
    release = build_release(config, _MODEL, predictor, df)

    release_json(release)  # the real one serializes fine

    broken = release.model_copy(
        update={"metrics": Metrics(brier_score=float("nan"), margin_mae=9.1, n_games=7)}
    )
    with pytest.raises(ValueError, match="not JSON compliant"):
        release_json(broken)


class _Bucket:
    """Stands in for everything `_publish` reads out of s3.

    Counts what it was asked for, because reading the odds database once and
    each league's seasons once is the whole reason publishing every model
    happens in one process: separately, those reads run about a minute per
    model.
    """

    bucket = "a-bucket"

    def __init__(self) -> None:
        self.odds_reads = 0
        self.season_reads: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
        monkeypatch.setattr(publish, "CASSANDRA_HOME", home)
        monkeypatch.setattr(
            publish, "Config", type("Config", (), {"init_from_file": lambda: self})
        )
        monkeypatch.setattr(publish, "OddsDatabase", self)
        monkeypatch.setattr(publish, "read_all_seasons", self._read_all_seasons)

    async def from_s3(self, bucket: str) -> OddsDatabase:
        assert bucket == self.bucket
        self.odds_reads += 1
        return OddsDatabase({})

    async def _read_all_seasons(self, league: str, bucket: str):
        assert bucket == self.bucket
        self.season_reads.append(league)
        for season in _seasons():
            yield season


def _write_configs(root: Path, configs: dict[tuple[str, str], PredictorConfig]) -> list:
    """`configs` on disk, and the `_models` listing that finds them."""
    listing = []
    for (league, model), config in configs.items():
        path = root / league / f"{model}_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config.model_dump_json())
        listing.append((league, model, path))
    return listing


def test_publishing_everything_reads_each_league_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default: every model, every league, one process, one set of reads.

    Two models in one league and one in another, so a per-model season read
    would show up as four reads instead of two, and a per-model odds read as
    three instead of one.
    """
    configs = {
        ("mens", "elo"): _config("EloPredictor", league="mens"),
        ("mens", "glicko_full"): _config(
            "GlickoPredictor", league="mens", home_advantage=88.0, initial_rd=205.0
        ),
        ("nfl", "elo"): _config("EloPredictor", league="nfl"),
    }
    listing = _write_configs(tmp_path / "configs", configs)
    monkeypatch.setattr(publish, "_models", lambda leagues: listing)
    bucket = _Bucket()
    bucket.install(monkeypatch, tmp_path / "home")

    out = tmp_path / "releases"
    failures = asyncio.run(publish._publish(None, [], out, upload=False))

    assert failures == []
    assert bucket.odds_reads == 1
    assert bucket.season_reads == ["mens", "nfl"]
    for league, model in configs:
        release = ModelRelease.model_validate_json(
            (out / "models" / league / model / "latest.json").read_text()
        )
        assert (release.league, release.model) == (league, model)
        assert release.ratings, "a published release should rate somebody"


def test_one_bad_model_doesnt_cost_the_others_their_releases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model that can't be built is a failure, and only its own.

    The run keeps going and every other model still gets its release; the
    failure comes back in the list so `jobs.py publish` can exit non-zero
    rather than calling a half-finished stage green.
    """
    configs = {
        ("mens", "busted"): _config("NoSuchPredictor", league="mens"),
        ("mens", "elo"): _config("EloPredictor", league="mens"),
    }
    listing = _write_configs(tmp_path / "configs", configs)
    monkeypatch.setattr(publish, "_models", lambda leagues: listing)
    _Bucket().install(monkeypatch, tmp_path / "home")

    out = tmp_path / "releases"
    failures = asyncio.run(publish._publish(None, [], out, upload=False))

    assert failures == ["mens/busted"]
    assert not (out / "models" / "mens" / "busted").exists()
    assert (out / "models" / "mens" / "elo" / "latest.json").exists()


def test_a_model_with_no_ratings_is_skipped_rather_than_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A flat model has no ratings to publish, and that isn't a failure.

    It's a checked-in baseline in every league, so it's in the listing and
    will be reached on every run. Counting it as a failure would leave
    `jobs.py publish` exiting 1 for every league forever -- a red stage with
    nothing to fix -- so it comes back out of `failures` while still
    producing no release of its own.
    """
    configs = {
        # Built here rather than through `_config`, whose default params are
        # Elo's: FlatPredictor takes none, and passing them raises a TypeError
        # in the constructor -- which would make this pass without ever
        # reaching the ratings call it exists to cover.
        ("mens", "flat"): PredictorConfig(
            predictor_class="FlatPredictor", league="mens", target=0.0, params={}
        ),
        ("mens", "elo"): _config("EloPredictor", league="mens"),
    }
    listing = _write_configs(tmp_path / "configs", configs)
    monkeypatch.setattr(publish, "_models", lambda leagues: listing)
    _Bucket().install(monkeypatch, tmp_path / "home")

    out = tmp_path / "releases"
    failures = asyncio.run(publish._publish(None, [], out, upload=False))

    assert failures == []
    assert not (out / "models" / "mens" / "flat").exists()
    assert (out / "models" / "mens" / "elo" / "latest.json").exists()


def test_a_named_config_publishes_only_that_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--config` skips discovery rather than filtering it."""
    configs = {("mens", "elo"): _config("EloPredictor", league="mens")}
    [(_, _, config_path)] = _write_configs(tmp_path / "configs", configs)

    def _no_discovery(leagues):
        raise AssertionError("--config should not go looking for other models")

    monkeypatch.setattr(publish, "_models", _no_discovery)
    _Bucket().install(monkeypatch, tmp_path / "home")

    out = tmp_path / "releases"
    failures = asyncio.run(publish._publish(str(config_path), [], out, upload=False))

    assert failures == []
    assert (out / "models" / "mens" / "elo" / "latest.json").exists()


def test_the_league_flag_takes_one_or_several() -> None:
    """fire hands over a string for one and a tuple for a comma-separated list."""
    assert publish._as_leagues(None) == []
    assert publish._as_leagues("mens") == ["mens"]
    assert publish._as_leagues(("mens", "nfl")) == ["mens", "nfl"]


def test_a_config_in_the_wrong_leagues_directory_is_refused(tmp_path: Path) -> None:
    """Directory and config have to agree on the league.

    `evaluate_models.py` labels a model by its directory, so a config that
    disagrees would be published under one league and scored under another.
    """
    config_path = tmp_path / "elo_result.json"
    config_path.write_text(_config("EloPredictor", league="nfl").model_dump_json())

    assert publish._job(config_path, "nfl").league == "nfl"
    with pytest.raises(ValueError, match="does not match directory"):
        publish._job(config_path, "mens")


def test_upload_archives_the_run_before_pointing_latest_at_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bucket gets the same two objects, in the order that stays rollable.

    latest.json is what the API serves, so writing it before its copy under
    runs/ opens a window where the release being handed out has nothing to
    roll back to.
    """
    config = _config()
    predictor, df = _run(config)
    release = build_release(config, _MODEL, predictor, df)
    put: list[tuple[str, str, bytes]] = []

    async def _fake_save(bucket: str, key: str, data: bytes) -> None:
        put.append((bucket, key, data))

    monkeypatch.setattr(publish, "save_data_to_s3", _fake_save)

    run_key, latest_key = asyncio.run(publish.upload_release(release, "a-bucket"))

    prefix = f"models/{_LEAGUE}/{_MODEL}"
    assert [(bucket, key) for bucket, key, _ in put] == [
        ("a-bucket", f"{prefix}/runs/{release.run_id}.json"),
        ("a-bucket", f"{prefix}/latest.json"),
    ]
    assert (run_key, latest_key) == (put[0][1], put[1][1])
    # Same bytes in both places, and the same bytes write_release would leave
    # on disk, so a local look at the artifact is a look at what got served.
    assert put[0][2] == put[1][2] == release_json(release).encode()


def test_an_optimization_config_is_refused_by_name(tmp_path: Path) -> None:
    """The mistake the two config shapes invite.

    `models/<league>/*.json` are optimization configs -- parameter ranges and
    n_iter -- and look enough like a model config to reach for. Pydantic would
    report two missing fields; this says which file to use instead.
    """
    config_path = tmp_path / "glicko_full.json"
    config_path.write_text(
        json.dumps(
            {
                "predictor_class": "GlickoPredictor",
                "league": _LEAGUE,
                "parameters": {"k": [1, 79]},
                "n_iter": 60,
            }
        )
    )

    with pytest.raises(ValueError, match="not a predictor config"):
        publish._load_config(config_path)


def test_the_model_name_drops_the_result_suffix() -> None:
    """So a release lines up by name with the rows in the evaluations csv."""
    assert publish._model_name(Path("models/mens/glicko_full_result.json")) == (
        "glicko_full"
    )
    assert publish._model_name(Path("models/mens/flat_result.json")) == "flat"

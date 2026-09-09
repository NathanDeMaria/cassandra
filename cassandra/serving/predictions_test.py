"""Tests for the stored predictions.

The property that matters most here isn't a schema detail: it's that a row
holds the forecast the model made *before* the game updated it. Everything
downstream -- the webapp dropping its hindsight dagger, an honest ATS
record -- rests on that, and it is invisible in the data. A row recomputed
after the fact looks exactly like a real one, only better.
"""

import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from call_it_what_you_want import TeamNamer
from endgame.types import Game, Season, Week

from cassandra.odds import Odds, OddsDatabase
from cassandra.predictor import EloPredictor
from cassandra.prob_to_margin import (
    BaseProbToMarginPredictor,
    LogisticProbToMarginPredictor,
)
from cassandra.save_predictions import join_with_odds

from .predictions import (
    PREDICTION_COLUMNS,
    predictions_frame,
    predictions_path,
    read_predictions,
    upsert_predictions,
    write_predictions,
)

_LEAGUE = "test_league"
_RUN = "2026-09-09T00:00:00Z"


def _game(
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    week: int,
    neutral_site: bool = False,
    year: int = 2026,
) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=neutral_site,
        completed=True,
        date=datetime(year, 3, week, tzinfo=timezone.utc),
        game_id=f"{year}-{week}-{away}@{home}",
        status="STATUS_FINAL",
    )


def _season(year: int, *weeks: list[Game]) -> Season:
    return Season(
        year=year,
        weeks=[Week(games=games, number=n) for n, games in enumerate(weeks, start=1)],
    )


def _frame(
    seasons: list[Season],
    odds: dict[str, Odds] | None = None,
    margin_predictor: BaseProbToMarginPredictor | None = None,
    run_id: str = _RUN,
) -> pd.DataFrame:
    predictor = EloPredictor(_LEAGUE)
    predictions = list(
        join_with_odds(
            predictor,
            seasons,
            OddsDatabase(odds or {}),
            post_callbacks=False,
            roll_over_final_season=False,
        )
    )
    # `join_with_odds` doesn't take a namer; the league has no registry, so
    # nothing is renamed either way. Asserted rather than assumed, since the
    # whole point of these keys is that they match a release's.
    assert TeamNamer.for_league(_LEAGUE).apply(seasons[0].weeks[0].games[0]).home
    return predictions_frame(
        pd.DataFrame([asdict(p) for p in predictions]),
        run_id,
        margin_predictor or LogisticProbToMarginPredictor(25.0),
    )


def _lopsided_season() -> Season:
    """Team A wins four straight, then loses the fifth.

    By week 5 Elo has Team A as a heavy favorite, so the upset is the case
    where a pre-game forecast and one recomputed afterwards are far apart
    and easy to tell from each other.
    """
    return _season(
        2026,
        *[[_game("Team A", "Team B", 90, 40, week)] for week in range(1, 5)],
        [_game("Team A", "Team B", 40, 90, 5)],
    )


def test_a_row_is_the_forecast_from_before_the_game() -> None:
    """The upset keeps the confidence it was given, not one hindsight fixed."""
    frame = _frame([_lopsided_season()]).sort_values("week")
    upset = frame.iloc[4]
    before = frame.iloc[3]

    assert (int(upset["home_score"]), int(upset["away_score"])) == (40, 90)
    # Still the favorite, by a lot, in the row for the game it lost.
    assert float(upset["team1_win_prob"]) > 0.7
    assert float(upset["predicted_margin"]) > 0
    # And it really was the pre-game number: the four wins before it moved
    # the probability up, so a post-update value would be *lower* than the
    # week-4 row rather than higher.
    assert float(upset["team1_win_prob"]) > float(before["team1_win_prob"])


def test_every_played_game_gets_exactly_one_row() -> None:
    frame = _frame([_lopsided_season()])

    assert list(frame.columns) == list(PREDICTION_COLUMNS)
    assert len(frame) == 5
    assert frame["game_id"].is_unique
    assert (frame["run_id"] == _RUN).all()


def test_predicted_margin_comes_from_the_run_s_calibration() -> None:
    """Stored rather than re-derived, because the fit changes every run."""
    seasons = [_lopsided_season()]

    gentle = _frame(seasons, margin_predictor=LogisticProbToMarginPredictor(10.0))
    steep = _frame(seasons, margin_predictor=LogisticProbToMarginPredictor(40.0))

    assert (gentle["team1_win_prob"] == steep["team1_win_prob"]).all()
    # Same forecasts, different margins -- which is exactly what a consumer
    # re-deriving against a later fit would get wrong.
    assert (steep["predicted_margin"].abs() > gentle["predicted_margin"].abs()).all()


def test_a_neutral_site_game_says_so() -> None:
    """The model gives nobody an advantage there, and it changes how it reads."""
    seasons = [
        _season(
            2026,
            [_game("Team A", "Team B", 70, 60, 1)],
            [_game("Team A", "Team B", 70, 60, 2, neutral_site=True)],
        )
    ]

    frame = _frame(seasons).sort_values("week")

    assert frame["neutral_site"].tolist() == [False, True]
    probs = [float(p) for p in frame["team1_win_prob"]]
    assert probs[1] < probs[0]


def test_the_book_s_number_rides_along_when_there_is_one() -> None:
    """And is null, not 0, for a game nobody put a line on."""
    seasons = [
        _season(
            2026,
            [_game("Team A", "Team B", 70, 60, 1)],
            [_game("Team A", "Team B", 70, 60, 2)],
        )
    ]
    lined = "2026-1-Team B@Team A"

    frame = _frame(seasons, odds={lined: Odds(game_id=lined, spread=-6.5)}).set_index(
        "game_id"
    )

    assert frame.loc[lined, "spread"] == -6.5
    assert pd.isna(frame.loc["2026-2-Team B@Team A", "spread"])


def test_rows_are_sorted_by_date(tmp_path: Path) -> None:
    """What lets a consumer prune row groups for a +/- 7 day window."""
    frame = _frame([_lopsided_season()])
    path = predictions_path(tmp_path, _LEAGUE, "elo")

    write_predictions(frame.iloc[::-1], path)

    assert read_predictions(path)["date"].is_monotonic_increasing


def test_predictions_round_trip_through_parquet(tmp_path: Path) -> None:
    lined = "2026-1-Team B@Team A"
    frame = _frame([_lopsided_season()], odds={lined: Odds(game_id=lined, spread=-6.5)})
    path = predictions_path(tmp_path, _LEAGUE, "elo")

    write_predictions(frame, path)

    pd.testing.assert_frame_equal(frame, read_predictions(path))


def test_upserting_the_same_rows_twice_leaves_the_same_bytes(tmp_path: Path) -> None:
    frame = _frame([_lopsided_season()])
    path = predictions_path(tmp_path, _LEAGUE, "elo")

    upsert_predictions(frame, path)
    first = path.read_bytes()
    upsert_predictions(frame, path)

    assert path.read_bytes() == first


def test_a_corrected_score_replaces_a_row_rather_than_appending(
    tmp_path: Path,
) -> None:
    """Games get re-fetched, which is why the key is `game_id`."""
    frame = _frame([_lopsided_season()])
    path = predictions_path(tmp_path, _LEAGUE, "elo")
    upsert_predictions(frame, path)
    game_id = frame["game_id"].iloc[0]
    corrected = frame.head(1).assign(home_score=91, run_id="later-run")

    merged = upsert_predictions(corrected, path)

    assert len(merged) == len(frame)
    row = merged[merged["game_id"] == game_id]
    assert row["home_score"].tolist() == [91]
    assert row["run_id"].tolist() == ["later-run"]


def test_reading_predictions_that_arent_there_yet_is_empty(tmp_path: Path) -> None:
    frame = read_predictions(predictions_path(tmp_path, _LEAGUE, "elo"))

    assert frame.empty
    assert list(frame.columns) == list(PREDICTION_COLUMNS)


def test_the_path_is_the_bucket_layout(tmp_path: Path) -> None:
    assert predictions_path(tmp_path, "mens", "glicko_full") == (
        tmp_path / "models" / "mens" / "glicko_full" / "predictions.parquet"
    )


def test_the_schemas_import_without_pyarrow() -> None:
    """The boundary the pyproject spends a paragraph on, for these two files.

    pyarrow is ~152MB and lives in the `fit` group; `cassandra.serving` is
    the half a webapp installs without it. Building rows and knowing the
    schema must not need it -- only the reads and writes do, and pandas
    resolves the engine inside those.

    A subprocess with pyarrow blocked at the import hook, because pandas
    imports it eagerly when it is installed, so `sys.modules` here says
    nothing.
    """
    source = """
import sys

class Blocked:
    def find_module(self, name, path=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("pyarrow is not installed")
        return None

    def find_spec(self, name, path=None, target=None):
        return self.find_module(name, path)

sys.meta_path.insert(0, Blocked())
for name in [n for n in sys.modules if n.startswith("pyarrow")]:
    del sys.modules[name]

from cassandra.serving import (
    HISTORY_COLUMNS,
    PREDICTION_COLUMNS,
    RatingHistory,
    predictions_frame,
)

assert "team" in HISTORY_COLUMNS and "predicted_margin" in PREDICTION_COLUMNS
assert len(RatingHistory().frame("run")) == 0
print("pyarrow" in sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"

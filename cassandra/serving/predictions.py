"""`predictions.parquet`: what the model said before each game was played.

The walk-forward already computes a genuine out-of-sample forecast for
every game -- `update_game` predicts and *then* updates, in every stateful
predictor -- and then throws it away once the metrics are scored. This
keeps it.

Why that matters: without it, a consumer showing an against-the-spread
record for completed games has to re-predict them from the current release,
which has already trained on the results. That is hindsight, and the webapp
currently marks those rows as such. With this file the rule becomes

    completed game  -> show the stored pre-game prediction
    unplayed game   -> predict live from the release

and both halves are honest, so there is no third case and no dagger.

## The invariant

That rule only holds if this file and `latest.json` come out of the same
run and carry the same `run_id`. A game listed in the release's
`trained_through.processed_game_ids` with no row here is the failure mode:
the consumer would fall through to a live prediction for a game the model
has memorized and print it as a forecast.

So: `publish` writes the release, this file and `history.parquet` together
or not at all, stamped with one `run_id`. A consumer that finds
"trained on it, but no prediction row" should treat it as a mismatch and
suppress the row rather than render a number it can't vouch for.

## Grades, not verdicts

The columns are inputs -- the forecast, the line, the final score -- and
the consumer grades them. That is deliberate, because "did it cover"
already has two answers in this codebase and they differ on exactly one
kind of game.

`model_eval.score_predictions`, which is what a release's
`metrics.against_spread_accuracy` means, uses

    team1_covered = spread + team1_mov > 0

so a game that lands exactly on the number counts as team1 *not*
covering, and the bet is graded won or lost accordingly. The webapp grades
a push as neither a hit nor a miss. Both are defensible; they are not the
same number, and a stored verdict column would silently pick one. If a
derived column is ever added here it should keep `score_predictions`'
definition and be named `correct_bet` after it, so which convention is in
the file is answerable by reading the column name.
"""

from pathlib import Path

import pandas as pd

from cassandra.prob_to_margin import BaseProbToMarginPredictor

from . import _parquet
from .layout import model_dir

PREDICTION_DTYPES: dict[str, str] = {
    # ESPN's competition id: the key the webapp already stores games by,
    # and the same id `trained_through.processed_game_ids` holds.
    "game_id": "object",
    "date": _parquet.DATETIME,
    "year": "int64",
    "week": "int64",
    "home_team": "object",
    "away_team": "object",
    # The model gives nobody an advantage in a neutral game, which changes
    # how the number reads -- and a bowl schedule is a lot of them.
    "neutral_site": "bool",
    # team1 is the home team. Kept under this name rather than renamed to
    # `home_win_prob`, because `GameDfColumns.TEAM1_WIN_PROB` and the
    # predictions csv already say team1, and a second vocabulary for the
    # same number is how a `1 -` ends up in the wrong place.
    "team1_win_prob": "float64",
    # This run's calibration applied to that probability; positive means
    # the home team wins by that much. Stored rather than left to be
    # re-derived: the prob-to-margin fit is refit every run, and a
    # consumer grading an old forecast against a newer fit would be
    # grading it against a mapping that didn't exist when it was made.
    "predicted_margin": "float64",
    "home_score": "int64",
    "away_score": "int64",
    # The book's number, from the home side, as `join_with_odds` attached
    # it. Null for a game no book in the odds database had a line on.
    "spread": "float64",
    "run_id": "object",
}

PREDICTION_COLUMNS: tuple[str, ...] = tuple(PREDICTION_DTYPES)

PREDICTIONS_KEY: tuple[str, ...] = ("game_id",)

# Sorted by date, because the consumer reads a +/- 7 day window and prunes
# row groups on the footer's min/max -- the same trick it already uses on
# endgame's play-by-play. `game_id` is only the tiebreak: two games kick
# off at the same instant constantly, and a sort that leaves those in
# input order would make a re-upsert of unchanged rows rewrite the file.
_SORT: tuple[str, ...] = ("date", "game_id")


def predictions_frame(
    predictions: pd.DataFrame,
    run_id: str,
    margin_predictor: BaseProbToMarginPredictor,
) -> pd.DataFrame:
    """Turn a replay's predictions into the file's rows.

    `predictions` is the frame `save_predictions.build_predictions_df`
    produces -- one row per `_Prediction`, which stays the single
    definition of what a prediction row *is*; this adds the two things a
    consumer can't recover from it (the run that made it, and the margin
    that run's calibration implied) and drops the ones it can (`team1_win`
    is `home_score > away_score`).

    `week_number` becomes `week` here, and nowhere else: the csv keeps the
    name it has always had.
    """
    if predictions.empty:
        return _parquet.empty(PREDICTION_DTYPES)
    rows = predictions.rename(columns={"week_number": "week"}).assign(
        predicted_margin=lambda x: margin_predictor.predict_margins(
            x["team1_win_prob"].to_numpy()
        ),
        run_id=run_id,
    )
    return _parquet.ordered(_parquet.normalized(rows, PREDICTION_DTYPES), _SORT)


def predictions_path(root: Path, league: str, model: str) -> Path:
    """Where one model's stored predictions live, under the bucket's layout."""
    return model_dir(root, league, model) / "predictions.parquet"


def read_predictions(path: Path) -> pd.DataFrame:
    """The stored predictions, or an empty frame if there aren't any yet."""
    return _parquet.read(path, PREDICTION_DTYPES, _SORT)


def predictions_bytes(frame: pd.DataFrame) -> bytes:
    """The file's exact contents for `frame`, in memory. See `history_bytes`."""
    return _parquet.to_bytes(
        _parquet.ordered(_parquet.normalized(frame, PREDICTION_DTYPES), _SORT)
    )


def write_predictions(frame: pd.DataFrame, path: Path) -> None:
    """Rewrite the whole file. For the full replay, which walks every game."""
    _parquet.write_bytes(predictions_bytes(frame), path)


def upsert_predictions(rows: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge `rows` into the stored predictions on `game_id`.

    For the daily refresh. Upsert rather than append for the reason
    `trained_through.processed_game_ids` is a set of ids: games get
    re-fetched and scores corrected, so the same game arrives more than
    once and the last version of it is the right one. Returns what was
    written.
    """
    merged = _parquet.upserted(
        rows, read_predictions(path), PREDICTIONS_KEY, PREDICTION_DTYPES, _SORT
    )
    _parquet.write(merged, path)
    return merged

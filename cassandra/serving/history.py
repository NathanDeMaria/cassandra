"""`history.parquet`: what every team was rated at the end of every week.

A release is a point-in-time snapshot with no previous rating in it, so
nothing reading one can say "up 40 points and four spots since last week",
and the `runs/{run_id}.json` archive is only as dense as someone's manual
publishes. This is the dense version: one row per (team, year, week), long
format, one file per (league, model).

The rows come out of the replay itself, through
`save_predictions.generate_predictions`' `week_observer` -- `RatingHistory`
below is that observer. Each row is taken after the week's games and before
`predictor.pass_week()`, so it is the rating the team *finished the week
with* rather than one already aged by Glicko's weekly rd inflation.

Two writers, splitting the way the publish pipeline does. A full replay
walks every season and so is the only thing that can rewrite the file end
to end (`write_history`). The daily refresh only ever sees the current
week, so it merges its rows in (`upsert_history`) -- and it has to be an
upsert rather than an append, because a week's ratings move as its games
land and the job has to be safe to re-run.

Deliberately no rank column. Rank depends on which teams are in the table,
and the webapp drops programs that folded before it counts, so a rank
computed here would disagree with the leaderboard it is supposed to
explain. Ship the ratings; let the API rank what it is showing.
"""

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from cassandra.predictor import Rating

from . import _parquet
from .layout import model_dir

# The file's schema: column order and dtype, pinned. See `_parquet`.
HISTORY_DTYPES: dict[str, str] = {
    "team": "object",
    "year": "int64",
    "week": "int64",
    "date": _parquet.DATETIME,
    "rating": "float64",
    # Glicko's rating deviation. Null for the Elo family rather than 0,
    # which is a meaningful and very wrong deviation -- the same choice
    # `predictor.types.Rating` and `TeamRating` already make.
    "rd": "float64",
    "wins": "int64",
    "losses": "int64",
    "run_id": "object",
}

HISTORY_COLUMNS: tuple[str, ...] = tuple(HISTORY_DTYPES)

# What identifies a row, and what an upsert replaces on.
HISTORY_KEY: tuple[str, ...] = ("team", "year", "week")

# Total over the rows, so the order is a function of the data alone.
_SORT: tuple[str, ...] = HISTORY_KEY


class WeekSnapshot(NamedTuple):
    """One week, as the replay hands it to a `week_observer`.

    Everything a row needs, so an observer never has to re-derive anything
    or reach back into the season it came from.

    `date` is the last *completed* game of the week. The natural reading is
    `endgame`'s `week.end`, and for a fully played week the two are the
    same number -- but season pickles now carry fixtures, so `week.end` on
    a week in progress is a game nobody has played, and dating a rating
    snapshot in the future is worse than the round trip through the games.
    It's the end rather than the start because the snapshot is what the
    team finished the week with.

    `wins` and `losses` are season-to-date through this week, counted over
    the games the replay actually walked. They're here rather than left to
    the API because the observer sits inside the game walk, where the tally
    is free -- `ratings_from_predictor` is right that a record isn't the
    *predictor's* to know, and this isn't the predictor. A tie counts for
    neither, matching `publish._with_records`, so a team whose only result
    was a draw is 0-0 and still very much playing -- which is why `played`
    is its own set rather than something read off these two.

    `played` is every team that has appeared in a game this season up to
    and including this week. `ratings` is everyone the predictor rates,
    which is a different and much larger group: a rating never leaves a
    predictor, so by 2026 it still holds programs that folded in 2012.
    Carrying both lets an observer decide which it wants -- `RatingHistory`
    writes rows for the intersection, see there for why.
    """

    league: str
    year: int
    week: int
    date: datetime
    ratings: Mapping[str, Rating]
    played: frozenset[str]
    wins: Mapping[str, int]
    losses: Mapping[str, int]


# What `generate_predictions` calls once per played week.
WeekObserver = Callable[[WeekSnapshot], None]


class RatingHistory:
    """Collects `WeekSnapshot`s into history rows.

    A callable object rather than a closure so a caller can hold the rows:
    `publish` passes one in as the `week_observer`, then asks it for the
    frame once the replay is done.

    The `run_id` is supplied to `frame` rather than to the constructor,
    because the replay happens before the run has a name -- publish derives
    the id from the `created_at` it stamps on the release afterwards. Asking
    for it at the end is also what keeps the stamping honest: the same
    argument goes into `predictions_frame` and into the release, at one
    call site, so the three artifacts can't disagree about which run they
    came from.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, object]] = []

    def __call__(self, snapshot: WeekSnapshot) -> None:
        for team, rating in snapshot.ratings.items():
            if team not in snapshot.played:
                # A team the predictor still rates but that isn't playing
                # this season. Ratings are forever -- nothing removes a
                # team from one -- so without this every program that ever
                # existed gets a row every week for the rest of history: on
                # ncaafb that was a quarter of the file, and 57 programs
                # whose last game was in 2014 were still drawing a flat
                # line through 2026.
                #
                # Season-to-date rather than the season's whole roster, so
                # a team's line starts at its first game rather than at a
                # rating it carried in from last year under this year's
                # label. Once it has played it keeps getting rows through
                # its idle weeks, which is what a chart wants.
                #
                # Iterating `ratings` and filtering, rather than walking
                # `played` and looking rating up, because the two can
                # disagree: a predictor is free to rate whoever it likes,
                # and this way the file only ever holds teams it actually
                # rates.
                continue
            self._rows.append(
                {
                    "team": team,
                    "year": snapshot.year,
                    "week": snapshot.week,
                    "date": snapshot.date,
                    "rating": rating.rating,
                    "rd": rating.rd,
                    "wins": snapshot.wins.get(team, 0),
                    "losses": snapshot.losses.get(team, 0),
                }
            )

    def __len__(self) -> int:
        return len(self._rows)

    def frame(self, run_id: str) -> pd.DataFrame:
        """The collected rows, stamped with the run, in the file's schema."""
        if not self._rows:
            return _parquet.empty(HISTORY_DTYPES)
        rows = pd.DataFrame(self._rows).assign(run_id=run_id)
        return _parquet.ordered(_parquet.normalized(rows, HISTORY_DTYPES), _SORT)


def history_path(root: Path, league: str, model: str) -> Path:
    """Where one model's history lives, under the bucket's layout."""
    return model_dir(root, league, model) / "history.parquet"


def read_history(path: Path) -> pd.DataFrame:
    """The stored history, or an empty frame if there isn't one yet."""
    return _parquet.read(path, HISTORY_DTYPES, _SORT)


def history_bytes(frame: pd.DataFrame) -> bytes:
    """The file's exact contents for `frame`, in memory.

    For a caller that puts the same bytes in two places -- `publish` writes
    them to disk and to s3 -- so the local artifact and the served one are
    the same object rather than two serializations that ought to agree.
    """
    return _parquet.to_bytes(
        _parquet.ordered(_parquet.normalized(frame, HISTORY_DTYPES), _SORT)
    )


def write_history(frame: pd.DataFrame, path: Path) -> None:
    """Rewrite the whole file. For the full replay, which walks every season."""
    _parquet.write_bytes(history_bytes(frame), path)


def upsert_history(rows: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge `rows` into the stored history on (team, year, week).

    For the daily refresh, which only sees the current week and must not
    drop the fifteen seasons it never walked. Returns what was written.
    """
    merged = _parquet.upserted(
        rows, read_history(path), HISTORY_KEY, HISTORY_DTYPES, _SORT
    )
    _parquet.write(merged, path)
    return merged


def tally(
    wins: dict[str, int], losses: dict[str, int], home: str, away: str, margin: int
) -> None:
    """Add one played game to a season-to-date record, in place.

    Shared with nothing yet, but named and separate because the tie rule is
    a decision rather than arithmetic: a 0 margin counts for neither side.
    `publish._with_records` makes the same call for the release's records,
    and a table where a team's history disagrees with its release row about
    how many games it has lost is the kind of thing nobody debugs twice.
    """
    if margin == 0:
        return
    winner, loser = (home, away) if margin > 0 else (away, home)
    wins[winner] = wins.get(winner, 0) + 1
    losses[loser] = losses.get(loser, 0) + 1

"""Tests for the rating history: the observer, and the file it writes.

Seasons are hand-written, and the replay under them is the real
`generate_predictions` -- what these are checking is *when* a snapshot is
taken relative to the rollovers, and that is a property of the walk rather
than of any predictor.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from call_it_what_you_want import TeamNamer
from endgame.types import Game, Season, Week

from cassandra.predictor import EloPredictor, FlatPredictor, GlickoPredictor, Predictor
from cassandra.predictor.opponent_prior import OpponentPriorManager
from cassandra.save_predictions import generate_predictions

from .history import (
    HISTORY_COLUMNS,
    HISTORY_KEY,
    RatingHistory,
    history_bytes,
    history_path,
    read_history,
    upsert_history,
    write_history,
)

# A league nobody has run, so no opponent priors on disk get picked up.
_LEAGUE = "test_league"
_RUN = "2026-09-09T00:00:00Z"


def _game(
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    year: int,
    week: int,
    completed: bool = True,
) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=False,
        completed=completed,
        # A day per week, because `iter_weeks` refuses a season whose weeks
        # overlap in time.
        date=datetime(year, 3, week, tzinfo=timezone.utc),
        game_id=f"{year}-{week}-{away}@{home}",
        status="STATUS_FINAL" if completed else "STATUS_SCHEDULED",
    )


def _season(year: int, *weeks: list[Game]) -> Season:
    return Season(
        year=year,
        weeks=[Week(games=games, number=n) for n, games in enumerate(weeks, start=1)],
    )


def _two_seasons() -> list[Season]:
    """2025: A beats B, then C beats A. 2026: A beats B twice."""
    return [
        _season(
            2025,
            [_game("Team A", "Team B", 80, 60, 2025, 1)],
            [_game("Team C", "Team A", 70, 60, 2025, 2)],
        ),
        _season(
            2026,
            [_game("Team A", "Team B", 80, 61, 2026, 1)],
            [_game("Team A", "Team B", 90, 51, 2026, 2)],
        ),
    ]


def _replay(
    predictor: Predictor,
    seasons: list[Season],
    roll_over_final_season: bool = True,
) -> pd.DataFrame:
    history = RatingHistory()
    list(
        generate_predictions(
            predictor,
            seasons,
            namer=TeamNamer.empty(),
            week_observer=history,
            roll_over_final_season=roll_over_final_season,
        )
    )
    return history.frame(_RUN)


def _glicko(
    weekly_rd_increase: float = 1.0, initial_rd: float = 216.0
) -> GlickoPredictor:
    return GlickoPredictor(
        _LEAGUE,
        weekly_rd_increase=weekly_rd_increase,
        initial_rd=initial_rd,
        opponent_prior_manager=OpponentPriorManager(_LEAGUE),
    )


def _records(frame: pd.DataFrame) -> dict[tuple[str, int, int], tuple[int, int]]:
    """(team, year, week) -> (wins, losses), as plain python."""
    return {
        (str(team), int(year), int(week)): (int(wins), int(losses))
        for team, year, week, wins, losses in zip(
            frame["team"],
            frame["year"],
            frame["week"],
            frame["wins"],
            frame["losses"],
            strict=True,
        )
    }


def test_a_replay_writes_one_row_per_team_per_week() -> None:
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())

    assert list(frame.columns) == list(HISTORY_COLUMNS)
    assert not frame.duplicated(subset=list(HISTORY_KEY)).any()
    # A row per team that has played *this season* so far. A and B from 2025
    # week 1; C joins in week 2. In 2026 only A and B play, so C -- still
    # rated, and rated for good -- gets no rows at all.
    assert frame.groupby(["year", "week"]).size().to_dict() == {
        (2025, 1): 2,
        (2025, 2): 3,
        (2026, 1): 2,
        (2026, 2): 2,
    }
    assert (frame["run_id"] == _RUN).all()


def test_the_snapshot_is_dated_by_the_last_game_of_its_week() -> None:
    """A time axis, so a chart doesn't have to plot week numbers that reset."""
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())

    weeks = frame.drop_duplicates(["year", "week"])
    dates = {
        (int(year), int(week)): date
        for year, week, date in zip(
            weeks["year"], weeks["week"], weeks["date"], strict=True
        )
    }
    assert dates[(2025, 1)] == pd.Timestamp("2025-03-01", tz="UTC")
    assert dates[(2026, 2)] == pd.Timestamp("2026-03-02", tz="UTC")


def test_the_last_week_of_a_season_is_recorded_before_the_rollover() -> None:
    """The season's final standing, not next season's opening one.

    `season_regression=1.0` snaps every rating onto its anchor in the
    rollover, so a snapshot taken on the wrong side of `pass_season` is
    exactly 1500 for everybody and this fails loudly rather than by a few
    points.
    """
    frame = _replay(
        EloPredictor(_LEAGUE, season_regression=1.0),
        _two_seasons(),
        roll_over_final_season=False,
    )

    final_2025 = frame[(frame["year"] == 2025) & (frame["week"] == 2)]
    assert (final_2025["rating"] != 1500.0).all()
    # And the rollover really did happen -- otherwise the assertion above
    # passes for the wrong reason.
    assert frame[(frame["year"] == 2026) & (frame["week"] == 1)]["rating"].nunique() > 1


def test_glicko_records_the_rd_from_before_pass_week() -> None:
    """`pass_week` inflates every rd; the snapshot is what the week ended at."""
    seasons = [_season(2025, [_game("Team A", "Team B", 80, 60, 2025, 1)])]
    predictor = _glicko(weekly_rd_increase=40.0, initial_rd=400.0)

    frame = _replay(predictor, seasons, roll_over_final_season=False)

    recorded = frame.set_index("team")["rd"]
    after_the_week = {t: r.rd for t, r in predictor.ratings.items()}
    for team, rd in after_the_week.items():
        assert rd is not None
        # Strictly smaller: the recorded value is the pre-inflation one.
        assert recorded[team] < rd


def test_elo_records_a_null_rd_rather_than_a_zero() -> None:
    """0 is a meaningful and very wrong rating deviation."""
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())

    assert frame["rd"].isna().all()
    assert not (frame["rd"] == 0).any()


def test_records_are_season_to_date_and_reset_at_the_boundary() -> None:
    records = _records(_replay(EloPredictor(_LEAGUE), _two_seasons()))

    # 2025: A wins week 1, loses week 2.
    assert records[("Team A", 2025, 1)] == (1, 0)
    assert records[("Team A", 2025, 2)] == (1, 1)
    # 2026 starts over rather than carrying 1-1 forward.
    assert records[("Team A", 2026, 1)] == (1, 0)
    assert records[("Team A", 2026, 2)] == (2, 0)
    # Team C played only in 2025, so 2026 has no row for it at all rather
    # than a 0-0 one -- see `test_a_team_that_isnt_playing_gets_no_rows`.
    assert ("Team C", 2026, 2) not in records


def test_a_tie_counts_for_neither_side() -> None:
    """Matching `publish._with_records`, so a history and a release agree."""
    seasons = [_season(2025, [_game("Team A", "Team B", 70, 70, 2025, 1)])]

    frame = _replay(EloPredictor(_LEAGUE), seasons, roll_over_final_season=False)

    assert frame[["wins", "losses"]].to_numpy().sum() == 0


def test_an_unplayed_game_changes_nothing_it_touches() -> None:
    """The §0 bug, from the history's side.

    A fixture arrives 0-0 and would otherwise be replayed as a draw: it
    would move both ratings, add a row for a week nobody played, and put a
    loss on the home team's record.
    """
    week = [
        _game("Team A", "Team B", 80, 60, 2025, 1),
        _game("Team C", "Team D", 0, 0, 2025, 1, completed=False),
    ]
    upcoming = [_game("Team A", "Team C", 0, 0, 2025, 2, completed=False)]

    frame = _replay(
        EloPredictor(_LEAGUE),
        [_season(2025, week, upcoming)],
        roll_over_final_season=False,
    )

    # Only the played week, and only the teams that played it. Team C and
    # Team D are absent because their game hasn't happened, not merely
    # unrated: replaying it 0-0 would have rated both of them.
    assert frame["week"].unique().tolist() == [1]
    assert sorted(frame["team"]) == ["Team A", "Team B"]
    # One result: 1-0 and 0-1, and nothing charged for the fixture.
    assert _records(frame) == {
        ("Team A", 2025, 1): (1, 0),
        ("Team B", 2025, 1): (0, 1),
    }


def test_a_predictor_with_no_ratings_runs_clean_and_writes_nothing() -> None:
    """FlatPredictor rates nobody, and that isn't an error to handle per week.

    No rows rather than rows of 1500s: "this model rates nobody" and "this
    model rates everyone the same" are different claims, and only one of
    them is true.
    """
    history = RatingHistory()

    results = list(
        generate_predictions(
            FlatPredictor(_LEAGUE),
            _two_seasons(),
            namer=TeamNamer.empty(),
            week_observer=history,
        )
    )

    assert len(results) == 4
    assert len(history) == 0
    assert list(history.frame(_RUN).columns) == list(HISTORY_COLUMNS)


def test_an_empty_history_still_has_the_schema(tmp_path: Path) -> None:
    """So the first upsert concatenates rather than special-casing."""
    path = history_path(tmp_path, _LEAGUE, "elo")

    write_history(RatingHistory().frame(_RUN), path)

    assert list(read_history(path).columns) == list(HISTORY_COLUMNS)
    assert read_history(path).empty


def test_reading_a_history_that_isnt_there_yet_is_empty(tmp_path: Path) -> None:
    frame = read_history(history_path(tmp_path, _LEAGUE, "elo"))

    assert frame.empty
    assert list(frame.columns) == list(HISTORY_COLUMNS)


def test_history_round_trips_through_parquet(tmp_path: Path) -> None:
    """Same rows and same dtypes, which is what makes an upsert safe."""
    frame = _replay(_glicko(), _two_seasons())
    path = history_path(tmp_path, _LEAGUE, "glicko")

    write_history(frame, path)

    back = read_history(path)
    pd.testing.assert_frame_equal(frame, back)


def test_upserting_the_same_rows_twice_leaves_the_same_bytes(tmp_path: Path) -> None:
    """The refresh runs daily; a no-op day must not churn the object."""
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())
    path = history_path(tmp_path, _LEAGUE, "elo")

    upsert_history(frame, path)
    first = path.read_bytes()
    upsert_history(frame, path)

    assert path.read_bytes() == first


def test_upsert_replaces_a_week_rather_than_appending(tmp_path: Path) -> None:
    """Ratings move within a week as its games land."""
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())
    path = history_path(tmp_path, _LEAGUE, "elo")
    upsert_history(frame, path)
    corrected = frame[
        (frame["team"] == "Team A") & (frame["year"] == 2026) & (frame["week"] == 2)
    ].assign(rating=1234.5, run_id="later-run")

    merged = upsert_history(corrected, path)

    assert len(merged) == len(frame)
    row = merged[
        (merged["team"] == "Team A") & (merged["year"] == 2026) & (merged["week"] == 2)
    ]
    assert row["rating"].tolist() == [1234.5]
    assert row["run_id"].tolist() == ["later-run"]
    # Everything it didn't name is untouched, including the run that wrote it.
    assert (merged[merged["team"] == "Team B"]["run_id"] == _RUN).all()


def test_upsert_keeps_the_seasons_it_never_walked(tmp_path: Path) -> None:
    """The refresh only ever sees the current week."""
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())
    path = history_path(tmp_path, _LEAGUE, "elo")
    upsert_history(frame, path)

    merged = upsert_history(frame[frame["year"] == 2026], path)

    assert sorted(merged["year"].unique()) == [2025, 2026]


def test_rows_are_sorted_by_their_key(tmp_path: Path) -> None:
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())
    path = history_path(tmp_path, _LEAGUE, "elo")

    upsert_history(frame.iloc[::-1], path)

    back = read_history(path)
    assert back[list(HISTORY_KEY)].apply(tuple, axis=1).is_monotonic_increasing


def test_the_path_is_the_bucket_layout(tmp_path: Path) -> None:
    assert history_path(tmp_path, "mens", "glicko_full") == (
        tmp_path / "models" / "mens" / "glicko_full" / "history.parquet"
    )


@pytest.mark.parametrize("column", ["rating", "wins", "losses"])
def test_no_column_is_silently_object_typed(column: str) -> None:
    """An object column is how a None slips into a number and reaches parquet."""
    frame = _replay(_glicko(), _two_seasons())

    assert frame[column].dtype != object


def test_the_in_memory_bytes_are_what_a_write_leaves_on_disk(tmp_path: Path) -> None:
    """What lets `publish` put the same object on disk and in the bucket.

    Serializing twice -- once for each destination -- would leave nothing
    holding the two equal, and a difference would surface as the API
    reading a file that isn't the one you checked locally.
    """
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())
    path = history_path(tmp_path, _LEAGUE, "elo")

    write_history(frame, path)

    assert history_bytes(frame) == path.read_bytes()


def test_a_team_that_isnt_playing_this_season_gets_no_rows() -> None:
    """A rating is forever; a history row shouldn't be.

    Nothing ever removes a team from a predictor's ratings, so without this
    every program that ever existed draws a flat line to the present. On
    ncaafb that was a quarter of the file -- 873 teams rated in 2026 against
    675 that played -- including 57 whose last game was before 2015.
    """
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())

    assert "Team C" in set(frame[frame["year"] == 2025]["team"])
    # Still rated in 2026 -- this is a row rule, not a forgetting rule.
    assert "Team C" not in set(frame[frame["year"] == 2026]["team"])


def test_a_teams_line_starts_at_its_first_game_of_the_season() -> None:
    """Season-to-date, not the season's whole roster.

    Before its first game a team's rating is last year's, carried in under
    this year's label, and it has no record. A line that starts when the
    team does is the honest one.
    """
    frame = _replay(EloPredictor(_LEAGUE), _two_seasons())

    in_2025 = frame[frame["year"] == 2025]
    # Team C debuts in week 2, so it has no week 1 row even though the
    # predictor was already rating A and B by then.
    assert set(in_2025[in_2025["week"] == 1]["team"]) == {"Team A", "Team B"}
    assert "Team C" in set(in_2025[in_2025["week"] == 2]["team"])


def test_an_idle_week_still_gets_a_row_once_a_team_has_played() -> None:
    """The rule is "played this season", not "played this week".

    Otherwise a team's line goes dotted through its bye weeks, and "up 40
    points since last week" has nothing to subtract from.
    """
    seasons = [
        _season(
            2025,
            [_game("Team A", "Team B", 80, 60, 2025, 1)],
            [_game("Team A", "Team C", 70, 65, 2025, 2)],
        )
    ]

    frame = _replay(EloPredictor(_LEAGUE), seasons, roll_over_final_season=False)

    # Team B sat out week 2 and still has a row for it.
    week2 = frame[frame["week"] == 2]
    assert set(week2["team"]) == {"Team A", "Team B", "Team C"}


def test_a_team_whose_only_result_was_a_tie_is_still_playing() -> None:
    """`played` is its own set, not something read off wins and losses.

    A tie counts toward neither column, so a team inferred from the record
    would vanish from the file for the week it drew.
    """
    seasons = [_season(2025, [_game("Team A", "Team B", 70, 70, 2025, 1)])]

    frame = _replay(EloPredictor(_LEAGUE), seasons, roll_over_final_season=False)

    assert set(frame["team"]) == {"Team A", "Team B"}
    assert frame[["wins", "losses"]].to_numpy().sum() == 0

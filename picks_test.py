from datetime import UTC, datetime, timedelta

from endgame.types import Game

from picks import QB_UNREAD_WINDOW, qb_reads

NOW = datetime(2026, 10, 21, 3, 0, tzinfo=UTC)  # Tue 22:00 CT


def _game(away: str, home: str, ago: timedelta, completed: bool = True) -> Game:
    return Game(
        home=home,
        home_score=0,
        away=away,
        away_score=0,
        neutral_site=False,
        completed=completed,
        date=NOW - ago,
        game_id=f"{away}@{home}",
    )


def test_only_recent_unread_games_are_read_off_espn() -> None:
    week = timedelta(days=7)
    games = [
        _game("Toledo", "Ohio", 2 * week),  # in the index: Toledo has plays
        _game("Ohio", "Kent", 2 * week),  # in the index: Kent has plays
        # A D-II opponent ESPN never had plays for, weeks ago: the sweeps
        # have run since, so it's missing for good and isn't read again.
        _game("Wheeling", "Toledo", week),
        _game("Akron", "Kent", timedelta(hours=3)),  # tonight's, not swept
        _game("Toledo", "Kent", -week, completed=False),
    ]
    stored = {"Toledo@Ohio", "Ohio@Kent"}
    upcoming = [games[-1]]
    unread, needy, wanted = qb_reads(games, stored, upcoming, NOW)
    assert unread == {"Akron@Kent"}
    assert needy == {"Kent"}
    # Kent's whole season, so the rule knows who its starter is.
    assert wanted == ["Akron@Kent", "Ohio@Kent"]
    # The old one is read while a sweep could still be behind on it.
    later = NOW - week + QB_UNREAD_WINDOW - timedelta(hours=1)
    unread, _, _ = qb_reads(games, stored, upcoming, later)
    assert "Wheeling@Toledo" in unread

"""Tests for reading a starter in doubt off his team's last game."""

from datetime import UTC, datetime, timedelta

import pytest
from endgame.types import Game

from cassandra.predictor.qb_out import QB_OUT_DECAY
from cassandra.qb_starts import (
    AVAILABILITY_OUT,
    GameQb,
    availabilities,
    availability,
    doubt,
    doubts,
)

ACE, BACKUP = "ace a", "backup b"
NAMES = {ACE: "A.Ace", BACKUP: "B.Backup"}


def _game(n: int, game_id: str | None = None, completed: bool = True) -> Game:
    return Game(
        game_id=game_id or f"g{n}",
        home="Home",
        away="Away",
        # 7pm Central on a Saturday, 00:00 UTC on the Sunday: named the Saturday.
        date=datetime(2026, 9, 6, tzinfo=UTC) + timedelta(weeks=n),
        home_score=24 if completed else 0,
        away_score=17 if completed else 0,
        neutral_site=False,
        completed=completed,
    )


def _qb(
    starter: str = ACE,
    attempts: dict[str, int] | None = None,
    snaps: tuple[str, ...] | None = None,
    last: dict[str, tuple[int | None, int | None]] | None = None,
) -> GameQb:
    attempts = attempts if attempts is not None else {starter: 30}
    return GameQb(
        starter=starter,
        attempts=attempts,
        names={k: NAMES.get(k, k) for k in attempts},
        snaps=list(snaps if snaps is not None else attempts),
        last_snap=last if last is not None else {k: (4, 3) for k in attempts},
    )


def _season(*last: GameQb, starts: int = 4) -> list[tuple[Game, GameQb]]:
    played = [(_game(n), _qb()) for n in range(starts)]
    return played + [(_game(starts + i), qb) for i, qb in enumerate(last)]


def test_a_starter_who_played_through_is_not_in_doubt() -> None:
    assert doubt("Home", _season(_qb())) is None


def test_a_starter_who_missed_the_last_game_is_in_doubt() -> None:
    found = doubt("Home", _season(_qb(BACKUP)))

    assert found is not None
    assert found.kind == "missed"
    assert found.sentence == (
        "Home: A.Ace (4 starts) didn't play Oct 03; B.Backup started."
    )


def test_a_starter_who_left_a_close_game_early_is_in_doubt() -> None:
    """South Florida's starter on Oct 31: 3 passes, gone in the first."""
    last = _qb(attempts={ACE: 3, BACKUP: 10}, last={ACE: (1, -10), BACKUP: (4, 20)})

    found = doubt("Home", _season(last))

    assert found is not None
    assert found.kind == "left early"
    assert found.sentence == (
        "Home: A.Ace (5 starts) threw 3 of 13 passes Oct 03, last snap in the "
        "1st quarter; B.Backup threw 10."
    )


def test_a_starter_rested_in_a_rout_is_not_in_doubt() -> None:
    last = _qb(attempts={ACE: 15, BACKUP: 12}, last={ACE: (3, 28), BACKUP: (4, 35)})

    assert doubt("Home", _season(last)) is None


def test_a_first_half_exit_in_a_rout_is_still_in_doubt() -> None:
    """A big lead in the second quarter isn't a reason to sit him yet."""
    last = _qb(attempts={ACE: 8, BACKUP: 20}, last={ACE: (2, 24), BACKUP: (4, 30)})

    found = doubt("Home", _season(last))

    assert found is not None and found.kind == "left early"


def test_a_backup_who_barely_threw_is_not_a_cover() -> None:
    """A run-heavy afternoon: the starter's share is low on few passes, and
    the other man threw a trick play."""
    last = _qb(attempts={ACE: 6, BACKUP: 4})

    assert doubt("Home", _season(last)) is None


def test_a_starter_who_came_off_the_bench_is_in_doubt() -> None:
    last = _qb(BACKUP, attempts={BACKUP: 22, ACE: 8})

    found = doubt("Home", _season(last))

    assert found is not None
    assert found.kind == "came off the bench"
    assert found.sentence == (
        "Home: B.Backup started Oct 03 over A.Ace (4 starts), who threw 8 of 30 passes."
    )


def test_the_doubt_lasts_through_the_absence() -> None:
    found = doubt("Home", _season(_qb(BACKUP), _qb(BACKUP), _qb(BACKUP)))

    assert found is not None
    assert found.sentence.startswith("Home: A.Ace (4 starts) didn't play")


def test_a_cover_with_more_starts_is_the_starter() -> None:
    """Out for the season: by now the team is the backup's."""
    covering = [_qb(BACKUP)] * 3

    assert doubt("Home", _season(*covering, starts=2)) is None


def test_a_starter_back_from_an_absence_is_not_in_doubt() -> None:
    assert doubt("Home", _season(_qb(BACKUP), _qb())) is None


def test_a_preseason_starter_back_from_injury_is_not_the_backup() -> None:
    """Hurt in week two, back in week nine: the cover has more starts this
    season, but the job was never his, so the starter's return is no doubt.
    Read by most starts, it was -- Utah State's Hillstead, Nov 8 2026."""
    played = [(_game(0), _qb()), (_game(1), _qb())]
    played += [(_game(n), _qb(BACKUP)) for n in range(2, 4)]
    # the cover ties him, then ace is back for two games
    played += [(_game(n), _qb()) for n in range(4, 6)]

    assert doubt("Home", played) is None


def test_last_seasons_starter_back_from_missing_the_opener_is_no_doubt() -> None:
    played = [(_game(n), _qb(BACKUP)) for n in range(2)]
    played += [(_game(n), _qb()) for n in range(2, 4)]

    assert doubt("Home", played) is not None
    assert doubt("Home", played, last_season={ACE: 12, BACKUP: 0}) is None


def test_the_doubt_is_priced_by_how_the_last_game_went() -> None:
    assert availability("Home", _season(_qb())) == 0.0
    assert availability("Home", _season(_qb(BACKUP))) == pytest.approx(
        AVAILABILITY_OUT["missed"] * QB_OUT_DECAY
    )
    left = _qb(attempts={ACE: 8, BACKUP: 20}, last={ACE: (2, 0), BACKUP: (4, 3)})
    assert availability("Home", _season(left)) == pytest.approx(
        AVAILABILITY_OUT["left early"]
    )


def test_a_long_absence_weighs_less_each_game() -> None:
    """The flag would decay if he sits again, and so does the price."""
    one = availability("Home", _season(_qb(BACKUP)))
    three = availability("Home", _season(_qb(BACKUP), _qb(BACKUP), _qb(BACKUP)))

    assert three == pytest.approx(one * QB_OUT_DECAY**2)


def test_availabilities_price_each_teams_next_game_only() -> None:
    season = [_game(n) for n in range(4)]
    later = [_game(8, completed=False), _game(9, completed=False)]
    starts = {g.game_id: {"Home": _qb(), "Away": _qb()} for g in season[:3]}
    starts["g3"] = {"Home": _qb(BACKUP), "Away": _qb()}

    found = availabilities(season + later, starts, later)

    # One game missed already: if he sits again it's the second of it.
    assert found == {
        "g8": {"Home": pytest.approx(AVAILABILITY_OUT["missed"] * QB_OUT_DECAY)}
    }


def test_doubts_reads_each_team_after_its_last_game() -> None:
    season = [_game(n) for n in range(4)] + [_game(9, completed=False)]
    starts = {g.game_id: {"Home": _qb(), "Away": _qb()} for g in season[:3]}
    starts["g3"] = {"Home": _qb(BACKUP), "Away": _qb()}
    # Plays for a game that isn't final yet are no evidence of anything.
    starts["g9"] = {"Away": _qb(BACKUP)}

    found = doubts(season, starts)

    assert found.keys() == {"Home"}
    assert found["Home"].kind == "missed"


def test_a_game_without_plays_is_passed_over() -> None:
    """No plays (a D-II opponent ESPN doesn't cover) isn't an absence."""
    season = [_game(n) for n in range(4)]
    starts = {g.game_id: {"Home": _qb()} for g in season[:3]}

    assert doubts(season, starts) == {}

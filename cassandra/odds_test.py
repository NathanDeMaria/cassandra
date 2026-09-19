from datetime import UTC, datetime

from .odds import (
    Odds,
    OddsDatabase,
    OddsSnapshot,
    _parse_odds,
    _parse_snapshot,
    read_at_from_key,
)


def _record(game_id: str, **entry: object) -> dict:
    return {"competition_id": game_id, "odds": [entry]}


def test_reads_the_spread_off_the_first_entry() -> None:
    assert _parse_odds(_record("1", spread=-6.5, overUnder=47.5)) == Odds(
        game_id="1", spread=-6.5
    )


def test_a_game_with_the_line_off_the_board_has_no_odds() -> None:
    """The shape a book leaves behind when it pulls the spread.

    Everything else is still there -- an over/under, a moneyline, a
    `pointSpread` whose line reads "OFF" -- and only `spread` is missing. This
    used to be a KeyError, and because `read_all_odds` reads every league's
    odds regardless of the league being run, a single game like it took down
    every optimize, evaluate and publish run.
    """
    entry = {
        "overUnder": 47.5,
        "moneyline": {},
        "pointSpread": {"home": {"close": {"line": "OFF"}}},
    }

    assert _parse_odds(_record("401858438", **entry)) is None


def test_a_record_with_no_entries_at_all_has_no_odds() -> None:
    assert _parse_odds({"competition_id": "1", "odds": []}) is None
    assert _parse_odds({"competition_id": "1", "odds": None}) is None
    assert _parse_odds({"competition_id": "1"}) is None


def test_a_game_with_no_odds_is_the_same_as_one_that_was_never_loaded() -> None:
    database = OddsDatabase({"1": Odds(game_id="1", spread=-6.5)})

    assert database.get_odds("1") == Odds(game_id="1", spread=-6.5)
    assert database.get_odds("401858438") is None


def test_read_time_comes_off_the_key_in_chicago_time() -> None:
    # 09:33 in Chicago on a September (CDT, UTC-5) morning is 14:33 UTC.
    assert read_at_from_key("odds/ncaafb/2026-09-12/09-33-near.json") == datetime(
        2026, 9, 12, 14, 33, tzinfo=UTC
    )
    # The pre-horizon keys have no suffix and the same clock.
    assert read_at_from_key("odds/nfl/2026-08-02/22-03.json") == datetime(
        2026, 8, 3, 3, 3, tzinfo=UTC
    )
    assert read_at_from_key("odds/nfl/README") is None


def test_a_snapshot_keeps_the_moneylines_and_the_read() -> None:
    entry = {
        "spread": -7.5,
        "moneyline": {
            "home": {"close": {"odds": "-310"}, "open": {"odds": "-280"}},
            "away": {"close": {"odds": "+250"}, "open": {"odds": "+230"}},
        },
    }
    at = datetime(2026, 9, 12, 14, 33, tzinfo=UTC)

    assert _parse_snapshot(_record("1", **entry), at) == OddsSnapshot(
        read_at=at, spread=-7.5, home_moneyline=-310.0, away_moneyline=250.0
    )


def test_a_snapshot_with_a_price_off_the_board_keeps_the_other_one() -> None:
    """The book takes the moneyline off a 50-point game but leaves the spread.

    `_parse_odds` reads only the spread and drops the whole game when that's
    what's off; a snapshot is a record of the moment and keeps whatever was
    up. Only a record with no book entry at all is nothing.
    """
    at = datetime(2026, 9, 12, 14, 33, tzinfo=UTC)
    off_moneyline = {
        "spread": -57.5,
        "moneyline": {"home": {"close": {"odds": "OFF"}}, "away": {}},
    }
    assert _parse_snapshot(_record("1", **off_moneyline), at) == OddsSnapshot(
        read_at=at, spread=-57.5, home_moneyline=None, away_moneyline=None
    )
    off_spread = {"moneyline": {"home": {"close": {"odds": "-150"}}}}
    assert _parse_snapshot(_record("1", **off_spread), at) == OddsSnapshot(
        read_at=at, spread=None, home_moneyline=-150.0, away_moneyline=None
    )
    assert _parse_snapshot({"competition_id": "1", "odds": []}, at) is None


def test_the_line_is_the_last_read_that_had_one() -> None:
    def at(hour: int) -> datetime:
        return datetime(2026, 9, 12, hour, tzinfo=UTC)

    history = {
        "1": [
            OddsSnapshot(at(10), -3.0, -150.0, 130.0),
            OddsSnapshot(at(11), -3.5, -160.0, 140.0),
            # The book pulled the spread in the final read; the moneyline stayed.
            OddsSnapshot(at(12), None, -160.0, 140.0),
        ],
        "2": [OddsSnapshot(at(10), None, -150.0, 130.0)],
    }
    database = OddsDatabase.from_history(history)

    assert database.get_odds("1") == Odds(game_id="1", spread=-3.5)
    assert database.get_odds("2") is None
    assert database.snapshots("1") == history["1"]
    assert database.snapshots("never priced") == []

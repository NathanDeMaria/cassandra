from .odds import Odds, OddsDatabase, _parse_odds


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

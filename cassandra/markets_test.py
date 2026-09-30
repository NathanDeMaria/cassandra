from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from .betting import (
    LineColumns,
    american_to_probability,
    line_windows,
    no_vig_home_probability,
)
from .markets import (
    close_probabilities,
    close_probability,
    game_snapshots,
    history_from_days,
    kalshi_fee,
    probability_to_american,
)
from .odds import OddsDatabase

FIELDS = ["at", "price", "bid", "ask", "volume"]
KICKOFF = datetime(2026, 9, 27, 20, 5, tzinfo=UTC)


def _hour(hours_before: int, seconds: int = 0) -> int:
    top = KICKOFF.replace(minute=0) - timedelta(hours=hours_before)
    return int(top.timestamp()) + seconds


def _game(home_rows, away_rows, game_id="401"):
    return {
        "game_id": game_id,
        "kickoff": KICKOFF.isoformat(),
        "home": {"code": "SF", "prices": home_rows},
        "away": {"code": "ARI", "prices": away_rows},
    }


def test_a_cost_is_a_moneyline() -> None:
    assert probability_to_american(0.6) == pytest.approx(-150)
    assert probability_to_american(0.25) == pytest.approx(300)
    assert probability_to_american(0.5) == pytest.approx(-100)
    costs = pd.Series([0.05, 0.25, 0.5, 0.6, 0.95])
    back = american_to_probability(costs.map(probability_to_american))
    assert back.to_numpy() == pytest.approx(costs.to_numpy())


def test_a_settled_or_empty_book_is_no_price() -> None:
    for cost in (None, 0.0, 1.0, 1.2):
        assert probability_to_american(cost) is None


def test_kalshis_fee_peaks_at_even_money() -> None:
    assert kalshi_fee(0.5) == pytest.approx(0.0175)
    assert kalshi_fee(0.9) == pytest.approx(0.0063)


def test_kalshi_is_priced_at_the_ask_plus_the_fee() -> None:
    # The last trade (0.70) is stale and the bid (0.77) isn't what buying
    # costs; the ask is, and the fee comes on top of it.
    game = _game(
        [[_hour(1), 0.70, 0.77, 0.78, 100.0]],
        [[_hour(1), None, 0.22, 0.23, 100.0]],
    )

    [snapshot] = game_snapshots(game, "kalshi", FIELDS)

    assert snapshot.spread is None
    home_cost = american_to_probability(pd.Series([snapshot.home_moneyline]))[0]
    away_cost = american_to_probability(pd.Series([snapshot.away_moneyline]))[0]
    assert home_cost == pytest.approx(0.78 + kalshi_fee(0.78))
    assert away_cost == pytest.approx(0.23 + kalshi_fee(0.23))


def test_polymarket_sides_pair_by_hour_and_read_at_the_later_stamp() -> None:
    # One price an hour, stamped a few seconds past it and not quite
    # together. The pair is known once both are.
    game = _game(
        [
            [_hour(2, 17), 0.36, None, None, None],
            [_hour(1, 18), 0.37, None, None, None],
        ],
        [
            [_hour(2, 20), 0.64, None, None, None],
            [_hour(1, 18), 0.63, None, None, None],
        ],
    )

    first, second = game_snapshots(game, "polymarket", FIELDS)

    assert first.read_at == datetime.fromtimestamp(_hour(2, 20), UTC)
    assert second.read_at == datetime.fromtimestamp(_hour(1, 18), UTC)
    market = no_vig_home_probability(
        pd.Series([second.home_moneyline]), pd.Series([second.away_moneyline])
    )
    assert market[0] == pytest.approx(0.37)


def test_an_hour_only_one_side_traded_keeps_the_other_side_empty() -> None:
    game = _game(
        [[_hour(2), 0.6, None, None, None], [_hour(1), 0.6, None, None, None]],
        [[_hour(1), 0.4, None, None, None]],
    )

    first, second = game_snapshots(game, "polymarket", FIELDS)

    assert first.home_moneyline is not None and first.away_moneyline is None
    assert second.home_moneyline is not None and second.away_moneyline is not None


def test_an_hour_with_neither_side_priced_is_no_snapshot() -> None:
    game = _game(
        [[_hour(1), 1.0, None, None, None]], [[_hour(1), 0.0, None, None, None]]
    )

    assert game_snapshots(game, "polymarket", FIELDS) == []


def test_a_game_read_twice_is_the_later_read() -> None:
    day = {"fields": FIELDS, "games": [_game([[_hour(1), 0.4, 0, 0, 0]], [])]}
    again = {"fields": FIELDS, "games": [_game([[_hour(1), 0.6, 0, 0, 0]], [])]}

    history = history_from_days("polymarket", [day, again])

    [snapshot] = history["401"]
    assert snapshot.home_moneyline == pytest.approx(probability_to_american(0.6))


def test_market_prices_make_moneyline_windows_and_no_spread() -> None:
    """What `betting.py --book kalshi` leans on: the windows work unchanged.

    The close is the last hour before kickoff; the rows after it are the
    game being played, which a bet before kickoff can't have had.
    """
    game = _game(
        [
            [_hour(30), None, 0.59, 0.60, 1.0],
            [_hour(1), None, 0.64, 0.65, 1.0],
            [_hour(-1), None, 0.98, 0.99, 1.0],
        ],
        [
            [_hour(30), None, 0.40, 0.41, 1.0],
            [_hour(1), None, 0.35, 0.36, 1.0],
            [_hour(-1), None, 0.01, 0.02, 1.0],
        ],
    )
    database = OddsDatabase.from_history(
        history_from_days("kalshi", [{"fields": FIELDS, "games": [game]}])
    )
    predictions = pd.DataFrame(
        [
            dict(
                game_id="401",
                home_team="SF",
                away_team="ARI",
                date=KICKOFF,
                home_score=24,
                away_score=17,
                team1_win_prob=0.7,
                predicted_margin=5.0,
                year=2026,
            ),
            # Each team's previous game, a week back, which opens the window.
            dict(
                game_id="400",
                home_team="SF",
                away_team="ARI",
                date=KICKOFF - timedelta(days=7),
                home_score=0,
                away_score=0,
                team1_win_prob=0.5,
                predicted_margin=0.0,
                year=2026,
            ),
        ]
    )

    lines = line_windows(predictions, database).set_index("game_id")

    assert database.get_odds("401") is None
    row = lines.loc["401"]
    assert pd.isna(row[LineColumns.ENTRY_SPREAD])
    assert pd.isna(row[LineColumns.CLOSE_SPREAD])
    assert row[LineColumns.ENTRY_READ_AT] == datetime.fromtimestamp(_hour(30), UTC)
    assert row[LineColumns.CLOSE_READ_AT] == datetime.fromtimestamp(_hour(1), UTC)
    assert row[LineColumns.CLOSE_HOME_ML] == pytest.approx(
        probability_to_american(0.65 + kalshi_fee(0.65))
    )


def test_a_book_too_wide_to_bet_is_no_price() -> None:
    # Days out, both asks near a dollar: no-vig would call it 50/50 whatever
    # the game. The next hour has a real market.
    game = _game(
        [[_hour(50), None, 0.01, 0.95, 1.0], [_hour(2), None, 0.64, 0.65, 1.0]],
        [[_hour(50), None, 0.02, 0.94, 1.0], [_hour(2), None, 0.35, 0.36, 1.0]],
    )

    [snapshot] = game_snapshots(game, "kalshi", FIELDS)

    assert snapshot.read_at == datetime.fromtimestamp(_hour(2), UTC)


def _database(venue: str, game: dict) -> OddsDatabase:
    return OddsDatabase.from_history(
        history_from_days(venue, [{"fields": FIELDS, "games": [game]}])
    )


def test_the_close_is_the_last_two_sided_read_before_kickoff() -> None:
    game = _game(
        [
            [_hour(3), 0.60, None, None, None],
            [_hour(1), 0.70, None, None, None],
            [_hour(-1), 0.99, None, None, None],
        ],
        [
            [_hour(3), 0.40, None, None, None],
            [_hour(1), 0.30, None, None, None],
            [_hour(-1), 0.01, None, None, None],
        ],
    )

    close = close_probability(_database("polymarket", game), "401", KICKOFF)

    assert close == pytest.approx(0.70)


def test_close_probabilities_take_the_first_venue_that_priced_the_game() -> None:
    kalshi = _database(
        "kalshi",
        _game([[_hour(1), None, 0.6, 0.61, 1.0]], [[_hour(1), None, 0.38, 0.39, 1.0]]),
    )
    polymarket = OddsDatabase.from_history(
        history_from_days(
            "polymarket",
            [
                {
                    "fields": FIELDS,
                    "games": [
                        _game([[_hour(1), 0.5, 0, 0, 0]], [[_hour(1), 0.5, 0, 0, 0]]),
                        _game(
                            [[_hour(1), 0.3, 0, 0, 0]],
                            [[_hour(1), 0.7, 0, 0, 0]],
                            game_id="402",
                        ),
                    ],
                }
            ],
        )
    )
    games = pd.DataFrame(
        {
            "game_id": ["401", "402", "403"],
            # Naive, as some season files store it: taken as UTC.
            "date": [KICKOFF.replace(tzinfo=None)] * 3,
        }
    )

    closes = close_probabilities([kalshi, polymarket], games)

    kalshi_home = 0.61 + kalshi_fee(0.61)
    kalshi_away = 0.39 + kalshi_fee(0.39)
    assert closes[0] == pytest.approx(kalshi_home / (kalshi_home + kalshi_away))
    assert closes[1] == pytest.approx(0.3)
    assert pd.isna(closes[2])

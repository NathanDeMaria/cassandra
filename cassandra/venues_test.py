import math

import pytest

from .kalshi_venue import KALSHI_VENUE
from .markets import kalshi_fee
from .venues import KALSHI_FEES, Book, Holding, Resting, held_on, pending_on


def test_kalshis_order_fee_rounds_each_order_up_to_the_cent():
    for price in (0.05, 0.42, 0.5, 0.93):
        for count in (1, 7, 40):
            assert KALSHI_FEES.order(price, count) == (
                math.ceil(round(100 * count * kalshi_fee(price), 6)) / 100
            )
    assert KALSHI_FEES.contract(0.5) == pytest.approx(0.0175)


def test_whats_held_for_a_side_counts_every_contract_that_pays_on_it():
    side = Book("E-H", "H", "Home", 0.4, 1, 0.42, 1, 0, 0)
    other = Book("E-A", "A", "Away", 0.55, 1, 0.58, 1, 0, 0)
    holdings = [
        Holding("E-H", "yes", 3, 1.2),  # YES on the side
        Holding("E-A", "no", 2, 0.9),  # NO on its opponent: the same result
        Holding("E-A", "yes", 1, 0.6),  # against it
        Holding("F-X", "yes", 9, 4.0),  # another game
    ]
    assert held_on(KALSHI_VENUE, holdings, side, other) == (5, 1)
    assert KALSHI_VENUE.held_as(holdings[1]) == "2 NO A"


def test_a_resting_order_is_on_its_way_and_a_rung_isnt():
    side = Book("E-H", "H", "Home", 0.4, 1, 0.42, 1, 0, 0)
    other = Book("E-A", "A", "Away", 0.55, 1, 0.58, 1, 0, 0)
    resting = [
        Resting("a", "E-H", "yes", 0.42, 6),  # the pick's own order
        Resting("b", "E-A", "no", 0.41, 2),  # the same result, the other way
        Resting("c", "E-H", "yes", 0.38, 9, rung=True),  # a rung: extra
        Resting("d", "E-A", "yes", 0.55, 3),  # against
        Resting("e", "F-X", "yes", 0.30, 4),  # another game
    ]
    assert pending_on(KALSHI_VENUE, resting, side, other) == (8, 3)


def test_a_network_failure_is_the_venues_error_and_a_read_tries_again(monkeypatch):
    import urllib.request

    from . import kalshi_trade, kalshi_venue, polymarket_us
    from .kalshi_trade import KalshiClient, KalshiError
    from .polymarket_us import PolymarketUsError

    calls: list[str] = []

    def down(request, timeout=None):
        calls.append(getattr(request, "method", None) or "GET")
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    monkeypatch.setattr(kalshi_trade.time, "sleep", lambda _: None)
    monkeypatch.setattr(polymarket_us.time, "sleep", lambda _: None)
    monkeypatch.setattr(kalshi_trade, "sign", lambda *_: "sig")
    client = KalshiClient("id", None)
    with pytest.raises(KalshiError, match="timed out"):
        client.account()
    assert len(calls) == 4  # the read and its three retries
    calls.clear()
    # A write isn't tried again: it may have landed.
    with pytest.raises(KalshiError, match="timed out"):
        client.buy("T-X", "yes", 1, 0.5)
    assert calls == ["POST"]
    with pytest.raises(KalshiError):
        kalshi_venue.KALSHI_VENUE.games("ncaafb")
    with pytest.raises(PolymarketUsError):
        polymarket_us.POLYMARKET_US.games("mens")

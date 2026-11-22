import base64
from datetime import UTC, date, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from . import polymarket_us as pm
from .venues import Fill, Holding


def _event(closed: bool = False, bid: str | None = "0.9150") -> dict:
    """An open football event as `GET /v1/events` lists it, cut down."""
    return {
        "id": "190001",
        "slug": "cfb-furman-vamil-2026-11-21",
        "startTime": "2026-11-21T19:00:00Z",
        "markets": [
            {"slug": "spread-x", "marketType": "spread"},
            {
                "slug": "aec-cfb-furman-vamil-2026-11-21",
                "marketType": "moneyline",
                "active": True,
                "closed": closed,
                "startDate": "2026-11-07T17:30:10Z",
                "bestBidQuote": None if bid is None else {"value": bid},
                "bestAskQuote": {"value": "0.9200", "currency": "USD"},
                "marketSides": [
                    {
                        "long": True,
                        "team": {
                            "name": "Furman",
                            "abbreviation": "furman",
                            "alias": "Paladins",
                            "safeName": "Furman",
                        },
                    },
                    {
                        "long": False,
                        "team": {
                            "name": "VMI (W)",
                            "abbreviation": "vamil",
                            "alias": "Keydets",
                            "safeName": "VMI",
                        },
                    },
                ],
            },
        ],
    }


def test_an_events_moneyline_is_a_game_with_each_side_in_its_own_terms():
    game = pm.game(_event())
    assert game is not None
    assert game.event == "aec-cfb-furman-vamil-2026-11-21"
    assert game.day == date(2026, 11, 21)
    long, short = game.sides
    assert (long.code, long.team, long.outcome) == ("furman", "Furman", "long")
    assert (long.bid, long.ask) == (0.915, 0.92)
    # The short side buys by selling into the bids.
    assert short.outcome == "short"
    assert short.ask == pytest.approx(0.085) and short.bid == pytest.approx(0.08)
    assert short.team == "VMI" and "VMI Keydets" in short.names
    assert long.opened == datetime(2026, 11, 7, 17, 30, 10, tzinfo=UTC)


def test_a_closed_or_one_sided_moneyline_isnt_a_game_to_price():
    assert pm.game(_event(closed=True)) is None
    game = pm.game(_event(bid=None))
    assert game is not None and game.sides[1].ask is None
    found, odd = pm.games([_event(closed=True), _event()])
    assert len(found) == 1 and odd == []


def test_backing_a_side_is_its_own_side_of_the_instrument():
    game = pm.game(_event())
    assert game is not None
    long, short = game.sides
    venue = pm.POLYMARKET_US
    [route] = venue.routes(short, long)
    assert (route.action, route.price) == ("short", pytest.approx(0.085))
    assert venue.pays(long, short, long.ticker, "long") is True
    assert venue.pays(long, short, long.ticker, "short") is False
    assert venue.pays(short, long, long.ticker, "short") is True
    assert venue.pays(long, short, "aec-cfb-other", "long") is None
    assert venue.league_of(long.ticker) == "ncaafb"
    assert venue.league_of("aec-wcbb-alc-bella-2026-11-20") == "womens"
    assert venue.league_of("nfl-x") is None


def test_whats_on_offer_reads_the_yes_book_for_both_sides():
    book = {
        "marketData": {
            "offers": [
                {"px": {"value": "0.55"}, "qty": "10"},
                {"px": {"value": "0.56"}, "qty": "20"},
            ],
            "bids": [
                {"px": {"value": "0.53"}, "qty": "7"},
                {"px": {"value": "0.50"}, "qty": "9"},
            ],
        }
    }
    assert pm.on_offer(book, "long", 0.55) == 10
    assert pm.on_offer(book, "long", 0.56) == 30
    # Buying the short side at most 0.48 sells YES at 0.52 or more.
    assert pm.on_offer(book, "short", 0.48) == 7
    assert pm.on_offer(book, "short", 0.50) == 16


def test_an_order_sends_a_short_in_yes_terms():
    long = pm.order_body("aec-x", "long", 5, 0.43)
    assert long["intent"] == "ORDER_INTENT_BUY_LONG"
    assert long["price"] == {"value": "0.4300", "currency": "USD"}
    assert long["tif"] == "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL"
    assert "goodTillTime" not in long
    short = pm.order_body("aec-x", "short", 5, 0.43, expires=1795200000)
    assert short["intent"] == "ORDER_INTENT_BUY_SHORT"
    assert short["price"]["value"] == "0.5700"
    assert short["tif"] == "TIME_IN_FORCE_GOOD_TILL_DATE"
    assert short["goodTillTime"] == "2026-11-20T18:40:00Z"
    with pytest.raises(ValueError):
        pm.order_body("aec-x", "yes", 5, 0.43)


def test_a_fill_is_read_from_its_order_and_priced_as_the_side_bought():
    response = {
        "id": "ord-1",
        "executions": [
            {"type": "EXECUTION_TYPE_PARTIAL_FILL", "lastShares": "3"},
            {
                "type": "EXECUTION_TYPE_FILL",
                "order": {
                    "cumQuantity": 5,
                    "leavesQuantity": 0,
                    "state": "ORDER_STATE_FILLED",
                    "avgPx": {"value": "0.5800"},
                    "commissionNotionalTotalCollected": {"value": "0.08"},
                },
            },
        ],
    }
    done = pm.fill(response, "short", 5)
    assert done._replace(price=None) == Fill("ord-1", 5.0, 0.0, None, 0.08)
    assert done.price == pytest.approx(0.42)
    resting = pm.fill(
        {
            "id": "ord-2",
            "executions": [
                {
                    "order": {
                        "cumQuantity": 0,
                        "leavesQuantity": 5,
                        "state": "ORDER_STATE_NEW",
                    }
                }
            ],
        },
        "long",
        5,
    )
    assert (resting.filled, resting.resting, resting.price) == (0.0, 5.0, None)


def test_a_fill_over_its_limit_stops_the_runs_short_orders():
    pm.check_fill(Fill("o", 5, 0, 0.42, 0.0), "short", 0.43)
    with pytest.raises(pm.PolymarketUsError, match="over its 0.43 limit"):
        pm.check_fill(Fill("o", 5, 0, 0.57, 0.0), "short", 0.43)


def test_positions_orders_and_balance_read_as_the_venues_shapes():
    found = pm.holdings(
        {
            "aec-a": {"netPositionDecimal": "-4.00", "cost": {"value": "1.60"}},
            "aec-b": {"netPositionDecimal": "3", "cost": {"value": "2.10"}},
            "aec-c": {"netPositionDecimal": "0"},
            "aec-d": {"netPositionDecimal": "2", "expired": True},
        }
    )
    assert found == [
        Holding("aec-a", "short", 4.0, 1.60),
        Holding("aec-b", "long", 3.0, 2.10),
    ]
    [order] = pm.resting(
        [
            {
                "id": "o1",
                "marketSlug": "aec-a",
                "intent": "ORDER_INTENT_BUY_SHORT",
                "price": {"value": "0.6000"},
                "leavesQuantity": 5,
            },
            {"id": "o2", "marketSlug": "aec-a", "intent": "ORDER_INTENT_SELL_LONG"},
        ]
    )
    assert (order.action, order.price, order.remaining) == ("short", 0.4, 5.0)
    # Shaped as the account read on 2026-11-22: a long, and a short holding
    # margin, which isn't what the positions are worth.
    money = pm.balance(
        [
            {
                "currency": "USD",
                "currentBalance": 52.0,
                "buyingPower": 50.0,
                "assetNotional": 0,
                "openOrders": 0.5,
                "marginRequirement": 2,
            }
        ],
        {
            "aec-b": {"netPositionDecimal": "9", "cashValue": {"value": "5.94"}},
            "aec-a": {"netPositionDecimal": "-3", "cashValue": {"value": "1.38"}},
            "aec-x": {
                "netPositionDecimal": "1",
                "cashValue": {"value": "0.9"},
                "expired": True,
            },
        },
    )
    assert money.cash == pytest.approx(50.0)
    assert money.invested == pytest.approx(5.94 + 1.38 + 0.5)


def test_a_request_is_signed_over_its_time_method_and_path():
    seed = bytes(range(32))
    key = pm.private_key(base64.b64encode(seed + bytes(32)).decode())
    signature = pm.sign(key, "1700000000000", "GET", "/v1/portfolio/positions?limit=5")
    public = Ed25519PrivateKey.from_private_bytes(seed).public_key()
    public.verify(
        base64.b64decode(signature), b"1700000000000GET/v1/portfolio/positions"
    )
    assert public.public_bytes(
        Encoding.Raw, PublicFormat.Raw
    ) == key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def test_the_api_promotion_prices_contracts_net_of_its_rebate():
    during = datetime(2026, 12, 1, tzinfo=UTC)
    after = datetime(2027, 1, 2, tzinfo=UTC)
    assert pm.FEES.contract(0.5, during) == pytest.approx(0.0695 * 0.3 * 0.25)
    assert pm.FEES.contract(0.5, after) == pytest.approx(0.095 * 0.25)
    # The order takes the whole fee when it fills, to the cent half to even:
    # the docs' 100 at 0.50 is $1.74.
    assert pm.FEES.order(0.5, 100, during) == 1.74
    assert pm.FEES.order(0.5, 2, during) == 0.03


def test_a_probe_uses_a_game_whose_short_side_is_a_clear_favorite():
    from datetime import timedelta

    now = datetime(2026, 11, 21, 12, tzinfo=UTC)
    soon = {**_event(bid="0.1500"), "startTime": "2026-11-21T13:00:00Z"}
    later = {**_event(bid="0.2000"), "startTime": "2026-11-21T19:00:00Z"}
    close = {**_event(bid="0.5000"), "startTime": "2026-11-21T19:00:00Z"}
    found = pm.probe_game([soon, later, close], now)
    assert found is not None
    read, kickoff = found
    # The short side asks 1 - 0.20: the soon game is too near kickoff, the
    # close one no favorite.
    assert read.sides[1].ask == pytest.approx(0.80)
    assert kickoff - now == timedelta(hours=7)
    assert pm.probe_game([close], now) is None


def test_the_book_says_which_way_a_no_buy_was_read():
    def book(offers: dict[float, float]) -> dict:
        return {
            "marketData": {
                "offers": [
                    {"px": {"value": str(p)}, "qty": str(q)} for p, q in offers.items()
                ]
            }
        }

    before = book({0.26: 40.0, 0.70: 5.0})
    assert pm.convention(before, book({0.26: 40.0, 0.70: 6.0}), 0.30) == "yes"
    assert pm.convention(before, book({0.26: 40.0, 0.30: 1.0, 0.70: 5.0}), 0.30) == "no"
    assert pm.convention(before, before, 0.30) is None


def test_the_listed_order_says_which_way_a_no_buy_was_read():
    # As the first probe's order was listed, 2026-11-22.
    order = {
        "intent": "ORDER_INTENT_BUY_SHORT",
        "side": "ORDER_SIDE_SELL",
        "price": {"value": "0.7", "currency": "USD"},
        "quantity": 1,
        "principal": {"value": "0.3", "currency": "USD"},
    }
    assert pm.order_convention(order, 0.30) == "yes"
    assert pm.order_convention({**order, "principal": {"value": "0.7"}}, 0.30) == "no"
    assert pm.order_convention({**order, "principal": None}, 0.30) is None

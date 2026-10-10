from datetime import UTC, date, datetime, timedelta

import pytest
from endgame.types import Game

from .markets import kalshi_fee
from .picks import (
    Book,
    Fixture,
    KalshiGame,
    Matched,
    PreviousGame,
    allocate,
    book,
    contracts_for,
    entry,
    event_day,
    fillable,
    kalshi_games,
    kelly,
    match,
    max_price,
    no_vig_home,
    order_fee,
    previous_games,
    price,
    routes,
    skip_reason,
)

SATURDAY = datetime(2026, 10, 17, 18, 0, tzinfo=UTC)


def _market(event: str, code: str, team: str, bid="0.4000", ask="0.4200", **extra):
    return {
        "event_ticker": event,
        "ticker": f"{event}-{code}",
        "yes_sub_title": team,
        "yes_bid_dollars": bid,
        "yes_bid_size_fp": "100.00",
        "yes_ask_dollars": ask,
        "yes_ask_size_fp": "250.00",
        "volume_fp": "10.00",
        "volume_24h_fp": "0.00",
        **extra,
    }


def _book(code: str, bid: float | None, ask: float | None, size=100.0) -> Book:
    return Book(f"KXNCAAFGAME-26OCT17X-{code}", code, code, bid, size, ask, size, 0, 0)


def _game(
    home: str, away: str, when=SATURDAY, completed=False, status="", game_id=None
):
    return Game(
        home=home,
        home_score=0,
        away=away,
        away_score=0,
        neutral_site=False,
        completed=completed,
        date=when,
        game_id=game_id or f"{away}@{home}",
        status=status,
    )


def test_the_event_ticker_carries_the_eastern_date() -> None:
    assert event_day("KXNCAAFGAME-26OCT17BRWNPRIN") == date(2026, 10, 17)
    assert event_day("KXNCAAFGAME-NODATE") is None


def test_an_empty_side_of_the_book_is_no_quote() -> None:
    read = book(_market("E-26OCT17AB", "A", "Alpha", bid="0.0000", ask="1.0000"))
    assert read.bid is None and read.ask is None and read.spread is None
    assert book(_market("E-26OCT17AB", "A", "Alpha")).spread == pytest.approx(0.02)


def test_an_event_is_a_game_when_it_has_two_teams() -> None:
    markets = [
        _market("KXNCAAFGAME-26OCT17AB", "B", "Beta"),
        _market("KXNCAAFGAME-26OCT17AB", "A", "Alpha"),
        _market("KXNCAAFGAME-26OCT17AB", "TIE", "Tie"),
        _market("KXNCAAFGAME-26OCT17CD", "C", "Gamma"),
    ]
    games, odd = kalshi_games(markets)
    assert [g.event for g in games] == ["KXNCAAFGAME-26OCT17AB"]
    assert [side.code for side in games[0].sides] == ["A", "B"]
    assert odd == ["KXNCAAFGAME-26OCT17CD"]


def _ids(table: dict[str, set[str]]):
    return lambda side: frozenset(table.get(side.code, set()))


def test_home_and_away_come_from_espn_not_kalshi() -> None:
    kalshi = KalshiGame(
        "E", date(2026, 10, 17), (_book("A", 0.3, 0.32), _book("B", 0.66, 0.68))
    )
    fixture = Fixture(_game("Beta", "Alpha"), home_id="2", away_id="1")
    matched, unmatched = match([fixture], [kalshi], _ids({"A": {"1"}, "B": {"2"}}))
    assert unmatched == []
    found = matched[fixture.game.game_id]
    assert found.home.code == "B" and found.away.code == "A"


def test_a_shared_code_is_settled_by_the_opponent() -> None:
    # Kalshi writes Washington State and Wayne State both as WSU.
    kalshi = KalshiGame(
        "E", date(2026, 10, 17), (_book("OSU", 0.5, 0.52), _book("WSU", 0.46, 0.48))
    )
    fixtures = [
        Fixture(_game("Washington State", "Oregon State"), "265", "204"),
        Fixture(_game("Wayne State", "Ferris State"), "131", "2222"),
    ]
    table = {"OSU": {"204"}, "WSU": {"265", "131"}}
    matched, _ = match(fixtures, [kalshi], _ids(table))
    assert list(matched) == ["Oregon State@Washington State"]


def test_a_game_two_fixtures_fit_is_not_matched() -> None:
    kalshi = KalshiGame(
        "E", date(2026, 10, 17), (_book("A", 0.3, 0.32), _book("B", 0.6, 0.62))
    )
    fixtures = [
        Fixture(_game("Beta", "Alpha", game_id="1"), "2", "1"),
        Fixture(
            _game("Beta", "Alpha", when=SATURDAY + timedelta(hours=3), game_id="2"),
            "2",
            "1",
        ),
    ]
    matched, unmatched = match(fixtures, [kalshi], _ids({"A": {"1"}, "B": {"2"}}))
    assert matched == {} and unmatched == ["E"]


def test_the_market_is_the_two_asks_with_fees_normalized() -> None:
    home, away = _book("H", 0.58, 0.60), _book("A", 0.42, 0.44)
    found = no_vig_home(home, away)
    assert found is not None
    probability, overround = found
    home_cost, away_cost = 0.60 + kalshi_fee(0.60), 0.44 + kalshi_fee(0.44)
    assert overround == pytest.approx(home_cost + away_cost)
    assert probability == pytest.approx(home_cost / (home_cost + away_cost))
    assert no_vig_home(home, _book("A", 0.42, None)) is None


def test_backing_a_team_takes_the_cheaper_of_its_yes_and_the_opponents_no() -> None:
    side, other = _book("H", 0.55, 0.60), _book("A", 0.42, 0.44)
    best = routes(side, other)[0]
    assert (best.ticker, best.action, best.price) == (
        other.ticker,
        "no",
        pytest.approx(0.58),
    )
    tied = routes(_book("H", 0.55, 0.58), other)[0]
    assert tied.action == "yes"


def test_the_fee_rounds_up_to_the_cent_per_order() -> None:
    assert order_fee(0.5, 1) == 0.02
    assert order_fee(0.5, 100) == 1.75
    assert order_fee(0.5, 0) == 0.0


def test_a_stake_buys_what_it_can_with_the_fee_in() -> None:
    count = contracts_for(20, 0.4)
    assert count * 0.4 + order_fee(0.4, count) <= 20
    assert (count + 1) * 0.4 + order_fee(0.4, count + 1) > 20


def test_kelly_backs_only_a_positive_expected_return() -> None:
    assert kelly(0.6, 0.5) == pytest.approx(0.2)
    assert kelly(0.5, 0.52) == 0.0


def test_a_budget_is_split_by_weight_and_spent_within_itself() -> None:
    prices, weights = [0.40, 0.20], [0.3, 0.1]
    counts = allocate(20, prices, weights)
    assert counts == [contracts_for(15, 0.40), contracts_for(5, 0.20)]
    spent = sum(n * p + order_fee(p, n) for n, p in zip(counts, prices))
    assert spent <= 20


def test_a_share_too_small_for_a_contract_goes_to_the_rest() -> None:
    # $1 at a 0.9 weight to 0.1: the favorite's 10 cents can't buy a contract
    # at 0.85, so the whole dollar goes to the other pick.
    assert allocate(1, [0.30, 0.85], [0.9, 0.1]) == [contracts_for(1, 0.30), 0]
    assert allocate(1, [0.30, 0.85], [0.0, 0.0]) == [0, 0]


def test_buying_yes_fills_against_no_bids() -> None:
    orderbook = {
        "orderbook_fp": {
            "no_dollars": [
                ["0.3800", "50.00"],
                ["0.3900", "20.00"],
                ["0.4000", "5.00"],
            ],
            "yes_dollars": [["0.5500", "70.00"]],
        }
    }
    # NO bids at 0.40 and 0.39 are YES asks at 0.60 and 0.61.
    assert fillable(orderbook, "yes", 0.61) == pytest.approx(25)
    assert fillable(orderbook, "yes", 0.60) == pytest.approx(5)
    assert fillable(orderbook, "no", 0.45) == pytest.approx(70)


def test_a_team_waits_on_its_last_game_but_not_a_canceled_one() -> None:
    week = timedelta(days=7)
    games = [
        _game("Alpha", "Beta", when=SATURDAY - week, completed=True, game_id="done"),
        _game(
            "Gamma",
            "Alpha",
            when=SATURDAY - timedelta(days=2),
            status="STATUS_CANCELED",
        ),
        _game("Delta", "Beta", when=SATURDAY - timedelta(hours=2), game_id="live"),
    ]
    previous = previous_games(games)
    assert previous("Alpha", SATURDAY) == PreviousGame("done", SATURDAY - week, True)
    assert previous("Beta", SATURDAY) == PreviousGame(
        "live", SATURDAY - timedelta(hours=2), False
    )
    assert previous("Nobody", SATURDAY) is None


def test_the_entry_waits_for_the_later_of_the_two_previous_games() -> None:
    earlier = PreviousGame("1", SATURDAY - timedelta(days=7), True)
    later = PreviousGame("2", SATURDAY - timedelta(days=5), True)
    assert entry([earlier, later]) == later.date + timedelta(hours=4)
    assert entry([None, earlier]) == earlier.date + timedelta(hours=4)
    assert entry([None, None]) is None


def _matched(home_ask: float, away_ask: float, spread=0.02) -> Matched:
    home = _book("H", home_ask - spread, home_ask)
    away = _book("A", away_ask - spread, away_ask)
    return Matched(KalshiGame("E", date(2026, 10, 17), (home, away)), home, away)


def test_the_model_takes_the_side_it_likes_more_than_the_market() -> None:
    priced = price(0.45, _matched(0.60, 0.42))
    assert priced is not None and priced.route is not None
    assert not priced.home
    assert priced.edge == pytest.approx(priced.home_market - 0.45)
    cost = priced.route.price + kalshi_fee(priced.route.price)
    assert priced.cost == pytest.approx(cost)
    assert priced.expected_return == pytest.approx(0.55 / cost - 1)


def test_the_max_price_is_the_last_cent_that_clears_the_edge() -> None:
    matched = _matched(0.60, 0.42)
    priced = price(0.80, matched)
    assert priced is not None and priced.route is not None
    highest = max_price(0.80, priced, matched.home)
    assert highest is not None and highest > priced.route.price

    def edge_at(ask: float) -> float:
        return 0.80 - (ask + kalshi_fee(ask)) / priced.overround

    assert edge_at(highest) >= 0.05 > edge_at(highest + 0.01)
    # NO on the opponent moves with the side's YES ask.
    cheaper = _matched(0.60, 0.42)._replace(away=_book("A", 0.42, 0.44))
    via_no = price(0.80, cheaper)
    assert via_no is not None and via_no.route is not None
    assert via_no.route.action == "no"
    found = max_price(0.80, via_no, cheaper.home)
    moved = max_price(0.80, via_no, cheaper.home._replace(ask=via_no.route.price))
    assert found is not None and moved is not None
    assert found == pytest.approx(moved - (0.60 - via_no.route.price))


def test_what_isnt_a_pick() -> None:
    priced = price(0.80, _matched(0.60, 0.42))
    assert priced is not None and priced.edge > 0.05
    fresh = timedelta(hours=2)

    def reason(side, other, waiting=(), since=fresh, **kwargs):
        return skip_reason(priced, side, other, list(waiting), since, **kwargs) or ""

    assert reason("FBS", "FBS") == ""
    assert reason("FBS", "FBS", since=None) == ""
    # The model's known miss: it under-rates FBS teams against lower ones.
    assert reason("FCS", "FBS") == "lower-division side vs FBS"
    assert reason("FBS", "FCS") == ""
    assert reason("FCS", "FCS") == ""
    assert reason("FBS", "FBS", waiting=["Alpha"]).startswith("previous game")
    assert reason("FBS", "FBS", since=timedelta(days=2)) == "past the entry window"
    assert reason("FBS", "FBS", min_edge=0.5) == "edge under threshold"
    wide = price(0.80, _matched(0.60, 0.55))
    assert wide is not None
    assert (skip_reason(wide, "FBS", "FBS", [], fresh) or "").startswith(
        "book too wide"
    )

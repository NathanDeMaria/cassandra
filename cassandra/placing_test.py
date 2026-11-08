import argparse
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import pandas as pd
import pytest

from . import placing
from .kalshi_trade import Fill, KalshiClient, KalshiError
from .picks import Book


class _Matched(NamedTuple):
    home: Book
    away: Book


def _book(ticker: str) -> Book:
    return Book(ticker, ticker, ticker, 0.40, 10, 0.42, 10, 0, 0)


@pytest.fixture
def no_books(monkeypatch):
    # `size` reads each pick's order book for what's on offer; none here.
    monkeypatch.setattr(placing, "public_get", lambda path: {"orderbook_fp": {}})


def _frame(held: list[float]) -> tuple[pd.DataFrame, dict]:
    rows, matched = [], {}
    for i, h in enumerate(held):
        game_id = f"g{i}"
        matched[game_id] = _Matched(_book(f"H{i}"), _book(f"A{i}"))
        rows.append(
            {
                "game_id": game_id,
                "edge": 0.20 - 0.02 * i,
                "held": h,
                "model": 0.60,
                "cost": 0.44,
                "limit": 0.43,
                "bet_home": True,
                "skip": None,
            }
        )
    return pd.DataFrame(rows), matched


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            "top": 2,
            "kelly_fraction": 0.25,
            "bankroll": 100.0,
            "max_bet": 10.0,
            "budget": None,
            "cash": 50.0,
            "slippage": 0.01,
            **kw,
        }
    )


@pytest.mark.parametrize("fraction", [0.25, None])
def test_a_pick_held_in_full_doesnt_take_a_place_in_the_top(no_books, fraction):
    frame, matched = _frame([50, 0, 3, 0])
    sized, picks = placing.size(frame, matched, _args(kelly_fraction=fraction))
    assert list(picks["game_id"]) == ["g1", "g2"]
    skips = dict(zip(sized["game_id"], sized["skip"]))
    assert skips["g0"].startswith("already held (50 of")
    assert skips["g3"] == "outside the top 2 by edge"


def test_contracts_line_up_with_their_picks(no_books):
    # The top is taken by edge, the picks come back in the frame's order;
    # each pick's contracts must be its own.
    frame, matched = _frame([0, 0, 0])
    frame.loc[0, "model"] = 0.50  # the biggest edge, but a smaller Kelly stake
    frame = frame.iloc[::-1].reset_index(drop=True)
    _, picks = placing.size(frame, matched, _args(top=3))
    by_game = dict(zip(picks["game_id"], picks["contracts"]))
    assert by_game["g0"] < by_game["g1"] == by_game["g2"]


def test_cash_cuts_every_order_and_names_the_ones_left_out(no_books):
    frame, matched = _frame([0, 0])
    sized, picks = placing.size(frame, matched, _args(cash=0.20))
    assert picks.empty
    assert set(sized["skip"]) == {"no cash left for it"}


def test_the_limit_is_slippage_past_the_ask_but_never_past_the_max_price():
    assert placing.limit_price(0.40, 0.45, 0.01) == 0.41
    assert placing.limit_price(0.40, 0.40, 0.01) == 0.40
    assert placing.limit_price(0.40, None, 0.01) == 0.41
    assert placing.limit_price(0.99, None, 0.01) == 0.99


def _priced(
    model: float = 0.60, max_price: float | None = 0.50
) -> tuple[pd.DataFrame, dict]:
    matched = {"g0": _Matched(_book("EV-HOME"), _book("EV-AWAY"))}
    frame = pd.DataFrame(
        [
            {
                "game_id": "g0",
                "event": "EV",
                "game": "Away Team @ Home Team",
                "bet": "Home Team",
                "bet_home": True,
                "model": model,
                "price": 0.42,
                "max_price": max_price,
                "skip": "past the entry window",
            }
        ]
    )
    return frame, matched


def _rung(ticker: str, book_side: str, yes_price: float) -> dict:
    return {
        "order_id": f"{ticker}-{book_side}",
        "client_order_id": "rung-x",
        "ticker": ticker,
        "outcome_side": "yes",
        "book_side": book_side,
        "yes_price_dollars": f"{yes_price:.2f}",
    }


def test_a_rung_stays_while_the_model_still_backs_its_price():
    frame, matched = _priced()
    # YES on the home team at 0.40, and NO on the away team at 0.40: both
    # back the home team, under the 0.50 that still clears the edge.
    orders = [_rung("EV-HOME", "bid", 0.40), _rung("EV-AWAY", "ask", 0.60)]
    assert placing.stale_rungs(orders, frame, matched) == []


def test_a_rung_goes_when_its_price_no_longer_clears_the_edge():
    frame, matched = _priced(max_price=0.38)
    [(order, why)] = placing.stale_rungs(
        [_rung("EV-HOME", "bid", 0.40)], frame, matched
    )
    assert why.startswith("0.40 is over the 0.38")
    frame, matched = _priced(max_price=None)
    [(_, why)] = placing.stale_rungs([_rung("EV-HOME", "bid", 0.40)], frame, matched)
    assert why == "no price clears the edge now"


def test_a_rung_goes_when_the_model_switches_sides_or_the_game_is_excluded():
    frame, matched = _priced()
    # YES on the away team: the pick now backs home.
    [(_, why)] = placing.stale_rungs([_rung("EV-AWAY", "bid", 0.55)], frame, matched)
    assert why == "the model backs Home Team now"
    found = placing.stale_rungs(
        [_rung("EV-HOME", "bid", 0.40)], frame, matched, exclude=["home team"]
    )
    assert [why for _, why in found] == ["excluded by hand"]
    # A game this run didn't price is left alone.
    assert placing.stale_rungs([_rung("OTHER-X", "bid", 0.4)], frame, matched) == []


def test_rungs_come_after_the_picks_out_of_the_cash(no_books):
    frame, matched = _frame([0])
    frame["event"] = "E0"
    frame["price"] = 0.42
    args = _args(rungs=(0.02, 0.04), rung_events=set(), cash=50.0)
    _, picks = placing.size(frame, matched, args)
    ladder = picks["rungs"].iloc[0]
    assert [p for p, _ in ladder] == [0.40, 0.38]
    # Too little cash for the rungs: the pick keeps its contracts.
    entry = picks["outlay"].iloc[0]
    _, tight = placing.size(
        frame, matched, _args(rungs=(0.02, 0.04), rung_events=set(), cash=entry + 0.5)
    )
    assert tight["contracts"].iloc[0] == picks["contracts"].iloc[0]
    assert tight["rung_outlay"].iloc[0] <= 0.5
    # A game with rungs already resting gets none.
    _, again = placing.size(
        frame, matched, _args(rungs=(0.02, 0.04), rung_events={"E0"})
    )
    assert again["rungs"].iloc[0] == []


def test_qb_news_names_the_picks_whose_news_names_a_quarterback():
    from .injuries import Note

    picks = pd.DataFrame({"game_id": ["g0", "g1", "g2"]})
    news = {
        "g0": placing.News(
            [
                Note("A", False, "The left tackle is out.", "story"),
                Note("A", True, "The starting quarterback is doubtful.", "story"),
            ],
            [],
        ),
        "g1": placing.News([Note("B", False, "A linebacker is out.", "story")], []),
    }
    assert placing.qb_news(picks, news) == {
        "g0": "The starting quarterback is doubtful."
    }


def test_a_pick_whose_side_has_its_starter_in_doubt_is_held_too():
    """The stories said nothing about South Florida's starter; the plays did.
    The doubt is read before the news, and only for the side bet on, which
    `qb_doubt` already is."""
    from .injuries import Note

    doubt = "South Florida Bulls: M.Van Buren (8 starts) threw 3 of 13 passes Oct 31"
    picks = pd.DataFrame(
        {"game_id": ["g0", "g1", "g2"], "qb_doubt": [doubt, None, doubt]}
    )
    news = {"g2": placing.News([Note("B", True, "The QB is out.", "story")], [])}
    assert placing.qb_news(picks, news) == {"g0": doubt, "g2": doubt}


def test_a_pick_says_its_edge_if_its_starter_sits():
    row = pd.Series({"bet": "Tulane Green Wave", "edge_qb_out": 0.0231})
    assert (
        placing.if_out_line(row) == "if Tulane Green Wave's starter is out: edge +0.023"
    )
    # A league whose model doesn't read quarterbacks has no such edge.
    assert placing.if_out_line(row.drop("edge_qb_out")) is None
    assert placing.if_out_line(pd.Series({"bet": "X", "edge_qb_out": None})) is None


def test_a_rung_goes_when_its_side_has_its_starter_in_doubt():
    frame, matched = _priced()
    frame["qb_doubt"] = "Home Team: A.Ace (5 starts) didn't play Oct 31"
    orders = [_rung("EV-HOME", "bid", 0.40)]
    # Only when the run holds on it: an unattended run, or --hold-qb by hand.
    assert placing.stale_rungs(orders, frame, matched) == []
    [(_, why)] = placing.stale_rungs(orders, frame, matched, hold_qb=True)
    assert why == "QB in doubt: Home Team: A.Ace (5 starts) didn't play Oct 31"


def test_a_leagues_exposure_is_its_positions_and_resting_orders():
    held = [
        {"ticker": "KXNCAAWBGAME-26NOV05UCFFLA-FLA", "market_exposure_dollars": "4.20"},
        {"ticker": "KXNCAAFGAME-26NOV07OSUPUR-OSU", "market_exposure_dollars": "9.00"},
    ]
    resting = [
        # 10 YES at 30c, and a NO bid at a 60c YES price: 5 NO at 40c.
        {
            "ticker": "KXNCAAWBGAME-26NOV05JACMIA-JAC",
            "side": "yes",
            "action": "buy",
            "yes_price_dollars": "0.30",
            "remaining_count_fp": "10",
        },
        {
            "ticker": "KXNCAAWBGAME-26NOV05HAMPSMU-SMU",
            "side": "no",
            "action": "buy",
            "yes_price_dollars": "0.60",
            "remaining_count_fp": "5",
        },
        {
            "ticker": "KXNCAAFGAME-26NOV07OSUPUR-PUR",
            "side": "yes",
            "action": "buy",
            "yes_price_dollars": "0.10",
            "remaining_count_fp": "50",
        },
    ]
    womens = placing.league_exposure(held, resting, ("KXNCAAWBGAME",))
    assert womens == pytest.approx(4.20 + 3.00 + 2.00)
    football = placing.league_exposure(held, resting, ("KXNCAAFGAME", "KXNCAAFCSGAME"))
    assert football == pytest.approx(9.00 + 5.00)
    # 40% of $100 with $9.20 out leaves $30.80, and never less than none.
    assert placing.exposure_room(100.0, 0.4, womens) == pytest.approx(30.80)
    assert placing.exposure_room(100.0, 0.05, womens) == 0.0


def test_a_longshot_is_sized_smaller_and_says_so(no_books):
    frame, matched = _frame([0, 0])
    frame["cost"] = [0.44, 0.12]
    frame["limit"] = [0.43, 0.11]
    frame["model"] = [0.60, 0.30]
    _, full = placing.size(frame, matched, _args())
    args = _args(longshot_price=0.15, longshot_scale=0.6)
    _, shrunk = placing.size(frame, matched, args)
    assert shrunk["contracts"].iloc[0] == full["contracts"].iloc[0]
    assert shrunk["contracts"].iloc[1] < full["contracts"].iloc[1]
    assert "0.6x under 15c" in placing.sizing(args)


def test_an_order_rests_no_later_than_kickoff():
    now = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)
    tip = now + timedelta(hours=3)
    assert placing.order_expiry(None, tip, now) is None
    assert placing.order_expiry(2.0, tip, now) == int(
        (now + timedelta(hours=2)).timestamp()
    )
    assert placing.order_expiry(8.0, tip, now) == int(tip.timestamp())


class _Exchange(KalshiClient):
    """The Kalshi client's side of placing: shards that a transfer reaches
    `lag` reads after it's sent, and orders that fail on the tickers named."""

    def __init__(self, shards, lag=0, dry_run=False, reject=(), refuse=False):
        self.shards = dict(shards)
        self.lag = lag
        self.dry_run = dry_run
        self.reject = reject
        self.refuse = refuse
        self.landing: list[list] = []
        self.log: list[tuple] = []

    def shard_balances(self):
        for move in self.landing:
            move[0] -= 1
            if move[0] < 0:
                self.shards[move[1]] = self.shards.get(move[1], 0) + move[2]
        self.landing = [move for move in self.landing if move[0] >= 0]
        return dict(self.shards)

    def transfer(self, source, destination, centicents):
        if self.refuse:
            raise KalshiError("POST transfer: 403 nope")
        self.log.append(("transfer", source, destination, centicents))
        self.shards[source] -= centicents
        self.landing.append([self.lag, destination, centicents])
        return "t1"

    def buy(self, ticker, action, count, limit, expires_at=None, client_order_id=None):
        self.log.append(("buy", ticker))
        if ticker in self.reject:
            raise KalshiError(
                'POST: 404 {"error":{"code":"insufficient_shard_balance"}}'
            )
        return Fill("o1", float(count), 0.0, limit, 0.01)


def _no_sleep(_: float) -> None:
    pass


def test_funding_moves_only_whats_short_and_sets_it_aside():
    exchange = _Exchange({0: 100_000, 3: 10_000})
    balances = exchange.shard_balances()
    placing.fund_shard(exchange, balances, 3, 0.50, sleep=_no_sleep)
    assert exchange.log == []  # $1 there, 50c needed
    assert balances[3] == 5_000
    placing.fund_shard(exchange, balances, 3, 2.00, sleep=_no_sleep)
    assert exchange.log == [("transfer", 0, 3, 15_000)]
    assert balances == {0: 85_000, 3: 5_000}  # what the landed read says, less this


def test_funding_waits_for_the_transfer_to_land():
    exchange = _Exchange({0: 100_000, 3: 0}, lag=3)
    slept: list[float] = []
    balances = {0: 100_000, 3: 0}
    placing.fund_shard(exchange, balances, 3, 1.00, sleep=slept.append)
    assert slept == [1.0, 1.0, 1.0]
    assert balances[3] == 0


def test_funding_gives_up_on_a_transfer_that_doesnt_land():
    exchange = _Exchange({0: 100_000, 3: 0}, lag=10**6)
    with pytest.raises(KalshiError, match="not in after"):
        placing.fund_shard(
            exchange,
            {0: 100_000, 3: 0},
            3,
            1.00,
            wait=timedelta(0),
            sleep=_no_sleep,
        )


def test_a_dry_run_moves_on_paper_and_doesnt_wait():
    exchange = _Exchange({0: 100_000, 3: 0}, lag=10**6, dry_run=True)
    balances = {0: 100_000, 3: 0}

    def boom(_: float) -> None:
        raise AssertionError("waited")

    placing.fund_shard(exchange, balances, 3, 1.00, sleep=boom)
    assert balances == {0: 90_000, 3: 0}


def test_an_account_with_no_breakdown_moves_nothing():
    exchange = _Exchange({})
    placing.fund_shard(exchange, {}, 3, 1.00, sleep=_no_sleep)
    assert exchange.log == []


def _placing(tmp_path, monkeypatch, tickers: list[str], shard: int = 3):
    monkeypatch.setattr(placing, "ORDERS_DIR", tmp_path)
    kickoff = datetime(2026, 11, 7, 18, 0, tzinfo=UTC)
    rows, matched = [], {}
    for i, ticker in enumerate(tickers):
        side = _book(ticker)._replace(shard=shard, bid=None)
        other = _book(f"{ticker}-OTHER")._replace(shard=shard, bid=None)
        matched[f"g{i}"] = _Matched(side, other)
        rows.append(
            {
                "game_id": f"g{i}",
                "game": f"A{i} @ B{i}",
                "bet": f"B{i}",
                "bet_home": True,
                "kickoff": kickoff + timedelta(hours=i),
                "contracts": 5,
                "limit": 0.43,
                "outlay": 2.25,
                "rung_outlay": 0.0,
                "model": 0.6,
                "edge": 0.1,
                "rungs": [],
            }
        )
    args = argparse.Namespace(rest=None, demo=False, dry_run=False)
    return pd.DataFrame(rows), matched, args, tmp_path / "picks.csv"


def test_a_pick_is_funded_before_its_order(tmp_path, monkeypatch):
    picks, matched, args, out = _placing(tmp_path, monkeypatch, ["W-A"])
    exchange = _Exchange({0: 100_000, 3: 0})
    placed = placing.place(exchange, picks, matched, args, out)
    assert exchange.log == [("transfer", 0, 3, 22_500), ("buy", "W-A")]
    assert placed is not None and placed["filled"].iloc[0] == 5


def test_a_rejected_order_is_logged_and_the_next_pick_goes_on(tmp_path, monkeypatch):
    picks, matched, args, out = _placing(tmp_path, monkeypatch, ["W-A", "W-B"])
    exchange = _Exchange({0: 100_000, 3: 100_000}, reject={"W-A"})
    placed = placing.place(exchange, picks, matched, args, out)
    assert placed is not None
    bad = placed[placed["error"].notna()].iloc[0]
    assert (bad["bet"], bad["game"], bad["ticker"]) == ("B0", "A0 @ B0", "W-A")
    assert pd.notna(bad["placed_at"])
    assert ("buy", "W-B") in exchange.log


def test_a_refused_transfer_sends_no_order(tmp_path, monkeypatch):
    picks, matched, args, out = _placing(tmp_path, monkeypatch, ["W-A"])
    exchange = _Exchange({0: 100_000, 3: 0}, refuse=True)
    placed = placing.place(exchange, picks, matched, args, out)
    assert exchange.log == []
    assert placed is not None and "nope" in placed["error"].iloc[0]


def test_no_shard_balances_doesnt_lose_the_runs_orders(tmp_path, monkeypatch):
    picks, matched, args, out = _placing(tmp_path, monkeypatch, ["F-A"], shard=0)
    exchange = _Exchange({0: 100_000})

    def limited() -> dict[int, int]:
        raise KalshiError("GET /portfolio/balance: 429")

    monkeypatch.setattr(exchange, "shard_balances", limited)
    placed = placing.place(exchange, picks, matched, args, out)
    assert exchange.log == [("buy", "F-A")]
    assert placed is not None and placed["filled"].iloc[0] == 5

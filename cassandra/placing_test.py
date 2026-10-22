import argparse
from typing import NamedTuple

import pandas as pd
import pytest

from . import placing
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

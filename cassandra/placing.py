"""From priced games to orders on Kalshi: sizing, the news, the prompt, placing.

`picks.py` prices every matched game and says which are picks; this is
the rest of a `--place` run. Its docstring says how a run sizes, asks and
places, from the bettor's side. Here:

- `size` turns the picks into contracts: each pick's target (`--kelly-fraction`
  of Kelly on the bankroll, or a share of `--budget`), less what's held,
  over the top `--top` with something left to buy, fit to the cash;
- `injury_news` reads each pick's two teams' last stories and box scores;
- `print_picks` and `print_orders` show them, `ask` asks, and `place`
  sends the limit orders, cheapest route first, and logs what filled.

The functions take `picks.py`'s argparse namespace for its knobs.
"""

import argparse
import gzip
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
from call_it_what_you_want import TeamNamer
from endgame.types import Game

from . import injuries
from .constants import CASSANDRA_HOME
from .kalshi_trade import KalshiClient, KalshiError, public_get
from .picks import (
    allocate,
    contracts_for,
    fillable,
    fit_to_cash,
    kelly,
    kelly_counts,
    order_fee,
    routes,
    top_up,
)

CENTRAL = ZoneInfo("America/Chicago")
ORDERS_DIR = CASSANDRA_HOME / "betting" / "orders"
SUMMARIES_DIR = CASSANDRA_HOME / "picks" / "summaries"
ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
)
#: How many of each team's last games' stories are read for injury news.
NEWS_GAMES = 2
#: Dollars split over a run's picks when neither --budget nor
#: --kelly-fraction says otherwise.
DEFAULT_BUDGET = 20.0


def fmt_game(game: Game) -> str:
    joiner = "vs" if game.neutral_site else "@"
    return f"{game.away} {joiner} {game.home}"


def sizing(args: argparse.Namespace) -> str:
    """How this run sizes its bets, in a line."""
    if args.kelly_fraction is not None:
        found = (
            f"{args.kelly_fraction:g} Kelly on a ${args.bankroll:,.2f} bankroll, "
            f"at most ${args.max_bet:g} a bet"
        )
    else:
        found = (
            f"${_budget(args):g} split by Kelly weight, at most ${args.max_bet:g} a bet"
        )
    if args.kelly_fraction is not None and args.budget is not None:
        found += f", ${args.budget:g} this run"
    if args.cash is not None:
        found += f", ${args.cash:,.2f} cash"
    return found


def _budget(args: argparse.Namespace) -> float:
    """`--budget`, or the old default of $20 when splitting with no fraction."""
    return DEFAULT_BUDGET if args.budget is None else args.budget


def limit_price(price: float, ceiling: float | None, slippage: float) -> float:
    """The most an order pays a contract: `--slippage` past the ask, never past
    the price that still clears the edge (`max_price`)."""
    limit = price + slippage
    if ceiling is not None and not pd.isna(ceiling):
        limit = min(limit, max(ceiling, price))
    return round(min(limit, 0.99), 2)


def _targets(picks: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    """Each pick's contracts before what's held: Kelly on the bankroll, or a
    Kelly-weighted share of the budget. Either way at most `--max-bet`."""
    limits = list(picks["limit"])
    if args.kelly_fraction is not None:
        return kelly_counts(
            list(picks["model"]),
            list(picks["cost"]),
            limits,
            args.bankroll,
            args.kelly_fraction,
            args.max_bet,
        )
    weights = [kelly(m, c) for m, c in zip(picks["model"], picks["cost"])]
    return [
        min(n, contracts_for(args.max_bet, limit))
        for n, limit in zip(allocate(_budget(args), limits, weights), limits)
    ]


def size(
    frame: pd.DataFrame, matched: dict, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The picks and their contracts, and every game with what was bet on it.

    Each pick gets a target: `--kelly-fraction` of Kelly on the bankroll,
    or with no fraction its Kelly-weighted share of `--budget`. Either way
    it's at most `--max-bet`, counted at the limit price, so an order
    filling at its worst still costs no more than that. What's already held
    on the pick's side counts toward its target, and only the rest is
    bought. The whole run's worst case then fits `--budget` and the cash.
    """
    frame = frame.copy()
    # The top `--top` are taken from the games with something left to buy,
    # so a pick already held in full doesn't take a place a new one could.
    # Under a split budget a pick's target depends on which others share
    # it, so the cut is retaken until none in it is held in full.
    ranked = frame[frame["skip"].isna()].sort_values("edge", ascending=False)
    while True:
        top = ranked.iloc[: args.top]
        targets = pd.Series(_targets(top, args), index=top.index, dtype=int)
        counts = pd.Series(
            [top_up(t, h) for t, h in zip(targets, top["held"])], index=top.index
        )
        full = top.index[(top["held"] > 0) & (counts == 0)]
        if full.empty:
            break
        for i in full:
            frame.at[i, "skip"] = (
                f"already held ({frame.at[i, 'held']:g} of {targets[i]})"
            )
        ranked = ranked.drop(full)
    frame.loc[ranked.index[args.top :], "skip"] = f"outside the top {args.top} by edge"
    caps = [c for c in (args.budget, args.cash) if c is not None]
    if caps:
        counts = pd.Series(
            fit_to_cash(list(counts), list(top["limit"]), min(caps)), index=top.index
        )
    for i in top.index[counts == 0]:
        frame.at[i, "skip"] = (
            "no cash left for it" if targets[i] else "less than a contract's stake"
        )
    picks = frame[frame["skip"].isna()].copy()
    contracts = list(counts[picks.index])
    offered = f"offered_within_{round(args.slippage * 100)}c"
    fills = []
    for _, row in picks.iterrows():
        found = matched[row["game_id"]]
        side, other = (
            (found.home, found.away) if row["bet_home"] else (found.away, found.home)
        )
        # Both routes to the same side count: YES on it and NO on the
        # opponent fill on the same result, and `place` tries both.
        on_offer = 0.0
        for route in routes(side, other):
            book = public_get(f"/markets/{route.ticker}/orderbook")
            on_offer += fillable(book, route.action, row["limit"])
        fills.append(on_offer)
    picks["contracts"] = contracts
    picks["fee"] = [order_fee(p, c) for p, c in zip(picks["limit"], contracts)]
    picks["outlay"] = picks["contracts"] * picks["limit"] + picks["fee"]
    picks[offered] = fills
    frame = frame.merge(
        picks[["game_id", "contracts", "fee", "outlay", offered]],
        on="game_id",
        how="left",
    )
    return frame, picks


def _espn_summary(game_id: str, final: bool) -> dict:
    """ESPN's summary of a game: box score and AP story. A final one is kept."""
    path = SUMMARIES_DIR / f"{game_id}.json.gz"
    if final and path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    request = urllib.request.Request(
        f"{ESPN_SUMMARY}?event={game_id}", headers={"User-Agent": "cassandra-picks"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            summary = json.load(response)
    except OSError as error:
        logging.warning("no ESPN summary for %s: %s", game_id, error)
        return {}
    # A recap can land after the final whistle, so don't keep a summary
    # without one.
    if final and (summary.get("article") or {}).get("story"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(summary).encode()))
    return summary


def injury_news(
    picks: pd.DataFrame, games: list[Game], namer: TeamNamer
) -> dict[str, "News"]:
    """Injury sentences on each pick's two teams, from their last two games' stories,
    and who threw for each in those games' box scores.

    The upcoming game's preview is read too, once ESPN has one.
    """
    by_id = {g.game_id: g for g in games}
    wanted: dict[str, list[tuple[str, bool, str]]] = {}
    recent: dict[str, dict[str, list[Game]]] = {}
    for game_id in picks["game_id"]:
        fixture = by_id[game_id]
        stories = [(fixture.game_id, False, "preview")]
        recent[game_id] = {}
        for team in (fixture.home, fixture.away):
            played = sorted(
                (g for g in games if g.completed and team in (g.home, g.away)),
                key=lambda g: g.date,
            )[-NEWS_GAMES:]
            recent[game_id][team] = list(reversed(played))
            for g in reversed(played):
                when = g.date.astimezone(CENTRAL)
                stories.append((g.game_id, True, f"{fmt_game(g)}, {when:%b %d}"))
        wanted[game_id] = list(dict.fromkeys(stories))
    calls = {(g, final) for stories in wanted.values() for g, final, _ in stories}
    with ThreadPoolExecutor(8) as pool:
        summaries = dict(zip(calls, pool.map(lambda c: _espn_summary(*c), calls)))
    news: dict[str, News] = {}
    for game_id, stories in wanted.items():
        fixture = by_id[game_id]
        teams: dict[str, str] = {
            espn_id: t
            for t in (fixture.home, fixture.away)
            if (espn_id := namer.espn_id(t))
        }
        read = [summaries[(g, final)] for g, final, _ in stories]
        # The pick's own teams first, so a sentence naming one of theirs and
        # an opponent's is theirs.
        rosters = injuries.rosters(read, teams) + injuries.opponents(read, teams)
        found: dict[str, injuries.Note] = {}
        for summary, (_, _, label) in zip(read, stories):
            got = injuries.story(summary)
            if got is None:
                continue
            kind, text = got
            for note in injuries.notes(text, rosters, f"{kind}: {label}"):
                # News of last week's opponent isn't news of this game.
                if note.team is None or note.team in teams.values():
                    found.setdefault(note.sentence, note)
        passers = []
        for espn_id, team in teams.items():
            thrown = []
            for g in recent[game_id][team]:
                who = injuries.passing(summaries[(g.game_id, True)], espn_id)
                # The starter, and anyone else who threw five times.
                shown = [p for i, p in enumerate(who) if i == 0 or p[2] >= 5]
                when = f"{g.date.astimezone(CENTRAL):%b %d}"
                thrown.append(
                    f"{when} "
                    + (", ".join(f"{n} {c}/{a}" for n, c, a in shown) or "no box score")
                )
            passers.append(f"{team}: {' · '.join(thrown)}")
        news[game_id] = News(list(found.values()), passers)
    return news


class News(NamedTuple):
    """What a pick's order is shown with: injury sentences, and who threw lately."""

    notes: list[injuries.Note]
    #: Per team, its last games' passers, e.g. "UCLA Bruins: Oct 17 ...".
    passers: list[str]


def print_orders(
    picks: pd.DataFrame,
    news: dict[str, News],
    args: argparse.Namespace,
) -> None:
    """The orders `--place` would send, numbered, with the injury news under each."""
    if picks.empty:
        return
    tif = (
        "immediate-or-cancel"
        if args.rest is None
        else f"resting {args.rest:g}h, then cancelled"
    )
    print(f"\n=== {'orders' if args.place else 'orders --place would send'} ({tif})")
    order = picks.sort_values(["kickoff", "edge"], ascending=[True, False])
    flagged = []
    for number, (_, row) in enumerate(order.iterrows(), start=1):
        notes, passers = news.get(row["game_id"], News([], []))
        qb = [n for n in notes if n.qb]
        mark = "QB!" if qb else ("inj" if notes else "   ")
        if qb:
            flagged.append(number)
        print(
            f"{number:>3}. {mark} {row['kickoff']:%a %H:%M}  {row['bet']:<28} "
            f"{row['buy']:<40} {row['contracts']:>3.0f} @ ≤{row['limit']:.2f}  "
            f"≤${row['outlay']:.2f}   ({row['game']}, edge {row['edge']:+.3f})"
            + (f"  [topping up: holds {row['held_as']}]" if row["held"] else "")
        )
        for line in passers:
            print(f"         passing  {line}")
        for note in sorted(notes, key=lambda n: not n.qb)[: args.news_lines]:
            who = note.team or "?"
            print(f"         {'QB ' if note.qb else ''}[{who}] {note.sentence[:220]}")
            print(f"            -- {note.source}")
        if len(notes) > args.news_lines:
            print(f"         (+{len(notes) - args.news_lines} more injury sentences)")
    print(
        f"\nworst case ${picks['outlay'].sum():,.2f} across {len(picks)} orders "
        f"({sizing(args)})"
    )
    if flagged:
        print(
            f"QB news on {', '.join(map(str, flagged))}: if a starter is out of the "
            "next game, quit and rerun with --qb-out TEAM so the model prices it."
        )


def ask(count: int) -> set[int] | None:
    """What to do with the orders: an empty set to place them all, the numbers
    to drop (the rest are re-sized), or None to stop."""
    while True:
        try:
            answer = (
                input(
                    "\nplace these orders? [y = place all / numbers to drop, "
                    "e.g. '3 7' / q = quit]: "
                )
                .strip()
                .lower()
            )
        except EOFError:
            return None
        if answer in ("y", "yes"):
            return set()
        if answer in ("q", "quit", "n", "no", ""):
            return None
        numbers = answer.replace(",", " ").split()
        if all(n.isdigit() and 1 <= int(n) <= count for n in numbers):
            return {int(n) for n in numbers}
        print(f"  '{answer}': y, q, or numbers from 1 to {count}")


def place(
    client: KalshiClient,
    picks: pd.DataFrame,
    matched: dict,
    args: argparse.Namespace,
    out: Path,
) -> None:
    """Send the orders: the cheapest route first, what it left on the other.

    Each is a limit order at the pick's `limit`, so nothing fills at a price
    that doesn't clear the edge. Immediate-or-cancel unless `--rest`.
    """
    expires = (
        None
        if args.rest is None
        else int((datetime.now(UTC) + timedelta(hours=args.rest)).timestamp())
    )
    rows = []
    for _, row in picks.sort_values("kickoff").iterrows():
        found = matched[row["game_id"]]
        side, other = (
            (found.home, found.away) if row["bet_home"] else (found.away, found.home)
        )
        wanted = int(row["contracts"])
        for route in routes(side, other):
            if wanted <= 0:
                break
            if route.price > row["limit"] + 1e-9 and expires is None:
                continue
            try:
                done = client.buy(
                    route.ticker, route.action, wanted, row["limit"], expires
                )
            except KalshiError as error:
                print(f"  !! {row['bet']}: {error}")
                rows.append(
                    {"bet": row["bet"], "ticker": route.ticker, "error": str(error)}
                )
                break
            rows.append(
                {
                    "placed_at": datetime.now(CENTRAL),
                    "game": row["game"],
                    "game_id": row["game_id"],
                    "bet": row["bet"],
                    "ticker": route.ticker,
                    "action": route.action,
                    "ordered": wanted,
                    "limit": row["limit"],
                    "filled": done.filled,
                    "resting": done.resting,
                    "fill_price": done.price,
                    "fee": done.fee,
                    "model": row["model"],
                    "edge": row["edge"],
                    "order_id": done.order_id,
                    "demo": args.demo,
                    "dry_run": args.dry_run,
                }
            )
            price = "" if done.price is None else f" at {done.price:.2f}"
            rest = f", {done.resting:.0f} resting" if expires and done.resting else ""
            print(
                f"  {row['bet']:<28} {route.action.upper()} {route.ticker}: "
                f"{done.filled:g}/{wanted} filled{price}{rest}"
            )
            # A resting order holds the rest; only an IOC's leftover moves on.
            # A dry run shows the first order only: nothing filled to say
            # what the second would be.
            wanted = 0 if expires or args.dry_run else top_up(wanted, done.filled)
    placed = pd.DataFrame(rows)
    if placed.empty:
        return
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    log = ORDERS_DIR / out.name.replace(".csv", "_orders.csv")
    placed.to_csv(log, index=False)
    if "filled" not in placed:
        print(f"\nnothing filled; errors: {log}")
        return
    filled = placed[placed["filled"].fillna(0) > 0]
    spent = (filled["filled"] * filled["fill_price"] + filled["fee"].fillna(0)).sum()
    print(
        f"\nfilled {filled['filled'].sum():.0f} contracts on "
        f"{filled['game_id'].nunique()} games for ${spent:,.2f}; orders: {log}"
    )


def print_picks(
    frame: pd.DataFrame, picks: pd.DataFrame, args: argparse.Namespace
) -> None:
    offered = f"offered_within_{round(args.slippage * 100)}c"
    shown = (frame if args.all else frame[frame["skip"].isna()]).sort_values(
        ["kickoff", "edge"], ascending=[True, False]
    )
    columns = {
        "kickoff": lambda s: s.dt.strftime("%a %H:%M"),
        "game": None,
        "bet": None,
        "divisions": None,
        "model": lambda s: s.map("{:.3f}".format),
        "market": lambda s: s.map("{:.3f}".format),
        "edge": lambda s: s.map("{:+.3f}".format),
        "buy": None,
        "price": lambda s: s.map("{:.2f}".format),
        "limit": lambda s: s.map("{:.2f}".format),
        "max_price": lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
        "ev": lambda s: s.map("{:+.1%}".format),
        "contracts": lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:.0f}"),
        "outlay": lambda s: s.map(lambda v: "" if pd.isna(v) else f"${v:.2f}"),
        offered: lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:,.0f}"),
        "spread": lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v * 100:.0f}c"),
        "hours_since_entry": lambda s: s.map(
            lambda v: "" if pd.isna(v) else f"{v:.0f}h"
        ),
        "no_plays": None,
    }
    if args.all:
        columns["skip"] = None
    table = pd.DataFrame(
        {
            name: (shown[name] if fmt is None else fmt(shown[name]))
            for name, fmt in columns.items()
        }
    )
    print(
        f"\n=== {len(picks)} picks (edge ≥ {args.min_edge:.0%} over Kalshi's no-vig "
        f"price), top {args.top} by edge; {sizing(args)}"
    )
    if not table.empty:
        with pd.option_context("display.width", 250, "display.max_colwidth", 60):
            print(table.to_string(index=False))
    if not picks.empty:
        print(
            f"\ntotal outlay ${picks['outlay'].sum():,.2f} on {len(picks)} picks; "
            f"model's expected return {(picks['ev'] * picks['outlay']).sum() / picks['outlay'].sum():+.1%} "
            "(the backtest's realized ROI was about 80% of what the model expected)"
        )
        dogs = (picks["price"] < 0.5).sum()
        print(f"{dogs} underdogs, {len(picks) - dogs} favorites")
        short = picks[picks[offered] < picks["contracts"]]
        if not short.empty:
            print(
                f"{len(short)} won't fill at the price (fewer contracts within "
                f"{round(args.slippage * 100)}c than the order): "
                + ", ".join(
                    f"{bet} {on_offer:.0f}/{wanted:.0f}"
                    for bet, on_offer, wanted in zip(
                        short["bet"], short[offered], short["contracts"]
                    )
                )
            )
    # The reason without its detail (who it's waiting on, how wide the book
    # is), so each reason is counted once.
    skipped = (
        frame["skip"].dropna().str.replace(r"\s*[:(].*", "", regex=True).value_counts()
    )
    if not skipped.empty:
        print("\nnot picks: " + ", ".join(f"{n} {why}" for why, n in skipped.items()))

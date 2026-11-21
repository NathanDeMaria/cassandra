"""From priced games to orders on the venues: sizing, the news, the prompt, placing.

`picks.py` prices every matched game on every venue and says which are
picks; this is the rest of a `--place` run. Its docstring says how a run
sizes, asks and places, from the bettor's side. Every function here works
on a game's offers (`cassandra.picks.Offer`, one a venue) and the venues'
accounts through `cassandra.venues`, so none of it is any one venue's. Here:

- `size` turns the picks into contracts: each pick's target (`--kelly-fraction`
  of Kelly on the bankroll, or a share of `--budget`), less what's held on
  every venue, over the top `--top` with something left to buy, fit to the
  cash; and, under `--kelly-fraction`, the rungs under each
  (`cassandra.picks.rungs`), on a venue that can mark them;
- `stale_rungs` says which rungs left resting by an earlier run the model
  no longer backs at their price, for `cancel` to take off the book;
- `injury_news` reads each pick's two teams' last stories and box scores;
- `print_picks` and `print_orders` show them, `ask` asks, and `place`
  sends the limit orders, the cheapest route on any venue first, and logs
  what filled.

The functions take `picks.py`'s argparse namespace for its knobs.
"""

import argparse
import gzip
import json
import logging
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
from call_it_what_you_want import TeamNamer
from endgame.types import Game

from . import injuries, venues
from .constants import CASSANDRA_HOME
from .picks import (
    Longshots,
    Offer,
    allocate,
    contracts_for,
    fit_to_cash,
    kelly,
    kelly_counts,
    order_fee,
    rungs,
    top_up,
)
from .venues import (
    KALSHI_FEES,
    Account,
    Book,
    Fees,
    Holding,
    Resting,
    Route,
    Venue,
)

CENTRAL = ZoneInfo("America/Chicago")
ORDERS_DIR = CASSANDRA_HOME / "betting" / "orders"
SUMMARIES_DIR = CASSANDRA_HOME / "picks" / "summaries"
#: ESPN's game summary (box score, plays, AP story) for each pick league.
ESPN_SUMMARY = {
    league: f"https://site.api.espn.com/apis/site/v2/sports/{sport}/summary"
    for league, sport in (
        ("ncaafb", "football/college-football"),
        ("mens", "basketball/mens-college-basketball"),
        ("womens", "basketball/womens-college-basketball"),
    )
}
#: A final summary still without an AP story this long after the start is
#: kept anyway: many small-college games never get one.
STORY_WAIT = timedelta(days=2)
#: How many of each team's last games' stories are read for injury news.
NEWS_GAMES = 2
#: A nominal budget, split over a run's picks to show them with contracts
#: when neither --budget nor --kelly-fraction says what to stake: a run
#: that places always has the league's Kelly fraction.
DISPLAY_BUDGET = 100.0


def cents(price: float) -> str:
    """A price to the cent, or the half cent a venue quotes in."""
    return (
        f"{price:.2f}"
        if abs(price * 100 - round(price * 100)) < 1e-6
        else f"{price:.3f}"
    )


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
    longshots = _longshots(args)
    if longshots.scale != 1.0 and longshots.price > 0:
        found += f", {longshots.scale:g}x under {longshots.price * 100:g}c"
    if args.kelly_fraction is not None and args.budget is not None:
        found += f", ${args.budget:,.2f} this run"
    if args.cash is not None:
        found += f", ${args.cash:,.2f} cash"
    return found


def _budget(args: argparse.Namespace) -> float:
    """`--budget`, or the nominal `DISPLAY_BUDGET` when splitting with no
    fraction."""
    return DISPLAY_BUDGET if args.budget is None else args.budget


def limit_price(price: float, ceiling: float | None, slippage: float) -> float:
    """The most an order pays a contract: `--slippage` past the ask, never past
    the price that still clears the edge (`max_price`). To the half cent, the
    finest a venue quotes."""
    limit = price + slippage
    if ceiling is not None and not pd.isna(ceiling):
        limit = min(limit, max(ceiling, price))
    return round(round(min(limit, 0.99) * 200) / 200, 3)


def offer_limit(offer: Offer, slippage: float) -> float | None:
    """The most an order on `offer`'s venue pays a contract (`limit_price`);
    None when it has nothing on offer."""
    if offer.priced.route is None:
        return None
    return limit_price(offer.priced.route.price, offer.max_price, slippage)


def good_offers(offers: Mapping[str, Sequence[Offer]], game_id: str) -> list[Offer]:
    """A game's offers that are picks on their own venue, cheapest first."""
    return [o for o in offers.get(game_id, ()) if o.skip is None]


def _sides(offer: Offer, bet_home: bool) -> tuple[Book, Book]:
    """The book of the side a pick backs on `offer`'s venue, and its opponent's."""
    home, away = offer.matched.home, offer.matched.away
    return (home, away) if bet_home else (away, home)


def _fees(picks: pd.DataFrame, offers: Mapping[str, Sequence[Offer]]) -> list[Fees]:
    """Each pick's fees: its cheapest venue's, which it's sized at."""
    return [
        offers[game_id][0].venue.fees if offers.get(game_id) else KALSHI_FEES
        for game_id in picks["game_id"]
    ]


def _targets(
    picks: pd.DataFrame, args: argparse.Namespace, fees: Sequence[Fees]
) -> list[int]:
    """Each pick's contracts before what's held: Kelly on the bankroll, or a
    Kelly-weighted share of the budget. Either way at most `--max-bet`."""
    limits = list(picks["limit"])
    longshots = _longshots(args)
    if args.kelly_fraction is not None:
        return kelly_counts(
            list(picks["model"]),
            list(picks["cost"]),
            limits,
            args.bankroll,
            args.kelly_fraction,
            args.max_bet,
            longshots,
            fees,
        )
    weights = [
        longshots.of(c) * kelly(m, c) for m, c in zip(picks["model"], picks["cost"])
    ]
    return [
        min(n, contracts_for(args.max_bet, limit, venue))
        for n, limit, venue in zip(
            allocate(_budget(args), limits, weights, fees), limits, fees
        )
    ]


def _longshots(args: argparse.Namespace) -> Longshots:
    """`--longshot-price` and `--longshot-scale`, or no shrink."""
    return Longshots(
        getattr(args, "longshot_price", 0.0), getattr(args, "longshot_scale", 1.0)
    )


def league_exposure(
    accounts: Mapping[str, tuple[Venue, Sequence[Holding], Sequence[Resting]]],
    league: str,
) -> float:
    """Dollars a league has at risk across every account: what its open
    positions cost, and the most its resting orders -- rungs too -- could
    still spend (`venues.league_exposure`)."""
    return sum(
        venues.league_exposure(venue, held, resting, league)
        for venue, held, resting in accounts.values()
    )


def exposure_room(bankroll: float, max_exposure: float, exposure: float) -> float:
    """What a league may still put at risk: `max_exposure` of the bankroll,
    less what it has there already, and never below nothing."""
    return max(max_exposure * bankroll - exposure, 0.0)


def size(
    frame: pd.DataFrame,
    offers: Mapping[str, Sequence[Offer]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The picks and their contracts, and every game with what was bet on it.

    Each pick gets a target: `--kelly-fraction` of Kelly on the bankroll,
    or with no fraction its Kelly-weighted share of `--budget`. Either way
    it's at most `--max-bet`, counted at the limit price of its cheapest
    venue (`offers`, each game's, cheapest pick first), so an order filling
    at its worst still costs no more than that. What's already held on the
    pick's side, on any venue, counts toward its target, and only the rest
    is bought. The whole run's worst case then fits `--budget` and the cash.
    """
    frame = frame.copy()
    # The top `--top` are taken from the games with something left to buy,
    # so a pick already held in full doesn't take a place a new one could.
    # Under a split budget a pick's target depends on which others share
    # it, so the cut is retaken until none in it is held in full.
    ranked = frame[frame["skip"].isna()].sort_values("edge", ascending=False)
    while True:
        top = ranked.iloc[: args.top]
        fees = _fees(top, offers)
        targets = pd.Series(_targets(top, args, fees), index=top.index, dtype=int)
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
    if args.top is not None:
        frame.loc[ranked.index[args.top :], "skip"] = (
            f"outside the top {args.top} by edge"
        )
    caps = [c for c in (args.budget, args.cash) if c is not None]
    if caps:
        counts = pd.Series(
            fit_to_cash(list(counts), list(top["limit"]), min(caps), fees),
            index=top.index,
        )
    for i in top.index[counts == 0]:
        frame.at[i, "skip"] = (
            "no cash left for it" if targets[i] else "less than a contract's stake"
        )
    picks = frame[frame["skip"].isna()].copy()
    contracts = list(counts[picks.index])
    fees = _fees(picks, offers)
    offered = f"offered_within_{round(args.slippage * 100)}c"
    fills = []
    for _, row in picks.iterrows():
        # Every route on every venue where it's a pick counts: they pay on
        # the same result, and `place` tries them all, each at its own limit.
        on_offer = 0.0
        for offer in good_offers(offers, row["game_id"]):
            limit = offer_limit(offer, args.slippage)
            if limit is None:
                continue
            for route in offer.venue.routes(*_sides(offer, row["bet_home"])):
                try:
                    on_offer += offer.venue.on_offer(route, limit)
                except Exception as error:
                    # What's on offer is shown, not placed by: an order
                    # takes what's there whatever this read says.
                    print(
                        f"  !! no {offer.venue.label} book for {route.ticker}: {error!r}"
                    )
        fills.append(on_offer)
    picks["contracts"] = contracts
    picks["fee"] = [
        order_fee(p, c, venue) for p, c, venue in zip(picks["limit"], contracts, fees)
    ]
    picks["outlay"] = picks["contracts"] * picks["limit"] + picks["fee"]
    picks[offered] = fills
    ladders, venues_ = _rungs(picks, offers, args, min(caps) if caps else None)
    # A column of lists, one ladder per pick; pandas-stubs types a column
    # assignment's value as scalars or a flat sequence.
    picks["rungs"] = ladders  # ty: ignore[invalid-assignment]
    picks["rung_venue"] = venues_
    picks["rung_outlay"] = [
        sum(n * p + _venue_fees(offers, g, v).order(p, n) for p, n in found)
        for g, found, v in zip(picks["game_id"], picks["rungs"], picks["rung_venue"])
    ]
    frame = frame.merge(
        picks[["game_id", "contracts", "fee", "outlay", offered, "rung_outlay"]],
        on="game_id",
        how="left",
    )
    return frame, picks


def _venue_fees(
    offers: Mapping[str, Sequence[Offer]], game_id: str, venue: str | None
) -> Fees:
    for offer in offers.get(game_id, ()):
        if offer.venue.name == venue:
            return offer.venue.fees
    return KALSHI_FEES


def rung_offer(offers: Sequence[Offer]) -> Offer | None:
    """Where a pick's rungs rest: its cheapest venue that can mark an order as
    a rung (`Venue.tags_orders`), if it's a pick there."""
    return next((o for o in offers if o.skip is None and o.venue.tags_orders), None)


def _rungs(
    picks: pd.DataFrame,
    offers: Mapping[str, Sequence[Offer]],
    args: argparse.Namespace,
    cap: float | None,
) -> tuple[list[list[tuple[float, int]]], list[str | None]]:
    """Each pick's rungs and the venue they rest on (`rung_offer`), fit to
    what the run's cap leaves after the picks.

    Only under `--kelly-fraction`, which says what a lower price is worth;
    a split budget has no stake to grow. A game with rungs resting from an
    earlier run (`args.rung_events`, by venue and event) gets no more.
    """
    unders = getattr(args, "rungs", ())
    none: list[list[tuple[float, int]]] = [[] for _ in range(len(picks))]
    if not unders or args.kelly_fraction is None or picks.empty:
        return none, [None] * len(picks)
    resting = getattr(args, "rung_events", set())
    found: list[list[tuple[float, int]]] = []
    where: list[str | None] = []
    fees: list[Fees] = []
    for _, row in picks.iterrows():
        offer = rung_offer(offers.get(row["game_id"], ()))
        if (
            offer is None
            or offer.priced.route is None
            or (offer.venue.name, offer.matched.game.event) in resting
        ):
            found.append([])
            where.append(None)
            fees.append(KALSHI_FEES)
            continue
        found.append(
            rungs(
                row["model"],
                offer.priced.route.price,
                row["held"] + row["contracts"],
                (row["held"] + row["contracts"]) * row["limit"] + row["fee"],
                unders,
                args.bankroll,
                # A longshot's rungs are shrunk with it.
                args.kelly_fraction * _longshots(args).of(row["cost"]),
                args.max_bet,
                offer.venue.fees,
            )
        )
        where.append(offer.venue.name)
        fees.append(offer.venue.fees)
    if cap is None:
        return found, where
    flat = [(i, p, n) for i, ladder in enumerate(found) for p, n in ladder]
    left = cap - picks["outlay"].sum()
    counts = fit_to_cash(
        [n for *_, n in flat],
        [p for _, p, _ in flat],
        max(left, 0),
        [fees[i] for i, _, _ in flat],
    )
    fitted: list[list[tuple[float, int]]] = [[] for _ in found]
    for (i, p, _), n in zip(flat, counts):
        if n > 0:
            fitted[i].append((p, n))
    return fitted, where


def stale_rungs(
    orders: Sequence[tuple[Venue, Resting]],
    frame: pd.DataFrame,
    offers: Mapping[str, Sequence[Offer]],
    exclude: Sequence[str] = (),
    hold_qb: bool = False,
) -> list[tuple[Venue, Resting, str]]:
    """The resting rungs to cancel, each with its venue and why.

    A rung rests until kickoff, long after its pick's entry window, so it's
    held to what the model says now, not to whether the game is still a
    pick: it goes when the model backs the other side, when the game names
    a team in `exclude` (`--exclude` and the overrides file), or when its
    price is over its venue's `max_price`, the most the side can cost there
    and clear the edge -- which a starter ruled out does. With `hold_qb` it
    also goes when the side it backs has its starter in doubt (`qb_doubt`):
    a pick in doubt waits for a person, and so does buying more of one. A
    game this run didn't price is left alone.
    """
    rows = {row["game_id"]: row for _, row in frame.iterrows()}
    by_event: dict[tuple[str, str], tuple[pd.Series, Offer]] = {}
    for game_id, found in offers.items():
        if game_id in rows:
            for offer in found:
                key = (offer.venue.name, offer.matched.game.event)
                by_event[key] = (rows[game_id], offer)
    stale = []
    for venue, order in orders:
        hit = by_event.get((venue.name, venue.event_of(order.ticker)))
        if hit is None:
            continue
        row, offer = hit
        backs = venue.pays(*_sides(offer, row["bet_home"]), order.ticker, order.action)
        if backs is None:
            continue
        ceiling = offer.max_price
        if not backs:
            why = f"the model backs {row['bet']} now"
        elif any(team.lower() in row["game"].lower() for team in exclude):
            why = "excluded by hand"
        elif hold_qb and isinstance(row.get("qb_doubt"), str):
            why = f"QB in doubt: {row['qb_doubt']}"
        elif offer.priced.route is None:
            continue
        elif ceiling is None or pd.isna(ceiling):
            why = "no price clears the edge now"
        elif order.price > ceiling + 1e-9:
            why = (
                f"{order.price:.2f} is over the {ceiling:.2f} that clears the edge now"
            )
        else:
            continue
        stale.append((venue, order, why))
    return stale


def cancel(
    accounts: Mapping[str, Account],
    stale: Sequence[tuple[Venue, Resting, str]],
    offers: Mapping[str, Sequence[Offer]],
    frame: pd.DataFrame,
) -> list[str]:
    """Take the stale rungs off the book; a line for each."""
    titles = dict(zip(frame["game_id"], frame["game"]))
    games = {
        (offer.venue.name, offer.matched.game.event): titles.get(game_id, game_id)
        for game_id, found in offers.items()
        for offer in found
    }
    lines = []
    for venue, order, why in stale:
        game = games.get((venue.name, venue.event_of(order.ticker)), order.ticker)
        try:
            reduced = accounts[venue.name].cancel(order)
        except Exception as error:
            lines.append(f"  !! couldn't cancel the rung on {game}: {error}")
            continue
        lines.append(
            f"  cancelled the {order.price:.2f} rung on {game} "
            f"({reduced:g} contracts): {why}"
        )
    for line in lines:
        print(line)
    return lines


def espn_summary(game_id: str, final: bool, league: str = "ncaafb") -> dict:
    """ESPN's summary of a game: box score and AP story. A final one is kept."""
    path = SUMMARIES_DIR / f"{game_id}.json.gz"
    if final and path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    request = urllib.request.Request(
        f"{ESPN_SUMMARY[league]}?event={game_id}",
        headers={"User-Agent": "cassandra-picks"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            summary = json.load(response)
    except OSError as error:
        logging.warning("no ESPN summary for %s: %s", game_id, error)
        return {}
    if final and _keep(summary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(summary).encode()))
    return summary


def _keep(summary: Mapping) -> bool:
    """Whether a final game's summary is done changing. A recap can land
    after the final whistle, so one without waits up to `STORY_WAIT`."""
    if (summary.get("article") or {}).get("story"):
        return True
    competition = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    completed = ((competition.get("status") or {}).get("type") or {}).get("completed")
    started = competition.get("date")
    if not completed or not started:
        return False
    start = datetime.fromisoformat(started.replace("Z", "+00:00"))
    return datetime.now(UTC) - start >= STORY_WAIT


def injury_news(
    picks: pd.DataFrame, games: list[Game], namer: TeamNamer
) -> dict[str, News]:
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
        summaries = dict(zip(calls, pool.map(lambda c: espn_summary(*c), calls)))
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


def qb_news(picks: pd.DataFrame, news: Mapping[str, News]) -> dict[str, str]:
    """The picks the model may have priced on a quarterback who won't play,
    each with why: the side bet on has its starter in doubt after his last
    game (`qb_doubt`, `cassandra.qb_starts`), or the injury news names a
    quarterback, its first sentence that does. The model reads a starter
    out only when told (`--qb-out`)."""
    found = {}
    for _, row in picks.iterrows():
        game_id = row["game_id"]
        doubt = row.get("qb_doubt")
        if isinstance(doubt, str):
            found[game_id] = doubt
            continue
        notes = news.get(game_id, News([], [])).notes
        qb = [note for note in notes if note.qb]
        if qb:
            found[game_id] = qb[0].sentence
    return found


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
        doubt = row.get("qb_doubt")
        doubt = doubt if isinstance(doubt, str) else None
        mark = "QB!" if qb or doubt else ("inj" if notes else "   ")
        if qb or doubt:
            flagged.append(number)
        print(
            f"{number:>3}. {mark} {row['kickoff']:%a %H:%M}  {row['bet']:<28} "
            f"{row['buy']:<40} {row['contracts']:>3.0f} @ ≤{cents(row['limit'])}  "
            f"≤${row['outlay']:.2f}   ({row['game']}, edge {row['edge']:+.3f})"
            + (f"  [topping up: holds {row['held_as']}]" if row["held"] else "")
        )
        if row.get("rungs"):
            print(f"         {rung_line(row['rungs'])}")
        if doubt:
            print(f"         QB doubt {doubt}")
        if (qb or doubt) and (line := if_out_line(row)):
            print(f"         {line}")
        for line in passers:
            print(f"         passing  {line}")
        for note in sorted(notes, key=lambda n: not n.qb)[: args.news_lines]:
            who = note.team or "?"
            print(f"         {'QB ' if note.qb else ''}[{who}] {note.sentence[:220]}")
            print(f"            -- {note.source}")
        if len(notes) > args.news_lines:
            print(f"         (+{len(notes) - args.news_lines} more injury sentences)")
    laddered = picks["rung_outlay"].sum() if "rung_outlay" in picks else 0.0
    print(
        f"\nworst case ${picks['outlay'].sum() + laddered:,.2f} across {len(picks)} orders "
        + (f"(${laddered:,.2f} of it rungs) " if laddered else "")
        + f"({sizing(args)})"
    )
    if flagged:
        print(
            f"QB news or doubt on {', '.join(map(str, flagged))}: if a starter is out "
            "of the next game, his 'starter is out' line is the edge; to size at that "
            "price, quit and rerun with --qb-out TEAM."
        )


def if_out_line(row: pd.Series) -> str | None:
    """The pick's edge with its side's starter out (`edge_qb_out`), or None
    for a league whose model doesn't read quarterbacks."""
    out = row.get("edge_qb_out")
    if out is None or pd.isna(out):
        return None
    return f"if {row['bet']}'s starter is out: edge {out:+.3f}"


def rung_line(found: list[tuple[float, int]]) -> str:
    return "rungs, resting to kickoff: " + ", ".join(f"{n} @ {p:.2f}" for p, n in found)


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
    accounts: Mapping[str, Account],
    picks: pd.DataFrame,
    offers: Mapping[str, Sequence[Offer]],
    args: argparse.Namespace,
    out: Path,
    cash: Mapping[str, float] | None = None,
) -> pd.DataFrame | None:
    """Send the orders: every route on every venue where the pick is a pick,
    cheapest first, each taking what the last one left.

    Each is a limit order at its venue's limit (`offer_limit`; the pick's
    own `limit` on its cheapest venue), so nothing fills at a price that
    doesn't clear the edge on that venue's book. One past the cheapest
    venue is held to the dollars the pick has left and that venue's `cash`,
    so the pick never costs more than it was sized at. Immediate-or-cancel,
    unless `--rest`: then what's on offer at the limits is taken first and
    the rest rests on the cheapest route, never past kickoff
    (`order_expiry`); a pick with one route at all rests there in one order.
    A venue is funded before its first order (`Account.fund`). An order or a
    transfer that fails is logged with its `error`, and the next venue, then
    the next pick, goes on.
    """
    rows = []
    left_on = dict(cash or {})
    for _, row in picks.sort_values("kickoff").iterrows():
        expires = order_expiry(args.rest, row["kickoff"], datetime.now(UTC))
        good = good_offers(offers, row["game_id"])
        if not good:
            continue
        best = good[0]
        tries: list[tuple[Offer, Book, Route, float]] = []
        for offer in good:
            side, other = _sides(offer, row["bet_home"])
            limit = row["limit"] if offer is best else offer_limit(offer, args.slippage)
            if limit is None:
                continue
            tries += [(offer, side, r, limit) for r in offer.venue.routes(side, other)]
        # Stable, so a venue's own order of its routes holds on a tie.
        tries.sort(key=lambda t: t[2].price + t[0].venue.fees.contract(t[2].price))
        one_order = expires is not None and len(tries) == 1
        wanted = int(row["contracts"])
        dollars = row["outlay"]
        funded: set[str] = set()
        failed: set[str] = set()

        def attempt(
            offer: Offer, side: Book, route: Route, limit: float, rest: int | None
        ) -> None:
            nonlocal wanted, dollars
            venue = offer.venue
            count = wanted
            if offer is not best:
                count = min(count, contracts_for(max(dollars, 0.0), limit, venue.fees))
            if venue.name in left_on:
                count = min(
                    count,
                    contracts_for(max(left_on[venue.name], 0.0), limit, venue.fees),
                )
            if count <= 0:
                return
            account = accounts[venue.name]
            if venue.name not in funded:
                need = (
                    row["outlay"]
                    + (
                        row.get("rung_outlay", 0.0)
                        if row.get("rung_venue") == venue.name
                        else 0.0
                    )
                    if offer is best
                    else count * limit + venue.fees.order(limit, count)
                )
                try:
                    account.fund(side, need)
                except Exception as error:
                    print(f"  !! {row['bet']}: {error}")
                    rows.append(_rejected(row, venue, side.ticker, error))
                    failed.add(venue.name)
                    return
                funded.add(venue.name)
            try:
                done = account.buy(route, count, limit, rest)
            except Exception as error:
                print(f"  !! {row['bet']}: {error}")
                rows.append(_rejected(row, venue, route.ticker, error))
                failed.add(venue.name)
                return
            rows.append(
                {
                    "placed_at": datetime.now(CENTRAL),
                    "venue": venue.name,
                    "game": row["game"],
                    "game_id": row["game_id"],
                    "bet": row["bet"],
                    "ticker": route.ticker,
                    "action": route.action,
                    "ordered": count,
                    "limit": limit,
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
            rest_note = f", {done.resting:.0f} resting" if rest and done.resting else ""
            on = "" if venue.name == "kalshi" else f" on {venue.label}"
            print(
                f"  {row['bet']:<28} {route.action.upper()} {route.ticker}{on}: "
                f"{done.filled:g}/{count} filled{price}{rest_note}"
            )
            spent = done.filled * (done.price or limit) + (done.fee or 0.0)
            held_back = done.resting * limit + (
                venue.fees.order(limit, done.resting) if done.resting else 0.0
            )
            dollars -= spent + held_back
            if venue.name in left_on:
                left_on[venue.name] -= spent + held_back
            # A resting order holds the rest. A dry run shows the first order
            # only: nothing filled to say what the next would be.
            wanted = (
                0
                if rest or args.dry_run
                else top_up(wanted, done.filled + done.resting)
            )

        for offer, side, route, limit in tries:
            if wanted <= 0:
                break
            if offer.venue.name in failed:
                continue
            if one_order:
                attempt(offer, side, route, limit, expires)
                continue
            if route.price > limit + 1e-9:
                continue
            attempt(offer, side, route, limit, None)
        if (
            wanted > 0
            and expires is not None
            and not one_order
            and tries
            and tries[0][0].venue.name not in failed
            and not args.dry_run
        ):
            attempt(*tries[0], expires)
        venue = row.get("rung_venue")
        if row.get("rungs") and venue and venue not in failed:
            offer = next(o for o in good if o.venue.name == venue)
            side, other = _sides(offer, row["bet_home"])
            found = offer.venue.routes(side, other)
            if found:
                rows += _place_rungs(accounts[venue], offer.venue, row, found[0], args)
    placed = pd.DataFrame(rows)
    if placed.empty:
        return None
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    log = ORDERS_DIR / out.name.replace(".csv", "_orders.csv")
    placed.to_csv(log, index=False)
    if "filled" not in placed:
        print(f"\nnothing filled; errors: {log}")
        return placed
    filled = placed[placed["filled"].fillna(0) > 0]
    spent = (filled["filled"] * filled["fill_price"] + filled["fee"].fillna(0)).sum()
    print(
        f"\nfilled {filled['filled'].sum():.0f} contracts on "
        f"{filled['game_id'].nunique()} games for ${spent:,.2f}; orders: {log}"
    )
    return placed


def _rejected(row: pd.Series, venue: Venue, ticker: str, error: Exception) -> dict:
    """An orders log's row for what a venue turned away, or an order that
    failed on the way -- a timeout, an answer that couldn't be read -- for the
    job's email and the morning summary (`picks_job.rejected_orders`). A
    failed order may still have landed; the next run reads what's held."""
    return {
        "placed_at": datetime.now(CENTRAL),
        "venue": venue.name,
        "game": row["game"],
        "game_id": row["game_id"],
        "bet": row["bet"],
        "ticker": ticker,
        "error": str(error) or repr(error),
    }


def order_expiry(
    rest_hours: float | None, kickoff: datetime, now: datetime
) -> int | None:
    """When a pick's order stops resting, as the venues take it (epoch seconds):
    `rest_hours` from now, or kickoff if that's sooner; None for
    immediate-or-cancel.

    The venues trade a game while it's played, so an order left past kickoff
    could fill at a pregame price on an in-game book. A football pick is
    usually days ahead, but a basketball entry can be hours before tip.
    """
    if rest_hours is None:
        return None
    return int(min(now + timedelta(hours=rest_hours), kickoff).timestamp())


def _place_rungs(
    account: Account,
    venue: Venue,
    row: pd.Series,
    route: Route,
    args: argparse.Namespace,
) -> list[dict]:
    """A pick's rungs, on its rung venue's cheapest route as read, resting
    until kickoff."""
    ladder = row.get("rungs") or []
    kickoff = int(row["kickoff"].timestamp())
    rows = []
    for at, count in ladder:
        try:
            done = account.buy(route, count, at, kickoff, rung=True)
        except Exception as error:
            print(f"  !! {row['bet']} rung at {at:.2f}: {error}")
            rows.append(_rejected(row, venue, route.ticker, error))
            break
        rows.append(
            {
                "placed_at": datetime.now(CENTRAL),
                "venue": venue.name,
                "game": row["game"],
                "game_id": row["game_id"],
                "bet": row["bet"],
                "ticker": route.ticker,
                "action": route.action,
                "ordered": count,
                "limit": at,
                "filled": done.filled,
                "resting": done.resting,
                "fill_price": done.price,
                "fee": done.fee,
                "model": row["model"],
                "edge": row["edge"],
                "order_id": done.order_id,
                "rung": True,
                "demo": args.demo,
                "dry_run": args.dry_run,
            }
        )
        print(
            f"  {row['bet']:<28} rung {route.action.upper()} {route.ticker} "
            f"@ {at:.2f}: {done.filled:g}/{count} filled, {done.resting:g} resting to kickoff"
        )
    return rows


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
        "venue": None,
        "buy": None,
        "price": lambda s: s.map(cents),
        "limit": lambda s: s.map(cents),
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
    rule = args.rule
    favorites = (
        f", {rule.edge_needed(1.0):.0%} at {rule.favorite_price * 100:g}c+"
        if rule.favorite_price is not None and rule.favorite_edge is not None
        else ""
    )
    if (
        rule.model_longshot_probability is not None
        and rule.model_longshot_edge is not None
    ):
        favorites += (
            f", {rule.model_longshot_edge:.0%} with the model under "
            f"{rule.model_longshot_probability:.0%}"
        )
    if rule.min_expected_return is not None:
        favorites += f", returning {rule.min_expected_return:.0%}+ at its price"
    top = "" if args.top is None else f", top {args.top} by edge"
    print(
        f"\n=== {len(picks)} picks (edge ≥ {rule.min_edge:.0%} over a venue's no-vig "
        f"price{favorites}){top}; {sizing(args)}"
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

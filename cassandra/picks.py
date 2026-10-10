"""What to bet on Kalshi now: the model's win probability against the live book.

`betting.py` grades a model's picks after the fact. On Kalshi's college
football moneylines it found an edge at the *entry* -- the first price after
both teams' previous game -- and only past a threshold. The roster model,
October 2026, edge over Kalshi's no-vig price (`no_vig_home`):

    edge          2025                2026
    0-5 points    -7% ROI (321)       -1% (154)
    5+ points     +13% ± 5% (374)     +21% ± 11% (159)

Under five points the overround eats the edge. Over five it held up in
both seasons. In 2026 it faded as the book caught up: +21% at entry, +19%
six hours later, +13% at twelve, +4% after a day. So the bet is placed the
night the previous week's games end, against the price up then. This
module is that bet, made live. It takes every open Kalshi game in the next
few days, finds its ESPN fixture, and prices it against the model the way
the backtest did.

**The edge** is the backtest's: each team's own YES ask plus Kalshi's taker
fee, normalized to sum to one, against the model's probability. A book
whose two costs sum past `markets.MAX_OVERROUND` wasn't a market in the
backtest and isn't one here.

**The price** is the cheapest way to back the side, which the backtest
didn't look for. A team's YES and its opponent's NO pay on the same result,
since football has no ties, and the opponent's NO costs 1 minus its YES
bid. Taking the cheaper of the two never makes a pick worse than the one
graded.

**What isn't a pick** (`skip_reason`):
- an edge under `MIN_EDGE`;
- the model backing a team below FBS against an FBS one. The model's known
  miss on those games is under-rating the FBS side: 10-20 point FBS
  favorites beat it by about 2 points in 2015-25. So its edge on the
  lower side is mostly that miss.
- a game where either team's previous game isn't final. The model hasn't
  seen that result yet, and the backtest's entry waited for it.
- a game whose entry was more than `MAX_SINCE_ENTRY` ago. By then the
  market has caught up, and the backtest's close showed no edge.

Everything here is a pure function of Kalshi's JSON, the fixtures and the
model's probabilities; `picks.py` at the repo root does the fetching.
"""

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from endgame.types import Game

from .betting import GAME_LENGTH
from .markets import MAX_OVERROUND, kalshi_fee

#: Kalshi's college football moneylines. FCS-only games are in their own
#: series, which is how gold-rush lists them too.
NCAAFB_SERIES = ("KXNCAAFGAME", "KXNCAAFCSGAME")

#: The smallest edge over the no-vig price worth betting. See the module
#: docstring: this is where the backtest's ROI turned positive in both
#: seasons it covers.
MIN_EDGE = 0.05

#: How long after the entry a price is still worth betting. The 2026 edge
#: over 5 points was +21% at the entry, +13% twelve hours later and +4% at a
#: day; favorites held theirs for the day, underdogs lost most of it by
#: twelve hours.
MAX_SINCE_ENTRY = timedelta(hours=24)

#: Kalshi files a game under its US Eastern date. So does `match`.
EASTERN = ZoneInfo("America/New_York")

#: Statuses of a game that will not be played when it says, so it is
#: neither a fixture to price nor a previous game to wait for.
NOT_HAPPENING = frozenset({"STATUS_CANCELED", "STATUS_POSTPONED"})

#: The division a team must be in for the model's edge on it to count
#: against a team from a lower one. See the module docstring.
TOP_DIVISION = "FBS"

_EVENT_DATE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS: dict[str, int] = {
    name: number
    for number, name in enumerate(
        "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), start=1
    )
}


class Book(NamedTuple):
    """One team's YES market: its best quotes and the contracts behind them.

    `bid` and `ask` are None when nobody is offering that side. Kalshi
    sends 0 or 1 then, which is not a price anyone could trade.
    """

    ticker: str
    code: str
    team: str
    bid: float | None
    bid_size: float
    ask: float | None
    ask_size: float
    volume: float
    volume_24h: float

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


class KalshiGame(NamedTuple):
    """One Kalshi event: a game, filed under its Eastern date, one market per team."""

    event: str
    day: date
    sides: tuple[Book, Book]


class Route(NamedTuple):
    """One way to back a team: which contract, which side of it, at what price.

    `action` is "yes" for buying the team's own market and "no" for buying
    NO on its opponent's.
    """

    ticker: str
    action: str
    price: float
    size: float


def event_day(event_ticker: str) -> date | None:
    """The date in an event ticker: `KXNCAAFGAME-26OCT17BRWNPRIN` is 2026-10-17."""
    found = _EVENT_DATE.search(event_ticker)
    if found is None or found.group(2) not in _MONTHS:
        return None
    year, month, day = found.groups()
    return date(2000 + int(year), _MONTHS[month], int(day))


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quote(value: Any) -> float | None:
    price = _number(value)
    return price if price is not None and 0 < price < 1 else None


def book(market: Mapping) -> Book:
    """A market as `GET /markets` lists it, read into a `Book`."""
    return Book(
        ticker=market["ticker"],
        code=market["ticker"].rsplit("-", 1)[-1],
        team=market.get("yes_sub_title") or "",
        bid=_quote(market.get("yes_bid_dollars")),
        bid_size=_number(market.get("yes_bid_size_fp")) or 0.0,
        ask=_quote(market.get("yes_ask_dollars")),
        ask_size=_number(market.get("yes_ask_size_fp")) or 0.0,
        volume=_number(market.get("volume_fp")) or 0.0,
        volume_24h=_number(market.get("volume_24h_fp")) or 0.0,
    )


def kalshi_games(markets: Iterable[Mapping]) -> tuple[list[KalshiGame], list[str]]:
    """Group a market listing into games, and name the events that aren't one.

    An event is a game when it has exactly two team markets. A tie market,
    where one is listed, isn't a team.
    """
    by_event: dict[str, list[Mapping]] = defaultdict(list)
    for market in markets:
        by_event[market["event_ticker"]].append(market)
    games, odd = [], []
    for event, event_markets in sorted(by_event.items()):
        teams = sorted((book(m) for m in event_markets), key=lambda b: b.ticker)
        teams = [b for b in teams if b.code != "TIE"]
        day = event_day(event)
        if len(teams) != 2 or day is None:
            odd.append(event)
            continue
        games.append(KalshiGame(event, day, (teams[0], teams[1])))
    return games, odd


class Fixture(NamedTuple):
    """An unplayed ESPN game, under canonical names, with both teams' ESPN ids."""

    game: Game
    home_id: str | None
    away_id: str | None


class Matched(NamedTuple):
    """The Kalshi game a fixture is, with its books in ESPN's home/away order."""

    kalshi: KalshiGame
    home: Book
    away: Book


def match(
    fixtures: Sequence[Fixture],
    games: Sequence[KalshiGame],
    side_ids: Callable[[Book], frozenset[str]],
) -> tuple[dict[str, Matched], list[str]]:
    """Which Kalshi game each fixture is, by game id; and the Kalshi events that fit none.

    gold-rush's rule, on the live listing: a side resolves to every ESPN
    team its code or name could be (`side_ids`). A Kalshi game is the one
    fixture on its Eastern day, or the day either side, whose two teams are
    its two sides. Kalshi writes Washington State and Wayne State both as
    `WSU`; the opponent and the date are what say which.
    """
    by_day: dict[date, list[Fixture]] = defaultdict(list)
    for fixture in fixtures:
        by_day[fixture.game.date.astimezone(EASTERN).date()].append(fixture)
    matched: dict[str, Matched] = {}
    unmatched = []
    for game in games:
        first, second = (side_ids(side) for side in game.sides)
        hits: list[tuple[str, Matched]] = []
        for day in (
            game.day,
            game.day - timedelta(days=1),
            game.day + timedelta(days=1),
        ):
            for fixture in by_day.get(day, []):
                if fixture.home_id in first and fixture.away_id in second:
                    hits.append((fixture.game.game_id, Matched(game, *game.sides)))
                elif fixture.home_id in second and fixture.away_id in first:
                    hits.append(
                        (fixture.game.game_id, Matched(game, *reversed(game.sides)))
                    )
            # The venue's own day first: a date it got wrong by one is the
            # likely miss, not a rematch two days later.
            if hits:
                break
        if len(hits) == 1:
            matched[hits[0][0]] = hits[0][1]
        else:
            unmatched.append(game.event)
    return matched, unmatched


def no_vig_home(home: Book, away: Book) -> tuple[float, float] | None:
    """The backtest's market: the home probability, and the overround it came from.

    Each side's cost is its YES ask plus the taker fee, as `markets._cost`
    prices a gold-rush row. None when either side has no ask.
    """
    if home.ask is None or away.ask is None:
        return None
    home_cost = home.ask + kalshi_fee(home.ask)
    away_cost = away.ask + kalshi_fee(away.ask)
    total = home_cost + away_cost
    return home_cost / total, total


def routes(side: Book, other: Book) -> list[Route]:
    """Every way to back `side`, cheapest first: its YES, or NO on `other`."""
    found = []
    if side.ask is not None:
        found.append(Route(side.ticker, "yes", side.ask, side.ask_size))
    if other.bid is not None:
        found.append(Route(other.ticker, "no", 1 - other.bid, other.bid_size))
    return sorted(found, key=lambda route: (route.price, route.action != "yes"))


def order_fee(price: float, contracts: int) -> float:
    """Kalshi's taker fee on one order: the per-contract fee, rounded up to the cent."""
    return math.ceil(round(100 * contracts * kalshi_fee(price), 6)) / 100


def contracts_for(stake: float, price: float) -> int:
    """How many contracts `stake` dollars buys at `price`, fee included."""
    count = int(stake // (price + kalshi_fee(price)))
    while count > 0 and count * price + order_fee(price, count) > stake:
        count -= 1
    return count


def fillable(orderbook: Mapping, action: str, limit: float) -> float:
    """Contracts on offer at or below `limit` for buying `action` in one market.

    Kalshi's book lists bids only. Buying YES fills against NO bids, at 1
    minus their price, and buying NO fills against YES bids the same way.
    `orderbook` is `GET /markets/{ticker}/orderbook`'s body.
    """
    levels = orderbook.get("orderbook_fp") or {}
    against = levels.get("no_dollars" if action == "yes" else "yes_dollars") or []
    return sum(
        float(size) for price, size in against if 1 - float(price) <= limit + 1e-9
    )


class PreviousGame(NamedTuple):
    """A team's last game before a fixture: its id, kickoff, and whether it's final."""

    game_id: str
    date: datetime
    completed: bool


def previous_games(
    games: Iterable[Game],
) -> Callable[[str, datetime], PreviousGame | None]:
    """Each team's last game before a moment, from a season's games.

    Canceled and postponed games are left out: there is no result to wait
    for.
    """
    by_team: dict[str, list[Game]] = defaultdict(list)
    for game in games:
        if game.status in NOT_HAPPENING:
            continue
        by_team[game.home].append(game)
        by_team[game.away].append(game)
    for played in by_team.values():
        played.sort(key=lambda g: g.date)

    def previous(team: str, before: datetime) -> PreviousGame | None:
        earlier = [g for g in by_team.get(team, []) if g.date < before]
        if not earlier:
            return None
        last = earlier[-1]
        return PreviousGame(last.game_id, last.date, last.completed)

    return previous


def entry(previous: Iterable[PreviousGame | None]) -> datetime | None:
    """When a fixture's entry price was up: once both teams' last games were over.

    `betting.line_windows`' rule, `GAME_LENGTH` after the later kickoff.
    None for a fixture neither team has played before, whose entry is
    whenever the market opened.
    """
    kickoffs = [game.date for game in previous if game is not None]
    return max(kickoffs) + GAME_LENGTH if kickoffs else None


class Priced(NamedTuple):
    """One matched game, priced: the backtest's edge and the best way to take it."""

    home_market: float
    overround: float
    home: bool
    edge: float
    route: Route | None
    cost: float | None
    expected_return: float | None


def price(model_home: float, matched: Matched) -> Priced | None:
    """The model's side of a game, its edge, and where to buy it. None for no market."""
    market = no_vig_home(matched.home, matched.away)
    if market is None:
        return None
    home_market, overround = market
    home = model_home > home_market
    side, other = (matched.home, matched.away) if home else (matched.away, matched.home)
    model = model_home if home else 1 - model_home
    found = routes(side, other)
    best = found[0] if found else None
    cost = None if best is None else best.price + kalshi_fee(best.price)
    return Priced(
        home_market=home_market,
        overround=overround,
        home=home,
        edge=abs(model_home - home_market),
        route=best,
        cost=cost,
        expected_return=None if cost is None else model / cost - 1,
    )


def skip_reason(
    priced: Priced,
    side_division: str | None,
    other_division: str | None,
    waiting_on: Sequence[str],
    since_entry: timedelta | None,
    min_edge: float = MIN_EDGE,
) -> str | None:
    """Why a priced game isn't a pick, or None if it is. See the module docstring."""
    if waiting_on:
        return f"previous game not final: {', '.join(waiting_on)}"
    if since_entry is not None and since_entry > MAX_SINCE_ENTRY:
        return "past the entry window"
    if priced.overround > MAX_OVERROUND:
        return f"book too wide ({priced.overround:.2f})"
    if priced.route is None:
        return "nothing on offer"
    if priced.edge < min_edge:
        return "edge under threshold"
    if other_division == TOP_DIVISION and side_division != TOP_DIVISION:
        return "lower-division side vs FBS"
    return None

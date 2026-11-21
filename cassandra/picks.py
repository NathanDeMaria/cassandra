"""What to bet now: the model's win probability against a venue's live book.

`betting.py` grades a model's picks after the fact. On Kalshi's college
football moneylines it found an edge at the *entry* -- the first price after
both teams' previous game -- and only past a threshold: under it the
overround eats the edge. Judged by the close, the smallest qualifying edges
were worth little, and so were favorites at a small edge, which Kelly
stakes the most -- so a favorite needs a bigger one. The edge faded as the
book caught up, so a price is taken only for a while after the entry. Every
one of those bars is a league's rules (`PickRule`, in s3's autobet.json);
none of their values are kept here.

The default model is the success-rate one (`glicko_margin_units_success`)
since 2026-10-14: it forecasts better held out, and on these books the two
can't be told apart -- over 2025-26 at the entry, brier 0.1535 against the
roster model's 0.1537 (the book's 0.1601), weight on the model 0.80 against
0.79. This module is that bet, made live. It takes every open Kalshi game,
finds its ESPN fixture, and prices it against the model the way the
backtest did.

**The edge** is the backtest's: each team's own YES ask plus Kalshi's taker
fee, normalized to sum to one, against the model's probability. A book
whose two costs sum past the rule's `max_overround` isn't a market.

**The price** is the cheapest way to back the side, which the backtest
didn't look for. A team's YES and its opponent's NO pay on the same result,
since football has no ties, and the opponent's NO costs 1 minus its YES
bid. Taking the cheaper of the two never makes a pick worse than the one
graded.

**What isn't a pick** (`skip_reason`):
- an edge under the rule's `min_edge`, or under its `favorite_edge` for a
  pick costing its `favorite_price` or more, or under its
  `model_longshot_edge` for a side the model gives under
  `model_longshot_probability` (`PickRule.edge_needed`), or a pick costing
  under its `min_price`;
- a pick returning under the rule's `min_expected_return` per dollar at its
  price: the model's probability over the cost, less one. The edge is
  against the no-vig market, which on a wide book sits well under a
  favorite's ask, so a heavy favorite can clear the edge while paying
  nearly what the model says it's worth. On 2025-26's books those picks
  were worth about nothing by the close, in all three leagues;
- the model backing a team below FBS against an FBS one. The model's known
  miss on those games is under-rating the FBS side: 10-20 point FBS
  favorites beat it by about 2 points in 2015-25. So its edge on the
  lower side is mostly that miss.
- a game where either team's previous game isn't final. The model hasn't
  seen that result yet, and the backtest's entry waited for it.
- a game whose entry was longer ago than the rule's `max_since_entry`. By
  then the market has caught up. The entry
  is the backtest's: the first price after both teams' previous games, so
  the later of those games ending and the market opening (`entry`). A game
  after a bye can have its previous games end before Kalshi lists it.

**Leagues.** A `PickLeague` says what differs by league: its Kalshi series,
its model, where its teams' Kalshi spellings are looked up, the division a
lower one's edge doesn't count against, whether its model reads
play-by-play (EPA and starting quarterbacks) or player ratings, and
whether a pick's injury news is read. The edge, the entry and the window
are the same for every league. `PICK_LEAGUES` holds the ones
`picks.py --league` takes.

Basketball (`WOMENS`, then `MENS`) prices on `glicko_players`. On Kalshi's
2025-26 books, at the entry and under football's kind of bar, womens picks
paid and mens a little (November 2026, at the 11-12 fits, which had seen
that season; the season's first days, which they hadn't, too). Two hours
before tip, mens lost. A basketball entry is nearly always the market's
opening: teams play every few days, and Kalshi lists a game a day or two
before tip, long after the previous final. Opening books are often a
cent or two on one side and near a dollar on the other. They are real --
contracts trade at those prices until someone sweeps them -- so they're
bet like any other, on the edge, and sized under the rules' cap.

The model's players' margin is ten-minute-version's daily ratings, filed
each morning with each player's chance of playing and each game's luck,
so a game also waits for those to cover both teams' previous games
(`ratings_waiting`). Waiting a morning past the final cost the backtest
little: in 2025-26 the book had opened before the morning's data on about
one game in eight, and waiting for it cost under a point of return.

**Venues.** The backtest was Kalshi's, and the rule is judged the same way
on every venue's own book (`cassandra.venues`): its no-vig market from its
own asks and fee, its cheapest route, its own listing time for the entry.
A game on two venues is two `Offer`s; it backs one side (`backed_side`),
and an order goes to its offers cheapest first (`ranked`).

Everything here is a pure function of the venues' JSON, the fixtures and
the model's probabilities; `picks.py` at the repo root does the fetching.
"""

import math
import re
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

from call_it_what_you_want import NCAAFB, NCAAMBB, NCAAWBB
from endgame.types import Game

from .betting import GAME_LENGTH
from .venues import KALSHI_FEES, Book, Fees, Route, Venue, VenueGame

#: Kalshi's college football moneylines. FCS-only games are in their own
#: series, which is how gold-rush lists them too.
NCAAFB_SERIES = ("KXNCAAFGAME", "KXNCAAFCSGAME")


class PickRule(NamedTuple):
    """What a priced game must clear to be a pick (`skip_reason`). Its values
    are a league's rules in s3's autobet.json (`picks_job.LeagueRules`);
    none are kept here.
    """

    #: The smallest edge over the no-vig price worth betting.
    min_edge: float
    #: How long after the entry a price is still worth taking. The edge fades
    #: as the book catches up.
    max_since_entry: timedelta
    #: The most a book's two costs may sum to and still be a market.
    max_overround: float
    #: A pick costing `favorite_price` or more, fee included, must clear
    #: `favorite_edge` instead: Kelly stakes those the most, and a small edge
    #: on them barely beats the fee. None for no such rule.
    favorite_price: float | None = None
    favorite_edge: float | None = None
    #: A pick costing less than this, fee included, isn't one.
    min_price: float = 0.0
    #: A pick the model itself gives under `model_longshot_probability` of
    #: winning must clear `model_longshot_edge` instead. It's the model's
    #: probability, not the price: a cheap price on a team the model rates
    #: well is a market behind, which the edge already measures, while a team
    #: the model rates a longshot is where its probabilities are least tested.
    #: None for no such rule.
    model_longshot_probability: float | None = None
    model_longshot_edge: float | None = None
    #: The least a pick must return per dollar by the model: its probability
    #: over what the pick costs, fee included, less one. The edge is the
    #: model against the market, not against the price paid, and a wide
    #: book's no-vig price sits well under a favorite's ask -- a 95c favorite
    #: can clear a 10-point edge and return 2c on the dollar. None for no
    #: such rule.
    min_expected_return: float | None = None

    def edge_needed(self, cost: float, model: float = 1.0) -> float:
        """The edge a pick costing `cost` a contract, fee included, that the
        model gives `model` of winning, must clear."""
        need = self.min_edge
        if (
            self.favorite_price is not None
            and self.favorite_edge is not None
            and cost >= self.favorite_price
        ):
            need = max(need, self.favorite_edge)
        if (
            self.model_longshot_probability is not None
            and self.model_longshot_edge is not None
            and model < self.model_longshot_probability
        ):
            need = max(need, self.model_longshot_edge)
        return need


#: Kalshi files a game under its US Eastern date. So does `match`.
EASTERN = ZoneInfo("America/New_York")

#: Statuses of a game that will not be played when it says, so it is
#: neither a fixture to price nor a previous game to wait for.
NOT_HAPPENING = frozenset({"STATUS_CANCELED", "STATUS_POSTPONED"})

#: The division a team must be in for the model's edge on it to count
#: against a team from a lower one. See the module docstring.
TOP_DIVISION = "FBS"


class PickLeague(NamedTuple):
    """What a league's picks run reads, and what it rules out. See the module
    docstring's "Leagues"."""

    #: Cassandra's name for it: its s3 seasons, models and team namer.
    name: str
    #: The Kalshi series its games are listed in.
    series: tuple[str, ...]
    #: The model it prices with unless `picks.py --model` says otherwise.
    model: str
    #: call-it-what-you-want's league, which Kalshi's spellings are filed under.
    teams: str
    #: `skip_reason`'s `top_division`; None for no such rule.
    top_division: str | None
    #: Whether the model reads play-by-play -- EPA and starting quarterbacks
    #: -- so a run checks who played quarterback and which games have plays.
    plays: bool
    #: Whether the model reads ten-minute-version's player ratings, so a game
    #: waits for them to cover both teams' previous games (`ratings_waiting`).
    players: bool
    #: Whether a pick's injury news is read off ESPN's stories
    #: (`placing.injury_news`, football's).
    news: bool


FOOTBALL = PickLeague(
    name="ncaafb",
    series=NCAAFB_SERIES,
    model="glicko_margin_units_success",
    teams=NCAAFB,
    top_division=TOP_DIVISION,
    plays=True,
    players=False,
    news=True,
)

#: Kalshi's womens college basketball moneylines.
NCAAWBB_SERIES = ("KXNCAAWBGAME",)

WOMENS = PickLeague(
    name="womens",
    series=NCAAWBB_SERIES,
    model="glicko_players",
    teams=NCAAWBB,
    top_division=None,
    plays=False,
    players=True,
    news=False,
)

#: Kalshi's mens college basketball moneylines.
NCAAMBB_SERIES = ("KXNCAAMBGAME",)

MENS = WOMENS._replace(name="mens", series=NCAAMBB_SERIES, teams=NCAAMBB)

PICK_LEAGUES: dict[str, PickLeague] = {
    league.name: league for league in (FOOTBALL, WOMENS, MENS)
}


def league_of(ticker: str) -> str | None:
    """The league a Kalshi market or event is in, by its series; None for
    one no `PickLeague` lists."""
    series = ticker.split("-", 1)[0]
    for league in PICK_LEAGUES.values():
        if series in league.series:
            return league.name
    return None


_EVENT_DATE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS: dict[str, int] = {
    name: number
    for number, name in enumerate(
        "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), start=1
    )
}


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
    except TypeError, ValueError:
        return None


def _quote(value: Any) -> float | None:
    price = _number(value)
    return price if price is not None and 0 < price < 1 else None


def _time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


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
        opened=_time(market.get("open_time")) if market.get("open_time") else None,
        shard=int(market.get("exchange_index") or 0),
    )


def kalshi_games(markets: Iterable[Mapping]) -> tuple[list[VenueGame], list[str]]:
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
        games.append(VenueGame(event, day, (teams[0], teams[1])))
    return games, odd


class Fixture(NamedTuple):
    """An unplayed ESPN game, under canonical names, with both teams' ESPN ids."""

    game: Game
    home_id: str | None
    away_id: str | None


class Matched(NamedTuple):
    """The venue's game a fixture is, with its books in ESPN's home/away order."""

    game: VenueGame
    home: Book
    away: Book


def match(
    fixtures: Sequence[Fixture],
    games: Sequence[VenueGame],
    side_ids: Callable[[Book], frozenset[str]],
) -> tuple[dict[str, Matched], list[str]]:
    """Which venue game each fixture is, by game id; and the venue's events
    that fit none.

    gold-rush's rule, on the live listing: a side resolves to every ESPN
    team its code or name could be (`side_ids`). A venue's game is the one
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


def no_vig_home(
    home: Book, away: Book, fees: Fees = KALSHI_FEES
) -> tuple[float, float] | None:
    """The backtest's market: the home probability, and the overround it came from.

    Each side's cost is its own contract's ask plus the venue's fee, as
    `markets._cost` prices a gold-rush row. None when either side has no ask.
    """
    if home.ask is None or away.ask is None:
        return None
    home_cost = home.ask + fees.contract(home.ask)
    away_cost = away.ask + fees.contract(away.ask)
    total = home_cost + away_cost
    return home_cost / total, total


def routes(side: Book, other: Book) -> list[Route]:
    """Every way to back `side` on Kalshi, cheapest first: its YES, or NO on
    `other`."""
    found = []
    if side.ask is not None:
        found.append(Route(side.ticker, "yes", side.ask, side.ask_size))
    if other.bid is not None:
        found.append(Route(other.ticker, "no", 1 - other.bid, other.bid_size))
    return sorted(found, key=lambda route: (route.price, route.action != "yes"))


def order_fee(price: float, contracts: int, fees: Fees = KALSHI_FEES) -> float:
    """The fee on one order filling at `price`: Kalshi's per-contract fee,
    rounded up to the cent, unless another venue's `fees`."""
    return fees.order(price, contracts)


def contracts_for(stake: float, price: float, fees: Fees = KALSHI_FEES) -> int:
    """How many contracts `stake` dollars buys at `price`, the order's fee
    included."""
    # The contract's fee in the end is never more than its order's share, so
    # this starts at or over the answer.
    count = int(stake // (price + fees.contract(price)))
    while count > 0 and count * price + fees.order(price, count) > stake:
        count -= 1
    return count


def _each(fees: Fees | Sequence[Fees], n: int) -> list[Fees]:
    """One venue's fees for every pick, or each pick's own."""
    return list(fees) if isinstance(fees, Sequence) else [fees] * n


def kelly(probability: float, cost: float) -> float:
    """The Kelly fraction for a contract paying 1 that costs `cost`, fee in."""
    return max(probability - cost, 0.0) / (1 - cost)


def allocate(
    budget: float,
    prices: Sequence[float],
    weights: Sequence[float],
    fees: Fees | Sequence[Fees] = KALSHI_FEES,
) -> list[int]:
    """Contracts for each pick when `budget` dollars are split by `weights`.

    A split of a small budget can leave a pick less than one contract's
    price. That pick is dropped, the smallest weight first, and the budget
    re-split among the rest, so the money goes to bets that can be placed
    rather than being left unspent.

    Split by Kelly weight, a fixed weekly budget over the top picks by edge
    beat the same budget split evenly over every pick on the 2025-26 Kalshi
    entries -- which left many picks below one contract. Ranking by expected
    return instead did a little better, but it buys longshots, and 23 weeks
    can't tell any of the three
    apart.
    """
    each = _each(fees, len(prices))
    active = [i for i, weight in enumerate(weights) if weight > 0]
    while active:
        total = sum(weights[i] for i in active)
        counts = {
            i: contracts_for(budget * weights[i] / total, prices[i], each[i])
            for i in active
        }
        unplaced = [i for i in active if counts[i] == 0]
        if not unplaced:
            return [counts.get(i, 0) for i in range(len(prices))]
        active.remove(min(unplaced, key=lambda i: weights[i]))
    return [0] * len(prices)


class Longshots(NamedTuple):
    """A pick costing under `price`, fee included, is staked at `scale` of
    what it would be otherwise. The default shrinks nothing.

    Womens basketball's backtest edge at the entry sat mostly in its
    cheapest picks, on books so early that many had traded nothing: a
    longshot's ask may be one contract nobody corrected. Until live fills
    say those prices are real, they can be staked smaller.
    """

    price: float = 0.0
    scale: float = 1.0

    def of(self, cost: float) -> float:
        """The share of its stake a pick costing `cost` gets."""
        return self.scale if cost < self.price else 1.0


def kelly_counts(
    probabilities: Sequence[float],
    costs: Sequence[float],
    limits: Sequence[float],
    bankroll: float,
    fraction: float,
    max_bet: float,
    longshots: Longshots = Longshots(),
    fees: Fees | Sequence[Fees] = KALSHI_FEES,
) -> list[int]:
    """Contracts for each pick staked on its own: `fraction` of Kelly on `bankroll`.

    Unlike `allocate`, no pick's stake depends on how many others there
    are, so picks that turn up an hour apart are sized alike. Each stake is
    at most `max_bet`, and the contracts are counted at the pick's limit
    price, fee included, so a fill at the limit costs no more than the stake.
    A longshot's stake is shrunk first (`Longshots`). Each pick's contracts
    are counted with its venue's `fees`.
    """
    return [
        contracts_for(
            min(fraction * longshots.of(c) * kelly(p, c) * bankroll, max_bet),
            limit,
            venue,
        )
        for p, c, limit, venue in zip(
            probabilities, costs, limits, _each(fees, len(limits))
        )
    ]


def rungs(
    probability: float,
    price: float,
    covered: float,
    covered_cost: float,
    unders: Sequence[float],
    bankroll: float,
    fraction: float,
    max_bet: float,
    fees: Fees = KALSHI_FEES,
) -> list[tuple[float, int]]:
    """The rungs under a pick: (price, contracts) for each of `unders`.

    Rungs are limit bids left on the book under a pick's price until
    kickoff, filling if the price comes back down to them. On the 2025-26
    entries, shallow ones filled often and returned more than the entry on
    the same games -- a lower price, and no taker fee. What fills is a
    little worse than the model says, the market having moved for a reason;
    deep enough, that took most of the gain. How deep is the league's
    rules' `rungs`.

    Each rung buys what Kelly adds at its lower price over what the pick
    and the rungs above it already cover (`covered` contracts, costing at
    most `covered_cost`), so with every rung filled the position is the
    Kelly stake at the lowest price, never more. All of it stays within
    `max_bet`, counted at each rung's limit with the taker fee.
    """
    found = []
    for under in unders:
        at = round(price - under, 2)
        if at < 0.01:
            break
        stake = min(
            fraction * kelly(probability, at + fees.contract(at)) * bankroll, max_bet
        )
        count = min(
            top_up(contracts_for(stake, at, fees), covered),
            contracts_for(max(max_bet - covered_cost, 0.0), at, fees),
        )
        if count <= 0:
            continue
        found.append((at, count))
        covered += count
        covered_cost += count * at + fees.order(at, count)
    return found


def shard_transfers(
    balances: Mapping[int, int], shard: int, need: int
) -> list[tuple[int, int]]:
    """What to move to `shard` so it holds `need`, in the units given: (source
    shard, amount) for each move, the fullest other shards first. Short of
    `need` when the shards don't hold that much between them; none when
    `shard` has enough, so cash already there stays put.
    """
    short = need - balances.get(shard, 0)
    moves = []
    for source, held in sorted(balances.items(), key=lambda item: -item[1]):
        if short <= 0:
            break
        if source == shard or held <= 0:
            continue
        amount = min(held, short)
        moves.append((source, amount))
        short -= amount
    return moves


def held_on(
    positions: Mapping[str, float], side: Book, other: Book
) -> tuple[float, float]:
    """Contracts already held for `side` winning, and for it losing.

    Kalshi signs a market's position: YES contracts positive, NO negative.
    YES on `side` and NO on `other` both pay when `side` wins.
    """
    on_side = positions.get(side.ticker, 0.0)
    on_other = positions.get(other.ticker, 0.0)
    return (
        max(on_side, 0.0) + max(-on_other, 0.0),
        max(-on_side, 0.0) + max(on_other, 0.0),
    )


def top_up(target: int, held: float) -> int:
    """Whole contracts still to buy for a pick sized at `target` with `held` in hand.

    Kalshi fills in hundredths of a contract, so what's held can be
    fractional; what's ordered here is whole.
    """
    return max(math.floor(target - held + 1e-9), 0)


def fit_to_cash(
    counts: Sequence[int],
    limits: Sequence[float],
    cash: float,
    fees: Fees | Sequence[Fees] = KALSHI_FEES,
) -> list[int]:
    """`counts` scaled down together until their worst-case cost fits `cash`,
    each order's fee its venue's (`fees`)."""
    each = _each(fees, len(limits))

    def cost(found: Sequence[int]) -> float:
        return sum(
            n * p + venue.order(p, n)
            for n, p, venue in zip(found, limits, each)
            if n > 0
        )

    total = cost(counts)
    if total <= cash:
        return list(counts)
    scale = cash / total
    scaled = [math.floor(n * scale) for n in counts]
    # Fees round up per order, so a proportional cut can still be a cent
    # over; trim the largest order until it fits.
    while cost(scaled) > cash and any(scaled):
        largest = max(range(len(scaled)), key=lambda i: scaled[i] * limits[i])
        scaled[largest] -= 1
    return scaled


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


def ratings_waiting(
    previous: Mapping[str, PreviousGame | None],
    newest: Mapping[str, date],
    cleared: Collection[str] | None = None,
) -> list[str]:
    """The teams whose player data hasn't caught up with their previous
    game: `newest` is each team's latest ratings row's `as_of`, the first
    day its fit left out, and `cleared` the games the morning's players run
    read -- the luck file's ids (`cassandra.predictor.luck`).

    A row covers a game when it's dated after the game's Eastern date, as
    Kalshi files it: a late tip out west is the next day in UTC, and a row
    dated that day might not have seen it. But a morning files a row for
    every team whether or not its last game's box score and plays were in
    yet, and a game they weren't in for is in neither the team's fit nor its
    missed-time estimate (the rows' `available`). The luck file gets a game
    only from a morning that read its box score, so with `cleared` a team
    also waits for its previous game to be in it.

    A team with no rows at all -- one the ratings don't reach -- is never
    waited for, since none are coming; the model prices it unblended.
    """
    return [
        team
        for team, game in previous.items()
        if game is not None
        and game.completed
        and team in newest
        and (
            newest[team] <= game.date.astimezone(EASTERN).date()
            or (cleared is not None and game.game_id not in cleared)
        )
    ]


def entry(
    previous: Iterable[PreviousGame | None], opened: datetime | None = None
) -> datetime | None:
    """When a fixture's entry price was up: the first price after both teams'
    last games were over, as in `betting.line_windows`.

    That's the later of `GAME_LENGTH` after the later kickoff and `opened`,
    when the market opened: after a bye, or two, the previous games can end
    before Kalshi lists the game. None when neither is known.
    """
    kickoffs = [game.date for game in previous if game is not None]
    known = [max(kickoffs) + GAME_LENGTH] if kickoffs else []
    if opened is not None:
        known.append(opened)
    return max(known) if known else None


class Priced(NamedTuple):
    """One matched game, priced: the backtest's edge and the best way to take it."""

    home_market: float
    overround: float
    home: bool
    edge: float
    route: Route | None
    cost: float | None
    expected_return: float | None

    @property
    def model(self) -> float:
        """The model's probability of the side bet on: the side is the one
        the model rates above the market, by `edge`."""
        return (self.home_market if self.home else 1 - self.home_market) + self.edge


def price(
    model_home: float,
    matched: Matched,
    fees: Fees = KALSHI_FEES,
    ways: Callable[[Book, Book], list[Route]] = routes,
    home: bool | None = None,
) -> Priced | None:
    """The model's side of a game, its edge, and where to buy it, on one
    venue: its `fees` and its `ways` to back a side (`Venue.routes`). None
    for no market.

    The side is the one the model rates above this venue's market, unless
    `home` says which: another venue's book can put the model on the other
    side of a close game, and a pick backs one side. Its edge on a side the
    model rates under the market is negative.
    """
    market = no_vig_home(matched.home, matched.away, fees)
    if market is None:
        return None
    home_market, overround = market
    if home is None:
        home = model_home > home_market
    side, other = (matched.home, matched.away) if home else (matched.away, matched.home)
    model = model_home if home else 1 - model_home
    found = ways(side, other)
    best = found[0] if found else None
    cost = None if best is None else best.price + fees.contract(best.price)
    return Priced(
        home_market=home_market,
        overround=overround,
        home=home,
        edge=(model_home - home_market) * (1 if home else -1),
        route=best,
        cost=cost,
        expected_return=None if cost is None else model / cost - 1,
    )


def max_price(
    model: float,
    priced: Priced,
    side: Book,
    rule: PickRule,
    fees: Fees = KALSHI_FEES,
) -> float | None:
    """The most the pick's route can cost, to the cent, and still clear the
    edge it needs at that price (`PickRule.edge_needed`) and the rule's
    `min_expected_return`.

    The edge is the model against the side's share of the two asks, so a
    rising price eats it about a point a cent. This holds the overround
    where it was read -- one side going up as the other comes down -- and
    lets the route move as far as the side's own ask can.
    """
    if priced.route is None or side.ask is None:
        return None
    shift = priced.route.price - side.ask
    found = None
    for cents in range(1, 100):
        ask = cents / 100
        route = ask + shift
        cost = route + fees.contract(route)
        need = rule.edge_needed(cost, model)
        if ask + fees.contract(ask) > (model - need) * priced.overround:
            break
        if (
            rule.min_expected_return is not None
            and model / cost - 1 < rule.min_expected_return
        ):
            break
        found = route
    if found is None:
        return None
    return min(math.floor(round(found * 100, 6)) / 100, 0.99)


def skip_reason(
    priced: Priced,
    side_division: str | None,
    other_division: str | None,
    waiting_on: Sequence[str],
    since_entry: timedelta | None,
    rule: PickRule,
    top_division: str | None = TOP_DIVISION,
    ratings_behind: Sequence[str] = (),
) -> str | None:
    """Why a priced game isn't a pick, or None if it is. See the module
    docstring; `rule` is the league's bar.

    `top_division` is the league's (`PickLeague.top_division`): a side from
    below it isn't backed against a team in it. None turns the rule off.
    `ratings_behind` are the teams `ratings_waiting` names.
    """
    if waiting_on:
        return f"previous game not final: {', '.join(waiting_on)}"
    if ratings_behind:
        return f"player data not in yet: {', '.join(ratings_behind)}"
    if since_entry is not None and since_entry > rule.max_since_entry:
        return "past the entry window"
    if priced.overround > rule.max_overround:
        return f"book too wide ({priced.overround:.2f})"
    if priced.route is None:
        return "nothing on offer"
    if priced.edge < rule.min_edge:
        return "edge under threshold"
    if priced.cost is not None and priced.edge < rule.edge_needed(
        priced.cost, priced.model
    ):
        # At no price at all only the model's own longshot bar is left.
        if priced.edge < rule.edge_needed(0.0, priced.model):
            return (
                f"edge under {(rule.model_longshot_edge or 0) * 100:g} points with the "
                f"model under {(rule.model_longshot_probability or 0) * 100:g}%"
            )
        return (
            f"edge under {(rule.favorite_edge or 0) * 100:g} points at "
            f"{(rule.favorite_price or 0) * 100:g}c+"
        )
    if (
        rule.min_expected_return is not None
        and priced.expected_return is not None
        and priced.expected_return < rule.min_expected_return
    ):
        return f"returns under {rule.min_expected_return * 100:g}% at its price"
    if priced.cost is not None and priced.cost < rule.min_price:
        return f"price under {rule.min_price * 100:g}c"
    if (
        top_division is not None
        and other_division == top_division
        and side_division != top_division
    ):
        return f"lower-division side vs {top_division}"
    return None


class Offer(NamedTuple):
    """One venue's price on a game's side: whether it's a pick there, and the
    most an order there may pay (`max_price`). The same game on another venue
    is another offer, priced on that venue's own book by the same rule."""

    venue: Venue
    matched: Matched
    priced: Priced
    max_price: float | None
    since_entry: timedelta | None
    skip: str | None

    @property
    def sides(self) -> tuple[Book, Book]:
        """The book of the side backed, and its opponent's."""
        home, away = self.matched.home, self.matched.away
        return (home, away) if self.priced.home else (away, home)


def backed_side(priced: Sequence[Priced]) -> bool:
    """Whether a game's pick backs its home side, from each venue's price of
    it: the side the model rates furthest above a venue's market. Venues
    disagree only on a game the model and the market call close."""
    return max(priced, key=lambda p: p.edge).home


def ranked(offers: Sequence[Offer]) -> list[Offer]:
    """A game's offers in the order an order goes to them: the picks, the
    cheapest first, then the rest in the order given."""
    picks = sorted(
        (o for o in offers if o.skip is None),
        key=lambda o: o.priced.cost if o.priced.cost is not None else math.inf,
    )
    return [*picks, *(o for o in offers if o.skip is not None)]

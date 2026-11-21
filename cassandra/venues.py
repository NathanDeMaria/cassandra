"""Where a pick can be bought: the interface every betting venue implements.

A venue is an exchange listing the games the model prices -- Kalshi
(`cassandra.kalshi_venue`) and Polymarket US (`cassandra.polymarket_us`).
Everything one trades on a game pays when one of the game's two teams wins,
so the interface is cut there. Each venue reads its own listing, books,
account and orders into the same few shapes:

- a game (`VenueGame`): its two sides' books (`Book`), each side's best
  quotes for its own contract, in that side's terms;
- the ways to back a side (`Venue.routes`), and what's on offer on one at or
  under a price (`Venue.on_offer`);
- which side a contract held, or an order resting, pays on (`Venue.pays`);
- its fees (`Fees`): what a contract costs past its price, which prices a
  pick, and what an order is charged when it fills, which is the cash it
  takes;
- an account (`Account`): its cash, what it holds (`Holding`), its resting
  orders (`Resting`), buying (`Fill`) and cancelling.

Above that the code is the same for every venue. `cassandra.picks` prices
each venue's book by the league's one rule. `picks.py` and
`cassandra.placing` add the accounts up -- one bankroll, one stake a game
whichever venues hold it, one exposure cap a league -- and send each order
to the cheapest venue first and what it leaves to the next.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import NamedTuple, Protocol

from .markets import kalshi_fee


class VenueError(RuntimeError):
    """A venue turned a request down, or couldn't be reached."""


class Book(NamedTuple):
    """One side of a game on a venue: the best quotes for its own contract,
    and the contracts behind them, in the side's terms.

    `bid` and `ask` are None when nobody is offering that side. Kalshi
    sends 0 or 1 then, which is not a price anyone could trade.
    """

    #: The contract that pays when this side wins: Kalshi's market ticker,
    #: Polymarket US's moneyline slug (the same for both its sides).
    ticker: str
    #: The venue's code for the team, and its name.
    code: str
    team: str
    bid: float | None
    bid_size: float
    ask: float | None
    ask_size: float
    volume: float
    volume_24h: float
    #: When the venue opened the market; None when the listing doesn't say.
    opened: datetime | None = None
    #: Kalshi's exchange shard it trades on, whose balance pays for an order
    #: on it. 0 elsewhere.
    shard: int = 0
    #: What the venue is told to buy for this side's own contract
    #: (`Route.action`): "yes" on Kalshi, "long" or "short" on Polymarket US.
    outcome: str = "yes"
    #: More spellings of the team, for matching it to ESPN's.
    names: tuple[str, ...] = ()

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


class VenueGame(NamedTuple):
    """One game on a venue, filed under its Eastern date: its two sides."""

    #: The venue's id for the game: Kalshi's event ticker, Polymarket US's
    #: moneyline slug. Every contract on the game says it (`Venue.event_of`).
    event: str
    day: date
    sides: tuple[Book, Book]


class Route(NamedTuple):
    """One way to back a team on a venue: which contract, what the venue is
    told to buy on it, at what price, and how many are offered there.

    `price` is always what backing the team costs a contract, before the fee,
    whatever terms the venue quotes it in.
    """

    ticker: str
    #: "yes" or "no" on Kalshi; "long" or "short" on Polymarket US.
    action: str
    price: float
    size: float


class Fill(NamedTuple):
    """What an order did when it was placed, priced from the side bought."""

    order_id: str
    filled: float
    resting: float
    price: float | None
    fee: float | None


class Holding(NamedTuple):
    """Contracts an account holds on one contract: what they are, and what
    they cost -- the dollars they have at risk."""

    ticker: str
    action: str
    contracts: float
    cost: float


class Resting(NamedTuple):
    """An order an account has resting on a venue's book."""

    order_id: str
    ticker: str
    action: str
    #: What it pays a contract of `action`, whatever terms the venue quotes.
    price: float
    remaining: float
    #: A bid under a pick's price left until kickoff (`cassandra.picks.rungs`),
    #: told apart from a pick's own order.
    rung: bool = False


class Balance(NamedTuple):
    """An account's money: what it can spend now, and what its open positions
    (and the cash its resting orders hold) are worth. The two make its share
    of the bankroll."""

    cash: float
    invested: float


def _cents_half_even(value: float) -> float:
    """`value` dollars to the cent, half to even. Through a short decimal, so
    a float a hair off a half cent rounds the way the exact sum would."""
    exact = Decimal(f"{value:.8f}")
    return float(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


class Fees(Protocol):
    """What a venue charges to buy.

    The two can differ: a rebate paid later makes a contract cheaper in the
    end than the cash its order takes when it fills.
    """

    def contract(self, price: float, when: datetime | None = None) -> float:
        """What one contract bought at `price` costs past its price, in the
        end: what pricing a pick counts."""
        ...

    def order(
        self, price: float, contracts: float, when: datetime | None = None
    ) -> float:
        """What an order of `contracts` filling at `price` is charged, to the
        cent: with the price, the cash it takes."""
        ...


class KalshiFees:
    """Kalshi's taker fee: 0.07 p(1 - p) a contract, each order's rounded
    up to the cent."""

    def contract(self, price: float, when: datetime | None = None) -> float:
        return kalshi_fee(price)

    def order(
        self, price: float, contracts: float, when: datetime | None = None
    ) -> float:
        return math.ceil(round(100 * contracts * kalshi_fee(price), 6)) / 100


KALSHI_FEES = KalshiFees()


class FeeTerm(NamedTuple):
    """A run of a fee schedule: Θ, the share of it rebated, and when it ends
    (None for no end)."""

    theta: float
    rebate: float
    until: datetime | None


class ScheduledFees:
    """A Θ p(1 - p) taker fee whose Θ and rebate change on set dates, each
    order's fee rounded to the cent half to even. The rebate is paid back
    after the trade, so a contract costs Θ (1 - rebate) p(1 - p) in the end
    while its order takes the whole fee when it fills."""

    def __init__(self, terms: Sequence[FeeTerm], clock: Callable[[], datetime]):
        self.terms = list(terms)
        self.clock = clock

    def term(self, when: datetime | None = None) -> FeeTerm:
        at = when or self.clock()
        for term in self.terms:
            if term.until is None or at < term.until:
                return term
        return self.terms[-1]

    def contract(self, price: float, when: datetime | None = None) -> float:
        term = self.term(when)
        return term.theta * (1 - term.rebate) * price * (1 - price)

    def order(
        self, price: float, contracts: float, when: datetime | None = None
    ) -> float:
        term = self.term(when)
        return _cents_half_even(term.theta * contracts * price * (1 - price))


class Venue(Protocol):
    """A venue's market data, which needs no account."""

    @property
    def name(self) -> str:
        """Its name in flags, logs and files, e.g. "kalshi"."""
        ...

    @property
    def label(self) -> str:
        """How a person reads it, e.g. "Kalshi"."""
        ...

    @property
    def spelled_as(self) -> str:
        """call-it-what-you-want's source its team spellings are filed under."""
        ...

    @property
    def fees(self) -> Fees: ...

    @property
    def tags_orders(self) -> bool:
        """Whether an order can be marked as a rung (`Resting.rung`) when it's
        placed. A rung that couldn't be told from a pick's own order would
        keep a later run off its game, so a venue that can't gets none."""
        ...

    def games(self, league: str) -> tuple[list[VenueGame], list[str]]:
        """Every open game in `league` (a `cassandra.picks.PickLeague` name),
        and the events listed that aren't a two-team game."""
        ...

    def routes(self, side: Book, other: Book) -> list[Route]:
        """Every way to back `side`, cheapest first."""
        ...

    def on_offer(self, route: Route, limit: float) -> float:
        """Contracts on offer on `route` now at or under `limit`, the price of
        backing the side."""
        ...

    def pays(self, side: Book, other: Book, ticker: str, action: str) -> bool | None:
        """Whether a contract (`ticker`, bought as `action`) pays when `side`
        wins: True, False when it pays on `other`, None when it's on neither."""
        ...

    def event_of(self, ticker: str) -> str:
        """The game (`VenueGame.event`) a contract is on."""
        ...

    def league_of(self, ticker: str) -> str | None:
        """The league a contract's game is in; None for one no pick league
        has."""
        ...

    def held_as(self, holding: Holding) -> str:
        """A holding in a few words, for a pick's `held_as`."""
        ...

    def account(self, demo: bool = False, dry_run: bool = False) -> Account:
        """The account its API key opens. `dry_run` reads it and sends
        nothing: an order or a cancel is printed instead."""
        ...

    def has_key(self) -> bool:
        """Whether an API key for the account is where `account` looks."""
        ...


class Account(Protocol):
    """A venue account, through its API key."""

    @property
    def venue(self) -> Venue: ...

    @property
    def dry_run(self) -> bool:
        """Read the account, send nothing."""
        ...

    def balance(self) -> Balance: ...

    def holdings(self) -> list[Holding]:
        """Every open position, each contract held."""
        ...

    def resting(self) -> list[Resting]: ...

    def buy(
        self,
        route: Route,
        contracts: int,
        limit: float,
        expires: int | None = None,
        rung: bool = False,
    ) -> Fill:
        """Buy up to `contracts` on `route` paying at most `limit` a contract
        (the price of backing the side). Immediate-or-cancel with no
        `expires`; otherwise what doesn't fill rests until then (epoch
        seconds)."""
        ...

    def cancel(self, order: Resting) -> float:
        """Take a resting order off the book: how many contracts that took off."""
        ...

    def fund(self, side: Book, dollars: float) -> None:
        """Have `dollars` where an order on `side` can spend them, or raise
        `VenueError`. Kalshi pays an order from its market's shard alone."""
        ...


def held_on(
    venue: Venue, holdings: Sequence[Holding], side: Book, other: Book
) -> tuple[float, float]:
    """Contracts an account holds for `side` winning, and for it losing."""
    mine = against = 0.0
    for holding in holdings:
        pays = venue.pays(side, other, holding.ticker, holding.action)
        if pays is True:
            mine += holding.contracts
        elif pays is False:
            against += holding.contracts
    return mine, against


def league_exposure(
    venue: Venue,
    holdings: Sequence[Holding],
    resting: Sequence[Resting],
    league: str,
) -> float:
    """Dollars a league has at risk on a venue: what its open positions cost,
    and the most its resting orders -- rungs too -- could still spend."""
    held = sum(h.cost for h in holdings if venue.league_of(h.ticker) == league)
    waiting = sum(
        o.remaining * o.price for o in resting if venue.league_of(o.ticker) == league
    )
    return held + waiting


def by_name(venues: Sequence[Venue]) -> Mapping[str, Venue]:
    return {venue.name: venue for venue in venues}

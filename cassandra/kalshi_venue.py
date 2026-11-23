"""Kalshi as a venue (`cassandra.venues`): its listing, books and account.

A Kalshi game is an event with one market per team, each a YES/NO contract
on that team winning. A team is backed by buying its YES, or NO on its
opponent's market, which pays on the same result since these games have no
ties; the order book of each market holds bids only, so a contract is bought
against the other side's bids. Its fee is 0.07 p(1 - p) a contract, each
order's rounded up to the cent. The HTTP client, signing and order bodies
are `cassandra.kalshi_trade`.
"""

import math
import time
import urllib.error
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from call_it_what_you_want import KALSHI

from .kalshi_trade import (
    CENTICENTS,
    KalshiClient,
    KalshiError,
    bought,
    event_of,
    is_rung,
    order_price,
    public_get,
    rung_id,
    signed_positions,
)
from .picks import (
    PICK_LEAGUES,
    fillable,
    kalshi_games,
    league_of,
    routes,
    shard_transfers,
)
from .venues import (
    KALSHI_FEES,
    Balance,
    Book,
    Fill,
    GameInfo,
    Holding,
    Resting,
    Route,
    Settled,
    Trade,
    VenueGame,
    other_side,
)

#: How long a run waits for a transfer between shards to land before it gives
#: up on the pick, and how often it looks. Kalshi processes them asynchronously.
FUND_WAIT = timedelta(seconds=30)
FUND_POLL = timedelta(seconds=1)


def markets(series: tuple[str, ...]) -> list[dict]:
    """Every open market in `series`, as `GET /markets` lists them."""
    found: list[dict] = []
    for ticker in series:
        cursor = None
        while True:
            params = {"series_ticker": ticker, "status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            page = public_get("/markets", params)
            found += page.get("markets") or []
            cursor = page.get("cursor")
            if not cursor or not page.get("markets"):
                break
    return found


def _public[T](read: Callable[..., T], *args: Any) -> T:
    """A public read, its failure -- an HTTP error, a timeout, a body that
    isn't JSON -- a `KalshiError` like the account's. `public_get` itself
    raises urllib's, which `cassandra.ledger` waits out a 429 by."""
    try:
        return read(*args)
    except (OSError, ValueError) as error:
        raise KalshiError(f"Kalshi's market data: {error}") from None


class Kalshi:
    """Kalshi's market data; `account` opens the account."""

    name = "kalshi"
    label = "Kalshi"
    spelled_as = KALSHI
    fees = KALSHI_FEES
    tags_orders = True

    def games(self, league: str) -> tuple[list[VenueGame], list[str]]:
        return kalshi_games(_public(markets, PICK_LEAGUES[league].series))

    def routes(self, side: Book, other: Book) -> list[Route]:
        return routes(side, other)

    def on_offer(self, route: Route, limit: float) -> float:
        book = _public(public_get, f"/markets/{route.ticker}/orderbook")
        return fillable(book, route.action, limit)

    def pays(self, side: Book, other: Book, ticker: str, action: str) -> bool | None:
        # YES on a team and NO on its opponent both pay when it wins.
        if ticker == side.ticker:
            return action == "yes"
        if ticker == other.ticker:
            return action == "no"
        return None

    def event_of(self, ticker: str) -> str:
        return event_of(ticker)

    def league_of(self, ticker: str) -> str | None:
        return league_of(ticker)

    def held_as(self, holding: Holding) -> str:
        return f"{holding.contracts:g} {self.contract_label(holding.ticker, holding.action)}"

    def contract_label(self, ticker: str, action: str) -> str:
        return f"{action.upper()} {code_of(ticker)}"

    def fetch_game(self, event: str) -> dict:
        return _read(f"/events/{event}")

    def read_game(self, listing: Mapping) -> GameInfo:
        return GameInfo(
            title=(listing.get("event") or {}).get("title") or "",
            sides=teams(listing),
            final=all(m.get("status") in FINAL for m in listing.get("markets", [])),
        )

    def side_key(self, ticker: str, action: str, game: GameInfo) -> str:
        # YES on a team pays it; NO on a team pays the other.
        code = code_of(ticker)
        return code if action == "yes" else other_side(code, game.sides)

    def fetch_prices(
        self, event: str, listing: Mapping, start: datetime, end: datetime
    ) -> dict:
        """Each team's hourly candles from `start` to `end`, by team code."""
        series = event.split("-", 1)[0]
        return {
            code_of(market["ticker"]): _read(
                f"/series/{series}/markets/{market['ticker']}/candlesticks",
                {
                    "start_ts": int(start.timestamp()),
                    "end_ts": int(end.timestamp()),
                    "period_interval": 60,
                },
            ).get("candlesticks", [])
            for market in listing.get("markets", [])
        }

    def no_vig(self, prices: Mapping, at: int) -> dict[str, float]:
        return candle_no_vig(prices, at)

    def account(self, demo: bool = False, dry_run: bool = False) -> KalshiAccount:
        return KalshiAccount(self, KalshiClient.from_env(demo=demo, dry_run=dry_run))

    def has_key(self) -> bool:
        try:
            KalshiClient.from_env()
        except SystemExit:
            return False
        return True


KALSHI_VENUE = Kalshi()


#: A market's statuses once its game is settled for good.
FINAL = {"finalized", "settled", "determined"}
#: Seconds a public read waits before each try again after a 429: the ledger
#: reads a few hundred candles a run, and runs close together trip the limit.
_PUBLIC_RETRIES = (2, 4, 8, 16, 32)


def _read(path: str, params: dict | None = None) -> dict:
    """A public read, waiting out Kalshi's rate limit."""
    for wait in _PUBLIC_RETRIES:
        try:
            return public_get(path, params)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(wait)
    return public_get(path, params)


def code_of(ticker: str) -> str:
    """A market ticker's team code: `...-26OCT24HOUUTAH-UTAH` -> `UTAH`."""
    return ticker.rsplit("-", 1)[1]


def teams(listing: Mapping) -> dict[str, str]:
    """An event's two teams: each market's code, and Kalshi's name for the team."""
    return {
        code_of(m["ticker"]): m.get("yes_sub_title") or code_of(m["ticker"])
        for m in listing.get("markets", [])
    }


def candle_no_vig(
    candles: Mapping[str, Sequence[Mapping]], at: int
) -> dict[str, float]:
    """Each team's price without the vig, from the last hourly candle ending
    at or before `at` (unix seconds): the mid of its YES bid and ask, the two
    teams' mids scaled to sum to one. Empty when a side has no quote."""
    mids = {}
    for code, series in candles.items():
        quoted = [
            c
            for c in series
            if c["end_period_ts"] <= at
            and (c.get("yes_bid") or {}).get("close_dollars")
            and (c.get("yes_ask") or {}).get("close_dollars")
        ]
        if not quoted:
            return {}
        last = quoted[-1]
        mids[code] = (
            float(last["yes_bid"]["close_dollars"])
            + float(last["yes_ask"]["close_dollars"])
        ) / 2
    total = sum(mids.values())
    return {code: mid / total for code, mid in mids.items()} if total else {}


def _num(value: Any) -> float:
    return float(value) if value not in (None, "") else 0.0


def _when(value: Any) -> datetime:
    """A Kalshi time; the epoch for none, as an old test's rows have."""
    if not value:
        return datetime.fromtimestamp(0, UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def held_contract(fill: Mapping) -> str:
    """What a fill bought on its ticker, "yes" or "no". Kalshi's V2 fills
    write a NO bought through the book as `action=sell, side=no` (see
    `cassandra.ledger`), and selling a YES nets to holding NO."""
    if fill.get("side") == "yes" and fill.get("action") == "buy":
        return "yes"
    return "no"


def as_trades(fills: Iterable[Mapping], since: datetime | None = None) -> list[Trade]:
    """`GET /portfolio/fills`' fills since `since`, each priced as the
    contract held (`held_contract`)."""
    found = []
    for fill in fills:
        at = _when(fill.get("created_time"))
        if since is not None and at < since:
            continue
        held = held_contract(fill)
        found.append(
            Trade(
                ticker=fill["ticker"],
                action=held,
                contracts=_num(fill.get("count_fp") or fill.get("count")),
                price=_num(fill.get(f"{held}_price_dollars")),
                fee=_num(fill.get("fee_cost")),
                at=at,
                order_id=fill.get("order_id"),
            )
        )
    return found


def as_settled(
    settlements: Iterable[Mapping], since: datetime | None = None
) -> list[Settled]:
    """`GET /portfolio/settlements`' settlements since `since`, each market's
    YES and NO held as a leg. The market's fees and payout go on its first
    leg. A sale before the end is booked as a buy of the other side: each
    YES-NO pair paid its dollar when it was netted, not at settlement, so the
    settlement's revenue leaves it out and the payout adds it back."""
    found = []
    for s in settlements:
        at = _when(s.get("settled_time"))
        if since is not None and at < since:
            continue
        yes, no = _num(s.get("yes_count_fp")), _num(s.get("no_count_fp"))
        fees = _num(s.get("fee_cost"))
        payout = _num(s.get("revenue")) / 100 + min(yes, no)
        for count, cost, action in (
            (yes, "yes_total_cost_dollars", "yes"),
            (no, "no_total_cost_dollars", "no"),
        ):
            if count:
                found.append(
                    Settled(
                        s["ticker"], action, count, _num(s.get(cost)), fees, payout, at
                    )
                )
                fees = payout = 0.0
    return found


def as_holdings(market_positions: Iterable[Mapping]) -> list[Holding]:
    """`GET /portfolio/positions`' positions, YES or NO, at what each cost
    (`market_exposure_dollars`) and the fees paid on it."""
    positions = list(market_positions)
    by_ticker = {p["ticker"]: p for p in positions}
    return [
        Holding(
            ticker,
            "yes" if count > 0 else "no",
            abs(count),
            _num(by_ticker[ticker].get("market_exposure_dollars")),
            _num(by_ticker[ticker].get("fees_paid_dollars")),
        )
        for ticker, count in signed_positions(positions).items()
    ]


def as_resting(orders: Iterable[Mapping]) -> list[Resting]:
    """`GET /portfolio/orders`' resting orders, each priced as the contract
    it buys."""
    return [
        Resting(
            order_id=str(order.get("order_id") or ""),
            ticker=order["ticker"],
            action=bought(order),
            price=order_price(order),
            remaining=float(order.get("remaining_count_fp") or 0),
            rung=is_rung(order),
        )
        for order in orders
    ]


class KalshiAccount:
    """A Kalshi account through `KalshiClient`, read into the venue's shapes."""

    def __init__(self, venue: Kalshi, client: KalshiClient) -> None:
        self.venue = venue
        self.client = client
        # Each shard's cash, read once a run when the first order needs it
        # (`fund`), and spent down as the run's orders are funded.
        self._shards: dict[int, int] | None = None

    @property
    def dry_run(self) -> bool:
        return self.client.dry_run

    def balance(self) -> Balance:
        cash, invested = self.client.account()
        return Balance(cash, invested)

    def holdings(self) -> list[Holding]:
        return as_holdings(self.client.market_positions())

    def resting(self) -> list[Resting]:
        return as_resting(self.client.resting_orders())

    def trades(self, since: datetime | None = None) -> list[Trade]:
        return as_trades(self.client.fills(), since)

    def settled(self, since: datetime | None = None) -> list[Settled]:
        return as_settled(self.client.settlements(), since)

    def credits(self, since: datetime) -> float:
        return 0.0

    def buy(
        self,
        route: Route,
        contracts: int,
        limit: float,
        expires: int | None = None,
        rung: bool = False,
    ) -> Fill:
        return self.client.buy(
            route.ticker,
            route.action,
            contracts,
            limit,
            expires,
            rung_id() if rung else None,
        )

    def cancel(self, order: Resting) -> float:
        return self.client.cancel(order.order_id, order.ticker)

    def fund(self, side: Book, dollars: float) -> None:
        if self._shards is None:
            try:
                self._shards = self.client.shard_balances()
            except KalshiError as error:
                # Not worth losing the run's orders over: one on a funded
                # shard still goes through, and one that isn't is rejected
                # and emailed.
                print(f"  !! no shard balances, so nothing moved: {error}")
                self._shards = {}
        fund_shard(self.client, self._shards, side.shard, dollars)


def fund_shard(
    client: KalshiClient,
    balances: dict[int, int],
    shard: int,
    dollars: float,
    wait: timedelta = FUND_WAIT,
    poll: timedelta = FUND_POLL,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Have `dollars` on `shard` for a pick's orders, moving it from the other
    shards if it isn't there, and set it aside in `balances`, this run's read
    of them in centicents (`KalshiClient.shard_balances`).

    What an order pays comes from its market's shard alone, and the leagues
    trade on different ones. A transfer is processed asynchronously, so this
    waits for the money to land; `KalshiError` if it hasn't by `wait`, or if
    Kalshi turns the transfer down. No breakdown to read means nothing to
    move. A dry run prints the moves and doesn't wait.
    """
    if not balances:
        return
    need = math.ceil(round(dollars * CENTICENTS, 6))
    moves = shard_transfers(balances, shard, need)
    for source, amount in moves:
        client.transfer(source, shard, amount)
        balances[source] -= amount
        balances[shard] = balances.get(shard, 0) + amount
        print(f"  moved ${amount / CENTICENTS:,.2f} from shard {source} to {shard}")
    if moves and not client.dry_run:
        deadline = datetime.now(UTC) + wait
        while (found := client.shard_balances()).get(shard, 0) < need:
            if datetime.now(UTC) >= deadline:
                raise KalshiError(
                    f"transfer to shard {shard} not in after {wait}: it holds "
                    f"{found.get(shard, 0) / CENTICENTS:,.2f}, this pick needs "
                    f"{need / CENTICENTS:,.2f}"
                )
            sleep(poll.total_seconds())
        balances.update(found)
    balances[shard] = balances.get(shard, 0) - need

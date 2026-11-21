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
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

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
    Holding,
    Resting,
    Route,
    VenueGame,
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


class Kalshi:
    """Kalshi's market data; `account` opens the account."""

    name = "kalshi"
    label = "Kalshi"
    spelled_as = KALSHI
    fees = KALSHI_FEES
    tags_orders = True

    def games(self, league: str) -> tuple[list[VenueGame], list[str]]:
        return kalshi_games(markets(PICK_LEAGUES[league].series))

    def routes(self, side: Book, other: Book) -> list[Route]:
        return routes(side, other)

    def on_offer(self, route: Route, limit: float) -> float:
        book = public_get(f"/markets/{route.ticker}/orderbook")
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
        return (
            f"{holding.contracts:g} {holding.action.upper()} "
            f"{holding.ticker.rsplit('-', 1)[-1]}"
        )

    def account(self, demo: bool = False, dry_run: bool = False) -> KalshiAccount:
        return KalshiAccount(self, KalshiClient.from_env(demo=demo, dry_run=dry_run))

    def has_key(self) -> bool:
        try:
            KalshiClient.from_env()
        except SystemExit:
            return False
        return True


KALSHI_VENUE = Kalshi()


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
        """Each market's position, YES or NO, at what it cost
        (`market_exposure_dollars`)."""
        found = []
        positions = self.client.market_positions()
        costs = {
            p["ticker"]: float(p.get("market_exposure_dollars") or 0) for p in positions
        }
        for ticker, count in signed_positions(positions).items():
            found.append(
                Holding(ticker, "yes" if count > 0 else "no", abs(count), costs[ticker])
            )
        return found

    def resting(self) -> list[Resting]:
        return [
            Resting(
                order_id=order["order_id"],
                ticker=order["ticker"],
                action=bought(order),
                price=order_price(order),
                remaining=float(order.get("remaining_count_fp") or 0),
                rung=is_rung(order),
            )
            for order in self.client.resting_orders()
        ]

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

"""Polymarket US as a venue (`cassandra.venues`): its listing, books and account.

Polymarket US is the CFTC-regulated exchange a US account trades on -- not
the international Polymarket, whose books and prices are its own. Market
data is public, at `gateway.polymarket.us`; the account is at
`api.polymarket.us`, signed with an Ed25519 key made at
polymarket.us/developer. The key is read from

    POLYMARKET_US_KEY_ID        the key's id, or the file ~/.polymarket_us/key_id
    POLYMARKET_US_SECRET_FILE   a file holding the base64 secret key, default
                                ~/.polymarket_us/secret_key

**A game** is an event with one moneyline market, a single YES/NO instrument
with a team on each side: YES backs the `long` team, NO the other. Both
sides trade in one book, quoted in YES terms, so the short team's ask is one
minus the best bid. A team is backed one way only, its own side of that
instrument (`ORDER_INTENT_BUY_LONG` or `ORDER_INTENT_BUY_SHORT`). Team codes
are the international venue's, filed in call-it-what-you-want under
`polymarket`. Football lists a game two to three weeks ahead, basketball
about two (Kalshi: two weeks, and four days), and basketball lists several
times as many games as Kalshi.

**Fees** are Θ p(1 - p) a contract, each order's rounded to the cent half to
even, and makers pay none. From 2026-11-21 through 2026-12-31, API taker
orders pay Θ 0.0695 and have 70% of it rebated (`FEE_TERMS`); a contract's
cost counts the rebate, and an order's cash the whole fee, which is what
leaves the account when it fills. After that, Θ is 0.095 under $10k of
30-day taker volume.

**What the docs don't settle**, and this assumes:

- the price of a NO buy is sent in YES terms, one minus the NO price: the
  partner order model says "`price` is always the YES price" and "BUY q NO
  @ p ≡ SELL q YES @ (1 − p)", and the retail API's concepts page says a NO
  at 0.40 is "really selling YES at $0.60". Every fill is checked against
  its limit (`check_fill`), and a run sends no more NO orders after one
  that paid over it;
- a NO position's `netPositionDecimal` is negative;
- a signed path leaves out the query string, as Kalshi's does;
- `goodTillTime` is an RFC 3339 time.

Make the first live order a small NO buy and read it back.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from call_it_what_you_want import POLYMARKET
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .venues import (
    Balance,
    Book,
    FeeTerm,
    Fill,
    Holding,
    Resting,
    Route,
    ScheduledFees,
    VenueError,
    VenueGame,
)

GATEWAY = "https://gateway.polymarket.us"
API = "https://api.polymarket.us"
EASTERN = ZoneInfo("America/New_York")
_KEY_DIR = Path.home() / ".polymarket_us"
# The retail API allows 25 requests a second from an address, shared.
_PACE = 1 / 12
_READ_RETRIES = (1.0, 2.0, 4.0)
# Events asked for a page at a time, and how far ahead a listing looks.
_PAGE = 100
_AHEAD = timedelta(days=30)
#: How long an order placed waits for its executions to come back with it.
_BLOCK_SECONDS = 4

#: Each pick league's Polymarket US league, and the series seen holding its
#: games. A series' name doesn't say its season -- `cbb-2025` (7) holds
#: November 2026's games -- so a listing asks every one seen and the one the
#: league endpoint calls active.
LEAGUES: dict[str, tuple[str, tuple[int, ...]]] = {
    "ncaafb": ("cfb", (5, 225)),
    "mens": ("cbb", (7,)),
    "womens": ("wcbb", (21,)),
}
_BY_SLUG = {slug: league for league, (slug, _) in LEAGUES.items()}

#: The API taker fee by date: before the promotion; the promotion's Θ and
#: rebate through 2026-12-31 Eastern; then the lowest volume tier's.
FEE_TERMS = (
    FeeTerm(0.095, 0.0, datetime(2026, 11, 21, 5, 1, tzinfo=UTC)),
    FeeTerm(0.0695, 0.70, datetime(2027, 1, 1, 5, 0, tzinfo=UTC)),
    FeeTerm(0.095, 0.0, None),
)
FEES = ScheduledFees(FEE_TERMS, lambda: datetime.now(UTC))

_WOMENS = " (W)"
_INTENT = {"long": "ORDER_INTENT_BUY_LONG", "short": "ORDER_INTENT_BUY_SHORT"}
_BOUGHT = {v: k for k, v in _INTENT.items()}
_RESTING_STATES = {
    "ORDER_STATE_NEW",
    "ORDER_STATE_PARTIALLY_FILLED",
    "ORDER_STATE_PENDING_NEW",
    "ORDER_STATE_PENDING_RISK",
}


class PolymarketUsError(VenueError):
    pass


def _money(value: Any) -> float | None:
    """An `Amount` (`{"value": "0.55", "currency": "USD"}`) or a number, in
    dollars; None for none."""
    if isinstance(value, Mapping):
        value = value.get("value")
    try:
        return None if value is None else float(value)
    except TypeError, ValueError:
        return None


def _quote(value: Any) -> float | None:
    found = _money(value)
    return found if found is not None and 0 < found < 1 else None


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        found = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return found if found.tzinfo else found.replace(tzinfo=UTC)


def moneyline(event: Mapping) -> Mapping | None:
    """An event's moneyline market, open or not; None for an event with none."""
    for market in event.get("markets") or []:
        if (
            market.get("marketType") == "moneyline"
            or market.get("sportsMarketType") == "moneyline"
        ):
            return market
    return None


def _names(team: Mapping) -> tuple[str, str, tuple[str, ...]]:
    """A team's code, name, and its other spellings. Women's teams off the
    international venue's roster are named "West Florida (W)"; ESPN's name,
    the fallback for a team nobody filed, is school and mascot."""
    name = (team.get("name") or "").removesuffix(_WOMENS)
    alias = team.get("alias") or ""
    others = (
        (team.get("safeName") or "").removesuffix(_WOMENS),
        alias,
        f"{name} {alias}" if name and alias else "",
    )
    return (
        team.get("abbreviation") or "",
        name,
        tuple(dict.fromkeys(n for n in others if n and n != name)),
    )


def game(event: Mapping) -> VenueGame | None:
    """An event's open moneyline as a game, or None when it hasn't one with a
    team on each side.

    The long side is first. Its bid and ask are the instrument's; the short
    side's are one minus the instrument's ask and bid. The listing carries
    no sizes: `PolymarketUs.on_offer` reads the book for those.
    """
    market = moneyline(event)
    if market is None or market.get("closed") or market.get("active") is False:
        return None
    sides = market.get("marketSides") or []
    longs = [s for s in sides if s.get("long") is True and s.get("team")]
    shorts = [s for s in sides if s.get("long") is False and s.get("team")]
    if len(sides) != 2 or len(longs) != 1 or len(shorts) != 1:
        return None
    kickoff = _time(event.get("startTime") or market.get("gameStartTime"))
    if kickoff is None:
        return None
    slug = market["slug"]
    bid, ask = _quote(market.get("bestBidQuote")), _quote(market.get("bestAskQuote"))
    opened = _time(market.get("startDate"))

    def side(found: Mapping, outcome: str) -> Book:
        code, name, others = _names(found["team"])
        long = outcome == "long"
        return Book(
            ticker=slug,
            code=code,
            team=name,
            bid=bid if long else (None if ask is None else round(1 - ask, 6)),
            bid_size=0.0,
            ask=ask if long else (None if bid is None else round(1 - bid, 6)),
            ask_size=0.0,
            volume=0.0,
            volume_24h=0.0,
            opened=opened,
            outcome=outcome,
            names=others,
        )

    return VenueGame(
        event=slug,
        day=kickoff.astimezone(EASTERN).date(),
        sides=(side(longs[0], "long"), side(shorts[0], "short")),
    )


def games(events: Sequence[Mapping]) -> tuple[list[VenueGame], list[str]]:
    """The open games in an event listing, and the events slugged like a
    game (`cfb-dart-prnce-2026-11-07`) without one to read."""
    found: dict[str, VenueGame] = {}
    odd = []
    for event in events:
        read = game(event)
        if read is not None:
            found.setdefault(read.event, read)
            continue
        parts = (event.get("slug") or "").split("-")
        market = moneyline(event)
        if (
            len(parts) == 6
            and all(p.isdigit() for p in parts[3:])
            and not (market or {}).get("closed")
        ):
            odd.append(event.get("slug") or str(event.get("id")))
    return sorted(found.values(), key=lambda g: (g.day, g.event)), sorted(set(odd))


def on_offer(book: Mapping, outcome: str, limit: float) -> float:
    """Contracts on offer at or under `limit` for buying `outcome` ("long" or
    "short") of a moneyline, from `GET /v1/markets/{slug}/book`.

    The book is in YES terms: the long side buys from the offers, the short
    side by selling into the bids, at one minus each bid.
    """
    data = book.get("marketData") or {}
    if outcome == "long":
        levels = [(_money(lv["px"]), lv["qty"]) for lv in data.get("offers") or []]
    else:
        levels = [
            (1 - (_money(lv["px"]) or 1), lv["qty"]) for lv in data.get("bids") or []
        ]
    return sum(
        float(qty)
        for price, qty in levels
        if price is not None and price <= limit + 1e-9
    )


def public_get(path: str, params: dict | None = None) -> dict:
    """A GET of the public market data. Anything that goes wrong -- an HTTP
    error, a timeout, a body that isn't JSON -- is tried again a few times,
    then raised as `PolymarketUsError`."""
    query = f"?{urllib.parse.urlencode(params, doseq=True)}" if params else ""
    request = urllib.request.Request(
        f"{GATEWAY}{path}{query}", headers={"User-Agent": "cassandra-picks"}
    )
    retries = list(_READ_RETRIES)
    while True:
        time.sleep(_PACE)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (OSError, ValueError) as error:
            if retries:
                time.sleep(retries.pop(0))
                continue
            raise PolymarketUsError(f"GET {path}: {error}") from None


def _iso(stamp: datetime) -> str:
    return stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def listing(league: str, now: datetime | None = None) -> list[dict]:
    """Every event in `league`'s series kicking off from twelve hours ago to
    `_AHEAD` from now."""
    slug, known = LEAGUES[league]
    found = public_get(f"/v2/leagues/{slug}")
    active = (found.get("league") or found).get("activeSeriesId")
    series = list(dict.fromkeys((*known, *([int(active)] if active else []))))
    now = now or datetime.now(UTC)
    events: list[dict] = []
    for one in series:
        offset = 0
        while True:
            page = public_get(
                "/v1/events",
                {
                    "seriesId": one,
                    "startTimeMin": _iso(now - timedelta(hours=12)),
                    "startTimeMax": _iso(now + _AHEAD),
                    "limit": _PAGE,
                    "offset": offset,
                },
            )
            found_events = page.get("events") or []
            events += found_events
            offset += len(found_events)
            if len(found_events) < _PAGE:
                break
    return events


class PolymarketUs:
    """Polymarket US's market data; `account` opens the account."""

    name = "polymarket_us"
    label = "Polymarket US"
    spelled_as = POLYMARKET
    fees = FEES
    # An order has no id of the client's own to mark a rung by.
    tags_orders = False

    def games(self, league: str) -> tuple[list[VenueGame], list[str]]:
        if league not in LEAGUES:
            return [], []
        return games(listing(league))

    def routes(self, side: Book, other: Book) -> list[Route]:
        # The opponent's side is the same instrument, so backing a team is
        # only ever its own side of it.
        if side.ask is None:
            return []
        return [Route(side.ticker, side.outcome, side.ask, side.ask_size)]

    def on_offer(self, route: Route, limit: float) -> float:
        return on_offer(
            public_get(f"/v1/markets/{route.ticker}/book"), route.action, limit
        )

    def pays(self, side: Book, other: Book, ticker: str, action: str) -> bool | None:
        if ticker != side.ticker:
            return None
        if action == side.outcome:
            return True
        if action == other.outcome:
            return False
        return None

    def event_of(self, ticker: str) -> str:
        return ticker

    def league_of(self, ticker: str) -> str | None:
        # `aec-cbb-stlou-stmry-2026-11-20`: the league is one of the first two.
        for part in ticker.split("-")[:2]:
            if part in _BY_SLUG:
                return _BY_SLUG[part]
        return None

    def held_as(self, holding: Holding) -> str:
        return f"{holding.contracts:g} {holding.action.upper()} on {self.label}"

    def account(self, demo: bool = False, dry_run: bool = False) -> PolymarketUsAccount:
        if demo:
            raise SystemExit(f"{self.label} has no demo exchange: drop --demo")
        return PolymarketUsAccount(self, PolymarketUsClient.from_env(dry_run=dry_run))

    def has_key(self) -> bool:
        return _key_paths()[0] is not None and _key_paths()[1].exists()


POLYMARKET_US = PolymarketUs()


def _key_paths() -> tuple[str | None, Path]:
    key_id = os.environ.get("POLYMARKET_US_KEY_ID")
    id_file = _KEY_DIR / "key_id"
    if not key_id and id_file.exists():
        key_id = id_file.read_text().strip()
    secret = Path(
        os.environ.get("POLYMARKET_US_SECRET_FILE") or _KEY_DIR / "secret_key"
    ).expanduser()
    return key_id or None, secret


def sign(
    private_key: Ed25519PrivateKey, timestamp_ms: str, method: str, path: str
) -> str:
    """The request's signature: timestamp, method and path, no query string,
    signed with Ed25519 and base64-encoded."""
    message = f"{timestamp_ms}{method}{path.split('?')[0]}".encode()
    return base64.b64encode(private_key.sign(message)).decode()


def private_key(secret: str) -> Ed25519PrivateKey:
    """The key from its base64 secret, whose first 32 bytes are the seed."""
    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(secret.strip())[:32])


def _amount(value: float) -> dict:
    return {"value": f"{value:.4f}", "currency": "USD"}


def order_body(
    slug: str, outcome: str, count: int, limit: float, expires: int | None = None
) -> dict:
    """`POST /v1/orders`'s body: buy `count` of `outcome` ("long"/"short") for
    at most `limit` a contract, the price of that side.

    The price goes in YES terms (see the module docstring): a short at most
    `limit` sells YES at least 1 - `limit`. With no `expires` the order is
    immediate-or-cancel; with one it rests until then. Either way the
    response waits for its executions.
    """
    if outcome not in _INTENT:
        raise ValueError(f"outcome is long or short, not {outcome!r}")
    if count < 1 or not 0 < limit < 1:
        raise ValueError(f"bad order: {count} at {limit}")
    body: dict[str, Any] = {
        "marketSlug": slug,
        "intent": _INTENT[outcome],
        "type": "ORDER_TYPE_LIMIT",
        "price": _amount(limit if outcome == "long" else 1 - limit),
        "quantity": count,
        "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL"
        if expires is None
        else "TIME_IN_FORCE_GOOD_TILL_DATE",
        "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
        "synchronousExecution": True,
        "maxBlockTime": str(_BLOCK_SECONDS),
    }
    if expires is not None:
        body["goodTillTime"] = _iso(datetime.fromtimestamp(expires, UTC))
    return body


def _side_price(yes_price: float | None, outcome: str) -> float | None:
    if yes_price is None:
        return None
    return yes_price if outcome == "long" else round(1 - yes_price, 6)


def fill(response: Mapping, outcome: str, count: int) -> Fill:
    """A create-order response read back: contracts filled, at what price for
    the side bought, the fee, and what's left resting.

    The order as it stands comes with each execution; the latest one's says
    it all. With none, the executions themselves are summed.
    """
    executions = response.get("executions") or []
    order = next(
        (e["order"] for e in reversed(executions) if isinstance(e.get("order"), dict)),
        None,
    )
    if order is not None:
        filled = float(order.get("cumQuantity") or 0)
        leaves = float(order.get("leavesQuantity") or 0)
        resting = leaves if order.get("state") in _RESTING_STATES else 0.0
        price = _money(order.get("avgPx")) if filled else None
        fee = _money(order.get("commissionNotionalTotalCollected"))
    else:
        trades = [
            e
            for e in executions
            if e.get("type") in ("EXECUTION_TYPE_FILL", "EXECUTION_TYPE_PARTIAL_FILL")
        ]
        filled = sum(float(e.get("lastShares") or 0) for e in trades)
        paid = sum(
            float(e.get("lastShares") or 0) * (_money(e.get("lastPx")) or 0)
            for e in trades
        )
        price = paid / filled if filled else None
        fee = sum(_money(e.get("commissionNotionalCollected")) or 0 for e in trades)
        resting = 0.0
    return Fill(
        order_id=str(response.get("id") or ""),
        filled=filled,
        resting=resting,
        price=_side_price(price, outcome),
        fee=fee,
    )


def check_fill(done: Fill, outcome: str, limit: float) -> None:
    """`PolymarketUsError` if an order filled over its limit: the side's price
    convention isn't what this module assumes."""
    if done.filled and done.price is not None and done.price > limit + 0.0051:
        raise PolymarketUsError(
            f"{outcome} order {done.order_id} filled {done.filled:g} at "
            f"{done.price:.4f}, over its {limit:.2f} limit: check how a "
            f"{outcome} price is sent (cassandra.polymarket_us)"
        )


def holdings(positions: Mapping[str, Mapping]) -> list[Holding]:
    """`GET /v1/portfolio/positions`'s map of positions, each a `Holding`:
    long for a positive net position, short for a negative one."""
    found = []
    for slug, position in positions.items():
        if position.get("expired"):
            continue
        net = float(
            position.get("netPositionDecimal") or position.get("netPosition") or 0
        )
        if not net:
            continue
        cost = abs(_money(position.get("cost")) or 0.0)
        found.append(Holding(slug, "long" if net > 0 else "short", abs(net), cost))
    return found


def resting(orders: Sequence[Mapping]) -> list[Resting]:
    """`GET /v1/orders/open`'s orders that buy a side, each priced as that
    side's contract."""
    found = []
    for order in orders:
        outcome = _BOUGHT.get(order.get("intent") or "")
        price = _side_price(_money(order.get("price")), outcome or "long")
        if outcome is None or price is None:
            continue
        found.append(
            Resting(
                order_id=str(order["id"]),
                ticker=order["marketSlug"],
                action=outcome,
                price=price,
                remaining=float(order.get("leavesQuantity") or 0),
            )
        )
    return found


def balance(balances: Sequence[Mapping]) -> Balance:
    """`GET /v1/account/balances`'s dollar balance: what can be spent now
    (`buyingPower`, open orders' cash already out), and what the positions
    are worth plus the cash open orders hold."""
    usd = next(
        (b for b in balances if (b.get("currency") or "USD") == "USD"),
        {},
    )
    current = float(usd.get("currentBalance") or 0)
    spendable = float(usd.get("buyingPower") or 0)
    cash = min(spendable, current)
    return Balance(cash, float(usd.get("assetNotional") or 0) + current - cash)


class PolymarketUsClient:
    """The authenticated retail API."""

    def __init__(
        self,
        key_id: str,
        key: Ed25519PrivateKey,
        base: str = API,
        dry_run: bool = False,
    ) -> None:
        self.key_id = key_id
        self.key = key
        self.base = base
        #: Read the account but never write to it.
        self.dry_run = dry_run

    @classmethod
    def from_env(cls, dry_run: bool = False) -> PolymarketUsClient:
        key_id, secret = _key_paths()
        if not key_id or not secret.exists():
            raise SystemExit(
                "no Polymarket US API key: set POLYMARKET_US_KEY_ID (or write "
                f"{_KEY_DIR / 'key_id'}) and put the secret key at {secret} (or "
                "set POLYMARKET_US_SECRET_FILE)"
            )
        return cls(key_id, private_key(secret.read_text()), dry_run=dry_run)

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        if self.dry_run and method != "GET":
            raise PolymarketUsError(f"dry run: refusing to send {method} {path}")
        timestamp = str(int(time.time() * 1000))
        headers = {
            "X-PM-Access-Key": self.key_id,
            "X-PM-Timestamp": timestamp,
            "X-PM-Signature": sign(self.key, timestamp, method, path),
            "User-Agent": "cassandra-picks",
            "Accept": "application/json",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base + path, data=data, headers=headers, method=method
        )
        retries = list(_READ_RETRIES) if method == "GET" else []
        while True:
            time.sleep(_PACE)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    text = response.read()
                    return json.loads(text) if text.strip() else {}
            except urllib.error.HTTPError as error:
                if error.code == 429 and retries:
                    time.sleep(retries.pop(0))
                    continue
                detail = error.read().decode(errors="replace")[:500]
                raise PolymarketUsError(
                    f"{method} {path}: {error.code} {detail}"
                ) from None
            except (OSError, ValueError) as error:
                # A timeout, a dropped connection, a body that isn't JSON. A
                # read is tried again; a write isn't, since it may have landed.
                if retries:
                    time.sleep(retries.pop(0))
                    continue
                raise PolymarketUsError(f"{method} {path}: {error}") from None

    def balances(self) -> list[dict]:
        return self._request("GET", "/v1/account/balances").get("balances") or []

    def positions(self) -> dict[str, dict]:
        found: dict[str, dict] = {}
        cursor = None
        while True:
            query = {"limit": 100, **({"cursor": cursor} if cursor else {})}
            page = self._request(
                "GET", f"/v1/portfolio/positions?{urllib.parse.urlencode(query)}"
            )
            found.update(page.get("positions") or {})
            cursor = page.get("nextCursor")
            if page.get("eof") or not cursor or not page.get("positions"):
                return found

    def open_orders(self) -> list[dict]:
        return self._request("GET", "/v1/orders/open").get("orders") or []

    def activities(self, types: tuple[str, ...] = ()) -> list[dict]:
        """Every activity, newest first: trades, resolutions, rebates."""
        found: list[dict] = []
        cursor = None
        while True:
            query: dict[str, Any] = {"limit": 100}
            if cursor:
                query["cursor"] = cursor
            if types:
                query["types"] = list(types)
            page = self._request(
                "GET",
                f"/v1/portfolio/activities?{urllib.parse.urlencode(query, doseq=True)}",
            )
            found += page.get("activities") or []
            cursor = page.get("nextCursor")
            if page.get("eof") or not cursor or not page.get("activities"):
                return found

    def create(self, body: dict) -> dict:
        if self.dry_run:
            print(f"    dry run, not sent: POST /v1/orders {json.dumps(body)}")
            return {}
        return self._request("POST", "/v1/orders", body)

    def cancel(self, order_id: str, slug: str) -> None:
        if self.dry_run:
            print(f"    dry run, not sent: POST /v1/order/{order_id}/cancel")
            return
        self._request("POST", f"/v1/order/{order_id}/cancel", {"marketSlug": slug})


class PolymarketUsAccount:
    """A Polymarket US account, read into the venue's shapes."""

    def __init__(self, venue: PolymarketUs, client: PolymarketUsClient) -> None:
        self.venue = venue
        self.client = client
        #: Set once a NO order fills over its limit: the run sends no more.
        self.shorts_stopped: str | None = None

    @property
    def dry_run(self) -> bool:
        return self.client.dry_run

    def balance(self) -> Balance:
        return balance(self.client.balances())

    def holdings(self) -> list[Holding]:
        return holdings(self.client.positions())

    def resting(self) -> list[Resting]:
        return resting(self.client.open_orders())

    def buy(
        self,
        route: Route,
        contracts: int,
        limit: float,
        expires: int | None = None,
        rung: bool = False,
    ) -> Fill:
        if rung:
            raise PolymarketUsError("no rungs here: an order can't be marked as one")
        if route.action == "short" and self.shorts_stopped:
            raise PolymarketUsError(
                f"no more NO orders this run: {self.shorts_stopped}"
            )
        body = order_body(route.ticker, route.action, contracts, limit, expires)
        if self.dry_run:
            self.client.create(body)
            return Fill("dry-run", 0.0, 0.0, None, None)
        done = fill(self.client.create(body), route.action, contracts)
        try:
            check_fill(done, route.action, limit)
        except PolymarketUsError as error:
            if route.action == "short":
                self.shorts_stopped = str(error)
            raise
        return done

    def cancel(self, order: Resting) -> float:
        self.client.cancel(order.order_id, order.ticker)
        return 0.0 if self.dry_run else order.remaining

    def fund(self, side: Book, dollars: float) -> None:
        # One balance pays for every market.
        return None

"""Kalshi's authenticated API: the account's balance, what it holds, and orders.

`picks.py --place` uses this to buy its picks. Reading markets needs no key;
everything here does. Kalshi signs each request with an API key made on the
account's settings page (Profile → API keys), whose private key is a PEM
file Kalshi shows once. The key is read from

    KALSHI_KEY_ID      the key's id, or the file ~/.kalshi/key_id
    KALSHI_KEY_FILE    the PEM's path, default ~/.kalshi/private_key.pem

and `--demo` reads `KALSHI_DEMO_KEY_ID` / `KALSHI_DEMO_KEY_FILE` (or
~/.kalshi/demo_key_id and ~/.kalshi/demo_private_key.pem) against Kalshi's
demo exchange, whose money isn't real.

Orders go to the V2 endpoint, `POST /portfolio/events/orders`, which quotes
every order from the YES side: buying a team's YES is a `bid` at its price,
and buying NO on its opponent at q is an `ask` -- selling YES -- at 1 - q.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

LIVE_API = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_API = "https://demo-api.kalshi.co/trade-api/v2"
_KEY_DIR = Path.home() / ".kalshi"
# Kalshi's basic tier allows about 10 writes and 20 reads a second.
_PACE = 1 / 8


def public_get(path: str, params: dict | None = None) -> dict:
    """A GET of Kalshi's public market data, which needs no key: markets,
    events, order books."""
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(
        f"{LIVE_API}{path}{query}", headers={"User-Agent": "cassandra-picks"}
    )
    time.sleep(_PACE)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def sign(private_key: Any, timestamp_ms: str, method: str, path: str) -> str:
    """Kalshi's request signature: timestamp, method and path, no query string.

    RSA keys sign with PSS over SHA-256, Ed25519 keys sign the text itself.
    """
    message = f"{timestamp_ms}{method}{path.split('?')[0]}".encode()
    if isinstance(private_key, Ed25519PrivateKey):
        signature = private_key.sign(message)
    else:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
    return base64.b64encode(signature).decode()


def _dollars(value: float) -> str:
    return f"{value:.2f}"


def order_body(
    ticker: str,
    action: str,
    count: int,
    limit: float,
    expires_at: int | None = None,
    client_order_id: str | None = None,
) -> dict:
    """A V2 order buying `count` contracts of `action` ("yes"/"no") for at most `limit`.

    `limit` is the price of the contract bought, so a NO at 38c is an ask
    on YES at 62c. With no `expires_at` the order is immediate-or-cancel:
    whatever is on offer at or under the limit fills and the rest is
    dropped. With one it rests until then.
    """
    if action not in ("yes", "no"):
        raise ValueError(f"action is yes or no, not {action!r}")
    if count < 1 or not 0 < limit < 1:
        raise ValueError(f"bad order: {count} at {limit}")
    yes_price = limit if action == "yes" else 1 - limit
    body = {
        "ticker": ticker,
        "client_order_id": client_order_id or str(uuid.uuid4()),
        "side": "bid" if action == "yes" else "ask",
        "count": str(count),
        "price": _dollars(yes_price),
        "time_in_force": "immediate_or_cancel"
        if expires_at is None
        else "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "cancel_order_on_pause": True,
    }
    if expires_at is not None:
        body["expiration_time"] = expires_at
    return body


class Fill(NamedTuple):
    """What an order did when it was placed, priced from the side bought."""

    order_id: str
    filled: float
    resting: float
    price: float | None
    fee: float | None


def fill(response: Mapping, action: str) -> Fill:
    """A V2 create-order response read back: contracts filled, at what price.

    The response's price is YES's, so a NO bought fills at 1 minus it.
    """
    filled = float(response.get("fill_count") or 0)
    average = response.get("average_fill_price")
    price = None if average is None else float(average)
    if price is not None and action == "no":
        price = 1 - price
    fee = response.get("average_fee_paid")
    return Fill(
        order_id=response.get("order_id", ""),
        filled=filled,
        resting=float(response.get("remaining_count") or 0),
        price=price,
        fee=None if fee is None else float(fee) * filled,
    )


def event_of(ticker: str) -> str:
    """The event a market belongs to: `KXNCAAFGAME-26OCT17BRWNPRIN-PRIN` → its event."""
    return ticker.rsplit("-", 1)[0]


class KalshiError(RuntimeError):
    pass


@dataclass
class KalshiClient:
    key_id: str
    private_key: Any
    base: str = LIVE_API
    #: Read the account but never write to it: `buy` prints the order it
    #: would send, and any request but a GET is refused before it's made.
    dry_run: bool = False

    @classmethod
    def from_env(cls, demo: bool = False, dry_run: bool = False) -> "KalshiClient":
        prefix, stem = ("KALSHI_DEMO_", "demo_") if demo else ("KALSHI_", "")
        key_id = os.environ.get(f"{prefix}KEY_ID")
        id_file = _KEY_DIR / f"{stem}key_id"
        if not key_id and id_file.exists():
            key_id = id_file.read_text().strip()
        key_file = Path(
            os.environ.get(f"{prefix}KEY_FILE") or _KEY_DIR / f"{stem}private_key.pem"
        ).expanduser()
        if not key_id or not key_file.exists():
            raise SystemExit(
                f"no Kalshi API key: set {prefix}KEY_ID (or write {id_file}) and "
                f"put the private key at {key_file} (or set {prefix}KEY_FILE)"
            )
        private_key = serialization.load_pem_private_key(
            key_file.read_bytes(), password=None
        )
        return cls(key_id, private_key, DEMO_API if demo else LIVE_API, dry_run)

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        if self.dry_run and method != "GET":
            raise KalshiError(f"dry run: refusing to send {method} {path}")
        url = urllib.parse.urlsplit(self.base + path)
        timestamp = str(int(time.time() * 1000))
        headers = {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": sign(
                self.private_key, timestamp, method, url.path
            ),
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
        time.sleep(_PACE)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise KalshiError(f"{method} {path}: {error.code} {detail}") from None

    def _pages(self, path: str, key: str, params: dict) -> list[dict]:
        found: list[dict] = []
        cursor = None
        while True:
            query = dict(params, limit=200, **({"cursor": cursor} if cursor else {}))
            page = self._request("GET", f"{path}?{urllib.parse.urlencode(query)}")
            found += page.get(key) or []
            cursor = page.get("cursor")
            if not cursor or not page.get(key):
                return found

    def account(self) -> tuple[float, float]:
        """Cash on hand, and what the open positions are worth, in dollars."""
        response = self._request("GET", "/portfolio/balance")
        if response.get("balance_dollars") is not None:
            cash = float(response["balance_dollars"])
        else:
            cash = float(response["balance"]) / 100
        return cash, float(response.get("portfolio_value") or 0) / 100

    def balance(self) -> float:
        """Cash on hand, in dollars."""
        return self.account()[0]

    def market_positions(self) -> list[dict]:
        """Every unsettled market position as Kalshi sends it: `position_fp`
        (YES positive, NO negative), `market_exposure_dollars` (what it cost),
        `fees_paid_dollars`, and so on."""
        return self._pages(
            "/portfolio/positions",
            "market_positions",
            {"count_filter": "position", "settlement_status": "unsettled"},
        )

    def positions(self) -> dict[str, float]:
        """Each market's open position: YES contracts positive, NO negative."""
        found: dict[str, float] = {}
        for position in self.market_positions():
            count = float(position.get("position_fp") or position.get("position") or 0)
            if count:
                found[position["ticker"]] = count
        return found

    def market(self, ticker: str) -> dict:
        """One market: its title, quotes and status."""
        return self._request("GET", f"/markets/{ticker}")["market"]

    def event(self, event_ticker: str) -> dict:
        """One event: its title, e.g. "Houston at Utah"."""
        return self._request("GET", f"/events/{event_ticker}")["event"]

    def fills(self) -> list[dict]:
        """Every fill the account has had: `ticker`, `order_id`, `count_fp`,
        `created_time`, `fee_cost`. Its `side`/`action` pair isn't the
        contract held -- a NO bought through the V2 ask shows as a `sell` of
        `no` -- so what's held is read from positions and settlements."""
        return self._pages("/portfolio/fills", "fills", {})

    def settlements(self) -> list[dict]:
        """Every market the account held to its settlement: each side's count
        and cost (`yes_count_fp`, `no_total_cost_dollars`), `revenue` in
        cents, and the fees paid trading it (`fee_cost`)."""
        return self._pages("/portfolio/settlements", "settlements", {})

    def resting_events(self) -> set[str]:
        """Every event with a resting order in it."""
        return {
            event_of(order["ticker"])
            for order in self._pages(
                "/portfolio/orders", "orders", {"status": "resting"}
            )
        }

    def held_events(self) -> dict[str, str]:
        """Every event the account is already in, and how: a position or a resting order."""
        held = {event_of(ticker): "position" for ticker in self.positions()}
        for event in self.resting_events():
            held.setdefault(event, "resting order")
        return held

    def buy(
        self,
        ticker: str,
        action: str,
        count: int,
        limit: float,
        expires_at: int | None = None,
    ) -> Fill:
        body = order_body(ticker, action, count, limit, expires_at)
        if self.dry_run:
            print(
                f"    dry run, not sent: POST /portfolio/events/orders {json.dumps(body)}"
            )
            return Fill("dry-run", 0.0, 0.0, None, None)
        return fill(self._request("POST", "/portfolio/events/orders", body), action)

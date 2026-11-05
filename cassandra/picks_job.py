"""What `jobs.py picks` keeps between runs: the autobet rules, what it has
emailed, a lock -- and the emails themselves.

The job runs `picks.py` on a schedule and after every ncaafb finals sweep,
and places the picks itself (`picks.py --place --yes`) by each league's
rules (`LeagueRules`) in `autobet.json`, the cash on hand the only budget,
each order resting `REST_HOURS` before it's cancelled. A league with a
`max_exposure` spends only what its share of the bankroll has left. The
rules' values live only in that file. `make pause` stops that and `make resume`
starts it again, the whole job or `ARGS="--league L"`'s; `make autobet`
changes a league's sizing, or adds one. Each pick's
rungs (`picks.py`'s "Rungs") rest until kickoff, and every run with the key
cancels the ones the model no longer backs.

Two emails, and only two (Nathan, 2026-11-01):

- **A hold**, the moment a run makes one: a pick `picks.py --hold-qb` kept
  back because its injury news names a quarterback, which the model doesn't
  read. It needs a person's OK; the email has the line that places it.
  Emailed once, not every hour it stays held (`unsent(held=True)`).
- **The morning summary** (`jobs.py picks_summary`, 8am): every bet placed
  since the last one, what's still resting, what settled, the holds still
  waiting, and the account. Placements and cancelled rungs go there rather
  than in an email each.

Objects in s3, each small JSON or TOML:

- `cassandra/betting/autobet.json` (batch bucket): `paused`, and under
  `leagues` each league's rules (`Autobet`, `LeagueRules`). No file, or no
  rules for a league, means nothing of it is placed. A file from before
  `leagues`, its `max_bet` and `kelly_fraction` at the top, reads as
  football's.
- `cassandra/betting/sent.json` (batch bucket): every pick emailed, by game
  id -- holds marked as such -- and each email's numbered list, so "place 1
  and 3 from the 22:40 email" names games rather than positions in a list
  that has moved since.
- `cassandra/betting/summary.json` (batch bucket): when the last morning
  summary went out, which is where the next one starts.
- `cassandra/picks.lock` (temp bucket): held for a run's length, so an
  hourly run and a finals-triggered one never price -- or place -- at once.
  Written with `If-None-Match: *`, so only one of two racing runs gets it.
  One older than `LOCK_STALE` is a run that died holding it, and is taken.

The overrides file (`cassandra.overrides`) lives at
`cassandra/betting/overrides.toml` for the job; `make push-overrides`
copies a laptop's up.

Every run copies its picks file, and the orders file when it placed, to
`cassandra/betting/picks/` and `orders/` (`keep`). They say what the model
gave each bet: `make ledger` reads them, and so does the morning summary.
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import botocore.session
import pandas as pd
from botocore.exceptions import ClientError

from cassandra.constants import CASSANDRA_HOME

SENT_KEY = "cassandra/betting/sent.json"
AUTOBET_KEY = "cassandra/betting/autobet.json"
SUMMARY_KEY = "cassandra/betting/summary.json"
OVERRIDES_KEY = "cassandra/betting/overrides.toml"
#: Where a run writes every game it priced (`picks.py`) and the orders it
#: sent (`placing.place`), by the s3 prefix `keep` copies them to.
ORDERS_PREFIX = "cassandra/betting/orders/"
RUN_FILES = {
    "cassandra/betting/picks/": CASSANDRA_HOME / "betting" / "picks",
    ORDERS_PREFIX: CASSANDRA_HOME / "betting" / "orders",
}
LOCK_KEY = "cassandra/picks.lock"
#: A run takes about five minutes; one holding the lock this long has died.
LOCK_STALE = timedelta(minutes=30)
#: How long a pick's order rests before it's cancelled. The backtest's edge
#: was mostly gone twelve hours after the entry; a later run re-prices the
#: game and places again if it's still a pick.
REST_HOURS = 12.0
#: How far back the first morning summary looks, with no last one to start at.
SUMMARY_DAY = timedelta(hours=24)
#: Emails kept in `sent.json`: a week of hourly runs and then some.
EMAILS_KEPT = 200


def client(service: str) -> Any:
    """A plain synchronous botocore client: what boto3 would give, without it."""
    return botocore.session.get_session().create_client(service)


def _code(error: ClientError) -> str:
    return error.response.get("Error", {}).get("Code", "")


def read_json(s3: Any, bucket: str, key: str) -> tuple[dict | None, str | None]:
    """An object's JSON and its ETag, or (None, None) when there isn't one."""
    try:
        found = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if _code(error) in ("NoSuchKey", "404"):
            return None, None
        raise
    return json.loads(found["Body"].read()), found["ETag"]


def write_json(s3: Any, bucket: str, key: str, value: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, indent=1, default=str).encode(),
        ContentType="application/json",
    )


# ------------------------------------------------------------------------------
# The lock
# ------------------------------------------------------------------------------


def acquire(s3: Any, bucket: str, now: datetime, owner: str) -> bool:
    """Take the run lock, or say another run has it."""
    for _ in range(2):
        try:
            s3.put_object(
                Bucket=bucket,
                Key=LOCK_KEY,
                Body=json.dumps({"owner": owner, "at": now.isoformat()}).encode(),
                IfNoneMatch="*",
            )
            return True
        except ClientError as error:
            if _code(error) not in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise
        try:
            held = s3.head_object(Bucket=bucket, Key=LOCK_KEY)
        except ClientError as error:
            if _code(error) in ("NoSuchKey", "404"):
                continue
            raise
        if now - held["LastModified"] < LOCK_STALE:
            return False
        s3.delete_object(Bucket=bucket, Key=LOCK_KEY)
    return False


def release(s3: Any, bucket: str) -> None:
    s3.delete_object(Bucket=bucket, Key=LOCK_KEY)


def keep(s3: Any, bucket: str, since: datetime, orders: bool = True) -> list[str]:
    """Copy the picks and orders files written since `since` -- this run's --
    up to s3, for `make ledger`. The keys they went to. `orders=False` keeps
    only the picks: a dry run's orders were never sent."""
    kept = []
    for prefix, directory in RUN_FILES.items():
        if not orders and prefix.rstrip("/").endswith("orders"):
            continue
        for path in sorted(directory.glob("*.csv")):
            if datetime.fromtimestamp(path.stat().st_mtime, UTC) < since:
                continue
            s3.put_object(Bucket=bucket, Key=prefix + path.name, Body=path.read_bytes())
            kept.append(prefix + path.name)
    return kept


# ------------------------------------------------------------------------------
# Autobet
# ------------------------------------------------------------------------------


#: The league an old, flat `autobet.json` -- one set of rules, from before
#: there were two leagues -- is read as.
FIRST_LEAGUE = "ncaafb"


@dataclass
class LeagueRules:
    """How the job sizes one league's bets. See `picks.py`'s "Sizing".

    The stake has no default: a league is placed only on a fraction and a
    cap someone wrote into `autobet.json`. The limits default to none.
    """

    kelly_fraction: float
    max_bet: float
    #: The most of the bankroll the league's open positions and resting
    #: orders may hold (`picks.py --max-exposure`); None for no cap.
    max_exposure: float | None = None
    #: A pick costing under `longshot_price` is staked at `longshot_scale`
    #: of its Kelly (`cassandra.picks.Longshots`).
    longshot_price: float = 0.0
    longshot_scale: float = 1.0
    #: A pick costing under this isn't placed (`picks.py --min-price`).
    min_price: float = 0.0
    #: Priced, not placed: the league's own `make pause`.
    paused: bool = False

    @classmethod
    def from_json(cls, value: dict) -> "LeagueRules":
        missing = [k for k in ("kelly_fraction", "max_bet") if value.get(k) is None]
        if missing:
            raise ValueError(f"league rules without {' or '.join(missing)}: {value}")
        exposure = value.get("max_exposure")
        return cls(
            kelly_fraction=float(value["kelly_fraction"]),
            max_bet=float(value["max_bet"]),
            max_exposure=None if exposure is None else float(exposure),
            longshot_price=float(value.get("longshot_price", 0.0)),
            longshot_scale=float(value.get("longshot_scale", 1.0)),
            min_price=float(value.get("min_price", 0.0)),
            paused=bool(value.get("paused", False)),
        )

    def to_json(self) -> dict:
        return {
            "kelly_fraction": self.kelly_fraction,
            "max_bet": self.max_bet,
            "max_exposure": self.max_exposure,
            "longshot_price": self.longshot_price,
            "longshot_scale": self.longshot_scale,
            "min_price": self.min_price,
            "paused": self.paused,
        }

    def argv(self) -> list[str]:
        """`picks.py`'s sizing flags for these rules."""
        found = [
            "--kelly-fraction",
            f"{self.kelly_fraction:g}",
            "--max-bet",
            f"{self.max_bet:g}",
        ]
        if self.max_exposure is not None:
            found += ["--max-exposure", f"{self.max_exposure:g}"]
        if self.longshot_price > 0 and self.longshot_scale != 1.0:
            found += [
                "--longshot-price",
                f"{self.longshot_price:g}",
                "--longshot-scale",
                f"{self.longshot_scale:g}",
            ]
        if self.min_price > 0:
            found += ["--min-price", f"{self.min_price:g}"]
        return found

    def describe(self) -> str:
        found = f"{self.kelly_fraction:g} Kelly, at most ${self.max_bet:g} a bet"
        if self.max_exposure is not None:
            found += f", at most {self.max_exposure:.0%} of the bankroll at risk"
        if self.longshot_price > 0 and self.longshot_scale != 1.0:
            found += f", {self.longshot_scale:g}x stakes under {self.longshot_price * 100:g}c"
        if self.min_price > 0:
            found += f", nothing under {self.min_price * 100:g}c"
        return found


@dataclass
class Autobet:
    """The rules the job places by. No file in s3 means none: nothing is
    placed.

    A league with no rules isn't placed: each one is opted into with `make
    autobet ARGS="--league ..."`. Every league sizes on the whole bankroll
    and spends the same cash, so a league that shouldn't take what another
    needs has a `max_exposure`.
    """

    paused: bool = False
    leagues: dict[str, LeagueRules] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: dict) -> "Autobet":
        if "leagues" not in value:
            if value.get("kelly_fraction") is None and value.get("max_bet") is None:
                return cls(paused=bool(value.get("paused", False)))
            # The flat file of one league's rules, and the pause, as written
            # before there were two.
            return cls(
                paused=bool(value.get("paused", False)),
                leagues={FIRST_LEAGUE: LeagueRules.from_json(value)},
            )
        return cls(
            paused=bool(value.get("paused", False)),
            leagues={
                league: LeagueRules.from_json(rules)
                for league, rules in value["leagues"].items()
            },
        )

    def to_json(self) -> dict:
        return {
            "paused": self.paused,
            "leagues": {
                league: rules.to_json() for league, rules in self.leagues.items()
            },
        }

    def placing(self, league: str) -> bool:
        """Whether the job places `league`'s picks: it has rules, and neither
        it nor the whole job is paused."""
        rules = self.leagues.get(league)
        return not self.paused and rules is not None and not rules.paused

    def describe(self) -> str:
        if self.paused:
            return (
                "autobet paused: runs price and cancel stale rungs, and place nothing"
            )
        leagues = "; ".join(
            f"{league} {'paused' if rules.paused else rules.describe()}"
            for league, rules in sorted(self.leagues.items())
        )
        return (
            f"autobet on: {leagues or 'no leagues'}; the cash on hand the budget, "
            f"orders resting {REST_HOURS:g}h"
        )


def autobet(s3: Any, bucket: str) -> Autobet:
    found, _ = read_json(s3, bucket, AUTOBET_KEY)
    return Autobet() if found is None else Autobet.from_json(found)


# ------------------------------------------------------------------------------
# What's been emailed
# ------------------------------------------------------------------------------


def unsent(picks: pd.DataFrame, sent: dict, held: bool = False) -> pd.DataFrame:
    """The picks no email has carried: a new game, or the same game's other side.

    With `held`, the picks a run held on a QB check (`picks.py --hold-qb`), for
    which an email counts only if it carried the pick *as held*: a game first
    emailed as an ordinary pick, or placed, and held later is news. Iowa
    State, Oregon and Mississippi State were held silently on 2026-11-01
    because earlier emails had carried them as picks.
    """
    games = sent.get("games", {})
    fresh = pd.Series(
        [
            game_id not in games
            or games[game_id].get("bet") != bet
            or (held and not games[game_id].get("held"))
            for game_id, bet in zip(picks["game_id"], picks["bet"])
        ],
        index=picks.index,
        dtype=bool,
    )
    # A mask, not a list: with no picks the list is `[]`, and `picks[[]]`
    # selects no columns rather than no rows, which lost `game_id` on a
    # pickless run that only cancelled rungs (2026-10-25).
    return picks[fresh]


def record(sent: dict, picks: pd.DataFrame, at: datetime, subject: str) -> dict:
    """`sent` with this email's picks in it, numbered as the email numbers them."""
    games = dict(sent.get("games", {}))
    listed = []
    for _, row in ordered(picks).iterrows():
        games[row["game_id"]] = {
            "bet": row["bet"],
            "game": row["game"],
            "edge": round(float(row["edge"]), 4),
            "at": at.isoformat(),
            # A held pick carries its QB sentence; see `unsent(held=True)`.
            "held": isinstance(row.get("qb_news"), str),
        }
        listed.append({"game_id": row["game_id"], "bet": row["bet"]})
    emails = [
        *sent.get("emails", []),
        {"at": at.isoformat(), "subject": subject, "picks": listed},
    ]
    return {"games": games, "emails": emails[-EMAILS_KEPT:]}


def ordered(picks: pd.DataFrame) -> pd.DataFrame:
    """The order `picks.py` numbers its orders in."""
    return picks.sort_values(["kickoff", "edge"], ascending=[True, False])


def only_args(game_ids: list[str]) -> str:
    return " ".join(f"--only {game_id}" for game_id in game_ids)


def hold_email(held: pd.DataFrame, at: datetime, slippage: float) -> tuple[str, str]:
    """The email asking for an OK: a subject a phone shows whole, and each
    held pick with its quarterback sentence and the line that places it,
    numbered as `picks.py` numbers them (so "place 2" works in reply)."""
    offered = f"offered_within_{round(slippage * 100)}c"
    names = ", ".join(ordered(held)["bet"])
    count = f"{len(held)} Kalshi pick{'s' if len(held) != 1 else ''}"
    need = "needs" if len(held) == 1 else "need"
    subject = f"{count} {need} your OK, QB check ({at:%a %H:%M} CT): {names}"[:99]
    lines = [
        f"Held: {count} that may be priced on a quarterback who won't play: the "
        "side bet on had its starter miss, leave or come off the bench in its "
        "last game, or the injury news names a quarterback. The model reads "
        "neither. Check the starter; to place one, run its line or say "
        'which in a Claude session ("place 1 from the hold email") -- it '
        "re-prices first. Unplaced, it's left alone; the morning summary lists "
        "it until then.",
        "",
    ]
    for number, (_, row) in enumerate(ordered(held).iterrows(), start=1):
        lines.append(
            f"{number}. {row['bet']} -- {row['game']}, {row['kickoff']:%a %H:%M} CT"
        )
        lines.append(f"   QB: {row['qb_news'][:240]}")
        lines.append(
            f"   {row['buy']}: {row['contracts']:.0f} @ <= {row['limit']:.2f}, "
            f"at most ${row['outlay']:.2f}"
        )
        since = row.get("hours_since_entry")
        lines.append(
            f"   model {row['model']:.3f} vs market {row['market']:.3f} "
            f"(edge {row['edge']:+.3f}); {row.get(offered, 0):.0f} offered within "
            f"{round(slippage * 100)}c"
            + ("" if since is None or pd.isna(since) else f"; {since:.0f}h since entry")
        )
        if row.get("no_plays"):
            lines.append(f"   last game's play-by-play not in yet: {row['no_plays']}")
        lines.append(f'   place: make bet ARGS="--only {row["game_id"]}"')
        lines.append("")
    if len(held) > 1:
        lines.append(
            f'All of them: make bet ARGS="{only_args(list(ordered(held)["game_id"]))}"'
        )
    return subject, "\n".join(lines)


# ------------------------------------------------------------------------------
# The morning summary
# ------------------------------------------------------------------------------


def held_contract(fill: Mapping) -> str:
    """What a fill bought on its ticker, "yes" or "no". Kalshi's V2 fills
    write a NO bought through the book as `action=sell, side=no` (see
    `cassandra.ledger`), and selling a YES nets to holding NO."""
    if fill.get("side") == "yes" and fill.get("action") == "buy":
        return "yes"
    return "no"


def placed_bets(fills: Sequence[Mapping], orders: pd.DataFrame) -> pd.DataFrame:
    """The games `fills` bought, a row each: the team, contracts, average
    price, cost with fees, and -- from the job's orders logs, by order id --
    the model and edge it was placed on and the contracts that came from
    rungs. A fill no log has (a bet made by hand) is named by its ticker."""
    from cassandra.kalshi_trade import event_of

    logged = (
        orders.drop_duplicates("order_id").set_index("order_id")
        if "order_id" in orders
        else pd.DataFrame()
    )
    rows = []
    for fill in fills:
        held = held_contract(fill)
        count = float(fill.get("count_fp") or fill.get("count") or 0)
        price = float(fill.get(f"{held}_price_dollars") or 0)
        code = fill["ticker"].rsplit("-", 1)[-1]
        event = event_of(fill["ticker"])
        if fill["order_id"] in logged.index:
            log = logged.loc[fill["order_id"]]
            bet, game = log["bet"], log["game"]
            model, edge = float(log["model"]), float(log["edge"])
            # The orders log is a CSV: its flag comes back "True"/"False".
            rung = count if str(log.get("rung")).strip().lower() == "true" else 0.0
        else:
            bet, game = f"{held.upper()} {code}", event
            model = edge = math.nan
            rung = 0.0
        rows.append(
            {
                "event": event,
                "bet": bet,
                "game": game,
                "contracts": count,
                "paid": price * count,
                "cost": price * count + float(fill.get("fee_cost") or 0),
                "model": model,
                "edge": edge,
                "rung": rung,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "event",
                "bet",
                "game",
                "contracts",
                "price",
                "cost",
                "model",
                "edge",
                "rung",
            ]
        )
    found = (
        pd.DataFrame(rows)
        .groupby(["event", "bet"], as_index=False)
        .agg(
            game=("game", "first"),
            contracts=("contracts", "sum"),
            paid=("paid", "sum"),
            cost=("cost", "sum"),
            model=("model", "first"),
            edge=("edge", "first"),
            rung=("rung", "sum"),
        )
    )
    found["price"] = found["paid"] / found["contracts"]
    return found.drop(columns="paid").sort_values("cost", ascending=False)


def settled_games(settlements: Sequence[Mapping]) -> pd.DataFrame:
    """Each game `settlements` closed, its P&L with fees, and whether it won.
    A YES-NO pair (a sale before the end, booked as a buy of the other side)
    paid its dollar when it was netted, so it's counted back here as
    `cassandra.ledger` counts it."""
    from cassandra.kalshi_trade import event_of

    pnl: dict[str, float] = {}
    for s in settlements:
        yes, no = float(s.get("yes_count_fp") or 0), float(s.get("no_count_fp") or 0)
        cost = float(s.get("yes_total_cost_dollars") or 0) + float(
            s.get("no_total_cost_dollars") or 0
        )
        paid = float(s.get("revenue") or 0) / 100 + min(yes, no)
        event = event_of(s["ticker"])
        pnl[event] = pnl.get(event, 0.0) + paid - cost - float(s.get("fee_cost") or 0)
    found = pd.DataFrame({"event": list(pnl), "pnl": list(pnl.values())})
    found["won"] = found["pnl"] > 0
    return found.sort_values("pnl", ascending=False)


def summary_email(
    bets: pd.DataFrame,
    resting: Sequence[Mapping],
    settled: pd.DataFrame,
    holds: Sequence[Mapping],
    account: tuple[float, float],
    titles: Mapping[str, str],
    rules: Autobet,
    since: datetime,
    at: datetime,
) -> tuple[str, str]:
    """The morning email: what was placed since `since`, what still rests,
    what settled, the holds still waiting, and the account."""
    from cassandra.kalshi_trade import bought, is_rung, order_price

    won, lost = int(settled["won"].sum()), int((~settled["won"]).sum())
    subject = (
        f"Kalshi {at:%a %b %-d}: {len(bets)} bet{'s' if len(bets) != 1 else ''} "
        f"placed (${bets['cost'].sum():,.2f})"
        + (
            f", settled {won}-{lost} ({_money(settled['pnl'].sum())})"
            if len(settled)
            else ""
        )
    )
    lines = [f"Since {since:%a %b %-d %H:%M} CT. {rules.describe()}.", ""]
    lines.append(
        f"Placed: {len(bets)} game{'s' if len(bets) != 1 else ''}, "
        f"${bets['cost'].sum():,.2f} with fees" + (":" if len(bets) else ".")
    )
    for _, bet in bets.iterrows():
        model = (
            ""
            if pd.isna(bet["model"])
            else f"; model {bet['model']:.2f} (edge {bet['edge']:+.3f})"
        )
        rungs = f"; {bet['rung']:g} from rungs" if bet["rung"] else ""
        lines.append(
            f"  {bet['bet']} -- {bet['game']}: {bet['contracts']:g} @ "
            f"{bet['price']:.2f}, ${bet['cost']:.2f}{model}{rungs}"
        )
    lines.append("")
    if resting:
        worst = sum(
            float(o.get("remaining_count_fp") or 0)
            * (order_price(o) if bought(o) == "yes" else 1 - order_price(o))
            for o in resting
        )
        rungs = sum(1 for o in resting if is_rung(o))
        lines += [
            f"Resting: {len(resting)} order{'s' if len(resting) != 1 else ''}, up to "
            f"${worst:,.2f} if they all fill ({rungs} of them rungs, until kickoff).",
            "",
        ]
    if len(settled):
        lines.append(f"Settled: {won}-{lost}, {_money(settled['pnl'].sum())}")
        for _, game in settled.iterrows():
            title = titles.get(game["event"], game["event"])
            lines.append(
                f"  {'won ' if game['won'] else 'lost'} {_money(game['pnl']):>8}  {title}"
            )
        lines.append("")
    if holds:
        lines.append(f"Waiting on you: {len(holds)} held on a QB check, not placed")
        lines += [
            f'  {h["bet"]} -- {h["game"]}: make bet ARGS="--only {h["game_id"]}"'
            for h in holds
        ]
        lines.append("")
    cash, positions = account
    lines.append(f"Account: ${cash:,.2f} cash, ${positions:,.2f} in open positions.")
    return subject, "\n".join(lines)


def _money(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def holds_since(sent: dict, since: datetime) -> list[dict]:
    """The picks emailed as held since `since`, newest first, as `sent.json`
    keeps them: bet, game and game id."""
    found = [
        {"game_id": game_id, **game}
        for game_id, game in sent.get("games", {}).items()
        if game.get("held") and datetime.fromisoformat(game["at"]) >= since
    ]
    return sorted(found, key=lambda h: h["at"], reverse=True)


def now_utc() -> datetime:
    return datetime.now(UTC)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    """`main`'s commands: `pause` and `resume`, the whole job or `--league`'s;
    `autobet --league L` to set (or with `--remove`, drop) one league's rules,
    a league not listed yet needing its fraction and max bet; and
    `push-overrides` and `status`."""
    parser = argparse.ArgumentParser(prog="python -m cassandra.picks_job")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help in (
        ("pause", "stop placing; runs still price"),
        ("resume", "place the picks again"),
    ):
        command = commands.add_parser(name, help=help)
        command.add_argument("--league", help="only this league (default: all)")
    rules = commands.add_parser("autobet", help="change how the job sizes its bets")
    rules.add_argument("--league", default=FIRST_LEAGUE)
    rules.add_argument("--max-bet", type=float)
    rules.add_argument("--kelly-fraction", type=float)
    rules.add_argument("--max-exposure", type=float, metavar="SHARE")
    rules.add_argument(
        "--no-max-exposure", action="store_true", help="lift the league's cap"
    )
    rules.add_argument("--longshot-price", type=float, metavar="COST")
    rules.add_argument("--longshot-scale", type=float)
    rules.add_argument("--min-price", type=float, metavar="COST")
    rules.add_argument(
        "--remove", action="store_true", help="drop the league: it isn't placed"
    )
    commands.add_parser("push-overrides", help="copy the overrides file up for the job")
    commands.add_parser("status", help="the autobet rules, and the last emails' picks")
    return parser.parse_args(argv)


def changed(current: Autobet, args: argparse.Namespace) -> Autobet:
    """The rules after a `pause`, `resume` or `autobet` command."""
    leagues = {name: replace(rules) for name, rules in current.leagues.items()}
    found = Autobet(current.paused, leagues)
    if args.command in ("pause", "resume"):
        paused = args.command == "pause"
        if args.league is None:
            found.paused = paused
        elif args.league in leagues:
            leagues[args.league].paused = paused
        else:
            raise ValueError(
                f"no rules for {args.league}: add them with `make autobet "
                f'ARGS="--league {args.league} ..."`'
            )
        return found
    if args.remove:
        leagues.pop(args.league, None)
        return found
    if args.league not in leagues:
        if args.kelly_fraction is None or args.max_bet is None:
            raise ValueError(
                f"{args.league} has no rules yet: give it --kelly-fraction and --max-bet"
            )
        leagues[args.league] = LeagueRules(args.kelly_fraction, args.max_bet)
    rules = leagues[args.league]
    for name in (
        "kelly_fraction",
        "max_bet",
        "max_exposure",
        "longshot_price",
        "longshot_scale",
        "min_price",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(rules, name, value)
    if args.no_max_exposure:
        rules.max_exposure = None
    return found


def main(argv: list[str] | None = None) -> None:
    """`make pause` / `make resume` / `make autobet` / `make push-overrides` /
    `make picks-status`."""
    from endgame_aws import Config

    from cassandra.placing import CENTRAL

    args = parse(argv)
    s3 = client("s3")
    bucket = Config.init_from_file().bucket
    current = autobet(s3, bucket)
    if args.command in ("pause", "resume", "autobet"):
        try:
            current = changed(current, args)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        write_json(s3, bucket, AUTOBET_KEY, current.to_json())
        print(current.describe())
    elif args.command == "push-overrides":
        local = CASSANDRA_HOME / "betting" / "overrides.toml"
        s3.put_object(Bucket=bucket, Key=OVERRIDES_KEY, Body=local.read_bytes())
        print(f"{local} -> s3://{bucket}/{OVERRIDES_KEY}")
    else:
        print(current.describe())
        last, _ = read_json(s3, bucket, SUMMARY_KEY)
        if last:
            at = datetime.fromisoformat(last["at"]).astimezone(CENTRAL)
            print(f"last morning summary: {at:%a %H:%M} CT")
        sent, _ = read_json(s3, bucket, SENT_KEY)
        for email in (sent or {}).get("emails", [])[-5:]:
            at = datetime.fromisoformat(email["at"]).astimezone(CENTRAL)
            print(f"\n{at:%a %H:%M} CT  {email['subject']}")
            for number, pick in enumerate(email["picks"], start=1):
                print(f"  {number}. {pick['bet']}  --only {pick['game_id']}")


if __name__ == "__main__":
    main()

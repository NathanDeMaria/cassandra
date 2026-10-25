"""The Kalshi ledger: every game bet, what it paid, and why it went that way.

    python ledger.py             # or `make ledger`
    python ledger.py --offline   # no s3: skip the cloud job's orders and the backup
    python ledger.py --digest    # also print each settled game still without a theme

Each run rebuilds the whole history from the account (read-only, like
`make positions`), Kalshi's hourly prices for each game, and ESPN's box
score and AP recap for each settled one. It writes two files to
~/.cassandra/betting/ledger/: `ledger.csv`, one row per game bet, and
`report.md`. The report has the overall numbers, a table per result type
and tag, every week's games with their themes, and the open bets. A
final game's prices and summary are kept, so a rerun only fetches what's
new or still open.

A theme is written by hand -- or by a Claude session -- into `themes.toml`
there, keyed by the game's Kalshi event (the `=== KX...` line of
`--digest`):

    [KXNCAAFGAME-26OCT16MEMTULN]
    type = "turnovers"
    theme = "Turnover avalanche"
    why = "Tulane forced six turnovers, one a pick-6 at the half; Memphis outgained them 488-325."

`type` is one of `cassandra.ledger.TYPES`. Until a game has a theme it
counts under the type its box score suggests, marked `?`.

The cloud job prices every run in its container. It copies each run's
picks file to s3 (and its orders file, when it placed), and this pulls
them down, so its bets have their model price too. A bet with neither file still
counts: it just has no model EV.
"""

import argparse
import json
import math
import time
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from cassandra import ledger
from cassandra.kalshi_trade import KalshiClient, public_get
from cassandra.picks_job import RUN_FILES, client
from cassandra.placing import espn_summary

#: Each game's Kalshi event and hourly prices, once final.
KALSHI_DIR = ledger.LEDGER_DIR / "kalshi"
S3_LEDGER = "cassandra/betting/ledger/"
#: A market in any of these is done moving, so its prices can be kept.
FINAL = {"finalized", "settled", "determined"}


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--offline", action="store_true", help="skip s3; Kalshi and ESPN only"
    )
    parser.add_argument(
        "--digest",
        action="store_true",
        help="print what each settled game without a theme is written from",
    )
    return parser.parse_args()


def _s3():
    """An s3 client and the batch bucket, or None without an AWS login."""
    try:
        from endgame_aws import Config

        s3 = client("s3")
        bucket = Config.init_from_file().bucket
        s3.head_bucket(Bucket=bucket)
        return s3, bucket
    except Exception as error:  # noqa: BLE001 -- any failure means "no s3 today"
        print(f"no s3 ({type(error).__name__}): the cloud job's bets go unpriced")
        return None


def _download(s3, bucket: str, key: str, path: Path) -> None:
    path.write_bytes(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def pull_down(s3, bucket: str) -> None:
    """The cloud job's picks and orders files, and the themes if this machine has none."""
    fetched = 0
    for prefix, directory in RUN_FILES.items():
        directory.mkdir(parents=True, exist_ok=True)
        pages = s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        )
        for page in pages:
            for item in page.get("Contents", []):
                path = directory / Path(item["Key"]).name
                if not path.exists():
                    _download(s3, bucket, item["Key"], path)
                    fetched += 1
    if fetched:
        print(f"{fetched} picks/orders files from the cloud job")
    if not ledger.THEMES.exists():
        try:
            _download(s3, bucket, S3_LEDGER + "themes.toml", ledger.THEMES)
            print(f"themes restored from s3://{bucket}/{S3_LEDGER}themes.toml")
        except s3.exceptions.NoSuchKey:
            pass


def push_up(s3, bucket: str) -> None:
    """Back the ledger up: the themes are the one part that can't be rebuilt."""
    for path in (
        ledger.THEMES,
        ledger.LEDGER_DIR / "ledger.csv",
        ledger.LEDGER_DIR / "report.md",
        ledger.LEDGER_DIR / "wins-losses.md",
    ):
        if path.exists():
            s3.put_object(
                Bucket=bucket, Key=S3_LEDGER + path.name, Body=path.read_bytes()
            )
    print(f"backed up to s3://{bucket}/{S3_LEDGER}")


def _cached(path: Path, fetch, keep) -> dict:
    """A JSON fetch, kept on disk once `keep(it)` says it won't change."""
    if path.exists():
        return json.loads(path.read_text())
    found = fetch()
    if keep(found):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(found))
    return found


def _public_get(path: str, params: dict | None = None) -> dict:
    """`public_get`, waiting out Kalshi's rate limit: a ledger run reads a few
    hundred candles, and runs close together trip it."""
    for wait in (2, 4, 8, 16, 32):
        try:
            return public_get(path, params)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(wait)
    return public_get(path, params)


def event_json(event: str) -> dict:
    return _cached(
        KALSHI_DIR / f"{event}.json",
        lambda: _public_get(f"/events/{event}"),
        lambda e: all(m.get("status") in FINAL for m in e.get("markets", [])),
    )


def candles(
    event: str, tickers: list[str], start: datetime, end: datetime, final: bool
) -> dict:
    """Each team's hourly candles from `start` to `end`, by team code."""
    series = event.split("-", 1)[0]

    def fetch() -> dict:
        return {
            ledger.code_of(ticker): _public_get(
                f"/series/{series}/markets/{ticker}/candlesticks",
                {
                    "start_ts": int(start.timestamp()),
                    "end_ts": int(end.timestamp()),
                    "period_interval": 60,
                },
            ).get("candlesticks", [])
            for ticker in tickers
        }

    return _cached(KALSHI_DIR / f"{event}_candles.json", fetch, lambda _: final)


def _timestamp(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    found = pd.Timestamp(value)
    return found.tz_localize(UTC) if found.tzinfo is None else found


def build(client: KalshiClient) -> tuple[pd.DataFrame, dict[str, dict]]:
    """The ledger, one row per game bet, and each settled game's facts."""
    fills = client.fills()
    settlements = client.settlements()
    positions = client.market_positions()
    tickers = {f["ticker"] for f in fills} | {s["ticker"] for s in settlements}
    tickers |= {p["ticker"] for p in positions}
    events = {e: event_json(e) for e in sorted({ledger.event_of(t) for t in tickers})}
    frame = ledger.bets(fills, settlements, positions, events)
    if frame.empty:
        return frame, {}
    runs = ledger.read_runs(ledger.PICKS_DIR)
    orders = ledger.read_orders(ledger.ORDERS_DIR)
    themes = ledger.read_themes(ledger.THEMES)
    now = ledger.now_utc()

    rows, all_facts = [], {}
    for _, bet in frame.iterrows():
        row = bet.to_dict()
        codes = list(ledger.teams(events[bet["event"]]))
        found = ledger.bet_time(row, codes, runs, orders)
        row.update(found)
        if row.get("bet_home") is not None:
            # Kalshi's "X vs Y" doesn't say who's home.
            them = ledger.teams(events[bet["event"]])[
                ledger.other(bet["bet_code"], codes)
            ]
            us = bet["bet_short"]
            row["title"] = f"{them} @ {us}" if row["bet_home"] else f"{us} @ {them}"
        for key in ("model", "market"):
            row.setdefault(key, math.nan)
        row["edge"] = row["model"] - row["market"]

        summary = {}
        if bet["settled"] and row.get("game_id"):
            summary = espn_summary(row["game_id"], final=True)
        kickoff = _timestamp(row.get("kickoff"))
        if kickoff is None and summary:
            kickoff = _timestamp(summary["header"]["competitions"][0].get("date"))
        row["kickoff"] = kickoff
        when = kickoff or row["first_fill"]
        row["week"] = ledger.week_of(when.to_pydatetime()) if when is not None else None

        first = (
            row["first_fill"]
            if not pd.isna(row["first_fill"])
            else now - timedelta(days=10)
        )
        start = first - timedelta(hours=2)
        end = min(kickoff + timedelta(hours=1), now) if kickoff is not None else now
        tickers = [m["ticker"] for m in events[bet["event"]]["markets"]]
        prices = candles(bet["event"], tickers, start, end, final=bool(bet["settled"]))
        at = kickoff if (bet["settled"] and kickoff is not None) else now
        mids = ledger.no_vig(prices, int(at.timestamp()))
        price = mids.get(bet["bet_code"], math.nan)
        row["close"] = price if bet["settled"] else math.nan
        row["mark"] = math.nan if bet["settled"] else price
        row["clv"] = 100 * (row["close"] - row["avg_price"])

        row.update(
            {"score": None, "read": None, "tags": [], "type": None, "type_from": None}
        )
        if bet["settled"] and summary and row.get("bet_home") is not None:
            f = ledger.facts(summary, bool(row["bet_home"]))
            all_facts[bet["event"]] = f
            won = bool(bet["won"])
            row["score"] = f["score"]
            row["margin"] = f["margin"]
            row["headline"] = f["headline"]
            row["read"] = ledger.box_read(f, won)
            row["tags"] = ledger.tags(f, won)
            row["type"] = ledger.suggest(f, row["tags"], won, row["avg_price"])
            row["type_from"] = "box score"
        theme = themes.get(bet["event"])
        if theme is not None:
            row.update(
                type=theme.type, theme=theme.theme, why=theme.why, type_from="theme"
            )
        rows.append(row)
    return pd.DataFrame(rows), all_facts


def main() -> None:
    args = parse()
    ledger.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    s3 = None if args.offline else _s3()
    if s3 is not None:
        pull_down(*s3)

    client = KalshiClient.from_env(dry_run=True)
    frame, facts = build(client)
    if frame.empty:
        print("no bets on the account")
        return
    as_of = ledger.now_utc()
    out = frame.drop(columns=["order_ids"]).copy()
    out["tags"] = out["tags"].map(lambda ts: ";".join(ts))
    out.sort_values(["week", "kickoff"]).to_csv(
        ledger.LEDGER_DIR / "ledger.csv", index=False
    )
    page = ledger.report(frame, as_of)
    (ledger.LEDGER_DIR / "report.md").write_text(page)
    (ledger.LEDGER_DIR / "wins-losses.md").write_text(ledger.wins_losses(frame, as_of))

    settled = frame[frame["settled"]].copy()
    card = ledger.scorecard(settled.assign(won=settled["won"].astype(bool)))
    pending = settled[settled["type_from"] != "theme"]
    unpriced = frame[frame["model"].isna()]
    print(
        f"{len(frame)} games bet: {len(settled)} settled, {len(frame) - len(settled)} open"
    )
    if card.get("n"):
        print(
            f"settled {card['wins']}-{card['losses']}, P&L {card['pnl']:+.2f} on "
            f"{card['staked']:.2f} ({card['roi']:+.1%}); model EV {card['model_ev']:+.2f}, "
            f"close EV {card['close_ev']:+.2f}, CLV {card['clv']:+.1f}c"
        )
    if len(unpriced):
        print(f"no model price for: {', '.join(unpriced['title'])}")
    no_facts = settled[settled["score"].isna()]
    if len(no_facts):
        print(f"no ESPN game for: {', '.join(no_facts['title'])}")
    print(f"{len(pending)} settled games without a theme ({ledger.THEMES})")
    print(f"report: {ledger.LEDGER_DIR / 'report.md'}")
    print(f"wins and losses: {ledger.LEDGER_DIR / 'wins-losses.md'}")
    if args.digest:
        print(f"\ntypes: {', '.join(ledger.TYPES)}\n")
        for _, row in pending.sort_values("kickoff").iterrows():
            if row["event"] in facts:
                print(ledger.digest(row, facts[row["event"]]))
    if s3 is not None:
        push_up(*s3)


if __name__ == "__main__":
    main()

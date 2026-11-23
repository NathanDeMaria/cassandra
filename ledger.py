"""The ledger: every game bet, on Kalshi and Polymarket US, what it paid,
and why it went that way.

    python ledger.py             # or `make ledger`
    python ledger.py --offline   # no s3: skip the cloud job's orders and the backup
    python ledger.py --digest    # also print each settled game still without a theme

Each run rebuilds the whole history from the accounts (read-only, like
`make positions`), each venue's hourly prices for each game, and ESPN's box
score and AP recap for each settled one. Polymarket US's bets are read when
its key is set up (`cassandra.polymarket_us`); a game there is keyed by its
market (`aec-...`) where Kalshi's is keyed by its event. It writes two files to
~/.cassandra/betting/ledger/: `ledger.csv`, one row per game bet, and
`report.md`. The report has the overall numbers, a table per result type
and tag, every week's games with their themes, and the open bets. A
final game's prices and summary are kept, so a rerun only fetches what's
new or still open.

A theme is written by hand -- or by a Claude session -- into `themes.toml`
there, keyed by the game's venue id (the `=== KX...` or `=== aec-...` line of
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
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from cassandra import ledger
from cassandra.kalshi_venue import KALSHI_VENUE
from cassandra.picks_job import RUN_FILES, client
from cassandra.placing import espn_summary
from cassandra.polymarket_us import POLYMARKET_US
from cassandra.venues import Account, Venue, other_side

#: Every venue whose bets the ledger reads, when its key is set up.
VENUES: tuple[Venue, ...] = (KALSHI_VENUE, POLYMARKET_US)

#: Each game's Kalshi event and hourly prices, once final.
S3_LEDGER = "cassandra/betting/ledger/"
#: A market in any of these is done moving, so its prices can be kept.


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


def _venue_dir(venue: Venue) -> Path:
    """Where a venue's listings and prices are kept: `kalshi/`, `polymarket_us/`."""
    return ledger.LEDGER_DIR / venue.name


def game_listing(venue: Venue, event: str) -> dict:
    """A game's listing (`Venue.fetch_game`), kept once the game is settled."""
    return _cached(
        _venue_dir(venue) / f"{event}.json",
        lambda: venue.fetch_game(event),
        lambda found: venue.read_game(found).final,
    )


def game_prices(
    venue: Venue,
    event: str,
    listing: dict,
    start: datetime,
    end: datetime,
    final: bool,
) -> dict:
    """A game's hourly prices from `start` to `end` (`Venue.fetch_prices`),
    kept once it's over."""
    return _cached(
        _venue_dir(venue) / f"{event}_candles.json",
        lambda: venue.fetch_prices(event, listing, start, end),
        lambda _: final,
    )


def _timestamp(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    found = pd.Timestamp(value)
    return found.tz_localize(UTC) if found.tzinfo is None else found


def build(
    accounts: Sequence[tuple[Venue, Account]],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """The ledger, one row per game bet on every account, and each settled
    game's facts."""
    runs = ledger.read_runs(ledger.PICKS_DIR)
    orders = ledger.read_orders(ledger.ORDERS_DIR)
    themes = ledger.read_themes(ledger.THEMES)
    now = ledger.now_utc()

    rows, all_facts = [], {}
    for venue, account in accounts:
        trades, settled, holdings = (
            account.trades(),
            account.settled(),
            account.holdings(),
        )
        tickers = {t.ticker for t in trades} | {s.ticker for s in settled}
        tickers |= {h.ticker for h in holdings}
        listings = {
            e: game_listing(venue, e)
            for e in sorted({venue.event_of(t) for t in tickers})
        }
        games = {e: venue.read_game(found) for e, found in listings.items()}
        frame = ledger.bets(venue, games, holdings, settled, trades)
        for _, bet in frame.iterrows():
            row = bet.to_dict()
            game = games[bet["event"]]
            row.update(ledger.bet_time(row, venue, game, runs, orders))
            if row.get("bet_home") is not None:
                # A venue's "X vs Y" doesn't say who's home.
                them = game.sides[other_side(bet["bet_code"], game.sides)]
                us = bet["bet_short"]
                row["title"] = f"{them} @ {us}" if row["bet_home"] else f"{us} @ {them}"
            for key in ("model", "market"):
                row.setdefault(key, math.nan)
            row["edge"] = row["model"] - row["market"]

            summary = {}
            league = ledger.league_of(bet["event"]) or "ncaafb"
            if bet["settled"] and row.get("game_id"):
                summary = espn_summary(row["game_id"], final=True, league=league)
            kickoff = _timestamp(row.get("kickoff"))
            if kickoff is None and summary:
                kickoff = _timestamp(summary["header"]["competitions"][0].get("date"))
            if kickoff is None and game.kickoff is not None:
                kickoff = _timestamp(game.kickoff)
            row["kickoff"] = kickoff
            when = kickoff or row["first_fill"]
            row["week"] = (
                ledger.week_of(when.to_pydatetime()) if when is not None else None
            )

            first = (
                row["first_fill"]
                if not pd.isna(row["first_fill"])
                else now - timedelta(days=10)
            )
            start = first - timedelta(hours=2)
            end = min(kickoff + timedelta(hours=1), now) if kickoff is not None else now
            prices = game_prices(
                venue,
                bet["event"],
                listings[bet["event"]],
                start,
                end,
                final=bool(bet["settled"]),
            )
            at = kickoff if bet["settled"] else now
            # A settled game with no kickoff has no close: the price now is
            # after the final.
            price = (
                math.nan
                if at is None
                else venue.no_vig(prices, int(at.timestamp())).get(
                    bet["bet_code"], math.nan
                )
            )
            row["close"] = price if bet["settled"] else math.nan
            row["mark"] = math.nan if bet["settled"] else price
            row["clv"] = 100 * (row["close"] - row["avg_price"])
            read_result(row, summary, league, themes, all_facts)
            rows.append(row)
    return pd.DataFrame(rows), all_facts


def accounts() -> list[tuple[Venue, Account]]:
    """Every venue account with its key set up, read-only. One that can't be
    opened is left out, and says so."""
    found = []
    for venue in VENUES:
        if not venue.has_key():
            print(f"no {venue.label} key: its bets are left out")
            continue
        try:
            found.append((venue, venue.account(dry_run=True)))
        except Exception as error:
            print(f"!! {venue.label} left out, couldn't open its account: {error!r}")
    if not found:
        raise SystemExit("no venue account to read")
    return found


def read_result(
    row: dict, summary: dict, league: str, themes: dict, all_facts: dict
) -> None:
    """A settled game's result read off its ESPN summary -- score, margin,
    the box score's read, tags, a suggested type -- and its written theme
    over the suggestion, into `row`."""
    row.update(
        {"score": None, "read": None, "tags": [], "type": None, "type_from": None}
    )
    if row["settled"] and summary and row.get("bet_home") is not None:
        f = ledger.facts(summary, bool(row["bet_home"]), league)
        all_facts[row["event"]] = f
        won = bool(row["won"])
        row["score"] = f["score"]
        row["margin"] = f["margin"]
        row["headline"] = f["headline"]
        row["read"] = ledger.box_read(f, won)
        row["tags"] = ledger.tags(f, won)
        row["type"] = ledger.suggest(f, row["tags"], won, row["avg_price"])
        row["type_from"] = "box score"
    theme = themes.get(row["event"])
    if theme is not None:
        row.update(type=theme.type, theme=theme.theme, why=theme.why, type_from="theme")


def main() -> None:
    args = parse()
    ledger.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    s3 = None if args.offline else _s3()
    if s3 is not None:
        pull_down(*s3)

    frame, facts = build(accounts())
    if frame.empty:
        print("no bets on the account")
        return
    as_of = ledger.now_utc()
    out = frame.drop(columns=["order_ids"]).copy()
    out["tags"] = out["tags"].map(lambda ts: ";".join(ts))  # ty: ignore[no-matching-overload]
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
        print(f"\nfootball types: {', '.join(ledger.FOOTBALL_TYPES)}")
        print(f"basketball types: {', '.join(ledger.BASKETBALL_TYPES)}\n")
        for _, row in pending.sort_values("kickoff").iterrows():
            if row["event"] in facts:
                print(ledger.digest(row, facts[row["event"]]))
    if s3 is not None:
        push_up(*s3)


if __name__ == "__main__":
    main()

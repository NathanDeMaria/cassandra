import io
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from cassandra import picks_job
from cassandra.picks_job import Autobet, LeagueRules

NOW = datetime(2026, 10, 21, 3, 30, tzinfo=UTC)  # Tue 22:30 CT


class FakeS3:
    """The four calls the job makes, with the two conditions it relies on."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, datetime, str]] = {}
        self.version = 0

    def _missing(self, op: str) -> ClientError:
        return ClientError({"Error": {"Code": "NoSuchKey"}}, op)

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **_):
        held = self.objects.get((Bucket, Key))
        if IfNoneMatch == "*" and held is not None:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if IfMatch is not None and (held is None or held[2] != IfMatch):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.version += 1
        self.objects[(Bucket, Key)] = (Body, NOW, f'"{self.version}"')

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self._missing("GetObject")
        body, _, etag = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ETag": etag}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"LastModified": self.objects[(Bucket, Key)][1]}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


def _picks(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "bet": bet,
                "game": f"{bet} @ Somebody",
                "edge": edge,
                "kickoff": pd.Timestamp("2026-10-27 18:00", tz="America/Chicago"),
                "buy": "BUY YES KXNCAAFGAME-X",
                "contracts": 10,
                "limit": 0.55,
                "outlay": 5.6,
                "model": 0.62,
                "market": 0.55,
                "offered_within_1c": 30,
                "hours_since_entry": 1.0,
                "no_plays": "",
            }
            for game_id, bet, edge in rows
        ]
    )


def test_one_lock_between_two_runs():
    s3 = FakeS3()
    assert picks_job.acquire(s3, "tmp", NOW, "a")
    assert not picks_job.acquire(s3, "tmp", NOW, "b")
    picks_job.release(s3, "tmp")
    assert picks_job.acquire(s3, "tmp", NOW, "b")


def test_a_dead_runs_lock_is_taken():
    s3 = FakeS3()
    assert picks_job.acquire(s3, "tmp", NOW, "a")
    later = NOW + picks_job.LOCK_STALE + timedelta(minutes=1)
    assert picks_job.acquire(s3, "tmp", later, "b")


def test_only_unemailed_picks_or_a_flipped_side_are_new():
    sent = picks_job.record({}, _picks(("1", "Toledo Rockets", 0.07)), NOW, "s")
    now = _picks(
        ("1", "Toledo Rockets", 0.08),
        ("2", "Ohio Bobcats", 0.06),
    )
    assert list(picks_job.unsent(now, sent)["game_id"]) == ["2"]
    flipped = _picks(("1", "Ball State Cardinals", 0.06))
    assert list(picks_job.unsent(flipped, sent)["game_id"]) == ["1"]


def test_a_held_pick_is_news_until_an_email_carried_it_as_held():
    # Emailed, then placed, as an ordinary pick ...
    sent = picks_job.record({}, _picks(("1", "Toledo Rockets", 0.07)), NOW, "s")
    held = _picks(("1", "Toledo Rockets", 0.08)).assign(qb_news="QB doubtful.")
    assert list(picks_job.unsent(held, sent, held=True)["game_id"]) == ["1"]
    assert picks_job.unsent(held, sent).empty
    # ... then held once: the next hour's hold isn't news.
    sent = picks_job.record(sent, held, NOW, "held")
    assert sent["games"]["1"]["held"] is True
    assert picks_job.unsent(held, sent, held=True).empty


def test_record_keeps_each_emails_numbered_list():
    sent = picks_job.record(
        {}, _picks(("2", "Ohio Bobcats", 0.06), ("1", "Toledo Rockets", 0.09)), NOW, "s"
    )
    # Same kickoff, so numbered by edge, biggest first -- as picks.py numbers.
    assert [p["game_id"] for p in sent["emails"][-1]["picks"]] == ["1", "2"]
    assert json.loads(json.dumps(sent, default=str))


def test_keep_copies_up_only_this_runs_picks_and_orders(tmp_path, monkeypatch):
    picks_dir, orders_dir = tmp_path / "picks", tmp_path / "orders"
    picks_dir.mkdir()
    orders_dir.mkdir()
    monkeypatch.setattr(picks_job, "RUN_FILES", {"p/": picks_dir, "o/": orders_dir})
    # A second back: a file written right after can carry an mtime a hair
    # before a timestamp taken just before it.
    started = datetime.now(UTC) - timedelta(seconds=1)
    earlier = picks_dir / "ncaafb_m_20261017-2252.csv"
    earlier.write_text("an earlier run's")
    hour_ago = started.timestamp() - 3600
    os.utime(earlier, (hour_ago, hour_ago))
    (picks_dir / "ncaafb_m_20261021-0130.csv").write_text("priced")
    (orders_dir / "ncaafb_m_20261021-0130_orders.csv").write_text("placed")
    s3 = FakeS3()
    assert picks_job.keep(s3, "b", started) == [
        "p/ncaafb_m_20261021-0130.csv",
        "o/ncaafb_m_20261021-0130_orders.csv",
    ]
    assert s3.objects[("b", "o/ncaafb_m_20261021-0130_orders.csv")][0] == b"placed"


def test_keep_leaves_a_dry_runs_orders_behind(tmp_path, monkeypatch):
    picks_dir, orders_dir = tmp_path / "picks", tmp_path / "orders"
    picks_dir.mkdir()
    orders_dir.mkdir()
    monkeypatch.setattr(
        picks_job, "RUN_FILES", {"b/picks/": picks_dir, "b/orders/": orders_dir}
    )
    # A second back: a file written right after can carry an mtime a hair
    # before a timestamp taken just before it.
    started = datetime.now(UTC) - timedelta(seconds=1)
    (picks_dir / "ncaafb_m_20261021-0130.csv").write_text("priced")
    (orders_dir / "ncaafb_m_20261021-0130_orders.csv").write_text("not sent")
    assert picks_job.keep(FakeS3(), "b", started, orders=False) == [
        "b/picks/ncaafb_m_20261021-0130.csv"
    ]


def test_no_autobet_file_means_nothing_is_placed():
    s3 = FakeS3()
    rules = picks_job.autobet(s3, "b")
    assert rules == Autobet(paused=False, leagues={})
    assert not rules.placing("ncaafb")
    assert rules.describe().startswith("autobet on: no leagues")
    picks_job.write_json(s3, "b", picks_job.AUTOBET_KEY, Autobet(paused=True).to_json())
    assert picks_job.autobet(s3, "b").paused
    assert picks_job.autobet(s3, "b").describe().startswith("autobet paused")


def test_a_hold_email_says_why_and_how_to_place_each():
    held = _picks(("1", "Toledo Rockets", 0.09), ("2", "Ohio Bobcats", 0.12)).assign(
        qb_news="The starter left with an ankle injury and is doubtful."
    )
    at = NOW.astimezone(ZoneInfo("America/Chicago"))
    subject, text = picks_job.hold_email(held, at, 0.01)
    assert subject.startswith("2 Kalshi picks need your OK, QB check (Tue 22:30 CT)")
    assert subject.endswith("Ohio Bobcats, Toledo Rockets")
    assert text.startswith("Held: 2 Kalshi picks that may be priced on a quarterback")
    # Numbered as picks.py numbers them: same kickoff, the bigger edge first.
    assert "1. Ohio Bobcats" in text and "2. Toledo Rockets" in text
    assert "QB: The starter left with an ankle injury" in text
    assert 'make bet ARGS="--only 2"' in text
    assert 'All of them: make bet ARGS="--only 2 --only 1"' in text
    one, _ = picks_job.hold_email(held.iloc[:1], at, 0.01)
    assert one.startswith("1 Kalshi pick needs your OK")


def _fill(order_id, ticker, side, action, count, yes, fee=0.05):
    return {
        "order_id": order_id,
        "ticker": ticker,
        "side": side,
        "action": action,
        "count_fp": str(count),
        "yes_price_dollars": f"{yes:.2f}",
        "no_price_dollars": f"{1 - yes:.2f}",
        "fee_cost": str(fee),
    }


def test_the_summary_adds_up_each_games_fills_with_what_the_log_says():
    orders = pd.DataFrame(
        [
            {
                "order_id": "o1",
                "bet": "Toledo Rockets",
                "game": "Ohio @ Toledo",
                "model": 0.62,
                "edge": 0.08,
                "rung": False,
            },
            {
                "order_id": "o2",
                "bet": "Toledo Rockets",
                "game": "Ohio @ Toledo",
                "model": 0.62,
                "edge": 0.08,
                "rung": True,
            },
        ]
    )
    fills = [
        _fill("o1", "KXNCAAFGAME-26NOV07OHTOL-TOL", "yes", "buy", 6, 0.55),
        _fill("o2", "KXNCAAFGAME-26NOV07OHTOL-TOL", "yes", "buy", 4, 0.50),
        # A NO bought through the book, by hand: no log, named by its ticker.
        _fill("o3", "KXNCAAFGAME-26NOV07BALKSU-BAL", "no", "sell", 5, 0.30),
    ]
    bets = picks_job.placed_bets(fills, orders).set_index("bet")
    toledo = bets.loc["Toledo Rockets"]
    assert toledo["contracts"] == 10 and toledo["rung"] == 4
    assert toledo["price"] == pytest.approx((6 * 0.55 + 4 * 0.50) / 10)
    assert toledo["cost"] == pytest.approx(6 * 0.55 + 4 * 0.50 + 0.10)
    assert bets.loc["NO BAL"]["price"] == pytest.approx(0.70)
    assert pd.isna(bets.loc["NO BAL"]["model"])


def test_the_summary_counts_a_sale_before_the_end_as_paid():
    settled = picks_job.settled_games(
        [
            {
                "ticker": "E1-A",
                "yes_count_fp": "10",
                "yes_total_cost_dollars": "6",
                "no_count_fp": "0",
                "no_total_cost_dollars": "0",
                "revenue": 1000,
                "fee_cost": "0.1",
            },
            {
                "ticker": "E2-B",
                "yes_count_fp": "5",
                "yes_total_cost_dollars": "3",
                "no_count_fp": "0",
                "no_total_cost_dollars": "0",
                "revenue": 0,
                "fee_cost": "0.05",
            },
            # 11 YES at 0.88, sold at 0.97: booked as 11 NO at 0.03, revenue 0.
            {
                "ticker": "E3-C",
                "yes_count_fp": "11",
                "yes_total_cost_dollars": "9.68",
                "no_count_fp": "11",
                "no_total_cost_dollars": "0.33",
                "revenue": 0,
                "fee_cost": "0.1",
            },
        ]
    ).set_index("event")
    assert settled.loc["E1", "pnl"] == pytest.approx(3.9) and settled.loc["E1", "won"]
    assert (
        settled.loc["E2", "pnl"] == pytest.approx(-3.05)
        and not settled.loc["E2", "won"]
    )
    assert settled.loc["E3", "pnl"] == pytest.approx(11 - 9.68 - 0.33 - 0.1)


def test_the_morning_summary_reads_top_to_bottom():
    orders = pd.DataFrame(
        [
            {
                "order_id": "o1",
                "bet": "Toledo Rockets",
                "game": "Ohio @ Toledo",
                "model": 0.62,
                "edge": 0.08,
                "rung": False,
            }
        ]
    )
    bets = picks_job.placed_bets(
        [_fill("o1", "KXNCAAFGAME-26NOV07OHTOL-TOL", "yes", "buy", 6, 0.55)], orders
    )
    settled = picks_job.settled_games(
        [
            {
                "ticker": "E1-A",
                "yes_count_fp": "10",
                "yes_total_cost_dollars": "6",
                "no_count_fp": "0",
                "no_total_cost_dollars": "0",
                "revenue": 1000,
                "fee_cost": "0.1",
            }
        ]
    )
    resting = [
        {
            "ticker": "T-X",
            "client_order_id": "rung-1",
            "outcome_side": "yes",
            "book_side": "bid",
            "yes_price_dollars": "0.40",
            "remaining_count_fp": "5",
        }
    ]
    holds = [
        {
            "game_id": "9",
            "bet": "Ohio Bobcats",
            "game": "Ohio @ Toledo",
            "at": NOW.isoformat(),
        }
    ]
    central = ZoneInfo("America/Chicago")
    subject, text = picks_job.summary_email(
        bets,
        resting,
        settled,
        holds,
        (25.0, 120.0),
        {"E1": "Alpha at Beta"},
        Autobet(),
        (NOW - timedelta(days=1)).astimezone(central),
        NOW.astimezone(central),
    )
    assert subject == "Kalshi Tue Oct 20: 1 bet placed ($3.35), settled 1-0 (+$3.90)"
    assert (
        "Toledo Rockets -- Ohio @ Toledo: 6 @ 0.55, $3.35; model 0.62 (edge +0.080)"
        in text
    )
    assert "Resting: 1 order, up to $2.00 if they all fill (1 of them rungs" in text
    assert "won    +$3.90  Alpha at Beta" in text
    assert 'Ohio Bobcats -- Ohio @ Toledo: make bet ARGS="--only 9"' in text
    assert text.endswith("Account: $25.00 cash, $120.00 in open positions.")


def test_holds_since_lists_only_the_picks_emailed_as_held_lately():
    old = picks_job.record(
        {}, _picks(("1", "Toledo Rockets", 0.09)), NOW - timedelta(days=2), "s"
    )
    held = _picks(("2", "Ohio Bobcats", 0.12)).assign(qb_news="Doubtful.")
    sent = picks_job.record(old, held, NOW, "held")
    assert [
        h["game_id"] for h in picks_job.holds_since(sent, NOW - timedelta(days=1))
    ] == ["2"]


#: A league's required rules for the tests; the numbers are arbitrary.
REQUIRED = {
    "kelly_fraction": 0.3,
    "max_bet": 7.0,
    "min_edge": 0.06,
    "max_since_entry_hours": 20.0,
    "max_overround": 1.12,
}


def _rules(**optional: Any) -> LeagueRules:
    """`REQUIRED`'s rules, and any optional ones."""
    return LeagueRules(
        kelly_fraction=0.3,
        max_bet=7.0,
        min_edge=0.06,
        max_since_entry_hours=20.0,
        max_overround=1.12,
        **optional,
    )


REQUIRED_FLAGS = [
    "--kelly-fraction",
    "0.3",
    "--max-bet",
    "7",
    "--min-edge",
    "0.06",
    "--max-since-entry-hours",
    "20",
    "--max-overround",
    "1.12",
]


def test_an_autobet_file_round_trips_and_an_old_one_says_whats_missing():
    rules = Autobet(leagues={"ncaafb": _rules(rungs=(1.0, 3.0), top=6, rest_hours=5.0)})
    assert Autobet.from_json(rules.to_json()) == rules
    # The flat file from before leagues: football's, and short of a bar now.
    with pytest.raises(ValueError, match="without min_edge"):
        Autobet.from_json({"paused": False, "max_bet": 7.0, "kelly_fraction": 0.3})
    # The pause alone, as `make pause` wrote it with no rules: none to place.
    assert Autobet.from_json({"paused": True}) == Autobet(paused=True)


def test_a_league_needs_its_required_rules_written_down():
    with pytest.raises(ValueError, match="without max_overround"):
        LeagueRules.from_json(
            {k: v for k, v in REQUIRED.items() if k != "max_overround"}
        )
    with pytest.raises(ValueError, match="give it --kelly-fraction --max-bet"):
        picks_job.changed(Autobet(), picks_job.parse(["autobet", "--league", "womens"]))


def test_league_rules_are_picks_flags():
    rules = _rules(
        favorite_price=0.7,
        favorite_edge=0.11,
        max_exposure=0.4,
        slippage=0.02,
        top=6,
        rungs=(1.0, 3.0),
        rest_hours=5.0,
    )
    assert rules.argv() == [
        "--kelly-fraction", "0.3",
        "--max-bet", "7",
        "--min-edge", "0.06",
        "--max-since-entry", "20",
        "--max-overround", "1.12",
        "--favorite-price", "0.7",
        "--favorite-edge", "0.11",
        "--min-price", "0",
        "--max-exposure", "0.4",
        "--longshot-price", "0",
        "--longshot-scale", "1",
        "--slippage", "0.02",
        "--top", "6",
        "--rungs", "1,3",
        "--rest", "5",
    ]  # fmt: skip
    import picks

    args = picks.parse(["--place", *rules.argv()])
    picks.apply_rules(args, None)
    assert args.rule.min_edge == 0.06 and args.rule.favorite_edge == 0.11
    assert args.rungs == (0.01, 0.03) and args.rest == 5.0 and args.top == 6
    assert "--rungs" in _rules().argv()  # none, said so


def test_a_league_is_opted_in_paused_and_dropped_on_its_own():
    def run(rules: Autobet, *argv: str) -> Autobet:
        return picks_job.changed(rules, picks_job.parse(list(argv)))

    football = _rules()
    rules = run(
        Autobet(leagues={"ncaafb": football}),
        "autobet",
        "--league",
        "womens",
        *REQUIRED_FLAGS,
        "--max-exposure",
        "0.4",
        "--longshot-price",
        "0.15",
        "--longshot-scale",
        "0.6",
        "--rungs",
        "1,3",
    )
    womens = rules.leagues["womens"]
    assert womens == _rules(
        max_exposure=0.4,
        longshot_price=0.15,
        longshot_scale=0.6,
        rungs=(1.0, 3.0),
    )
    assert rules.leagues["ncaafb"] == football
    assert "womens 0.3 Kelly, at most $7 a bet, at most 40% of the bankroll" in (
        rules.describe()
    )
    # A league already listed changes only what's given.
    bigger = run(rules, "autobet", "--league", "womens", "--max-bet", "4")
    assert bigger.leagues["womens"].max_bet == 4.0
    assert bigger.leagues["womens"].max_exposure == 0.4

    paused = run(rules, "pause", "--league", "womens")
    assert not paused.placing("womens") and paused.placing("ncaafb")
    assert rules.placing("womens"), "the rules changed were a copy"
    assert "womens paused" in paused.describe()
    assert run(paused, "resume", "--league", "womens").placing("womens")
    everything = run(rules, "pause")
    assert not everything.placing("ncaafb") and not everything.placing("womens")

    cleared = run(
        rules, "autobet", "--league", "womens", "--clear", "max_exposure",
        "--clear", "rungs",
    )  # fmt: skip
    assert cleared.leagues["womens"].max_exposure is None
    assert cleared.leagues["womens"].rungs == ()
    with pytest.raises(ValueError, match="not an optional rule"):
        run(rules, "autobet", "--league", "womens", "--clear", "min_edge")
    dropped = run(rules, "autobet", "--league", "womens", "--remove")
    assert "womens" not in dropped.leagues
    with pytest.raises(ValueError, match="no rules for mens"):
        run(rules, "pause", "--league", "mens")

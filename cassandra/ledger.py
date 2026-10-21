"""The Kalshi bets' ledger: what each game bet cost, what it paid, and why.

`ledger.py` (`make ledger`) gathers the pieces and writes the report; this
is what it does with them:

- `bets` nets the account into one bet per game (a Kalshi event). What's
  held comes from Kalshi's own books -- settlements for games over,
  positions for the rest -- and the fills only say when, and in which
  orders.
- `bet_time` finds what the model said when the bet was made: the orders
  log `placing.place` keeps (its rows carry the order ids), or failing
  that the last picks run started before the first fill.
- `no_vig` reads the Kalshi price for each side from hourly candles: at
  kickoff for the closing line, now for an open bet's mark.
- `facts` reads a game's ESPN summary from the bet's side; `tags`,
  `box_read` and `suggest` turn that into result types that add up across
  weeks.
- `themes.toml` holds the written theme for each settled game, one of
  `TYPES` plus a phrase and a sentence or two on why. A game without one
  counts under the type `suggest` reads off its box score.
- `scorecard`, `breakdown` and `report` are the numbers and the page.

Everything is from the bet's side: "won" is the bet won, "margin" is the
bet team's points less its opponent's.
"""

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import tomllib

from . import injuries
from .constants import CASSANDRA_HOME
from .picks_job import RUN_FILES
from .placing import CENTRAL

LEDGER_DIR = CASSANDRA_HOME / "betting" / "ledger"
THEMES = LEDGER_DIR / "themes.toml"
#: Where `picks.py` writes every priced game of each run, and `place` its
#: orders; `ledger.py` brings the cloud job's copies down into the same places.
PICKS_DIR, ORDERS_DIR = RUN_FILES.values()

#: Why a game went the way it did, from the bet's side, one per game so the
#: weeks add up. The description is the test for using it.
TYPES = {
    "late_swing": "decided in the last minutes or overtime: a late drive, a "
    "walk-off kick, a late lead lost, a call at the goal line",
    "trenches": "the run game and the line of scrimmage decided it",
    "turnovers": "turnovers or return touchdowns decided it",
    "qb": "a quarterback decided it: an injury, a benching, a debut, a career "
    "day or a meltdown",
    "outclassed": "one side was better all game: a rout, or never in doubt",
    "upset": "the underdog outplayed the favorite outright, no single cause",
    "special_teams": "kicks, punts or returns decided it",
    "officiating": "a call or a review swung it",
}

#: The box-score read: a side outplayed the other when it gained at least
#: this many more yards, and this many more a play. Short of either, even.
EVEN_YARDS = 30
EVEN_YPP = 0.5
#: Tags' thresholds, in points and yards.
ONE_SCORE = 8
BLOWOUT = 21
BIG_LEAD = 10
RUN_GAME_YARDS = 120
TURNOVER_EDGE = 3
#: A comeback is also a winner whose win probability sank to 1 - this, at
#: least `WP_DROP` under its kickoff number.
BLOWN_WP = 0.85
WP_DROP = 0.25
#: "Late" is the final five minutes of the fourth quarter, or overtime.
LATE_SECONDS = 300
#: A second passer with this many attempts is a quarterback change.
QB_ATTEMPTS = 5

_STAMP = re.compile(r"_(\d{8}-\d{4})(?:_orders)?\.csv$")
_RETURN_TD = re.compile(r"\b(Interception|Fumble|Kickoff|Punt|Blocked \w+) Return\b")


# ------------------------------------------------------------------------------
# What's held
# ------------------------------------------------------------------------------


def code_of(ticker: str) -> str:
    """A market ticker's team code: `...-26OCT24HOUUTAH-UTAH` -> `UTAH`."""
    return ticker.rsplit("-", 1)[1]


def event_of(ticker: str) -> str:
    return ticker.rsplit("-", 1)[0]


def teams(event: Mapping) -> dict[str, str]:
    """An event's two teams: each market's code, and Kalshi's name for the team."""
    return {
        code_of(m["ticker"]): m.get("yes_sub_title") or code_of(m["ticker"])
        for m in event["markets"]
    }


def other(code: str, codes: Iterable[str]) -> str:
    """The other team of a two-team event."""
    rest = [c for c in codes if c != code]
    if len(rest) != 1:
        raise ValueError(f"{code} isn't one of two teams in {list(codes)}")
    return rest[0]


class Leg(NamedTuple):
    """Contracts in one market that pay if `code` wins."""

    code: str
    count: float
    cost: float
    fees: float


def _num(value) -> float:
    return float(value) if value not in (None, "") else 0.0


def settled_legs(settlement: Mapping, codes: Sequence[str]) -> list[Leg]:
    """A settlement's YES and NO sides, each as the team it paid on. The fees
    go on the first leg; a market is one trade's worth of fees."""
    code = code_of(settlement["ticker"])
    fees = _num(settlement.get("fee_cost"))
    found = []
    for count, cost, pays in (
        ("yes_count_fp", "yes_total_cost_dollars", code),
        ("no_count_fp", "no_total_cost_dollars", other(code, codes)),
    ):
        if _num(settlement.get(count)):
            found.append(
                Leg(pays, _num(settlement[count]), _num(settlement[cost]), fees)
            )
            fees = 0.0
    return found


def open_leg(position: Mapping, codes: Sequence[str]) -> Leg | None:
    """An open position as the team it pays on: YES counts are positive, NO negative."""
    count = _num(position.get("position_fp"))
    if not count:
        return None
    code = code_of(position["ticker"])
    return Leg(
        code if count > 0 else other(code, codes),
        abs(count),
        _num(position.get("market_exposure_dollars")),
        _num(position.get("fees_paid_dollars")),
    )


def bets(
    fills: Sequence[Mapping],
    settlements: Sequence[Mapping],
    positions: Sequence[Mapping],
    events: Mapping[str, Mapping],
) -> pd.DataFrame:
    """One row per game bet: the team it's on, what it cost, what it paid.

    `events` is each event's public JSON (its two markets). A game is
    settled once none of it is still held. A game held on both sides is
    called a bet on the bigger one, and flagged.
    """
    held: dict[str, list[Leg]] = {}
    paid: dict[str, float] = {}
    for s in settlements:
        event = event_of(s["ticker"])
        if event not in events:
            continue
        codes = list(teams(events[event]))
        held.setdefault(event, []).extend(settled_legs(s, codes))
        paid[event] = paid.get(event, 0.0) + _num(s.get("revenue")) / 100
    still_open = set()
    for p in positions:
        event = event_of(p["ticker"])
        if event not in events:
            continue
        leg = open_leg(p, list(teams(events[event])))
        if leg is not None:
            held.setdefault(event, []).append(leg)
            still_open.add(event)
    filled: dict[str, list[Mapping]] = {}
    for f in fills:
        filled.setdefault(event_of(f["ticker"]), []).append(f)

    rows = []
    for event, legs in held.items():
        names = teams(events[event])
        by_code: dict[str, float] = {}
        for leg in legs:
            by_code[leg.code] = by_code.get(leg.code, 0.0) + leg.count
        code = max(by_code, key=lambda c: by_code[c])
        contracts = by_code[code]
        side_cost = sum(leg.cost for leg in legs if leg.code == code)
        cost = sum(leg.cost for leg in legs)
        fees = sum(leg.fees for leg in legs)
        settled = event not in still_open
        payout = paid.get(event, 0.0) if settled else math.nan
        times = sorted(pd.Timestamp(f["created_time"]) for f in filled.get(event, []))
        rows.append(
            {
                "event": event,
                "title": (events[event].get("event") or {}).get("title") or event,
                "bet_code": code,
                "bet_short": names[code],
                "contracts": contracts,
                "avg_price": side_cost / contracts,
                "cost": cost,
                "fees": fees,
                "stake": cost + fees,
                "settled": settled,
                "payout": payout,
                "pnl": payout - cost - fees if settled else math.nan,
                "won": (payout > 0) if settled else None,
                "both_sides": len(by_code) > 1,
                "first_fill": times[0] if times else pd.NaT,
                "last_fill": times[-1] if times else pd.NaT,
                "order_ids": sorted({f["order_id"] for f in filled.get(event, [])}),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# What the model said when the bet was made
# ------------------------------------------------------------------------------


def stamp(path: Path) -> datetime | None:
    """When a picks run started, from its file name (`..._20261018-1212.csv`, Central)."""
    found = _STAMP.search(path.name)
    if not found:
        return None
    return datetime.strptime(found.group(1), "%Y%m%d-%H%M").replace(tzinfo=CENTRAL)


#: What a picks run's file gives the ledger, of its thirty-odd columns.
RUN_COLUMNS = [
    "event",
    "buy",
    "bet_home",
    "game",
    "game_id",
    "kickoff",
    "divisions",
    "model",
    "market",
]


def read_runs(directory: Path) -> pd.DataFrame:
    """Every picks run's priced games in one frame, oldest run first, each
    row with its run's start as `started`."""
    frames = []
    for path in directory.glob("*.csv"):
        when = stamp(path)
        if when is None or path.name.endswith("_orders.csv"):
            continue
        frame = pd.read_csv(
            path, dtype={"game_id": str}, usecols=lambda c: c in RUN_COLUMNS
        )
        if {"event", "buy"} <= set(frame.columns):
            frames.append(frame.assign(started=pd.Timestamp(when)))
    if not frames:
        return pd.DataFrame(columns=[*RUN_COLUMNS, "started"])
    runs = pd.concat(frames, ignore_index=True)
    return runs.sort_values("started", kind="stable", ignore_index=True)


def read_orders(directory: Path) -> pd.DataFrame:
    """Every order `placing.place` logged, with the model and edge it was sent on."""
    logs = [
        pd.read_csv(path, dtype={"game_id": str, "order_id": str})
        for path in sorted(directory.glob("*_orders.csv"))
    ]
    logs = [log for log in logs if "order_id" in log]
    return pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


def _row_side(row: pd.Series, codes: Sequence[str]) -> str | None:
    """The team a picks row's `buy` pays on: `YES ...-DEL` is DEL, `NO ...-DEL` the other."""
    buy = row.get("buy")
    if not isinstance(buy, str) or " " not in buy:
        return None
    action, ticker = buy.split(" ", 1)
    code = code_of(ticker)
    return code if action.upper() == "YES" else other(code, codes)


def _home_away(game: str) -> tuple[str, str]:
    away, home = game.split(" @ ", 1)
    return home, away


def oriented(row: pd.Series, code: str, codes: Sequence[str]) -> dict | None:
    """A picks row turned to the side `code`: its model and no-vig market
    for that team, and whether it's at home."""
    side = _row_side(row, codes)
    if side is None:
        return None
    same = side == code
    model, market = float(row["model"]), float(row["market"])
    bet_home = _bool(row["bet_home"]) == same
    home, away = _home_away(row["game"])
    return {
        "model": model if same else 1 - model,
        "market": market if same else 1 - market,
        "bet_home": bet_home,
        "bet": home if bet_home else away,
        "game": row["game"],
        "game_id": str(row["game_id"]),
        "kickoff": row["kickoff"],
        "divisions": row.get("divisions"),
    }


def _bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def bet_time(
    bet: Mapping,
    codes: Sequence[str],
    runs: pd.DataFrame,
    orders: pd.DataFrame,
) -> dict:
    """The model and market for the bet's side when it was made, and the game.

    The orders log is exact: the model each filled order was sent on,
    averaged over the contracts. Bets made by hand, or before the log, take
    the last picks run (`read_runs`) started before the first fill. The game
    itself (ESPN id, kickoff, divisions) comes from the latest run listing it.
    """
    found: dict = {"model_from": None}
    code = bet["bet_code"]
    listing = runs[runs["event"] == bet["event"]]
    for _, row in listing[::-1].iterrows():
        game = oriented(row, code, codes)
        if game is not None:
            found.update(
                {k: game[k] for k in ("bet", "game", "game_id", "kickoff", "divisions")}
            )
            found["bet_home"] = game["bet_home"]
            break

    ids = set(bet.get("order_ids") or [])
    if ids and not orders.empty:
        mine = orders[orders["order_id"].isin(ids) & (orders["filled"].fillna(0) > 0)]
        weights, models, markets = [], [], []
        for _, row in mine.iterrows():
            pays = code_of(row["ticker"])
            if str(row["action"]).lower() != "yes":
                pays = other(pays, codes)
            model, edge = float(row["model"]), float(row["edge"])
            same = pays == code
            weights.append(float(row["filled"]))
            models.append(model if same else 1 - model)
            markets.append(model - edge if same else 1 - (model - edge))
            found.setdefault("game_id", str(row["game_id"]))
            found.setdefault("game", row["game"])
            # An order buys its row's `bet`, so that team is `pays`.
            home, _ = _home_away(row["game"])
            found.setdefault("bet_home", (row["bet"] == home) == same)
        if weights:
            total = sum(weights)
            found["model"] = sum(w * m for w, m in zip(weights, models)) / total
            found["market"] = sum(w * m for w, m in zip(weights, markets)) / total
            found["model_from"] = "orders log"
            return found

    first = bet.get("first_fill")
    if first is not None and not pd.isna(first):
        before = listing[listing["started"] <= first]
        for _, row in before[::-1].iterrows():
            game = oriented(row, code, codes)
            if game is not None:
                found["model"], found["market"] = game["model"], game["market"]
                found["model_from"] = f"picks run {row['started']:%m-%d %H:%M}"
                break
    return found


# ------------------------------------------------------------------------------
# Prices
# ------------------------------------------------------------------------------


def no_vig(candles: Mapping[str, Sequence[Mapping]], at: int) -> dict[str, float]:
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


# ------------------------------------------------------------------------------
# The game, from the bet's side
# ------------------------------------------------------------------------------


def _stat(team: Mapping, name: str) -> str | None:
    for stat in team.get("statistics", []):
        if stat.get("name") == name:
            return stat.get("displayValue")
    return None


def _int(text: str | None) -> int:
    try:
        return int(text or 0)
    except ValueError:
        return 0


def _attempts(completions: str | None) -> int:
    """`27/49` -> 49."""
    if not completions or "/" not in completions:
        return 0
    return _int(completions.split("/")[1])


def passers(summary: Mapping, team_id: str) -> list[tuple[str, int, str]]:
    """A team's passers in a box score: name, attempts, and their line."""
    found = []
    for team in (summary.get("boxscore") or {}).get("players", []):
        if str(team["team"]["id"]) != str(team_id):
            continue
        for group in team.get("statistics", []):
            if group.get("name") != "passing":
                continue
            for athlete in group.get("athletes", []):
                name = athlete["athlete"]["displayName"].strip()
                if name == "Team":
                    continue
                stats = athlete.get("stats", [])
                line = (
                    f"{name} {stats[0]}, {stats[1]} yds, {stats[3]} TD, {stats[4]} INT"
                    if len(stats) >= 5
                    else name
                )
                found.append((name, _attempts(stats[0] if stats else None), line))
    return found


def _clock(play: Mapping) -> float:
    return float((play.get("clock") or {}).get("value") or 0)


def facts(summary: Mapping, bet_home: bool) -> dict:
    """What a final game's ESPN summary says, from the bet's side."""
    comp = summary["header"]["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    mine, theirs = (
        (sides["home"], sides["away"]) if bet_home else (sides["away"], sides["home"])
    )
    sign = 1 if bet_home else -1
    points_for, points_against = _int(mine.get("score")), _int(theirs.get("score"))
    margin = points_for - points_against
    final_sign = (margin > 0) - (margin < 0)

    scoring = summary.get("scoringPlays") or []
    path, half_margin, lead, deficit = [], 0, 0, 0
    go_ahead, previous = None, 0
    for play in scoring:
        m = sign * (play["homeScore"] - play["awayScore"])
        period = play["period"]["number"]
        lead, deficit = max(lead, m), min(deficit, m)
        if period <= 2:
            half_margin = m
        now = (m > 0) - (m < 0)
        if now == final_sign and previous != final_sign:
            go_ahead = play
        previous = now
        team = (play.get("team") or {}).get("abbreviation", "")
        path.append(
            f"Q{period} {(play.get('clock') or {}).get('displayValue', '')} "
            f"{team}: {play.get('text', '').strip()} ({play['awayScore']}-{play['homeScore']})"
        )
    periods = max([p["period"]["number"] for p in scoring], default=4)
    periods = max(periods, (comp.get("status") or {}).get("period") or 4)
    late = go_ahead is not None and (
        go_ahead["period"]["number"] >= 5
        or (go_ahead["period"]["number"] == 4 and _clock(go_ahead) <= LATE_SECONDS)
    )

    wp = [w["homeWinPercentage"] for w in summary.get("winprobability") or []]
    wp = wp if bet_home else [1 - w for w in wp]
    # The last entry is the end of the game, sometimes scored oddly.
    core = wp[1:-1] if len(wp) > 3 else wp

    stats = {t["homeAway"]: t for t in (summary.get("boxscore") or {}).get("teams", [])}
    mine_stats = stats.get("home" if bet_home else "away", {})
    their_stats = stats.get("away" if bet_home else "home", {})

    def ypp(team: Mapping) -> float:
        plays = _int(_stat(team, "rushingAttempts")) + _attempts(
            _stat(team, "completionAttempts")
        )
        return _int(_stat(team, "totalYards")) / plays if plays else math.nan

    my_passers = passers(summary, mine["team"]["id"])
    their_passers = passers(summary, theirs["team"]["id"])
    returns = [
        ((p.get("team") or {}).get("abbreviation", ""), p.get("text", "").strip())
        for p in scoring
        if _RETURN_TD.search(p.get("text", ""))
    ]
    my_abbreviation = mine["team"]["abbreviation"]
    line = ((summary.get("pickcenter") or [{}])[0] or {}).get("details")
    article = summary.get("article") or {}
    return {
        "score": f"{mine['team']['abbreviation']} {points_for}–{points_against} "
        f"{theirs['team']['abbreviation']}" + (" (OT)" if periods > 4 else ""),
        "points_for": points_for,
        "points_against": points_against,
        "margin": margin,
        "overtime": periods > 4,
        "half_margin": half_margin,
        "largest_lead": lead,
        "largest_deficit": deficit,
        "go_ahead": None
        if go_ahead is None
        else f"Q{go_ahead['period']['number']} {(go_ahead.get('clock') or {}).get('displayValue', '')}",
        "late_decider": late,
        "wp_start": wp[0] if wp else math.nan,
        "wp_peak": max(core) if core else math.nan,
        "wp_low": min(core) if core else math.nan,
        "yards_for": _int(_stat(mine_stats, "totalYards")),
        "yards_against": _int(_stat(their_stats, "totalYards")),
        "rush_for": _int(_stat(mine_stats, "rushingYards")),
        "rush_against": _int(_stat(their_stats, "rushingYards")),
        "ypp_for": ypp(mine_stats),
        "ypp_against": ypp(their_stats),
        "turnovers_for": _int(_stat(mine_stats, "turnovers")),
        "turnovers_against": _int(_stat(their_stats, "turnovers")),
        "third_downs_for": _stat(mine_stats, "thirdDownEff"),
        "third_downs_against": _stat(their_stats, "thirdDownEff"),
        "passers_for": [p[2] for p in my_passers],
        "passers_against": [p[2] for p in their_passers],
        "qb_change_for": sum(p[1] >= QB_ATTEMPTS for p in my_passers) > 1,
        "qb_change_against": sum(p[1] >= QB_ATTEMPTS for p in their_passers) > 1,
        "return_tds": [f"{team}: {text}" for team, text in returns],
        "return_tds_for": sum(team == my_abbreviation for team, _ in returns),
        "return_tds_against": sum(team != my_abbreviation for team, _ in returns),
        "line": line,
        "headline": article.get("headline"),
        "recap": " ".join(injuries.sentences(article.get("story") or "")),
        "scoring": path,
    }


def box_read(f: Mapping, won: bool) -> str:
    """Did the result follow the box score, turnovers aside? A side
    outplayed the other when it gained more yards and more a play (by
    `EVEN_YARDS` and `EVEN_YPP`): a win while outplaying is earned, while
    outplayed is lucky; a loss is unlucky or deserved the same way. Total
    yards alone favor the side that held the ball; a play alone, the side
    with the long plays; so it takes both."""
    yards = f["yards_for"] - f["yards_against"]
    per_play = f["ypp_for"] - f["ypp_against"]
    if math.isnan(per_play) or abs(yards) < EVEN_YARDS or abs(per_play) < EVEN_YPP:
        return "even"
    if (yards > 0) != (per_play > 0):
        return "even"
    if won:
        return "earned" if yards > 0 else "lucky"
    return "unlucky" if yards > 0 else "deserved"


def tags(f: Mapping, won: bool) -> list[str]:
    """What the game was like, in words that add up across weeks.

    The tags describe the game, whoever won it -- a `comeback` is the
    winner's -- so a tag's record says how the bets in such games went.
    The two QB tags are the exception, being about one side.
    """
    found = []
    margin = abs(f["margin"])
    if f["overtime"]:
        found.append("overtime")
    elif margin <= ONE_SCORE:
        found.append("one_score")
    if margin >= BLOWOUT:
        found.append("blowout")
    if f["late_decider"]:
        found.append("late_go_ahead")
    # The winner's deepest hole: in points, or in a win probability that
    # fell well below where it started -- an underdog that stayed one
    # didn't come back from anything.
    deficit = f["largest_deficit"] if won else -f["largest_lead"]
    start = f["wp_start"] if won else 1 - f["wp_start"]
    low = f["wp_low"] if won else 1 - f["wp_peak"]
    sank = not math.isnan(low) and low <= 1 - BLOWN_WP and start - low >= WP_DROP
    if deficit <= -BIG_LEAD or sank:
        found.append("comeback")
    elif deficit == 0:
        found.append("wire_to_wire")
    winner_edge = f["turnovers_against"] - f["turnovers_for"]
    winner_edge = winner_edge if won else -winner_edge
    if winner_edge >= TURNOVER_EDGE:
        found.append("turnover_edge")
    elif winner_edge <= -2:
        found.append("winner_lost_turnovers")
    rush_edge = f["rush_for"] - f["rush_against"]
    if (rush_edge if won else -rush_edge) >= RUN_GAME_YARDS:
        found.append("run_game")
    if f["return_tds"]:
        found.append("return_td")
    if f["qb_change_for"]:
        found.append("qb_change_us")
    if f["qb_change_against"]:
        found.append("qb_change_them")
    return found


def suggest(f: Mapping, found: Sequence[str], won: bool, price: float) -> str:
    """A first guess at the game's type from the box score, for games no one
    has written a theme for yet."""
    winner_returns = f["return_tds_for"] if won else f["return_tds_against"]
    if "overtime" in found or ("late_go_ahead" in found and "one_score" in found):
        return "late_swing"
    # A blowout's second quarterback is usually the backup at the end.
    if "blowout" in found:
        return "outclassed"
    if "turnover_edge" in found or winner_returns:
        return "turnovers"
    if ("qb_change_us" in found and not won) or ("qb_change_them" in found and won):
        return "qb"
    if "run_game" in found:
        return "trenches"
    if (price < 0.5) == won:
        return "upset"
    return "outclassed"


# ------------------------------------------------------------------------------
# Themes
# ------------------------------------------------------------------------------


class Theme(NamedTuple):
    type: str
    theme: str
    why: str


def read_themes(path: Path) -> dict[str, Theme]:
    """`themes.toml`: a table per event ticker with `type` (one of `TYPES`),
    `theme` (a few words) and `why` (a sentence or two)."""
    if not path.exists():
        return {}
    found = {}
    for event, entry in tomllib.loads(path.read_text()).items():
        kind = entry.get("type")
        if kind not in TYPES:
            raise ValueError(
                f"{path}: {event} has type {kind!r}; use one of {', '.join(TYPES)}"
            )
        found[event] = Theme(kind, entry.get("theme", ""), entry.get("why", ""))
    return found


# ------------------------------------------------------------------------------
# Weeks
# ------------------------------------------------------------------------------


def week_of(kickoff: datetime) -> date:
    """The Saturday a game's week is named for. A week runs Tuesday to
    Monday, Central, so Thursday and Friday games go with the Saturday after
    and a Sunday or Labor Day game with the one before."""
    day = kickoff.astimezone(CENTRAL).date()
    if day.weekday() in (6, 0):
        return day - timedelta(days=(day.weekday() - 5) % 7)
    return day + timedelta(days=5 - day.weekday())


# ------------------------------------------------------------------------------
# Numbers
# ------------------------------------------------------------------------------


def _brier(probabilities: pd.Series, outcomes: pd.Series) -> float:
    return float(((probabilities - outcomes) ** 2).mean())


def scorecard(settled: pd.DataFrame) -> dict:
    """The standard numbers over settled bets: record, money, what the model
    and the closing line expected, closing line value, calibration."""
    if settled.empty:
        return {"n": 0}
    won = settled["won"].astype(bool)
    priced = settled[settled["model"].notna()]
    closed = settled[settled["close"].notna()]
    staked = settled["stake"].sum()
    pnl = settled["pnl"].sum()
    model_ev = (
        priced["contracts"] * (priced["model"] - priced["avg_price"]) - priced["fees"]
    ).sum()
    close_ev = (
        closed["contracts"] * (closed["close"] - closed["avg_price"]) - closed["fees"]
    ).sum()
    sd = math.sqrt(
        (priced["contracts"] ** 2 * priced["model"] * (1 - priced["model"])).sum()
    )
    priced_pnl = priced["pnl"].sum()
    clv = 100 * (closed["close"] - closed["avg_price"])
    outcomes = won.astype(float)
    return {
        "n": len(settled),
        "wins": int(won.sum()),
        "losses": int((~won).sum()),
        "staked": staked,
        "fees": settled["fees"].sum(),
        "returned": settled["payout"].sum(),
        "pnl": pnl,
        "roi": pnl / staked if staked else math.nan,
        "priced": len(priced),
        "model_ev": model_ev,
        "model_roi": model_ev / priced["stake"].sum() if len(priced) else math.nan,
        "close_ev": close_ev,
        "close_roi": close_ev / closed["stake"].sum() if len(closed) else math.nan,
        "sd": sd,
        "z": (priced_pnl - model_ev) / sd if sd else math.nan,
        "wins_model": priced["model"].sum(),
        "wins_price": settled["avg_price"].sum(),
        "wins_close": closed["close"].sum(),
        "clv": clv.mean() if len(clv) else math.nan,
        "beat_close": int((clv > 0.5).sum()),
        "closed": len(closed),
        "brier_model": _brier(priced["model"], outcomes[priced.index])
        if len(priced)
        else math.nan,
        "brier_market": _brier(priced["market"], outcomes[priced.index])
        if len(priced)
        else math.nan,
        "brier_close": _brier(closed["close"], outcomes[closed.index])
        if len(closed)
        else math.nan,
    }


def breakdown(settled: pd.DataFrame, by: pd.Series | str) -> pd.DataFrame:
    """The scorecard's main columns for each group."""
    keys = settled[by] if isinstance(by, str) else by
    rows = []
    for key, group in settled.groupby(keys, sort=True):
        card = scorecard(group)
        rows.append(
            {
                "group": key,
                "n": card["n"],
                "record": f"{card['wins']}–{card['losses']}",
                "staked": card["staked"],
                "pnl": card["pnl"],
                "roi": card["roi"],
                "model_ev": card["model_ev"],
                "clv": card["clv"],
            }
        )
    return pd.DataFrame(rows)


def tag_table(settled: pd.DataFrame) -> pd.DataFrame:
    """Each tag's games, record and P&L. A game carries several tags."""
    rows = []
    for tag in sorted({t for ts in settled["tags"] for t in ts}):
        group = settled[[tag in ts for ts in settled["tags"]]]
        card = scorecard(group)
        rows.append(
            {
                "group": tag,
                "n": card["n"],
                "record": f"{card['wins']}–{card['losses']}",
                "staked": card["staked"],
                "pnl": card["pnl"],
                "roi": card["roi"],
                "model_ev": card["model_ev"],
                "clv": card["clv"],
            }
        )
    return (
        pd.DataFrame(rows).sort_values("n", ascending=False) if rows else pd.DataFrame()
    )


# ------------------------------------------------------------------------------
# The page
# ------------------------------------------------------------------------------


def _money(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{'+' if x > 0 else '−' if x < 0 else ''}${abs(x):,.2f}"


def _pct(x: float) -> str:
    if x is None or math.isnan(x):
        return "–"
    return f"{round(x, 2) + 0.0:+.0%}".replace("-", "−")


def _cents(x: float) -> str:
    if x is None or math.isnan(x):
        return "–"
    return f"{round(x, 1) + 0.0:+.1f}¢".replace("-", "−")


def _table(
    frame: pd.DataFrame, label: str, describe: Mapping[str, str] | None = None
) -> list[str]:
    head = f"| {label} | games | W–L | staked | P&L | ROI | model EV | avg CLV |"
    if describe:
        head += " what it means |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for _, r in frame.iterrows():
        line = (
            f"| {r['group']} | {r['n']} | {r['record']} | ${r['staked']:,.2f} | "
            f"{_money(r['pnl'])} | {_pct(r['roi'])} | {_money(r['model_ev'])} | {_cents(r['clv'])} |"
        )
        if describe:
            line += f" {describe.get(r['group'], '')} |"
        lines.append(line)
    return lines


def _card_lines(card: Mapping) -> list[str]:
    if not card.get("n"):
        return ["Nothing settled yet."]
    lines = [
        "| | |",
        "|---|---|",
        f"| Record | **{card['wins']}–{card['losses']}** |",
        f"| Staked (fees ${card['fees']:,.2f}) | ${card['staked']:,.2f} |",
        f"| Returned | ${card['returned']:,.2f} |",
        f"| **Net** | **{_money(card['pnl'])} (ROI {_pct(card['roi'])})** |",
        f"| Model's expected P&L at bet time | {_money(card['model_ev'])} ({_pct(card['model_roi'])})"
        + (
            f", {card['priced']} of {card['n']} bets priced"
            if card["priced"] < card["n"]
            else ""
        )
        + " |",
        f"| Closing market's expected P&L | {_money(card['close_ev'])} ({_pct(card['close_roi'])}) |",
        f"| Wins: actual / model / price paid / close | {card['wins']} / {card['wins_model']:.1f} / "
        f"{card['wins_price']:.1f} / {card['wins_close']:.1f} |",
        f"| Closing line value | {_cents(card['clv'])} a bet; beat the close on {card['beat_close']} of {card['closed']} |",
        f"| Brier: model / market at bet time / close | {card['brier_model']:.3f} / "
        f"{card['brier_market']:.3f} / {card['brier_close']:.3f} |",
        f"| Luck | P&L sd under the model ${card['sd']:,.2f}; the result is {card['z']:+.1f} sd from its EV |",
    ]
    return lines


def _game_cell(row: pd.Series) -> str:
    model = "" if pd.isna(row.get("model")) else f" ({row['model']:.0%})"
    close = "" if pd.isna(row.get("close")) else f" → {100 * row['close']:.0f}¢"
    return f"{row['bet_short']} {100 * row['avg_price']:.0f}¢{close}{model}"


def _theme_cell(row: pd.Series) -> str:
    if row.get("type_from") == "theme":
        return f"**{row['theme']}.** {row['why']}"
    return f"_pending_ ({', '.join(row['tags']) or 'no tags'})"


def _games(rows: pd.DataFrame) -> list[str]:
    lines = [
        "| Game | Bet @ price → close (model) | Final | P&L | Type | Box score | Theme |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in rows.sort_values("pnl", ascending=False).iterrows():
        kind = r["type"] if r["type_from"] == "theme" else f"{r['type']}?"
        lines.append(
            f"| {r['title']} | {_game_cell(r)} | {r['score']} | {_money(r['pnl'])} | "
            f"{kind} | {r['read']} | {_theme_cell(r)} |"
        )
    return lines


def report(ledger: pd.DataFrame, as_of: datetime) -> str:
    """The whole ledger as markdown: overall, by result type, by week, open."""
    settled = ledger[ledger["settled"]].copy()
    settled["won"] = settled["won"].astype(bool)
    open_ = ledger[~ledger["settled"]]
    card = scorecard(settled)
    weeks = sorted(settled["week"].unique(), reverse=True) if len(settled) else []
    out = [
        "# Kalshi ledger",
        "",
        f"Read {as_of.astimezone(CENTRAL):%Y-%m-%d %H:%M} CT: {len(ledger)} games bet, "
        f"{len(settled)} settled over {len(weeks)} week{'s' if len(weeks) != 1 else ''}, "
        f"{len(open_)} open.",
        "",
        "## Overall",
        "",
        *_card_lines(card),
    ]
    if len(settled):
        by_week = breakdown(settled, "week")
        by_week["group"] = by_week["group"].map(lambda d: f"{d:%b %d}")
        out += ["", "### By week", "", *_table(by_week, "week of")]

        out += [
            "",
            "## Result types",
            "",
            "One type per game: the written theme's, or a `?` guess from the box score "
            "until there is one.",
            "",
            *_table(
                breakdown(settled, "type").sort_values("n", ascending=False),
                "type",
                TYPES,
            ),
            "",
            "**Box-score read** -- did the side that outgained the other, in yards "
            f"(by {EVEN_YARDS}+) and yards a play (by {EVEN_YPP}+), win? Turnovers aside, so a "
            "lucky win is one the box score didn't earn.",
            "",
            *_table(breakdown(settled, "read"), "read"),
            "",
            "**Game flow** -- a game carries every tag that fits.",
            "",
            *_table(tag_table(settled), "tag"),
            "",
            "## Splits",
            "",
            *_table(
                breakdown(
                    settled,
                    settled["avg_price"].map(
                        lambda p: "favorite (paid ≥ 50¢)"
                        if p >= 0.5
                        else "underdog (< 50¢)"
                    ),
                ),
                "side",
            ),
            "",
            *_table(breakdown(settled, settled["divisions"].fillna("?")), "matchup"),
            "",
            *_table(
                breakdown(
                    settled,
                    settled["bet_home"].map({True: "home", False: "away"}).fillna("?"),
                ),
                "venue",
            ),
            "",
            *_table(
                breakdown(
                    settled,
                    (settled["model"] - settled["market"]).map(
                        lambda e: "?"
                        if pd.isna(e)
                        else "edge 5–10"
                        if e < 0.10
                        else "edge 10–15"
                        if e < 0.15
                        else "edge 15+"
                    ),
                ),
                "edge at bet time",
            ),
            "",
            *_table(
                breakdown(
                    settled,
                    (100 * (settled["close"] - settled["avg_price"])).map(
                        lambda c: "?"
                        if pd.isna(c)
                        else "beat the close"
                        if c > 0.5
                        else "level"
                        if c >= -0.5
                        else "close moved away"
                    ),
                ),
                "closing line",
            ),
        ]
    for week in weeks:
        rows = settled[settled["week"] == week]
        wcard = scorecard(rows)
        out += [
            "",
            f"## Week of {week:%b %d}: {wcard['wins']}–{wcard['losses']}, "
            f"{_money(wcard['pnl'])} on ${wcard['staked']:,.2f} ({_pct(wcard['roi'])})",
            "",
            f"Model expected {_money(wcard['model_ev'])}; the close {_money(wcard['close_ev'])}; "
            f"CLV {_cents(wcard['clv'])} a bet.",
            "",
            "**Wins**",
            "",
            *_games(rows[rows["won"]]),
            "",
            "**Losses**",
            "",
            *_games(rows[~rows["won"]]),
        ]
    if len(open_):
        staked = open_["stake"].sum()
        ev = (
            open_["contracts"] * (open_["model"] - open_["avg_price"]) - open_["fees"]
        ).sum()
        mark = (
            open_["contracts"] * (open_["mark"] - open_["avg_price"]) - open_["fees"]
        ).sum()
        out += [
            "",
            f"## Open: {len(open_)} games, ${staked:,.2f} staked",
            "",
            f"Model EV {_money(ev)}; marked to the no-vig mid now {_money(mark)}. "
            f"Moved our way on {int((open_['mark'] > open_['avg_price'] + 0.005).sum())} of {len(open_)}.",
            "",
            "| Kickoff | Game | Bet @ price (model) | Now | Move | Stake |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in open_.sort_values("kickoff").iterrows():
            kickoff = (
                "?"
                if pd.isna(r["kickoff"])
                else f"{pd.Timestamp(r['kickoff']).tz_convert(CENTRAL):%a %m-%d %H:%M}"
            )
            model = "" if pd.isna(r["model"]) else f" ({r['model']:.0%})"
            now = "–" if pd.isna(r["mark"]) else f"{100 * r['mark']:.0f}¢"
            move = (
                "–"
                if pd.isna(r["mark"])
                else _cents(100 * (r["mark"] - r["avg_price"]))
            )
            out.append(
                f"| {kickoff} | {r['title']} | {r['bet_short']} {100 * r['avg_price']:.0f}¢{model} | "
                f"{now} | {move} | ${r['stake']:,.2f} |"
            )
    out += [
        "",
        "## How it's read",
        "",
        "- Held, cost and paid are Kalshi's settlements and positions; fees are in the stake.",
        "- Model and market are the bet side's at bet time: the orders log's, or else the "
        "last picks run before the first fill. Market is no-vig.",
        "- The close is the no-vig mid of both teams' YES markets in the last hourly candle "
        "ending by kickoff; an open bet's mark is the same, now. CLV is close less price paid.",
        "- Themes are `~/.cassandra/betting/ledger/themes.toml`; `make ledger ARGS=--digest` "
        "lists the settled games still without one.",
        "",
    ]
    return "\n".join(out)


def digest(row: pd.Series, f: Mapping) -> str:
    """Everything a theme is written from, for one settled game."""
    lines = [
        f"=== {row['event']}",
        f"{row['title']}: bet {row['bet_short']} @ {row['avg_price']:.2f}"
        + ("" if pd.isna(row.get("model")) else f", model {row['model']:.2f}")
        + ("" if pd.isna(row.get("close")) else f", close {row['close']:.2f}")
        + f" | {row['divisions']} | {'WON' if row['won'] else 'LOST'} {f['score']} | P&L {_money(row['pnl'])}",
        f"line {f['line']} | WP start {f['wp_start']:.2f} peak {f['wp_peak']:.2f} low {f['wp_low']:.2f} "
        f"| half {f['half_margin']:+d}, biggest lead {f['largest_lead']}, deficit {f['largest_deficit']}, "
        f"winner went ahead for good {f['go_ahead']}",
        f"us:   {f['yards_for']} yds ({f['ypp_for']:.1f}/play), {f['rush_for']} rush, "
        f"{f['turnovers_for']} TO, 3rd {f['third_downs_for']} | {'; '.join(f['passers_for'])}",
        f"them: {f['yards_against']} yds ({f['ypp_against']:.1f}/play), {f['rush_against']} rush, "
        f"{f['turnovers_against']} TO, 3rd {f['third_downs_against']} | {'; '.join(f['passers_against'])}",
        f"tags: {', '.join(row['tags'])} | box score: {row['read']} | suggested type: {row['type']}",
        "scoring:",
        *(f"  {s}" for s in f["scoring"]),
        f"HEADLINE: {f['headline']}",
        f["recap"][:3000],
        "",
    ]
    return "\n".join(lines)


def now_utc() -> datetime:
    return datetime.now(UTC)

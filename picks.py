"""What to bet on Kalshi this week: the model against the live order books.

    python picks.py                    # refresh from s3, tonight's scores from ESPN
    python picks.py --offline          # no s3: the copies the last run saved
    python picks.py --budget 25 --top 8

Run it once the last of the night's games is final. That's when the
backtest bought, and the edge it measured was gone within a day (see
`cassandra.picks`). Each run does five things:

1. **Inputs.** It downloads the model's fit, the predictor data (EPA,
   quarterback index, roster measures) and every stored season from s3,
   and keeps a copy under `~/.cassandra/picks/`. `--offline` reads that copy
   instead, for when the AWS login has lapsed. The 2026 season is then
   topped up from ESPN directly (`--no-espn` to skip), since tonight's
   scores won't reach s3 until endgame's 08:00 pull. Tonight's games have
   no EPA until then either, so the model rates them on the score alone.
   Replayed on four 2026 weeks, that kept 101 of 116 picks and the same
   ROI (+19% vs +20%). The `no plays` column says which teams it applies
   to. Who played quarterback tonight is read straight off ESPN's plays,
   by the rule the sweep uses (`cassandra.qb_out_build`); on the games the
   stored index already had, that agreed on 42 of its 43 flags.
2. **Replay.** The model is replayed through every result, warmed on its
   search's own priors as `betting.py`'s replay is. The predictor is left
   where the season is now, and predicts the fixtures.
3. **Market.** It reads every open Kalshi college football game, matches
   each to its ESPN fixture, and takes each side's ask and size.
4. **Price.** It computes the backtest's edge, the cheapest way to take
   it, and Kalshi's fee. `cassandra.picks.skip_reason` says which games
   aren't picks.
5. **Size.** The `--top` picks by edge split `--budget`
   dollars by Kelly weight (`cassandra.picks.allocate`), and for each it
   reads the order book: how many contracts are offered within
   `--slippage` of the best price.

Every matched game, pick or not, goes to
`~/.cassandra/betting/picks/<league>_<model>_<when>.csv`, so a week's bets
can be graded against the close later.

A fixture assumes both starting quarterbacks play unless told otherwise:
`--qb-out TEAM` says a team's starter misses its next game, and the fit's
quarterback penalty (about 2 points in college) applies. That is the only
injury the model reads, so check the news on any pick you're about to
make; `--exclude TEAM` drops a game and re-splits the budget over the rest.
"""

import argparse
import asyncio
import gzip
import json
import logging
import pickle
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
from call_it_what_you_want import ESPN, KALSHI, NCAA, NCAAFB, TeamNamer, default_teams
from endgame.async_tools import apply_in_parallel
from endgame.football_plays import FootballLeague, get_game_plays
from endgame.ncaafb import get_season
from endgame.types import Game, Season, merge_weekly_seasons
from endgame_aws import Config
from endgame_aws.pbp_transform import transform_game_to_table

from cassandra.batch.artifacts import download, download_predictor_data
from cassandra.constants import CASSANDRA_HOME
from cassandra.exhibitions import without_exhibitions
from cassandra.model_eval import rebuild_priors
from cassandra.odds import OddsDatabase
from cassandra.picks import (
    EASTERN,
    MIN_EDGE,
    NCAAFB_SERIES,
    NOT_HAPPENING,
    Book,
    Fixture,
    allocate,
    entry,
    event_day,
    fillable,
    kalshi_games,
    kelly,
    match,
    max_price,
    order_fee,
    previous_games,
    price,
    routes,
    skip_reason,
)
from cassandra.predictor import OptimizationConfig, Predictor, load_predictor
from cassandra.predictor import frame as frames
from cassandra.predictor.adjustments import MatchupSources
from cassandra.predictor.config import load_predictor_class
from cassandra.predictor.epa import EpaIndex
from cassandra.predictor.opponent_prior import OpponentPriorManager
from cassandra.predictor.qb_out import QbOutIndex, load_qb_out
from cassandra.qb_out_build import build as build_qb_out
from cassandra.replay_cache import AUTHORED_DIR, GENERATED_DIR
from cassandra.residuals import team_tiers
from cassandra.save_predictions import join_with_odds, read_all_seasons

LEAGUE = "ncaafb"
DEFAULT_MODEL = "glicko_margin_units_roster"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
# Kalshi's basic tier allows about 20 requests a second; gold-rush paces
# itself at 8, and so does this.
_KALSHI_PACE = 1 / 8
CENTRAL = ZoneInfo("America/Chicago")
PICKS_DIR = CASSANDRA_HOME / "picks"
PLAYS_DIR = PICKS_DIR / "plays"
OUT_DIR = CASSANDRA_HOME / "betting" / "picks"


def _seasons_cache(league: str) -> Path:
    return PICKS_DIR / f"{league}_seasons.pkl"


async def _refresh_from_s3(league: str, model: str) -> list[Season]:
    bucket = Config.init_from_file().bucket
    fit = await download(bucket, f"models/{league}/{model}_result.json")
    if not fit:
        raise SystemExit(f"no fit for {league}/{model} in s3://{bucket}")
    await download_predictor_data(bucket)
    seasons = [season async for season in read_all_seasons(league, bucket)]
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    _seasons_cache(league).write_bytes(pickle.dumps(seasons))
    return seasons


def _from_cache(league: str) -> list[Season]:
    path = _seasons_cache(league)
    if not path.exists():
        raise SystemExit(f"{path} doesn't exist yet: run once without --offline")
    return pickle.loads(path.read_bytes())


class _OneWeek:
    """`qb_out_build.PlayWeeks` over plays fetched here, all filed as week 1.

    The build keys what it reads by game id, so one week holding every game
    reads the same as the store's real weeks would.
    """

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    async def load_week(self, league: str, season: int, week: int) -> pa.Table | None:
        return self._table if week == 1 else None


async def _game_plays(game_id: str) -> list[dict]:
    """A finished game's drives from ESPN, kept: they don't change after."""
    path = PLAYS_DIR / f"{game_id}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    drives = await get_game_plays(game_id, FootballLeague.ncaafb)
    if drives:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(drives).encode()))
    return drives


async def _qb_out_tonight(
    league: str, season: Season, games: list[Game], upcoming: list[Game]
) -> dict[str, set[str]]:
    """Who was missing their starter in the games the stored index hasn't read.

    That index is the sweep's, over the plays endgame has stored, and
    tonight's games aren't in it until the morning. They're read here the
    way the sweep will read them: `qb_out_build.build` over ESPN's plays.
    The rule needs a team's season to know who its starter is, so every
    finished game of a team with an unread one is fetched; only the unread
    games' flags are kept.

    Only teams with a game coming up, and with plays at all -- a team with
    no game in the EPA index has none on ESPN, which is most of D-II/III.
    """
    epa = EpaIndex.for_league(league)
    stored = {g.game_id for g in games if g.completed and epa.get(g.game_id)}
    with_plays = {t for g in games if g.game_id in stored for t in (g.home, g.away)}
    playing = {t for g in upcoming for t in (g.home, g.away)} & with_plays
    unread = {
        g.game_id
        for g in games
        if g.completed and g.game_id not in stored and {g.home, g.away} & playing
    }
    needy = {t for g in games if g.game_id in unread for t in (g.home, g.away)}
    needy &= playing
    wanted = sorted(
        {g.game_id for g in games if g.completed and {g.home, g.away} & needy}
    )
    if not wanted:
        return {}
    drives = [d async for d in apply_in_parallel(_game_plays, [(g,) for g in wanted])]
    table = pa.concat_tables(
        transform_game_to_table(d, g, league, season.year, 1)
        for g, d in zip(wanted, drives)
    )
    found = await build_qb_out(league, [season], _OneWeek(table))
    return {g: out & needy for g, out in found.items() if g in unread and out & needy}


def _qb_out_named(names: list[str], upcoming: list[Game]) -> dict[str, set[str]]:
    """`--qb-out` news: each named team out in its next game."""
    named: dict[str, set[str]] = {}
    playing = {t for g in upcoming for t in (g.home, g.away)}
    for name in names:
        teams = sorted(t for t in playing if t.lower() == name.lower()) or sorted(
            t for t in playing if name.lower() in t.lower()
        )
        if len(teams) != 1:
            raise SystemExit(
                f"--qb-out {name!r} matches {teams or 'no team with a game coming up'}"
            )
        game = min(
            (g for g in upcoming if teams[0] in (g.home, g.away)), key=lambda g: g.date
        )
        named.setdefault(game.game_id, set()).add(teams[0])
    return named


def _age(path: Path) -> str:
    if not path.exists():
        return "missing"
    when = datetime.fromtimestamp(path.stat().st_mtime, CENTRAL)
    return f"{when:%a %b %d %H:%M} CT"


async def _with_espn(seasons: list[Season]) -> list[Season]:
    """The latest season topped up from ESPN, as endgame's own pull merges it."""
    latest = max(seasons, key=lambda s: s.year)
    fresh = await get_season(latest.year, use_cache=False, include_unplayed=True)
    merged = merge_weekly_seasons([latest, fresh])
    return [merged if s is latest else s for s in seasons]


def _replay(
    league: str, model: str, seasons: list[Season], sources: MatchupSources
) -> tuple[Predictor, pd.DataFrame]:
    """The model through every result, warmed the way `betting.py`'s replay is."""
    result = GENERATED_DIR / league / f"{model}_result.json"
    search = AUTHORED_DIR / league / f"{model}.json"
    odds = OddsDatabase({})
    overrides: dict = {"sources": sources}
    if search.exists():
        config = OptimizationConfig.model_validate_json(search.read_text())
        weeks = frames.weeks_per_season([len(season.weeks) for season in seasons])
        # A private priors file: the shared one is rebuilt by whatever else
        # is replaying on this machine.
        priors = PICKS_DIR / f"{league}_{model}_priors.json"
        if rebuild_priors(
            load_predictor_class(config.predictor_class),
            league,
            frames.to_params(config.frame, config.fixed, weeks),
            seasons,
            odds,
            priors_path=priors,
        ):
            overrides["opponent_prior_manager"] = OpponentPriorManager(
                league, path=priors
            )
    predictor = load_predictor(result, **overrides)
    predictions = join_with_odds(
        predictor, seasons, odds, post_callbacks=False, roll_over_final_season=False
    )
    return predictor, pd.DataFrame([asdict(p) for p in predictions])


def _kalshi_get(path: str, params: dict | None = None) -> dict:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(
        f"{KALSHI_API}{path}{query}", headers={"User-Agent": "cassandra-picks"}
    )
    time.sleep(_KALSHI_PACE)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _kalshi_markets(series: tuple[str, ...]) -> list[dict]:
    markets: list[dict] = []
    for ticker in series:
        cursor = None
        while True:
            params = {"series_ticker": ticker, "status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            page = _kalshi_get("/markets", params)
            markets += page.get("markets") or []
            cursor = page.get("cursor")
            if not cursor or not page.get("markets"):
                break
    return markets


def _fmt_game(game: Game) -> str:
    joiner = "vs" if game.neutral_site else "@"
    return f"{game.away} {joiner} {game.home}"


def main() -> None:
    # The transform clamps the odd out-of-range yardline, which says nothing
    # about who played quarterback.
    logging.getLogger("endgame_aws.pbp_transform").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--days", type=float, default=8, help="price games kicking off this far ahead"
    )
    parser.add_argument(
        "--min-edge", type=float, default=MIN_EDGE, help="edge over the no-vig price"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=20.0,
        help="dollars across every pick, fees included",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="bet only this many picks, the biggest edges",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.01,
        help="how far past the best price to count contracts on offer",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TEAM",
        help="drop games naming this team, e.g. on injury news (repeatable)",
    )
    parser.add_argument(
        "--qb-out",
        action="append",
        default=[],
        metavar="TEAM",
        help="this team's starting QB is out of its next game (repeatable)",
    )
    parser.add_argument(
        "--offline", action="store_true", help="skip s3; read the last run's copies"
    )
    parser.add_argument(
        "--no-espn", action="store_true", help="don't top the season up from ESPN"
    )
    parser.add_argument(
        "--all", action="store_true", help="print every matched game, not just picks"
    )
    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    started = datetime.now(UTC)
    league, model = LEAGUE, args.model

    # 1. Inputs.
    if args.offline:
        seasons = _from_cache(league)
    else:
        seasons = asyncio.run(_refresh_from_s3(league, model))
    data = CASSANDRA_HOME / "predictor" / "data"
    print(f"fit:            {_age(GENERATED_DIR / league / f'{model}_result.json')}")
    print(f"seasons (s3):   {_age(_seasons_cache(league))}")
    print(f"EPA index:      {_age(data / f'{league}_epa.json')}")
    print(f"QB index:       {_age(data / f'{league}_qb_out.json')}")
    if not args.no_espn:
        seasons = asyncio.run(_with_espn(seasons))
        print(
            f"latest season:  topped up from ESPN at {datetime.now(CENTRAL):%H:%M} CT"
        )
    seasons = without_exhibitions(seasons, league)
    namer = TeamNamer.for_league(league)
    latest = max(seasons, key=lambda s: s.year)
    games = [namer.apply(g) for week in latest.weeks for g in week.games]
    now = datetime.now(UTC)
    horizon = now + timedelta(days=args.days)

    # The quarterbacks: the stored index, tonight's games read off ESPN's
    # plays as the sweep will read them, and `--qb-out` news for fixtures.
    upcoming = [
        g
        for g in games
        if not g.completed and g.status not in NOT_HAPPENING and now < g.date < horizon
    ]
    tonight = asyncio.run(_qb_out_tonight(league, latest, games, upcoming))
    named = _qb_out_named(args.qb_out, upcoming)
    flags = {game: set(out) for game, out in load_qb_out(league).items()}
    for extra in (tonight, named):
        for game, out in extra.items():
            flags.setdefault(game, set()).update(out)
    by_id = {g.game_id: g for g in games}
    for label, extra in (("QB out tonight", tonight), ("QB out next game", named)):
        for game, out in sorted(extra.items(), key=lambda item: by_id[item[0]].date):
            print(
                f"{label + ':':<16}{', '.join(sorted(out))}  ({_fmt_game(by_id[game])})"
            )
    sources = MatchupSources.for_league(league)._replace(qb_out=QbOutIndex(flags))

    # 2. Replay.
    clock = time.time()
    predictor, df = _replay(league, model, seasons, sources)
    print(f"replayed {len(df):,} games in {time.time() - clock:.0f}s")

    # 3. Fixtures and the market.
    previous = previous_games(games)
    now = datetime.now(UTC)
    # Every unplayed game is matched against, so a Kalshi game near the end
    # of the window still finds its fixture; only the window's are priced.
    fixtures = [
        Fixture(g, namer.espn_id(g.home), namer.espn_id(g.away))
        for g in games
        if not g.completed and g.status not in NOT_HAPPENING and now < g.date
    ]
    unfinished = [
        g
        for g in games
        if not g.completed and g.status not in NOT_HAPPENING and g.date <= now
    ]
    teams = default_teams(NCAA)

    def side_ids(side: Book) -> frozenset[str]:
        found = {
            t.espn_id
            for spelling in (side.code, side.team)
            if spelling
            for t in teams.find(spelling, source=KALSHI, league=NCAAFB)
        }
        if not found and side.team:
            found = {
                t.espn_id for t in teams.find(side.team, source=ESPN, league=NCAAFB)
            }
        return frozenset(found)

    markets = _kalshi_markets(NCAAFB_SERIES)
    kalshi, odd = kalshi_games(markets)
    # Today's events stay listed until they settle, and the ones under way
    # have no fixture left to match; from tomorrow (Eastern) on, a Kalshi
    # game with no fixture is a matching miss worth seeing.
    tomorrow = now.astimezone(EASTERN).date() + timedelta(days=1)
    in_window = [k for k in kalshi if k.day <= horizon.date()]
    matched, unmatched = match(fixtures, in_window, side_ids)
    missed = [e for e in unmatched if (event_day(e) or tomorrow) >= tomorrow]
    priced_ids = {f.game.game_id for f in fixtures if f.game.date < horizon}
    print(
        f"{len(priced_ids)} fixtures in the next {args.days:g} days, "
        f"{len(priced_ids & matched.keys())} of them on Kalshi"
        + (
            f"; {len(missed)} Kalshi games matched no fixture: {', '.join(missed)}"
            if missed
            else ""
        )
    )
    # Only the unfinished games a Kalshi game is waiting on are worth a line;
    # on a Saturday most of the rest are D-III games nobody lists.
    on_kalshi = {
        team
        for f in fixtures
        if f.game.game_id in priced_ids & matched.keys()
        for team in (f.game.home, f.game.away)
    }
    holding = [g for g in unfinished if {g.home, g.away} & on_kalshi]
    if holding:
        print(
            f"\n!! {len(holding)} games have kicked off and aren't final "
            "-- their teams' Kalshi games are held back:"
        )
        for g in holding:
            print(f"   {g.away} @ {g.home}  ({g.status or 'no status'})")
    if len(unfinished) > len(holding):
        print(
            f"   {len(unfinished) - len(holding)} other games aren't final, "
            "with no Kalshi game waiting on them"
        )

    # 4. Price.
    tier = team_tiers(df, league)
    epa = EpaIndex.for_league(league)
    rows = []
    year = latest.year

    def division(team: str) -> str | None:
        found_tier = tier(team, year) if tier else None
        return None if found_tier is None else found_tier[0]

    for fixture in fixtures:
        found = matched.get(fixture.game.game_id)
        if found is None or fixture.game.game_id not in priced_ids:
            continue
        game = fixture.game
        model_home = predictor.predict_game(game).team1_win_prob
        priced = price(model_home, found)
        if priced is None:
            continue
        side, other = (game.home, game.away) if priced.home else (game.away, game.home)
        last = {team: previous(team, game.date) for team in (game.home, game.away)}
        waiting = [t for t, p in last.items() if p is not None and not p.completed]
        entered = entry(last.values())
        since_entry = None if entered is None else now - entered
        no_plays = [
            t
            for t, p in last.items()
            if p is not None and p.completed and epa.get(p.game_id) is None
        ]

        reason = skip_reason(
            priced, division(side), division(other), waiting, since_entry, args.min_edge
        )
        book = found.home if priced.home else found.away
        route = priced.route
        model_side = model_home if priced.home else 1 - model_home
        rows.append(
            {
                "kickoff": game.date.astimezone(CENTRAL),
                "game": _fmt_game(game),
                "game_id": game.game_id,
                "event": found.kalshi.event,
                "bet": side,
                "bet_home": priced.home,
                "divisions": f"{division(side) or '?'} v {division(other) or '?'}",
                "model": model_side,
                "market": priced.home_market if priced.home else 1 - priced.home_market,
                "edge": priced.edge,
                "overround": priced.overround,
                "buy": None
                if route is None
                else f"{route.action.upper()} {route.ticker}",
                "price": None if route is None else route.price,
                "max_price": max_price(model_side, priced, book, args.min_edge),
                "cost": priced.cost,
                "ev": priced.expected_return,
                "size_at_price": None if route is None else route.size,
                "spread": book.spread,
                "volume_24h": book.volume_24h,
                "no_plays": ", ".join(no_plays),
                "hours_since_entry": None
                if since_entry is None
                else since_entry.total_seconds() / 3600,
                "skip": reason,
                "home_model": model_home,
                "home_ask": found.home.ask,
                "away_ask": found.away.ask,
                "home_bid": found.home.bid,
                "away_bid": found.away.bid,
                "read_at": now,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("\nno priced games")
        return

    # 5. Size the picks: the top `--top` by edge share `--budget`
    # by Kelly weight (see `allocate`).
    for team in args.exclude:
        named = frame["game"].str.contains(team, case=False, regex=False)
        frame.loc[named & frame["skip"].isna(), "skip"] = "excluded by hand"
    ranked = frame[frame["skip"].isna()].sort_values("edge", ascending=False)
    frame.loc[ranked.index[args.top :], "skip"] = f"outside the top {args.top} by edge"
    picks = frame[frame["skip"].isna()]
    weights = [kelly(m, c) for m, c in zip(picks["model"], picks["cost"])]
    counts = allocate(args.budget, list(picks["price"]), weights)
    unplaced = [i for i, n in zip(picks.index, counts) if n == 0]
    frame.loc[unplaced, "skip"] = "less than a contract's share of the budget"
    picks = frame[frame["skip"].isna()].copy()
    contracts = [n for n in counts if n > 0]
    fills = []
    for _, row in picks.iterrows():
        found = matched[row["game_id"]]
        side, other = (
            (found.home, found.away) if row["bet_home"] else (found.away, found.home)
        )
        # Both routes to the same side count: YES on it and NO on the
        # opponent fill on the same result.
        limit = row["price"] + args.slippage
        on_offer = 0.0
        for route in routes(side, other):
            book = _kalshi_get(f"/markets/{route.ticker}/orderbook")
            on_offer += fillable(book, route.action, limit)
        fills.append(on_offer)
    picks["contracts"] = contracts
    picks["fee"] = [order_fee(p, c) for p, c in zip(picks["price"], contracts)]
    picks["outlay"] = picks["contracts"] * picks["price"] + picks["fee"]
    picks[f"offered_within_{round(args.slippage * 100)}c"] = fills
    frame = frame.merge(
        picks[
            [
                "game_id",
                "contracts",
                "fee",
                "outlay",
                f"offered_within_{round(args.slippage * 100)}c",
            ]
        ],
        on="game_id",
        how="left",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{league}_{model}_{started.astimezone(CENTRAL):%Y%m%d-%H%M}.csv"
    frame.to_csv(out, index=False)

    _print(frame, picks, args)
    print(f"\nevery matched game: {out}")


def _print(frame: pd.DataFrame, picks: pd.DataFrame, args: argparse.Namespace) -> None:
    offered = f"offered_within_{round(args.slippage * 100)}c"
    shown = (frame if args.all else frame[frame["skip"].isna()]).sort_values(
        ["kickoff", "edge"], ascending=[True, False]
    )
    columns = {
        "kickoff": lambda s: s.dt.strftime("%a %H:%M"),
        "game": None,
        "bet": None,
        "divisions": None,
        "model": lambda s: s.map("{:.3f}".format),
        "market": lambda s: s.map("{:.3f}".format),
        "edge": lambda s: s.map("{:+.3f}".format),
        "buy": None,
        "price": lambda s: s.map("{:.2f}".format),
        "max_price": lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
        "ev": lambda s: s.map("{:+.1%}".format),
        "contracts": lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:.0f}"),
        "outlay": lambda s: s.map(lambda v: "" if pd.isna(v) else f"${v:.2f}"),
        offered: lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:,.0f}"),
        "spread": lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v * 100:.0f}c"),
        "hours_since_entry": lambda s: s.map(
            lambda v: "" if pd.isna(v) else f"{v:.0f}h"
        ),
        "no_plays": None,
    }
    if args.all:
        columns["skip"] = None
    table = pd.DataFrame(
        {
            name: (shown[name] if fmt is None else fmt(shown[name]))
            for name, fmt in columns.items()
        }
    )
    print(
        f"\n=== {len(picks)} picks (edge ≥ {args.min_edge:.0%} over Kalshi's no-vig "
        f"price), ${args.budget:g} split by Kelly weight over the top {args.top}"
    )
    if not table.empty:
        with pd.option_context("display.width", 250, "display.max_colwidth", 60):
            print(table.to_string(index=False))
    if not picks.empty:
        print(
            f"\ntotal outlay ${picks['outlay'].sum():,.2f} on {len(picks)} picks; "
            f"model's expected return {(picks['ev'] * picks['outlay']).sum() / picks['outlay'].sum():+.1%} "
            "(the backtest's realized ROI was about 80% of what the model expected)"
        )
        dogs = (picks["price"] < 0.5).sum()
        print(f"{dogs} underdogs, {len(picks) - dogs} favorites")
        short = picks[picks[offered] < picks["contracts"]]
        if not short.empty:
            print(
                f"{len(short)} won't fill at the price (fewer contracts within "
                f"{round(args.slippage * 100)}c than the order): "
                + ", ".join(
                    f"{bet} {on_offer:.0f}/{wanted:.0f}"
                    for bet, on_offer, wanted in zip(
                        short["bet"], short[offered], short["contracts"]
                    )
                )
            )
    # The reason without its detail (who it's waiting on, how wide the book
    # is), so each reason is counted once.
    skipped = (
        frame["skip"].dropna().str.replace(r"\s*[:(].*", "", regex=True).value_counts()
    )
    if not skipped.empty:
        print("\nnot picks: " + ", ".join(f"{n} {why}" for why, n in skipped.items()))


if __name__ == "__main__":
    main()

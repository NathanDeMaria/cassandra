"""What to bet on Kalshi this week: the model against the live order books.

    python picks.py                    # refresh from s3, tonight's scores from ESPN
    python picks.py --offline          # no s3: the copies the last run saved
    python picks.py --budget 25 --top 8
    python picks.py --place            # ...then buy them, after asking

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
5. **Size.** The `--top` picks by edge with something left to buy -- a
   pick already held in full doesn't count -- are sized (see "Sizing"),
   and for each it reads the order book: how many contracts are offered
   within `--slippage` of the best price.

Every matched game, pick or not, goes to
`~/.cassandra/betting/picks/<league>_<model>_<when>.csv`, so a week's bets
can be graded against the close later.

A fixture assumes both starting quarterbacks play unless told otherwise:
`--qb-out TEAM` says a team's starter misses its next game, and the fit's
quarterback penalty (about 2 points in college) applies. That is the only
injury the model reads, so check the news on any pick you're about to
make. When one of the last 18 hours' games has no plays on ESPN yet,
whether its teams' starters played is unknown, and their next games wait
for a later run ("QB check incomplete"). `--exclude TEAM` drops a game.
Under the orders, each pick lists the injury sentences in its two teams'
last two AP stories, and the next game's preview once there is one
(`cassandra.injuries`); `QB!` marks one naming a quarterback.

**Placing.** `--place` buys the picks, with a Kalshi API key (see
`cassandra.kalshi_trade` for where it's read from; `--demo` for the demo
exchange). It first reads the account: its cash, its open positions and
its resting orders. A `--budget` over the cash on hand stops the run. Then
it prints the numbered orders and asks: `y` places them, numbers drop those
games and re-size the rest, `q` stops. A drop never brings in a game that
wasn't on the first list, even when it frees a place in the `--top`.

**Sizing.** Each pick has a target number of contracts:

- with `--kelly-fraction F`, F of the Kelly stake on the bankroll -- the
  account's cash plus what its open positions are worth, or `--bankroll`.
  Each pick is sized on its own, so a run every half hour sizes a pick the
  same whichever run finds it, and there's no need to know how many picks
  the week holds;
- without it, a Kelly-weighted share of `--budget` (default $20) over this
  run's picks, as the backtest's weekly bet was sized.

A pick's target is at most `--max-bet`. Contracts already held on its side
count toward it and only the rest is bought, so a partial fill is topped up
on a later run while the pick still clears the edge. A game held on the
other side, or with a resting order, is skipped. If the run's orders would
cost more than `--budget` or the cash on hand, they're cut down together.

Each order is a limit order:

- its price is at most `--slippage` past the ask, and never past
  `max_price`, the most that still clears `--min-edge`;
- its contracts are counted at that price, so the worst fill costs no more
  than the pick's target;
- it's immediate-or-cancel, so what isn't on offer now is dropped (books
  this thin rarely hold a whole order). `--rest HOURS` leaves the rest at
  the limit until then instead -- the backtest's edge was mostly gone by
  twelve hours.

The cheapest route is tried first and what it left goes to the other one.
What filled is written to `~/.cassandra/betting/orders/`.

To rehearse without money, step by step:
- no `--place`: no key, nothing sent;
- `--place --dry-run`: the real key reads the balance and what's held,
  then the orders are printed rather than sent. The client refuses any
  request but a GET, and with a read-only key Kalshi would too;
- `--place --demo`: the whole thing on Kalshi's demo exchange, with fake
  money. It lists the FBS games under the same tickers but its books are
  empty, so an order is accepted and fills nothing (FCS games aren't
  there, and those orders error). With `--rest 1` the orders rest instead,
  and a rerun should skip those games as already held.
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
from concurrent.futures import ThreadPoolExecutor
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

from cassandra import injuries
from cassandra.batch.artifacts import download, download_predictor_data
from cassandra.constants import CASSANDRA_HOME
from cassandra.exhibitions import without_exhibitions
from cassandra.kalshi_trade import KalshiClient, KalshiError
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
    contracts_for,
    entry,
    event_day,
    fillable,
    fit_to_cash,
    held_on,
    kalshi_games,
    kelly,
    kelly_counts,
    match,
    max_price,
    order_fee,
    previous_games,
    price,
    routes,
    skip_reason,
    top_up,
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
DEFAULT_MODEL = "glicko_margin_units_success"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
# Kalshi's basic tier allows about 20 requests a second; gold-rush paces
# itself at 8, and so does this.
_KALSHI_PACE = 1 / 8
CENTRAL = ZoneInfo("America/Chicago")
PICKS_DIR = CASSANDRA_HOME / "picks"
PLAYS_DIR = PICKS_DIR / "plays"
OUT_DIR = CASSANDRA_HOME / "betting" / "picks"
ORDERS_DIR = CASSANDRA_HOME / "betting" / "orders"
SUMMARIES_DIR = PICKS_DIR / "summaries"
ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
)
#: How many of each team's last games' stories are read for injury news.
NEWS_GAMES = 2
#: How long after kickoff a game ESPN has no plays for is taken to be
#: missing them for now, not for good. Its teams' next games wait.
QB_PLAYS_GRACE = timedelta(hours=18)
#: Dollars split over a run's picks when neither --budget nor
#: --kelly-fraction says otherwise.
DEFAULT_BUDGET = 20.0


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
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Who was missing their starter in the games the stored index hasn't read,
    and the recent games that couldn't be read: no plays from ESPN yet.

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
        return {}, {}
    drives = [d async for d in apply_in_parallel(_game_plays, [(g,) for g in wanted])]
    # A game that ended in the last few hours with no plays from ESPN may
    # just not have them yet; who played quarterback in it is unknown, not
    # known to be the starter. Older ones without plays are games ESPN
    # never covered play by play.
    by_id = {g.game_id: g for g in games}
    recent = datetime.now(UTC) - QB_PLAYS_GRACE
    blind = {
        g: {by_id[g].home, by_id[g].away} & needy
        for g, d in zip(wanted, drives)
        if not d and g in unread and by_id[g].date > recent
    }
    table = pa.concat_tables(
        transform_game_to_table(d, g, league, season.year, 1)
        for g, d in zip(wanted, drives)
    )
    found = await build_qb_out(league, [season], _OneWeek(table))
    flags = {g: out & needy for g, out in found.items() if g in unread and out & needy}
    print(
        f"QB check:       {len(unread) - len(blind)} of {len(unread)} unread games "
        "read off ESPN's plays"
    )
    return flags, blind


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
        help="most dollars this run spends, fees included; split over the "
        f"picks by Kelly weight unless --kelly-fraction (default ${DEFAULT_BUDGET:g})",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        help="stake each pick on its own: this fraction of Kelly on the bankroll "
        "(the account's cash plus open positions with --place), e.g. 0.25",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        help="with --kelly-fraction: the bankroll to size on, instead of the account's",
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
    parser.add_argument(
        "--max-bet",
        type=float,
        default=5.0,
        help="most dollars on any one pick, fee included",
    )
    parser.add_argument(
        "--place",
        action="store_true",
        help="after showing the orders and asking, buy them on Kalshi",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="with --place: Kalshi's demo exchange and its key",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --place: read the account with the real key, then print "
        "the orders instead of sending them",
    )
    parser.add_argument(
        "--rest",
        type=float,
        metavar="HOURS",
        help="leave what doesn't fill resting at the limit this long "
        "(default: immediate-or-cancel)",
    )
    parser.add_argument(
        "--no-news", action="store_true", help="skip the injury read of ESPN's stories"
    )
    parser.add_argument(
        "--news-lines",
        type=int,
        default=4,
        help="injury sentences shown per pick",
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
    tonight, blind = asyncio.run(_qb_out_tonight(league, latest, games, upcoming))
    unread_teams = {t for teams in blind.values() for t in teams}
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
    # Most games ESPN has no plays for are D-II/III games it never covers
    # play by play; only the ones a Kalshi game is waiting on are worth a line.
    for game in sorted(blind, key=lambda g: by_id[g].date):
        if blind[game] & on_kalshi:
            print(
                f"!! no plays from ESPN yet for {_fmt_game(by_id[game])}: "
                f"{', '.join(sorted(blind[game] & on_kalshi))}'s next game waits "
                "for a run that can read who played quarterback"
            )
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
        unknown = sorted({game.home, game.away} & unread_teams)
        if reason is None and unknown:
            reason = f"QB check incomplete: {', '.join(unknown)}"
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

    # 5. Size the picks (`_size`), each at its limit price.
    client = None
    frame["held"] = 0.0
    args.cash = None
    if args.place:
        client = KalshiClient.from_env(demo=args.demo, dry_run=args.dry_run)
        cash, invested = client.account()
        positions = client.positions()
        resting = client.resting_events()
        args.cash = cash
        if args.kelly_fraction is not None and args.bankroll is None:
            args.bankroll = cash + invested
        mode = (" DEMO" if args.demo else "") + (" (dry run)" if args.dry_run else "")
        print(
            f"\nKalshi{mode}: ${cash:,.2f} cash, ${invested:,.2f} in "
            f"{len(positions)} open positions, {len(resting)} events with resting orders"
        )
        if args.budget is not None and args.budget > cash:
            raise SystemExit(
                f"--budget ${args.budget:g} is more than the ${cash:,.2f} on hand"
            )
        for i, row in frame.iterrows():
            found = matched[row["game_id"]]
            side, other = (
                (found.home, found.away)
                if row["bet_home"]
                else (found.away, found.home)
            )
            mine, against = held_on(positions, side, other)
            frame.at[i, "held"] = mine
            if not pd.isna(row["skip"]):
                continue
            if row["event"] in resting:
                frame.at[i, "skip"] = "resting order on Kalshi"
            elif against:
                frame.at[i, "skip"] = f"held on the other side ({against:g})"
    if args.kelly_fraction is not None and args.bankroll is None:
        raise SystemExit("--kelly-fraction without --place needs --bankroll")
    for team in args.exclude:
        named = frame["game"].str.contains(team, case=False, regex=False)
        frame.loc[named & frame["skip"].isna(), "skip"] = "excluded by hand"
    frame["limit"] = [
        _limit(p, m, args.slippage) for p, m in zip(frame["price"], frame["max_price"])
    ]
    priced_skips = frame["skip"].copy()
    dropped: set[str] = set()
    # The games on the first list asked about. Dropping some frees places
    # in the top `--top`, and a game nobody has read the news on shouldn't
    # take one: answering is about the list on the screen.
    listed: set[str] | None = None
    news: dict[str, list[injuries.Note]] = {}
    out = OUT_DIR / f"{league}_{model}_{started.astimezone(CENTRAL):%Y%m%d-%H%M}.csv"
    while True:
        frame["skip"] = priced_skips
        frame.loc[frame["game_id"].isin(dropped), "skip"] = "dropped at the prompt"
        if listed is not None:
            unlisted = ~frame["game_id"].isin(listed) & frame["skip"].isna()
            frame.loc[unlisted, "skip"] = "not on the list answered"
        sized, picks = _size(frame, matched, args)
        if listed is None:
            listed = set(picks["game_id"])
        if not args.no_news:
            fresh = picks[~picks["game_id"].isin(news.keys())]
            news.update(_injury_news(fresh, games, namer))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        sized.to_csv(out, index=False)
        _print(sized, picks, args)
        _print_orders(picks, news, args)
        print(f"\nevery matched game: {out}")
        if client is None or picks.empty:
            return
        answer = _ask(len(picks))
        if answer is None:
            print("nothing placed")
            return
        if not answer:
            break
        order = picks.sort_values(["kickoff", "edge"], ascending=[True, False])
        dropped |= {order["game_id"].iloc[i - 1] for i in answer}

    _place(client, picks, matched, args, out)


def _sizing(args: argparse.Namespace) -> str:
    """How this run sizes its bets, in a line."""
    if args.kelly_fraction is not None:
        found = (
            f"{args.kelly_fraction:g} Kelly on a ${args.bankroll:,.2f} bankroll, "
            f"at most ${args.max_bet:g} a bet"
        )
    else:
        found = (
            f"${_budget(args):g} split by Kelly weight, at most ${args.max_bet:g} a bet"
        )
    if args.kelly_fraction is not None and args.budget is not None:
        found += f", ${args.budget:g} this run"
    if args.cash is not None:
        found += f", ${args.cash:,.2f} cash"
    return found


def _budget(args: argparse.Namespace) -> float:
    """`--budget`, or the old default of $20 when splitting with no fraction."""
    return DEFAULT_BUDGET if args.budget is None else args.budget


def _limit(price: float, ceiling: float | None, slippage: float) -> float:
    """The most an order pays a contract: `--slippage` past the ask, never past
    the price that still clears the edge (`max_price`)."""
    limit = price + slippage
    if ceiling is not None and not pd.isna(ceiling):
        limit = min(limit, max(ceiling, price))
    return round(min(limit, 0.99), 2)


def _targets(picks: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    """Each pick's contracts before what's held: Kelly on the bankroll, or a
    Kelly-weighted share of the budget. Either way at most `--max-bet`."""
    limits = list(picks["limit"])
    if args.kelly_fraction is not None:
        return kelly_counts(
            list(picks["model"]),
            list(picks["cost"]),
            limits,
            args.bankroll,
            args.kelly_fraction,
            args.max_bet,
        )
    weights = [kelly(m, c) for m, c in zip(picks["model"], picks["cost"])]
    return [
        min(n, contracts_for(args.max_bet, limit))
        for n, limit in zip(allocate(_budget(args), limits, weights), limits)
    ]


def _size(
    frame: pd.DataFrame, matched: dict, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The picks and their contracts, and every game with what was bet on it.

    Each pick gets a target: `--kelly-fraction` of Kelly on the bankroll,
    or with no fraction its Kelly-weighted share of `--budget`. Either way
    it's at most `--max-bet`, counted at the limit price, so an order
    filling at its worst still costs no more than that. What's already held
    on the pick's side counts toward its target, and only the rest is
    bought. The whole run's worst case then fits `--budget` and the cash.
    """
    frame = frame.copy()
    # The top `--top` are taken from the games with something left to buy,
    # so a pick already held in full doesn't take a place a new one could.
    # Under a split budget a pick's target depends on which others share
    # it, so the cut is retaken until none in it is held in full.
    ranked = frame[frame["skip"].isna()].sort_values("edge", ascending=False)
    while True:
        top = ranked.iloc[: args.top]
        targets = pd.Series(_targets(top, args), index=top.index, dtype=int)
        counts = pd.Series(
            [top_up(t, h) for t, h in zip(targets, top["held"])], index=top.index
        )
        full = top.index[(top["held"] > 0) & (counts == 0)]
        if full.empty:
            break
        for i in full:
            frame.at[i, "skip"] = (
                f"already held ({frame.at[i, 'held']:g} of {targets[i]})"
            )
        ranked = ranked.drop(full)
    frame.loc[ranked.index[args.top :], "skip"] = f"outside the top {args.top} by edge"
    caps = [c for c in (args.budget, args.cash) if c is not None]
    if caps:
        counts[:] = fit_to_cash(list(counts), list(top["limit"]), min(caps))
    for i in top.index[counts == 0]:
        frame.at[i, "skip"] = (
            "no cash left for it" if targets[i] else "less than a contract's stake"
        )
    picks = frame[frame["skip"].isna()].copy()
    contracts = list(counts[picks.index])
    offered = f"offered_within_{round(args.slippage * 100)}c"
    fills = []
    for _, row in picks.iterrows():
        found = matched[row["game_id"]]
        side, other = (
            (found.home, found.away) if row["bet_home"] else (found.away, found.home)
        )
        # Both routes to the same side count: YES on it and NO on the
        # opponent fill on the same result, and `_place` tries both.
        on_offer = 0.0
        for route in routes(side, other):
            book = _kalshi_get(f"/markets/{route.ticker}/orderbook")
            on_offer += fillable(book, route.action, row["limit"])
        fills.append(on_offer)
    picks["contracts"] = contracts
    picks["fee"] = [order_fee(p, c) for p, c in zip(picks["limit"], contracts)]
    picks["outlay"] = picks["contracts"] * picks["limit"] + picks["fee"]
    picks[offered] = fills
    frame = frame.merge(
        picks[["game_id", "contracts", "fee", "outlay", offered]],
        on="game_id",
        how="left",
    )
    return frame, picks


def _espn_summary(game_id: str, final: bool) -> dict:
    """ESPN's summary of a game: box score and AP story. A final one is kept."""
    path = SUMMARIES_DIR / f"{game_id}.json.gz"
    if final and path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    request = urllib.request.Request(
        f"{ESPN_SUMMARY}?event={game_id}", headers={"User-Agent": "cassandra-picks"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            summary = json.load(response)
    except OSError as error:
        logging.warning("no ESPN summary for %s: %s", game_id, error)
        return {}
    # A recap can land after the final whistle, so don't keep a summary
    # without one.
    if final and (summary.get("article") or {}).get("story"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(summary).encode()))
    return summary


def _injury_news(
    picks: pd.DataFrame, games: list[Game], namer: TeamNamer
) -> dict[str, list[injuries.Note]]:
    """Injury sentences on each pick's two teams, from their last two games' stories.

    The upcoming game's preview is read too, once ESPN has one.
    """
    by_id = {g.game_id: g for g in games}
    wanted: dict[str, list[tuple[str, bool, str]]] = {}
    for game_id in picks["game_id"]:
        fixture = by_id[game_id]
        stories = [(fixture.game_id, False, "preview")]
        for team in (fixture.home, fixture.away):
            played = sorted(
                (g for g in games if g.completed and team in (g.home, g.away)),
                key=lambda g: g.date,
            )[-NEWS_GAMES:]
            for g in reversed(played):
                when = g.date.astimezone(CENTRAL)
                stories.append((g.game_id, True, f"{_fmt_game(g)}, {when:%b %d}"))
        wanted[game_id] = list(dict.fromkeys(stories))
    calls = {(g, final) for stories in wanted.values() for g, final, _ in stories}
    with ThreadPoolExecutor(8) as pool:
        summaries = dict(zip(calls, pool.map(lambda c: _espn_summary(*c), calls)))
    news: dict[str, list[injuries.Note]] = {}
    for game_id, stories in wanted.items():
        fixture = by_id[game_id]
        teams = {
            namer.espn_id(t): t
            for t in (fixture.home, fixture.away)
            if namer.espn_id(t)
        }
        read = [summaries[(g, final)] for g, final, _ in stories]
        # The pick's own teams first, so a sentence naming one of theirs and
        # an opponent's is theirs.
        rosters = injuries.rosters(read, teams) + injuries.opponents(read, teams)
        found: dict[str, injuries.Note] = {}
        for summary, (_, _, label) in zip(read, stories):
            got = injuries.story(summary)
            if got is None:
                continue
            kind, text = got
            for note in injuries.notes(text, rosters, f"{kind}: {label}"):
                # News of last week's opponent isn't news of this game.
                if note.team is None or note.team in teams.values():
                    found.setdefault(note.sentence, note)
        news[game_id] = list(found.values())
    return news


def _print_orders(
    picks: pd.DataFrame,
    news: dict[str, list[injuries.Note]],
    args: argparse.Namespace,
) -> None:
    """The orders `--place` would send, numbered, with the injury news under each."""
    if picks.empty:
        return
    tif = (
        "immediate-or-cancel"
        if args.rest is None
        else f"resting {args.rest:g}h, then cancelled"
    )
    print(f"\n=== {'orders' if args.place else 'orders --place would send'} ({tif})")
    order = picks.sort_values(["kickoff", "edge"], ascending=[True, False])
    flagged = []
    for number, (_, row) in enumerate(order.iterrows(), start=1):
        notes = news.get(row["game_id"], [])
        qb = [n for n in notes if n.qb]
        mark = "QB!" if qb else ("inj" if notes else "   ")
        if qb:
            flagged.append(number)
        print(
            f"{number:>3}. {mark} {row['kickoff']:%a %H:%M}  {row['bet']:<28} "
            f"{row['buy']:<40} {row['contracts']:>3.0f} @ ≤{row['limit']:.2f}  "
            f"≤${row['outlay']:.2f}   ({row['game']}, edge {row['edge']:+.3f})"
            + (f"  [topping up {row['held']:g} held]" if row["held"] else "")
        )
        for note in sorted(notes, key=lambda n: not n.qb)[: args.news_lines]:
            who = note.team or "?"
            print(f"         {'QB ' if note.qb else ''}[{who}] {note.sentence[:220]}")
            print(f"            -- {note.source}")
        if len(notes) > args.news_lines:
            print(f"         (+{len(notes) - args.news_lines} more injury sentences)")
    print(
        f"\nworst case ${picks['outlay'].sum():,.2f} across {len(picks)} orders "
        f"({_sizing(args)})"
    )
    if flagged:
        print(
            f"QB news on {', '.join(map(str, flagged))}: if a starter is out of the "
            "next game, quit and rerun with --qb-out TEAM so the model prices it."
        )


def _ask(count: int) -> set[int] | None:
    """What to do with the orders: an empty set to place them all, the numbers
    to drop (the rest are re-sized), or None to stop."""
    while True:
        try:
            answer = (
                input(
                    "\nplace these orders? [y = place all / numbers to drop, "
                    "e.g. '3 7' / q = quit]: "
                )
                .strip()
                .lower()
            )
        except EOFError:
            return None
        if answer in ("y", "yes"):
            return set()
        if answer in ("q", "quit", "n", "no", ""):
            return None
        numbers = answer.replace(",", " ").split()
        if all(n.isdigit() and 1 <= int(n) <= count for n in numbers):
            return {int(n) for n in numbers}
        print(f"  '{answer}': y, q, or numbers from 1 to {count}")


def _place(
    client: KalshiClient,
    picks: pd.DataFrame,
    matched: dict,
    args: argparse.Namespace,
    out: Path,
) -> None:
    """Send the orders: the cheapest route first, what it left on the other.

    Each is a limit order at the pick's `limit`, so nothing fills at a price
    that doesn't clear the edge. Immediate-or-cancel unless `--rest`.
    """
    expires = (
        None
        if args.rest is None
        else int((datetime.now(UTC) + timedelta(hours=args.rest)).timestamp())
    )
    rows = []
    for _, row in picks.sort_values("kickoff").iterrows():
        found = matched[row["game_id"]]
        side, other = (
            (found.home, found.away) if row["bet_home"] else (found.away, found.home)
        )
        wanted = int(row["contracts"])
        for route in routes(side, other):
            if wanted <= 0:
                break
            if route.price > row["limit"] + 1e-9 and expires is None:
                continue
            try:
                done = client.buy(
                    route.ticker, route.action, wanted, row["limit"], expires
                )
            except KalshiError as error:
                print(f"  !! {row['bet']}: {error}")
                rows.append(
                    {"bet": row["bet"], "ticker": route.ticker, "error": str(error)}
                )
                break
            rows.append(
                {
                    "placed_at": datetime.now(CENTRAL),
                    "game": row["game"],
                    "game_id": row["game_id"],
                    "bet": row["bet"],
                    "ticker": route.ticker,
                    "action": route.action,
                    "ordered": wanted,
                    "limit": row["limit"],
                    "filled": done.filled,
                    "resting": done.resting,
                    "fill_price": done.price,
                    "fee": done.fee,
                    "model": row["model"],
                    "edge": row["edge"],
                    "order_id": done.order_id,
                    "demo": args.demo,
                    "dry_run": args.dry_run,
                }
            )
            price = "" if done.price is None else f" at {done.price:.2f}"
            rest = f", {done.resting:.0f} resting" if expires and done.resting else ""
            print(
                f"  {row['bet']:<28} {route.action.upper()} {route.ticker}: "
                f"{done.filled:g}/{wanted} filled{price}{rest}"
            )
            # A resting order holds the rest; only an IOC's leftover moves on.
            # A dry run shows the first order only: nothing filled to say
            # what the second would be.
            wanted = 0 if expires or args.dry_run else top_up(wanted, done.filled)
    placed = pd.DataFrame(rows)
    if placed.empty:
        return
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    log = ORDERS_DIR / out.name.replace(".csv", "_orders.csv")
    placed.to_csv(log, index=False)
    if "filled" not in placed:
        print(f"\nnothing filled; errors: {log}")
        return
    filled = placed[placed["filled"].fillna(0) > 0]
    spent = (filled["filled"] * filled["fill_price"] + filled["fee"].fillna(0)).sum()
    print(
        f"\nfilled {filled['filled'].sum():.0f} contracts on "
        f"{filled['game_id'].nunique()} games for ${spent:,.2f}; orders: {log}"
    )


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
        "limit": lambda s: s.map("{:.2f}".format),
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
        f"price), top {args.top} by edge; {_sizing(args)}"
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

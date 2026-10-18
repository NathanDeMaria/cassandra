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
News that lasts -- a starter out for the season -- goes in
`~/.cassandra/betting/overrides.toml` instead, with the date it ends
(`cassandra.overrides`), and every run applies it until then.
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
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
from call_it_what_you_want import ESPN, KALSHI, NCAA, NCAAFB, TeamNamer, default_teams
from endgame.async_tools import apply_in_parallel
from endgame.football_plays import FootballLeague, get_game_plays
from endgame.ncaafb import get_season
from endgame.types import Game, Season, merge_weekly_seasons
from endgame_aws import Config
from endgame_aws.pbp_transform import transform_game_to_table

from cassandra import overrides, placing
from cassandra.batch.artifacts import download, download_predictor_data
from cassandra.constants import CASSANDRA_HOME
from cassandra.exhibitions import without_exhibitions
from cassandra.kalshi_trade import KalshiClient, public_get
from cassandra.markets import MAX_OVERROUND
from cassandra.model_eval import rebuild_priors
from cassandra.odds import OddsDatabase
from cassandra.picks import (
    EASTERN,
    MAX_SINCE_ENTRY,
    MIN_EDGE,
    NCAAFB_SERIES,
    NOT_HAPPENING,
    Book,
    Fixture,
    entry,
    event_day,
    held_on,
    kalshi_games,
    match,
    max_price,
    previous_games,
    price,
    skip_reason,
)
from cassandra.placing import CENTRAL, DEFAULT_BUDGET
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
PICKS_DIR = CASSANDRA_HOME / "picks"
PLAYS_DIR = PICKS_DIR / "plays"
OUT_DIR = CASSANDRA_HOME / "betting" / "picks"
OVERRIDES = CASSANDRA_HOME / "betting" / "overrides.toml"
#: How long after kickoff a game ESPN has no plays for is taken to be
#: missing them for now, not for good. Its teams' next games wait.
QB_PLAYS_GRACE = timedelta(hours=18)


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


def _overrides(args: argparse.Namespace) -> list[overrides.Override]:
    """The `--overrides` file's news still applying today, each printed."""
    if args.no_overrides:
        return []
    try:
        found = overrides.load(args.overrides)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    active, expired = overrides.split(found, datetime.now(CENTRAL).date())
    for override in active:
        print(
            f"{'override:':<16}{override.kind.replace('_', ' ')} {override.team} "
            f"through {override.until:%b %d}"
            + (f" ({override.note})" if override.note else "")
        )
    for override in expired:
        print(
            f"{'expired:':<16}{override.kind.replace('_', ' ')} {override.team} "
            f"ended {override.until:%b %d}, not applied -- renew or delete it in "
            f"{args.overrides}"
        )
    return active


def _qb_out_named(
    names: list[str], upcoming: list[Game], strict: bool = True
) -> dict[str, set[str]]:
    """`--qb-out` news: each named team out in its next game.

    A name matching no team with a game coming up is an error, unless not
    `strict`, when it's skipped.
    """
    named: dict[str, set[str]] = {}
    playing = {t for g in upcoming for t in (g.home, g.away)}
    for name in names:
        teams = sorted(t for t in playing if t.lower() == name.lower()) or sorted(
            t for t in playing if name.lower() in t.lower()
        )
        if not teams and not strict:
            continue
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


def _kalshi_markets(series: tuple[str, ...]) -> list[dict]:
    markets: list[dict] = []
    for ticker in series:
        cursor = None
        while True:
            params = {"series_ticker": ticker, "status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            page = public_get("/markets", params)
            markets += page.get("markets") or []
            cursor = page.get("cursor")
            if not cursor or not page.get("markets"):
                break
    return markets


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
        "--max-since-entry",
        type=float,
        default=MAX_SINCE_ENTRY.total_seconds() / 3600,
        metavar="HOURS",
        help="skip games whose entry (both previous games over) was longer ago; "
        "the 2025-26 edge was +15%% in the first 6h, +3%% at 12-24h",
    )
    parser.add_argument(
        "--max-overround",
        type=float,
        default=MAX_OVERROUND,
        help="skip books whose two costs sum past this: too wide to price",
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
        "--overrides",
        type=Path,
        default=OVERRIDES,
        help="--qb-out / --exclude news that lasts, each until a date "
        "(see cassandra.overrides)",
    )
    parser.add_argument(
        "--no-overrides", action="store_true", help="ignore the --overrides file"
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
    lasting = _overrides(args)
    named = _qb_out_named(args.qb_out, upcoming)
    # A lasting QB out for a team with no game in the window (a bye) has
    # nothing to apply to this run.
    for game, out in _qb_out_named(
        overrides.teams(lasting, "qb_out"), upcoming, strict=False
    ).items():
        named.setdefault(game, set()).update(out)
    args.exclude = [*args.exclude, *overrides.teams(lasting, "exclude")]
    flags = {game: set(out) for game, out in load_qb_out(league).items()}
    for extra in (tonight, named):
        for game, out in extra.items():
            flags.setdefault(game, set()).update(out)
    by_id = {g.game_id: g for g in games}
    for label, extra in (("QB out tonight", tonight), ("QB out next game", named)):
        for game, out in sorted(extra.items(), key=lambda item: by_id[item[0]].date):
            print(
                f"{label + ':':<16}{', '.join(sorted(out))}  ({placing.fmt_game(by_id[game])})"
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
                f"!! no plays from ESPN yet for {placing.fmt_game(by_id[game])}: "
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
            priced,
            division(side),
            division(other),
            waiting,
            since_entry,
            args.min_edge,
            timedelta(hours=args.max_since_entry),
            args.max_overround,
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
                "game": placing.fmt_game(game),
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

    # 5. Size the picks (`placing.size`), each at its limit price.
    client = None
    frame["held"] = 0.0
    frame["held_as"] = ""
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
            # The same outcome can be held through either contract, and a
            # top-up may buy the other one: say which is held.
            frame.at[i, "held_as"] = ", ".join(
                f"{count:g} {action} {ticker.rsplit('-', 1)[-1]}"
                for ticker, action, count in (
                    (side.ticker, "YES", positions.get(side.ticker, 0.0)),
                    (other.ticker, "NO", -positions.get(other.ticker, 0.0)),
                )
                if count > 0
            )
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
        placing.limit_price(p, m, args.slippage)
        for p, m in zip(frame["price"], frame["max_price"])
    ]
    priced_skips = frame["skip"].copy()
    dropped: set[str] = set()
    # Why each game was off the first list asked about. Dropping some frees
    # places in the top `--top`, and a game nobody has read the news on
    # shouldn't take one: answering is about the list on the screen. So
    # each later list keeps the first one's games and reasons.
    unlisted: pd.Series | None = None
    news: dict[str, placing.News] = {}
    out = OUT_DIR / f"{league}_{model}_{started.astimezone(CENTRAL):%Y%m%d-%H%M}.csv"
    while True:
        frame["skip"] = priced_skips
        frame.loc[frame["game_id"].isin(dropped), "skip"] = "dropped at the prompt"
        if unlisted is not None:
            off = frame["game_id"].isin(unlisted.index)
            frame.loc[off, "skip"] = frame.loc[off, "game_id"].map(unlisted)
        sized, picks = placing.size(frame, matched, args)
        if unlisted is None:
            unlisted = sized.set_index("game_id")["skip"].dropna()
        if not args.no_news:
            fresh = picks[~picks["game_id"].isin(news.keys())]
            news.update(placing.injury_news(fresh, games, namer))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        sized.to_csv(out, index=False)
        placing.print_picks(sized, picks, args)
        placing.print_orders(picks, news, args)
        print(f"\nevery matched game: {out}")
        if client is None or picks.empty:
            return
        answer = placing.ask(len(picks))
        if answer is None:
            print("nothing placed")
            return
        if not answer:
            break
        order = picks.sort_values(["kickoff", "edge"], ascending=[True, False])
        dropped |= {order["game_id"].iloc[i - 1] for i in answer}

    placing.place(client, picks, matched, args, out)


if __name__ == "__main__":
    main()

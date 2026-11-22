"""How past weeks' Kalshi bets did against what the model expected of them.

    python account_review.py                   # or `make account-review`
    python account_review.py --league womens --week YYYY-MM-DD --plot weeks.png

For each league with settled bets in the ledger (`make ledger`; the rows of
`~/.cassandra/betting/ledger/ledger.csv`), and each of its weeks -- by
kickoff, Tuesday to Monday, the ledger's week -- it takes every settled game
bet and prices it four ways:

- **as bet**: the model's probability the picks run logged when it placed;
- **replayed**: today's model and fit, replayed over only what was known
  when the bet was filled (`replay_cutoff`), and its probability for the
  bet's side;
- **market**: Kalshi's no-vig price at the bet;
- **close**: Kalshi's no-vig price at kickoff.

Each gives an expected net result and, drawing the week's bets 100,000
times, a distribution; the actual net (payout less what was paid) is then
placed within each. A week still holding open bets is read as far as it has
settled. Bets that share a state of the season share one replay.

What a replay knows depends on the league. Football's stands at the fill:
the results by then, and the quarterback flags of games already over (the
index's flags for later games come from those games' own plays).
Basketball's data lands once a day, before dawn, so its replay stands on
that morning, with the player ratings and crowd readings as published by
then (`MatchupSources.until`). Neither has the news a person fed in
(`--qb-out`, overrides), and each fit was trained on every game, so a
replay is a little better informed than the bettor was. The ledger holds no
balances, so a week is read as net dollars from its bets, to add to
whatever cash was on hand.
"""

import argparse
import asyncio
import logging
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from call_it_what_you_want import TeamNamer
from endgame.types import Season

import picks
from cassandra import account_value as av
from cassandra import ledger
from cassandra.constants import CASSANDRA_HOME
from cassandra.exhibitions import without_exhibitions
from cassandra.picks import PICK_LEAGUES, PickLeague
from cassandra.predictor.adjustments import MatchupSources
from cassandra.predictor.qb_out import QbOutIndex, load_qb_out

CENTRAL = ZoneInfo("America/Chicago")

#: By when a day's basketball results and player ratings are in, Central:
#: EndGame's pulls run at 4:30 and the ratings job at 5:00 (9:00 before
#: 2026-11-12, so an early bet then stood on a day less than this gives it).
RATINGS_IN = time(6)

#: Each view of a bet, and the ledger column its probability comes from.
VIEWS = (
    ("as bet", "model"),
    ("replayed", "replayed"),
    ("market", "market"),
    ("close", "close"),
)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--league",
        action="append",
        default=[],
        choices=sorted(PICK_LEAGUES),
        help="a league to review (repeatable; default: every one with settled bets)",
    )
    parser.add_argument(
        "--model", help="the model to replay (one --league only; default: its own)"
    )
    parser.add_argument(
        "--week",
        action="append",
        default=[],
        metavar="SATURDAY",
        help="a ledger week, by its Saturday, as YYYY-MM-DD (default: every one)",
    )
    parser.add_argument("--offline", action="store_true", help="skip s3")
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--plot", metavar="PNG", help="save a histogram per week")
    args = parser.parse_args()
    if args.model and len(args.league) != 1:
        parser.error("--model needs exactly one --league")
    return args


def read_ledger(weeks: list[str]) -> pd.DataFrame:
    """The ledger's game bets, one row per game bet, in the weeks asked for."""
    path = ledger.LEDGER_DIR / "ledger.csv"
    if not path.exists():
        raise SystemExit(f"{path} doesn't exist yet: run `make ledger`")
    frame = pd.read_csv(path, dtype={"game_id": str, "week": str})
    frame["first_fill"] = pd.to_datetime(frame["first_fill"], utc=True)
    stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    print(f"ledger: {len(frame)} game bets, written {stamp:%b %d %H:%M} UTC")
    return frame[frame["week"].isin(weeks)] if weeks else frame


def in_league(frame: pd.DataFrame, league: PickLeague) -> pd.DataFrame:
    """`frame`'s bets on `league`'s series."""
    return frame[frame["event"].str.split("-").str[0].isin(league.series)]


def replay_cutoff(league: PickLeague, fill: datetime) -> datetime:
    """The moment a replay of a bet filled at `fill` stands on.

    The fill itself, except in a league priced on player ratings: those are
    fitted once a day, and a picks run waits for them, so the bet stood on
    that morning's data -- the morning before, for a bet in the small hours.
    """
    if not league.players:
        return fill
    local = fill.astimezone(CENTRAL)
    day = local.date()
    if local.time() < RATINGS_IN:
        day -= timedelta(days=1)
    return datetime.combine(day, RATINGS_IN, CENTRAL)


def replayed(
    bets: pd.DataFrame,
    league: PickLeague,
    model: str,
    seasons: list[Season],
    namer: TeamNamer,
) -> pd.Series:
    """The model's probability for each bet's side, as it stood when it was filled.

    The season is cut at each bet's `replay_cutoff` (`av.as_of`); bets that
    share the same games over share one replay.
    """
    latest = max(seasons, key=lambda s: s.year)
    home = CASSANDRA_HOME / "picks"
    cache_path = home / f"review_{league.name}_{model}_replayed.csv"
    # The warm start for these replays, apart from `picks.py`'s own.
    priors = home / f"review_{league.name}_priors.json"
    cache = (
        pd.read_csv(cache_path, dtype={"key": str}).set_index("key")["p"].to_dict()
        if cache_path.exists()
        else {}
    )
    cuts = {
        index: replay_cutoff(league, fill.to_pydatetime())
        for index, fill in bets["first_fill"].items()
    }

    def key(index) -> str:
        return f"{bets.at[index, 'event']}|{cuts[index]:%Y%m%d%H%M%S}"

    found: dict = {i: cache[key(i)] for i in bets.index if key(i) in cache}
    states: dict[frozenset[str], tuple[Season, datetime, list]] = {}
    for index in bets.index:
        if index in found:
            continue
        cut = av.as_of(latest, cuts[index])
        over = frozenset(
            g.game_id for week in cut.weeks for g in week.games if g.completed
        )
        states.setdefault(over, (cut, cuts[index], []))[2].append(index)
    print(
        f"{len(bets)} bets: {len(found)} replayed before, the rest over "
        f"{len(states)} states of the season"
    )
    flagged = load_qb_out(league.name)
    in_order = sorted(states.items(), key=lambda state: state[1][1])
    for number, (over, (cut, moment, indexes)) in enumerate(in_order, 1):
        # The index flags a team missing its starter in a game from that
        # game's plays, so a flag for a game not yet played is a leak.
        flags = {game: set(out) for game, out in flagged.items() if game in over}
        sources = (
            MatchupSources.for_league(league.name)
            ._replace(qb_out=QbOutIndex(flags))
            .until(moment.astimezone(CENTRAL).date())
        )
        state_seasons = [cut if s is latest else s for s in seasons]
        predictor, _ = picks._replay(league.name, model, state_seasons, sources, priors)
        games = {g.game_id: namer.apply(g) for week in cut.weeks for g in week.games}
        for index in indexes:
            row = bets.loc[index]
            game = games.get(row["game_id"])
            if game is None or game.game_id in over:
                found[index] = float("nan")
                continue
            home_wins = predictor.predict_game(game).team1_win_prob
            found[index] = home_wins if row["bet_home"] else 1 - home_wins
            cache[key(index)] = found[index]
        pd.Series(cache, name="p").rename_axis("key").reset_index().to_csv(
            cache_path, index=False
        )
        print(
            f"  replay {number}/{len(states)}: {len(over)} games over, {len(indexes)} bets"
        )
    return pd.Series(found, dtype=float)


def slate(bets: pd.DataFrame, column: str) -> list[av.Bet]:
    """The bets as `Bet`s at the probability in `column`.

    A bet with none there (the picks run's file is gone, no price was read,
    or no game to replay) takes the next of replayed, as bet, and the close
    that it has, so every view covers the same bets.
    """
    fallbacks = (column, "replayed", "model", "close")
    return [
        av.Bet(
            f"{row['title']}: {row['bet_short']}",
            row["stake"],
            row["contracts"],
            next(row[c] for c in fallbacks if pd.notna(row[c])),
        )
        for _, row in bets.iterrows()
    ]


def review(bets: pd.DataFrame, draws: int) -> dict:
    """Each view's `av.expectation` of `bets`, by view name."""
    actual = bets["payout"].sum()
    return {
        name: av.expectation(slate(bets, column), actual, draws)
        for name, column in VIEWS
    }


def show(heading: str, views: dict) -> None:
    print(heading)
    print(f"  {'':<10}{'expected net':>13}{'sd':>8}{'P(gain)':>9}{'actual at':>11}")
    for name, view in views.items():
        print(
            f"  {name:<10}{view['expected']:>+13,.2f}{view['sd']:>8,.2f}"
            f"{view['p_gain']:>9.0%}{view['percentile']:>10.0%}ile"
        )


def together(heading: str, totals: list[dict]) -> None:
    """Weeks' views added up: expected nets and variances sum across weeks."""
    print(heading)
    for name, _ in VIEWS:
        expected = sum(v[name]["expected"] for v in totals)
        sd = sum(v[name]["sd"] ** 2 for v in totals) ** 0.5
        actual = sum(v[name]["actual"] for v in totals)
        print(
            f"  {name:<10}expected {expected:>+8,.2f}  sd {sd:>6,.2f}  actual {actual:>+8,.2f}  "
            f"({(actual - expected) / sd:+.2f} sd)"
        )


def main() -> None:
    logging.getLogger("endgame_aws.pbp_transform").setLevel(logging.ERROR)
    logging.getLogger("endgame").setLevel(logging.WARNING)
    args = parse()
    frame = read_ledger(args.week)
    leagues = [PICK_LEAGUES[name] for name in args.league] or [
        league
        for league in PICK_LEAGUES.values()
        if in_league(frame, league)["settled"].any()
    ]
    panels, every, reviewed = [], [], 0
    for league in leagues:
        rows = in_league(frame, league)
        bets = rows[rows["settled"]].copy()
        if bets.empty:
            print(f"\n{league.name}: no settled bets")
            continue
        still_open = rows[~rows["settled"]].groupby("week").size()
        model = args.model or league.model
        print(f"\n== {league.name} ({model}): {len(bets)} settled bets")
        seasons = (
            picks._from_cache(league.name)
            if args.offline
            else asyncio.run(picks._refresh_from_s3(league.name, model))
        )
        seasons = without_exhibitions(seasons, league.name)
        namer = TeamNamer.for_league(league.name)
        bets["replayed"] = replayed(bets, league, model, seasons, namer)

        totals = []
        for week, group in bets.groupby("week"):
            views = review(group, args.draws)
            won = int(group["won"].sum())
            pending = int(still_open.get(week, 0))
            so_far = f" (so far: {pending} still open)" if pending else ""
            show(
                f"\nweek of {week}{so_far}: {len(group)} bets, {won}-{len(group) - won}, "
                f"paid ${views['as bet']['staked']:,.2f}, "
                f"actual net {views['as bet']['actual']:+,.2f}",
                views,
            )
            panels.append(
                {
                    "label": f"{league.name}, week of {week}{so_far}: "
                    f"{won}-{len(group) - won} on ${views['replayed']['staked']:,.0f} staked",
                    "pnl": views["replayed"]["pnl"],
                    "expected": views["replayed"]["expected"],
                    "actual": views["replayed"]["actual"],
                }
            )
            totals.append(views)
        if len(totals) > 1:
            together(f"\n{league.name}, all weeks together:", totals)
        every.extend(totals)
        reviewed += 1

    if reviewed > 1:
        together("\nevery league together:", every)
    if args.plot and panels:
        av.plot_weeks(panels, args.plot)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()

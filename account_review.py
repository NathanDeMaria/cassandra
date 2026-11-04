"""How past weeks' Kalshi bets did against what the model expected of them.

    python account_review.py                   # or `make account-review`
    python account_review.py --week YYYY-MM-DD --plot weeks.png

For each settled week in the ledger (`make ledger`; the rows of
`~/.cassandra/betting/ledger/ledger.csv`), it takes every game bet that
week -- by kickoff, Tuesday to Monday, the ledger's week -- and prices each
bet three ways:

- **as bet**: the model's probability the picks run logged when it placed;
- **replayed**: today's model and fit, replayed over only the games that
  were over when the bet was filled (`cassandra.account_value.as_of`), and
  its probability for the bet's side;
- **market**: Kalshi's no-vig price at the bet.

Each gives an expected net result and, drawing the week's bets 100,000
times, a distribution; the actual net (payout less what was paid) is then
placed within the replayed one. Bets that share a state of the season
share one replay.

The replay keeps the quarterback flags of games already over and drops the
ones the index holds for later games, which come from those games' own
plays. It doesn't have the news a person fed in (`--qb-out`, overrides),
tonight's plays, or EPA that hadn't been swept yet, and the fit itself was
trained on every game, so the replay is a little better informed than the
bettor was. The ledger holds no balances, so the week is read as net dollars
from its bets, to add to whatever cash was on hand.
"""

import argparse
import asyncio
import logging
from datetime import UTC, datetime

import pandas as pd
from call_it_what_you_want import TeamNamer

import picks
from cassandra import account_value as av
from cassandra import ledger
from cassandra.constants import CASSANDRA_HOME
from cassandra.exhibitions import without_exhibitions
from cassandra.predictor.adjustments import MatchupSources
from cassandra.predictor.qb_out import QbOutIndex, load_qb_out

#: The warm start for these replays, apart from `picks.py`'s own.
PRIORS = CASSANDRA_HOME / "picks" / "review_priors.json"


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=picks.FOOTBALL.model)
    parser.add_argument(
        "--week",
        action="append",
        default=[],
        metavar="SATURDAY",
        help="a ledger week, by its Saturday, as YYYY-MM-DD (default: every settled one)",
    )
    parser.add_argument("--offline", action="store_true", help="skip s3")
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--plot", metavar="PNG", help="save a histogram per week")
    return parser.parse_args()


def settled_bets(weeks: list[str]) -> pd.DataFrame:
    """The ledger's game bets for weeks that are over, one row per game bet."""
    path = ledger.LEDGER_DIR / "ledger.csv"
    if not path.exists():
        raise SystemExit(f"{path} doesn't exist yet: run `make ledger`")
    frame = pd.read_csv(path, dtype={"game_id": str})
    frame["first_fill"] = pd.to_datetime(frame["first_fill"], utc=True)
    done = frame.groupby("week")["settled"].all()
    over = [str(w) for w, ok in done.items() if ok and (not weeks or w in weeks)]
    stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    print(
        f"ledger: {len(frame)} game bets, written {stamp:%b %d %H:%M} UTC; "
        f"settled weeks {', '.join(over) or 'none'}"
    )
    return frame[frame["week"].isin(over)].copy()


def replayed(bets: pd.DataFrame, seasons, namer, model: str) -> pd.Series:
    """The model's probability for each bet's side, as it stood when it was filled.

    The season is cut where each fill was (`av.as_of`); fills that share
    the same games over share one replay.
    """
    latest = max(seasons, key=lambda s: s.year)
    cache_path = CASSANDRA_HOME / "picks" / f"review_{model}_replayed.csv"
    cache = (
        pd.read_csv(cache_path, dtype={"key": str}).set_index("key")["p"].to_dict()
        if cache_path.exists()
        else {}
    )

    def key(index) -> str:
        return f"{bets.at[index, 'event']}|{bets.at[index, 'first_fill']:%Y%m%d%H%M%S}"

    found: dict = {i: cache[key(i)] for i in bets.index if key(i) in cache}
    states: dict[frozenset[str], list] = {}
    for index, fill in bets["first_fill"].items():
        if index in found:
            continue
        cut = av.as_of(latest, fill.to_pydatetime())
        over = frozenset(
            g.game_id for week in cut.weeks for g in week.games if g.completed
        )
        states.setdefault(over, [cut, []])[1].append(index)
    print(
        f"{len(bets)} bets: {len(found)} replayed before, the rest over "
        f"{len(states)} states of the season"
    )
    for number, (over, (cut, indexes)) in enumerate(states.items(), 1):
        # The index flags a team missing its starter in a game from that
        # game's plays, so a flag for a game not yet played is a leak.
        flags = {
            game: set(out)
            for game, out in load_qb_out(picks.FOOTBALL.name).items()
            if game in over
        }
        sources = MatchupSources.for_league(picks.FOOTBALL.name)._replace(
            qb_out=QbOutIndex(flags)
        )
        state_seasons = [cut if s is latest else s for s in seasons]
        predictor, _ = picks._replay(
            picks.FOOTBALL.name, model, state_seasons, sources, PRIORS
        )
        games = {g.game_id: namer.apply(g) for week in cut.weeks for g in week.games}
        for index in indexes:
            row = bets.loc[index]
            game = games.get(row["game_id"])
            if game is None or game.game_id in over:
                found[index] = float("nan")
                continue
            home = predictor.predict_game(game).team1_win_prob
            found[index] = home if row["bet_home"] else 1 - home
            cache[key(index)] = found[index]
        pd.Series(cache, name="p").rename_axis("key").reset_index().to_csv(
            cache_path, index=False
        )
        print(
            f"  replay {number}/{len(states)}: {len(over)} games over, {len(indexes)} bets"
        )
    return pd.Series(found)


def slate(bets: pd.DataFrame, column: str) -> list[av.Bet]:
    """The week's bets as `Bet`s at the probability in `column`.

    A bet with none there (the picks run's file is gone, or no price was
    read) takes the replayed probability, so every view covers the same bets.
    """
    return [
        av.Bet(
            f"{row['title']}: {row['bet_short']}",
            row["stake"],
            row["contracts"],
            row[column] if pd.notna(row[column]) else row["replayed"],
        )
        for _, row in bets.iterrows()
    ]


def main() -> None:
    logging.getLogger("endgame_aws.pbp_transform").setLevel(logging.ERROR)
    logging.getLogger("endgame").setLevel(logging.WARNING)
    args = parse()
    bets = settled_bets(args.week)
    if bets.empty:
        return
    league = picks.FOOTBALL.name
    seasons = (
        picks._from_cache(league)
        if args.offline
        else asyncio.run(picks._refresh_from_s3(league, args.model))
    )
    seasons = without_exhibitions(seasons, league)
    namer = TeamNamer.for_league(league)
    bets["replayed"] = replayed(bets, seasons, namer, args.model)
    bets["payout_actual"] = bets["payout"]

    panels, totals = [], []
    for week, group in bets.groupby("week"):
        actual = group["payout_actual"].sum()
        views = {
            name: av.expectation(slate(group, column), actual, args.draws)
            for name, column in (
                ("as bet", "model"),
                ("replayed", "replayed"),
                ("market", "market"),
            )
        }
        won = int(group["won"].sum())
        print(
            f"\nweek of {week}: {len(group)} bets, {won}-{len(group) - won}, "
            f"paid ${views['as bet']['staked']:,.2f}, actual net {views['as bet']['actual']:+,.2f}"
        )
        print(f"  {'':<10}{'expected net':>13}{'sd':>8}{'P(gain)':>9}{'actual at':>11}")
        for name, view in views.items():
            print(
                f"  {name:<10}{view['expected']:>+13,.2f}{view['sd']:>8,.2f}"
                f"{view['p_gain']:>9.0%}{view['percentile']:>10.0%}ile"
            )
        panels.append(
            {
                "label": f"Week of {week}: {won}-{len(group) - won} on ${views['replayed']['staked']:,.0f} staked",
                "pnl": views["replayed"]["pnl"],
                "expected": views["replayed"]["expected"],
                "actual": views["replayed"]["actual"],
            }
        )
        totals.append((week, views))

    if len(totals) > 1:
        print("\nall weeks together:")
        for name in ("as bet", "replayed", "market"):
            expected = sum(v[name]["expected"] for _, v in totals)
            sd = sum(v[name]["sd"] ** 2 for _, v in totals) ** 0.5
            actual = sum(v[name]["actual"] for _, v in totals)
            print(
                f"  {name:<10}expected {expected:>+8,.2f}  sd {sd:>6,.2f}  actual {actual:>+8,.2f}  "
                f"({(actual - expected) / sd:+.2f} sd)"
            )
    if args.plot:
        av.plot_weeks(panels, args.plot)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()

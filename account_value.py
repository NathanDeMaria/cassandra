"""What the Kalshi account should be worth once its open bets settle, by the model.

    python account_value.py             # or `make account-value`
    python account_value.py --offline   # no s3: the copies the last picks run saved

Read-only. For each open position it takes what was paid (fees included) and
what it pays at resolution, prices the game with the same model, fit and
replay `picks.py` bets with (`--model` to change it), and works out the
expected account balance and its spread, drawing the whole slate 200,000
times. Positions in games already final (not yet paid by Kalshi) are
decided; the rest use the model's pre-game probability, even when under
way. Also shown is the same total at Kalshi's no-vig price, to see how much
of the expectation is the model disagreeing with the market.

Games are treated as independent. Any promotional credit in the cash
isn't separated out. QB news comes from the stored index and what `picks.py`
reads off ESPN's plays tonight, plus `--qb-out TEAM` and the overrides file.
"""

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from call_it_what_you_want import ESPN, KALSHI, NCAA, NCAAFB, TeamNamer, default_teams
from endgame.types import Game

import picks
from cassandra import account_value as av
from cassandra.exhibitions import without_exhibitions
from cassandra.kalshi_trade import KalshiClient, event_of, public_get
from cassandra.picks import (
    NOT_HAPPENING,
    Book,
    Fixture,
    kalshi_games,
    match,
    no_vig_home,
)
from cassandra.placing import CENTRAL
from cassandra.predictor.adjustments import MatchupSources
from cassandra.predictor.qb_out import QbOutIndex, load_qb_out

#: How far back a game still counts as a fixture: Kalshi pays a few hours
#: to a day after the final, and ESPN keeps the result.
LOOKBACK = timedelta(days=5)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=picks.FOOTBALL.model)
    parser.add_argument("--offline", action="store_true", help="skip s3")
    parser.add_argument("--no-espn", action="store_true", help="skip the ESPN top-up")
    parser.add_argument(
        "--qb-out",
        action="append",
        default=[],
        metavar="TEAM",
        help="this team's starting QB is out of its next game (repeatable)",
    )
    parser.add_argument("--draws", type=int, default=200_000)
    parser.add_argument(
        "--plot", metavar="PNG", help="also save a density histogram of the balances"
    )
    return parser.parse_args()


def holdings(client: KalshiClient) -> list[dict]:
    """Each open position: its ticker, side, contracts and what it cost with fees."""
    found = []
    for position in client.market_positions():
        count = float(position.get("position_fp") or 0)
        if not count:
            continue
        found.append(
            {
                "ticker": position["ticker"],
                "event": event_of(position["ticker"]),
                "yes": count > 0,
                "count": abs(count),
                "cost": float(position.get("market_exposure_dollars") or 0)
                + float(position.get("fees_paid_dollars") or 0),
            }
        )
    return found


def event_books(events: set[str]) -> list:
    """Kalshi's games for these events, whatever their status."""
    markets: list[dict] = []
    for event in sorted(events):
        markets += public_get("/markets", {"event_ticker": event}).get("markets") or []
    return kalshi_games(markets)[0]


def model_home_probabilities(args, seasons, namer, fixtures):
    """The model's home win probability for each fixture, by game id."""
    latest = max(seasons, key=lambda s: s.year)
    games = [namer.apply(g) for week in latest.weeks for g in week.games]
    now = datetime.now(UTC)
    upcoming = [
        g
        for g in games
        if not g.completed and g.status not in NOT_HAPPENING and g.date > now
    ]
    tonight, _, _ = asyncio.run(
        picks._qb_out_tonight(picks.FOOTBALL.name, latest, games, upcoming)
    )
    named = picks._qb_out_named(args.qb_out, upcoming)
    flags = {game: set(out) for game, out in load_qb_out(picks.FOOTBALL.name).items()}
    for extra in (tonight, named):
        for game, out in extra.items():
            flags.setdefault(game, set()).update(out)
    sources = MatchupSources.for_league(picks.FOOTBALL.name)._replace(
        qb_out=QbOutIndex(flags)
    )
    predictor, _ = picks._replay(picks.FOOTBALL.name, args.model, seasons, sources)
    return {
        f.game.game_id: predictor.predict_game(f.game).team1_win_prob for f in fixtures
    }


def decided(game: Game) -> float | None:
    """1 or 0 for a home win once the game is final, else None."""
    if not game.completed or game.home_score == game.away_score:
        return None
    return float(game.home_score > game.away_score)


def main() -> None:
    logging.getLogger("endgame_aws.pbp_transform").setLevel(logging.ERROR)
    logging.getLogger("endgame").setLevel(logging.WARNING)
    args = parse()
    client = KalshiClient.from_env(dry_run=True)
    cash, _ = client.account()
    held = holdings(client)
    if not held:
        print(f"cash ${cash:,.2f}; no open positions")
        return

    league = picks.FOOTBALL.name
    seasons = (
        picks._from_cache(league)
        if args.offline
        else asyncio.run(picks._refresh_from_s3(league, args.model))
    )
    if not args.no_espn:
        seasons = asyncio.run(picks._with_espn(seasons, league))
    seasons = without_exhibitions(seasons, league)
    namer = TeamNamer.for_league(league)
    latest = max(seasons, key=lambda s: s.year)
    now = datetime.now(UTC)
    fixtures = [
        Fixture(g, namer.espn_id(g.home), namer.espn_id(g.away))
        for g in (namer.apply(g) for week in latest.weeks for g in week.games)
        if g.status not in NOT_HAPPENING and g.date > now - LOOKBACK
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

    matched, unmatched = match(
        fixtures, event_books({h["event"] for h in held}), side_ids
    )
    by_id = {f.game.game_id: f.game for f in fixtures}
    home_model = model_home_probabilities(args, seasons, namer, fixtures)

    bets, rows = [], []
    for h in held:
        game_id = next(
            (
                g
                for g, m in matched.items()
                if h["ticker"] in (m.home.ticker, m.away.ticker)
            ),
            None,
        )
        if game_id is None:
            raise SystemExit(
                f"{h['ticker']} matched no ESPN game (unmatched: {unmatched})"
            )
        found, game = matched[game_id], by_id[game_id]
        final = decided(game)
        p_home = home_model[game_id] if final is None else final
        is_home = found.home.ticker == h["ticker"]
        market = no_vig_home(found.home, found.away)
        p_team = p_home if is_home else 1 - p_home
        team_market = (
            None if market is None else (market[0] if is_home else 1 - market[0])
        )
        label = f"{'YES' if h['yes'] else 'NO'} {h['ticker'].rsplit('-', 1)[-1]}"
        bet = av.Bet(
            f"{game.away} @ {game.home}: {label}",
            h["cost"],
            h["count"],
            av.held_probability(h["yes"], p_team),
            None
            if final is not None or team_market is None
            else av.held_probability(h["yes"], team_market),
        )
        bets.append(bet)
        rows.append(
            {
                "kickoff": game.date.astimezone(CENTRAL).strftime("%a %m-%d %H:%M"),
                "game": f"{game.away} @ {game.home}",
                "holding": label,
                "pays": bet.payout,
                "cost": bet.cost,
                "model": bet.probability,
                "market": bet.market,
                "ev": bet.expected - bet.cost,
                "state": "final" if final is not None else "",
            }
        )

    table = pd.DataFrame(rows).sort_values(["kickoff", "game"])
    with pd.option_context("display.width", 220, "display.max_colwidth", 50):
        print(
            table.to_string(
                index=False,
                na_rep="",
                formatters={
                    "pays": "${:,.2f}".format,
                    "cost": "${:,.2f}".format,
                    "model": "{:.1%}".format,
                    "market": lambda x: "" if pd.isna(x) else f"{x:.1%}",
                    "ev": "{:+,.2f}".format,
                },
            )
        )
    balances = av.simulate(bets, cash, args.draws)
    s = av.summary(bets, cash, balances)
    now_central = datetime.now(CENTRAL)
    print(
        f"\n{args.model}, {now_central:%a %b %d %H:%M} CT: {len(bets)} open bets\n"
        f"  cash now                ${s['cash']:>9,.2f}\n"
        f"  paid for the bets       ${s['staked']:>9,.2f}\n"
        f"  if every bet pays       ${s['best']:>9,.2f}\n"
        f"  expected after settling ${s['expected']:>9,.2f}  "
        f"(+${s['profit']:,.2f} on the bets, {s['roi']:+.1%} ROI)\n"
        f"  at Kalshi's prices      ${s['market_expected']:>9,.2f}\n"
        f"  standard deviation      ${s['sd']:>9,.2f}\n"
        f"  P(the bets net a gain)  {s['p_profit']:>9.1%}\n"
        f"  P(lose over 10% staked) {s['p_loss_over_10pct']:>9.1%}\n"
    )
    print("final balance percentiles:")
    for pct, value in s["percentiles"].items():
        print(f"  {pct:>2}th  ${value:>9,.2f}")
    print("\ndistribution:")
    print("\n".join(av.histogram(balances)))
    if args.plot:
        av.plot(balances, s, args.plot)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()

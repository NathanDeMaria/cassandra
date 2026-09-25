"""Ask whether a model's picks are worth betting, not just whether they cover.

    python betting.py --league ncaafb --model glicko_full
    python betting.py --league nfl --model glicko_full --season 2026
    python betting.py --league ncaafb --model glicko_full glicko_margin_units_offseason
    python betting.py --league mens --model glicko_full --book kalshi

All on the games the book has a price for. `--book espn`, the default, is
the odds pulls, which started in August 2026. `--book kalshi` and
`--book polymarket` are the prediction markets gold-rush pulls (see
`cassandra.markets`), read every hour -- but they trade who wins, not the
spread, so they get the moneyline reports only (market information,
closing line value, spread strategies and teams are `espn` only).
Polymarket's get the close alone: its file has no quotes to tell an early
price anyone could trade from an empty market's, so an entry read there
grades noise (`markets.ENTRY_VENUES`). See `cassandra.betting` for what
each number means and why it's there; the short version, in reading order:

- **Market information.** Whether the model knows anything the line
  doesn't: the weight its disagreement with the line deserves (0 = none,
  1 = the model is right and the line wrong), with a standard error, at the
  entry line and the close; and how much of the disagreement at entry the
  line moved toward by kickoff. These are the numbers to compare models
  on, and they settle long before a record does.

- **Closing line value.** Each pick is made at the *entry* line, the first
  read after both teams' previous game, and compared with the *close*. A
  model that's ahead of the market gets positive CLV on average; the ATS
  record at the entry line is what betting it would actually have done.
  `close lead` says how far before kickoff the close was read -- an hour
  at best, and a game the hourly pulls missed (every 2026 Saturday before
  09-19 came back cut to 25 events) "closes" at that morning's read, so
  its CLV is measured against a line hours stale.

- **Spread strategies.** Flat -110 bets by minimum edge, split favorite/
  underdog and home/away, with a p-value against break-even beside each
  ROI. Then the same by week, to see drift.

- **Moneyline.** The model's win probability against the book's no-vig
  probability (DraftKings' on `espn`): whose Brier is lower, the model's weight against the price,
  and what the simple strategies on the gap return. Read the favorite/
  underdog split before the ROI. An edge that lives entirely on underdogs
  is a model less confident than the market, not a model that knows more.

- **Moneyline CLV.** The first half again, on probability: the model's
  side at the entry read's no-vig price, and how far the close moved
  toward it. Every book with moneylines has it, markets included.

- **Teams.** The teams the model and the line disagree about most, and
  who the results sided with.

Replays and the odds history are read through `cassandra.replay_cache`, so
a second run -- or a second model on the same odds -- takes seconds. Pass
several `--model`s to get a comparison table at the end. The per-game table
behind each lands at `~/.cassandra/betting/`, one csv per league and model.
"""

import argparse
import asyncio
from datetime import timedelta

import pandas as pd
from endgame_aws import Config

from cassandra.betting import (
    BREAK_EVEN,
    LineColumns,
    by_week,
    clv_by_edge,
    line_windows,
    market_information,
    mean_with_error,
    moneyline_bets,
    moneyline_calibration,
    moneyline_clv,
    moneyline_clv_by_edge,
    probability_information,
    record,
    spread_bets,
    spread_strategies,
    team_disagreement,
)
from cassandra.constants import CASSANDRA_HOME
from cassandra.markets import ENTRY_VENUES, VENUES, market_database
from cassandra.model_eval import DEFAULT_FITTERS, score_predictions
from cassandra.odds import OddsDatabase
from cassandra.replay_cache import (
    DEFAULT_MAX_AGE,
    DEFAULT_ODDS_MAX_AGE,
    load_odds,
    load_replay,
)

_BETTING_DIR = CASSANDRA_HOME / "betting"


def _hours_before_kickoff(lines: pd.DataFrame, read_at: str) -> pd.Series:
    return (
        pd.to_datetime(lines["date"], utc=True)
        - pd.to_datetime(lines[read_at], utc=True)
    ).dt.total_seconds() / 3600


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _table(frame: pd.DataFrame) -> str:
    return _indent(frame.to_string(index=False, float_format="{:.3f}".format))


def _print_market_information(lines: pd.DataFrame) -> dict:
    print("\n--- market information  (weight: share of the model's disagreement with")
    print("    the line that the result bore out; 0 = the line knew it all)")
    summary = {}
    for at in ("entry", "close"):
        info = market_information(lines, at)
        if info.n < 3:
            print(f"  {at}: {info.n} games, nothing to fit")
            continue
        print(
            f"  {at:>5}: {info.n:>4} games   weight {info.weight:+.3f} ± {info.weight_se:.3f}"
            f"   MAE line {info.mae_line:.2f}, model {info.mae_model:.2f}, "
            f"blend {info.mae_blend:.2f}"
        )
        summary[f"weight_{at}"] = f"{info.weight:+.2f}±{info.weight_se:.2f}"
        if at == "entry" and info.move == info.move:
            print(
                f"         line moved toward the model by close: {info.move:+.3f} ± "
                f"{info.move_se:.3f} per point of disagreement at entry"
            )
            summary["move"] = f"{info.move:+.2f}±{info.move_se:.2f}"
    return summary


def _print_clv(lines: pd.DataFrame) -> dict:
    bets = spread_bets(lines)
    print(f"\n--- closing line value ({len(bets)} games with an entry and a close)")
    if bets.empty:
        print("  no game has a line read after its previous game; nothing to grade")
        return {}
    close_lead = _hours_before_kickoff(bets, LineColumns.CLOSE_READ_AT)
    entry_lead = _hours_before_kickoff(bets, LineColumns.ENTRY_READ_AT)
    print(
        f"  close lead: median {close_lead.median():.1f}h before kickoff, "
        f"p90 {close_lead.quantile(0.9):.1f}h;   "
        f"entry lead: median {entry_lead.median():.0f}h, "
        f"min {entry_lead.min():.0f}h"
    )
    clv = bets["clv"]
    print(
        f"  CLV: {mean_with_error(clv)} pts/bet; "
        f"positive {(clv > 0).mean():.1%}, negative {(clv < 0).mean():.1%}, "
        f"unmoved {(clv == 0).mean():.1%}"
    )
    entry, close = record(bets, "entry"), record(bets, "close")
    print(
        f"  ATS at entry {entry} ({entry.cover_rate:.3f});"
        f"  at close {close} ({close.cover_rate:.3f});  break-even {BREAK_EVEN:.3f}"
    )
    print("\n  by the model's edge over the entry line:")
    print(
        _indent(clv_by_edge(bets).to_string(index=False, float_format="{:.3f}".format))
    )

    strategies = spread_strategies(bets)
    shown = strategies[strategies["side"].isin(["all", "favorite", "underdog"])]
    print(
        "\n--- spread strategies at -110 (entry line; p_value one-sided vs break-even)"
    )
    print(_table(shown))
    print("\n  every bet, by week:")
    print(_table(by_week(bets)))
    return {
        "bets": len(bets),
        "clv": f"{clv.mean():+.2f}±{mean_with_error(clv).se:.2f}",
        "ats_entry": f"{entry.cover_rate:.3f}",
    }


def _print_moneyline_clv(lines: pd.DataFrame) -> None:
    bets = moneyline_clv(lines)
    print(
        f"\n--- moneyline CLV ({len(bets)} games priced both sides at entry and close)"
    )
    if bets.empty:
        return
    clv = bets["clv_probability"]
    print(
        f"  CLV: mean {clv.mean():+.4f} prob/bet, "
        f"positive {(clv > 0).mean():.1%}, negative {(clv < 0).mean():.1%}, "
        f"unmoved {(clv == 0).mean():.1%}"
    )
    print("\n  by the model's edge over the entry's no-vig price:")
    print(
        _indent(
            moneyline_clv_by_edge(bets).to_string(
                index=False, float_format="{:.3f}".format
            )
        )
    )


def _print_moneyline(lines: pd.DataFrame, reads: tuple[str, ...]) -> dict:
    summary = {}
    for at in reads:
        calibration = moneyline_calibration(lines, at)
        print(
            f"\n--- moneyline at {at} ({calibration['n']:.0f} games priced both "
            f"sides, hold {calibration['hold']:.2%})"
        )
        if not calibration["n"]:
            continue
        info = probability_information(lines, at)
        print(
            f"  brier: model {calibration['brier_model']:.4f}, "
            f"market (no-vig) {calibration['brier_market']:.4f}, "
            f"blend {info.brier_blend:.4f};  model weight {info.weight:+.3f} ± "
            f"{info.weight_se:.3f}"
        )
        if at == "close":
            summary["ml_weight"] = f"{info.weight:+.2f}±{info.weight_se:.2f}"
            summary["brier_vs_mkt"] = (
                f"{calibration['brier_model'] - calibration['brier_market']:+.4f}"
            )
        print(_table(moneyline_bets(lines, at)))
    return summary


def _print_teams(lines: pd.DataFrame, top: int) -> None:
    teams = team_disagreement(lines)
    teams = teams[teams["n"] >= 2].head(top)
    if teams.empty:
        return
    print(
        "\n--- teams the model and the close disagree about most "
        "(signed to the team; model_closer = share of games the model was nearer)"
    )
    print(_table(teams))


async def _lines(
    league: str, model: str, odds_db: OddsDatabase, args: argparse.Namespace
) -> pd.DataFrame:
    replay = await load_replay(
        league, model, refresh=args.refresh, max_age=timedelta(hours=args.max_age_hours)
    )
    print(replay.describe())
    predictions = replay.predictions.copy()
    # The same margin fit a release carries, so a pick here is the pick the
    # published model makes.
    scored = score_predictions(predictions, DEFAULT_FITTERS["logistic_mae"])
    predictions["predicted_margin"] = scored.margin_predictor.predict_margins(
        predictions["team1_win_prob"].to_numpy()
    )
    lines = line_windows(predictions, odds_db)
    if args.season is not None:
        lines = lines[lines["year"] == args.season]
    # A market read has no spread, so there it's the read that counts.
    entry = (
        LineColumns.ENTRY_SPREAD if args.book == "espn" else LineColumns.ENTRY_READ_AT
    )
    print(
        f"{len(predictions)} games replayed, {len(lines)} with a line read "
        f"before kickoff, {lines[entry].notna().sum()} of those "
        "with one read after both teams' previous game"
    )
    return lines


def _save(lines: pd.DataFrame, league: str, model: str, book: str) -> None:
    _BETTING_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if book == "espn" else f"_{book}"
    out = _BETTING_DIR / f"{league}_{model}{suffix}.csv"
    # Every lined game, with the spread grading on the ones that have an
    # entry line -- a moneyline-only game still belongs in the table.
    graded = spread_bets(lines)
    lines.merge(
        graded[["game_id", *graded.columns.difference(lines.columns)]],
        on="game_id",
        how="left",
    ).to_csv(out, index=False)
    print(f"\nper-game table: {out}")


async def _main(args: argparse.Namespace) -> None:
    if args.book == "espn":
        odds_db, read_at = await load_odds(
            refresh=args.refresh, max_age=timedelta(hours=args.odds_max_age_hours)
        )
        print(f"odds history read {read_at:%Y-%m-%d %H:%M} UTC")
    else:
        # Only the lines differ by book. The replay and the margin fit are
        # the same whichever one is read -- the fit trains on every game's
        # result, not on a line -- so the picks graded are the same picks.
        bucket = Config.init_from_file().bucket
        odds_db = await market_database(bucket, args.book, args.league)
        print(f"{args.book} prices read from s3://{bucket}")
    comparison = []
    for model in args.model:
        print(f"\n{'=' * 78}\n{args.league}/{model}")
        lines = await _lines(args.league, model, odds_db, args)
        row = {"model": model}
        if args.book == "espn":
            row |= _print_market_information(lines)
            row |= _print_clv(lines)
        else:
            print(
                f"\n--- no spreads on {args.book}: it trades who wins, so moneyline only"
            )
        if args.book == "espn" or args.book in ENTRY_VENUES:
            row |= _print_moneyline(lines, ("close", "entry"))
            _print_moneyline_clv(lines)
        else:
            row |= _print_moneyline(lines, ("close",))
            print(
                f"\n--- no entry or CLV on {args.book}: its early prices can't be "
                "told from an empty market's (see cassandra.markets.ENTRY_VENUES)"
            )
        if args.book == "espn":
            _print_teams(lines, args.top_teams)
        _save(lines, args.league, model, args.book)
        comparison.append(row)
    if len(comparison) > 1:
        print(f"\n{'=' * 78}\ncomparison (weights ± se; clv in points per bet)")
        print(_indent(pd.DataFrame(comparison).to_string(index=False)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--league", required=True)
    parser.add_argument("--model", required=True, nargs="+")
    parser.add_argument("--season", type=int, help="only this season's games")
    parser.add_argument(
        "--book",
        choices=("espn", *VENUES),
        default="espn",
        help="whose prices to grade against (default: espn, the odds pulls)",
    )
    parser.add_argument(
        "--top-teams", type=int, default=12, help="rows of the team table"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="replay and re-read the odds even if the cached copies are current",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE.total_seconds() / 3600,
        help="replay again when the cached replay is older than this",
    )
    parser.add_argument(
        "--odds-max-age-hours",
        type=float,
        default=DEFAULT_ODDS_MAX_AGE.total_seconds() / 3600,
        help="re-read the odds when the cached history is older than this",
    )
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()

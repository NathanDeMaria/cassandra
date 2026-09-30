"""Ask whether a model's picks are worth betting, not just whether they cover.

    python betting.py --league ncaafb --model glicko_full
    python betting.py --league nfl --model glicko_full --season 2026
    python betting.py --league mens --model glicko_full --book kalshi

Up to three reports, all on the games the book has a price for. `--book espn`, the
default, is the odds pulls, which started in August 2026. `--book kalshi`
and `--book polymarket` are the prediction markets gold-rush pulls (see
`cassandra.markets`), which go back to the 2025 seasons and are read every
hour -- but trade who wins, not the spread, so they get the moneyline
reports only. Polymarket's get the close alone: its file has no quotes to
tell an early price anyone could trade from an empty market's, so an entry
read there grades noise (`markets.ENTRY_VENUES`). See `cassandra.betting`
for what each number means and why it's there; the short version:

- **Closing line value**, on `espn` only. Each pick is made at the
  *entry* line, the first read after both teams' previous game, and
  compared with the *close*. A model that's ahead of the market gets
  positive CLV on average; the ATS record at the entry line is what
  betting it would actually have done.
  `close lead` says how far before kickoff the close was read -- an hour
  at best, and a game the hourly pulls missed (every 2026 Saturday before
  09-19 came back cut to 25 events) "closes" at that morning's read, so
  its CLV is measured against a line hours stale.

- **Moneyline.** The model's win probability against the book's no-vig
  probability (DraftKings' on `espn`): whose Brier is lower, and what the
  simple strategies on the gap return. Read the favorite/underdog split
  before the ROI. An edge that lives entirely on underdogs is a model less
  confident than the market, not a model that knows more than it.

- **Moneyline CLV.** The first half again, on probability: the model's
  side at the entry read's no-vig price, and how far the close moved
  toward it. Every book with moneylines has it, markets included.

The per-game table behind them lands at `~/.cassandra/betting/`, one csv per
league, model and book, for anything this doesn't print.
"""

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from endgame_aws import Config

from cassandra.betting import (
    LineColumns,
    clv_by_edge,
    line_windows,
    moneyline_bets,
    moneyline_calibration,
    moneyline_clv,
    moneyline_clv_by_edge,
    record,
    spread_bets,
)
from cassandra.constants import CASSANDRA_HOME
from cassandra.markets import ENTRY_VENUES, VENUES, market_database
from cassandra.model_eval import DEFAULT_FITTERS, score_predictions
from cassandra.odds import OddsDatabase
from cassandra.predictor import load_predictor
from cassandra.save_predictions import join_with_odds, read_rated_seasons

# The same two directories `evaluate_models` and `diagnose` read, and the
# same precedence: a freshly optimized model wins over the checked-in
# baseline of that name.
_AUTHORED_DIR = Path(__file__).parent / "models"
_GENERATED_DIR = CASSANDRA_HOME / "models"
_BETTING_DIR = CASSANDRA_HOME / "betting"


def _config_path(league: str, model: str) -> Path:
    generated = _GENERATED_DIR / league / f"{model}_result.json"
    if generated.exists():
        return generated
    authored = _AUTHORED_DIR / league / f"{model}.json"
    if authored.exists():
        return authored
    raise FileNotFoundError(
        f"no config for {league}/{model}: looked at {generated} and {authored}"
    )


def _hours_before_kickoff(lines: pd.DataFrame, read_at: str) -> pd.Series:
    return (
        pd.to_datetime(lines["date"], utc=True)
        - pd.to_datetime(lines[read_at], utc=True)
    ).dt.total_seconds() / 3600


def _print_clv(lines: pd.DataFrame) -> None:
    bets = spread_bets(lines)
    print(f"\n--- closing line value ({len(bets)} games with an entry and a close)")
    if bets.empty:
        print("  no game has a line read after its previous game; nothing to grade")
        return
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
        f"  CLV: mean {clv.mean():+.3f} pts/bet, "
        f"positive {(clv > 0).mean():.1%}, negative {(clv < 0).mean():.1%}, "
        f"unmoved {(clv == 0).mean():.1%}"
    )
    entry, close = record(bets, "entry"), record(bets, "close")
    print(
        f"  ATS at entry {entry} ({entry.cover_rate:.3f});"
        f"  at close {close} ({close.cover_rate:.3f})"
    )
    mov = bets["home_score"] - bets["away_score"]
    print(
        "  MAE vs result: entry line %.2f, close %.2f, model %.2f"
        % (
            (-bets[LineColumns.ENTRY_SPREAD] - mov).abs().mean(),
            (-bets[LineColumns.CLOSE_SPREAD] - mov).abs().mean(),
            (bets["predicted_margin"] - mov).abs().mean(),
        )
    )
    print("\n  by the model's edge over the entry line:")
    print(
        _indent(clv_by_edge(bets).to_string(index=False, float_format="{:.3f}".format))
    )


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


def _print_moneyline(lines: pd.DataFrame, reads: tuple[str, ...]) -> None:
    for at in reads:
        calibration = moneyline_calibration(lines, at)
        print(
            f"\n--- moneyline at {at} ({calibration['n']:.0f} games priced both "
            f"sides, hold {calibration['hold']:.2%})"
        )
        if not calibration["n"]:
            continue
        print(
            f"  brier: model {calibration['brier_model']:.4f}, "
            f"market (no-vig) {calibration['brier_market']:.4f}"
        )
        print(
            _indent(
                moneyline_bets(lines, at).to_string(
                    index=False, float_format="{:.3f}".format
                )
            )
        )


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


async def _main(league: str, model: str, season: int | None, book: str) -> None:
    config = _config_path(league, model)
    bucket = Config.init_from_file().bucket
    print(f"{league}/{model} from {config}, priced on {book}")
    print(f"loading {book} prices and {league} seasons from s3://{bucket}")
    if book == "espn":
        odds_db = await OddsDatabase.from_s3(bucket)
    else:
        # Only the lines differ by book. The replay and the margin fit are
        # the same whichever one is read -- the fit trains on every game's
        # result, not on a line -- so the picks graded are the same picks.
        odds_db = await market_database(bucket, book, league)
    seasons = await read_rated_seasons(league, bucket)
    predictor = load_predictor(config)
    predictions = pd.DataFrame(
        [
            asdict(p)
            for p in join_with_odds(predictor, seasons, odds_db, post_callbacks=False)
        ]
    )
    # The same margin fit a release carries, so a pick here is the pick the
    # published model makes.
    scored = score_predictions(predictions, DEFAULT_FITTERS["logistic_mae"])
    predictions["predicted_margin"] = scored.margin_predictor.predict_margins(
        predictions["team1_win_prob"].to_numpy()
    )
    lines = line_windows(predictions, odds_db)
    if season is not None:
        lines = lines[lines["year"] == season]
    # A market read has no spread, so there it's the read that counts.
    entry = LineColumns.ENTRY_SPREAD if book == "espn" else LineColumns.ENTRY_READ_AT
    print(
        f"{len(predictions)} games replayed, {len(lines)} with a line read "
        f"before kickoff, {lines[entry].notna().sum()} of those "
        "with one read after both teams' previous game"
    )
    if book == "espn":
        _print_clv(lines)
    else:
        print(f"\n--- no spreads on {book}: it trades who wins, so moneyline only")
    if book == "espn" or book in ENTRY_VENUES:
        _print_moneyline(lines, ("close", "entry"))
        _print_moneyline_clv(lines)
    else:
        _print_moneyline(lines, ("close",))
        print(
            f"\n--- no entry or CLV on {book}: its early prices can't be told "
            "from an empty market's (see cassandra.markets.ENTRY_VENUES)"
        )

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--league", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--season", type=int, help="only this season's games")
    parser.add_argument(
        "--book",
        choices=("espn", *VENUES),
        default="espn",
        help="whose prices to grade against (default: espn, the odds pulls)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.league, args.model, args.season, args.book))


if __name__ == "__main__":
    main()

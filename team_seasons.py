"""Find the team-seasons a model keeps getting wrong, and say how much is real.

    python team_seasons.py --league ncaafb --model glicko_margin_units --season 2026
    python team_seasons.py --league ncaafb --model glicko_full --division fbs --since 2014
    python team_seasons.py --league ncaafb --model glicko_full --by conference --season 2026
    python team_seasons.py --league ncaafb --model glicko_full --team "Indiana Hoosiers"

Reads the model's replay from the cache `cassandra.replay_cache` keeps, so
after the first run of a model it takes seconds. See `cassandra.team_seasons`
for what each column means; the short version, in the order to read it:

- **The summary line.** `tau` is how big persistent team-season errors are
  across the league; `noise` is what a season's schedule of games produces
  on its own. When noise is three or four times tau, most of any ranked
  list is luck, and `reliability` says how much of a typical row is signal.
- **The null counts.** How many team-seasons pass |z| 2 and 3, against how
  many random sign flips of the same games would. The excess is roughly how
  many real ones there are; it is usually small.
- **The ranked rows.** Sorted by `shrunk` -- the ridge estimate, opponents
  netted out and pulled toward zero by the noise -- with `p_sign` for how
  sure it is of the sign. `early` vs `late` says whether it's a preseason
  miss or an update that isn't keeping up; `mkt_res` near zero next to a big
  `raw` says the market saw it coming and the model didn't.

`--team` prints that team's game log instead: every game, what the model and
the closing line said, and what happened. The full table lands in
`~/.cassandra/team_seasons/` for anything this doesn't print.
"""

import argparse
import asyncio
from datetime import timedelta

import pandas as pd

from cassandra.constants import CASSANDRA_HOME
from cassandra.replay_cache import DEFAULT_MAX_AGE, load_replay
from cassandra.residuals import (
    add_residuals,
    conference_label,
    division_label,
    team_tiers,
)
from cassandra.team_seasons import (
    DEFAULT_NULL_DRAWS,
    game_log,
    group_season_table,
    null_counts,
    select,
    summarize,
    team_games,
    team_season_table,
)

_OUT_DIR = CASSANDRA_HOME / "team_seasons"

_TEAM_COLUMNS = {
    "team": "{:<24.24}",
    "year": "{}",
    "conference": "{:<16.16}",
    "n": "{}",
    "raw": "{:+.1f}",
    "z": "{:+.1f}",
    "beat": "{:.0%}",
    "early": "{:+.1f}",
    "late": "{:+.1f}",
    "shrunk": "{:+.2f}",
    "posterior_sd": "{:.2f}",
    "p_sign": "{:.2f}",
    "n_lined": "{}",
    "model_vs_market": "{:+.1f}",
    "market_residual": "{:+.1f}",
}
_HEADERS = {
    "posterior_sd": "±sd",
    "n_lined": "lined",
    "model_vs_market": "mdl-mkt",
    "market_residual": "mkt_res",
    "game_number": "g",
    "week_number": "wk",
}


def _show(rows: pd.DataFrame, columns: dict[str, str]) -> str:
    present = [c for c in columns if c in rows.columns]

    text = (
        rows[present]
        .rename(columns=_HEADERS)
        .to_string(
            index=False,
            na_rep="-",
            formatters={_HEADERS.get(c, c): columns[c].format for c in present},
        )
    )
    return "\n".join(f"  {line}" for line in text.splitlines())


def _print_team_seasons(scored: pd.DataFrame, args: argparse.Namespace) -> None:
    result = team_season_table(scored, team_tiers(scored, args.league))
    table = select(result.table, args.division, args.since, args.until)
    if args.season is not None:
        table = table[table["year"] == args.season]
    table = table[table["n"] >= args.min_games]
    if table.empty:
        raise SystemExit("no team-seasons left after the filters")
    summary = summarize(table, result.tau, result.sigma)
    print(
        f"\n{summary.team_seasons} team-seasons.  per-game residual sd "
        f"{summary.sigma:.1f};  persistent team-season spread: tau {summary.tau:.2f} "
        f"league-wide, {summary.excess_sd:.2f} in this set (observed "
        f"{summary.observed_sd:.2f} vs noise {summary.noise_sd:.2f})"
    )
    print(
        f"  reliability of a typical row {summary.reliability:.0%};  "
        f"early-vs-late corr {summary.early_late_corr:+.3f};  "
        f"year-to-year corr {summary.year_to_year_corr:+.3f}"
    )
    counts = null_counts(scored, table, result.sigma, draws=args.null_draws)
    print("\n  |z| past   observed   noise alone: mean, p95")
    for _, row in counts.iterrows():
        print(
            f"  {row['threshold']:>8.1f}   {row['observed']:>8}   "
            f"{row['null_mean']:>17.1f}, {row['null_p95']:.0f}"
        )

    print("\nmost under-rated (beat the model), by shrunk:")
    print(_show(table.head(args.top), _TEAM_COLUMNS))
    print("\nmost over-rated (fell short of the model), by shrunk:")
    print(_show(table.tail(args.top).iloc[::-1], _TEAM_COLUMNS))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"{args.league}_{args.model}.csv"
    result.table.to_csv(out, index=False)
    print(f"\nevery team-season: {out}")


def _print_groups(scored: pd.DataFrame, args: argparse.Namespace) -> None:
    tiers = team_tiers(scored, args.league)
    if tiers is None:
        raise SystemExit(f"--by {args.by}: no divisions on file for {args.league}")
    label = conference_label if args.by == "conference" else division_label
    years = scored["year"].to_numpy()
    home = [label(tiers(t, int(y))) for t, y in zip(scored["home_team"], years)]
    away = [label(tiers(t, int(y))) for t, y in zip(scored["away_team"], years)]
    result = group_season_table(scored, home, away)
    table = select(result.table, since=args.since, until=args.until)
    if args.season is not None:
        table = table[table["year"] == args.season]
    table = table[table["n"] >= args.min_games]
    print(
        f"\n{len(table)} {args.by}-seasons; tau {result.tau:.2f}, per-game sd "
        f"{result.sigma:.1f}; n counts only games against another {args.by}"
    )
    columns = {"group": "{:<40.40}"} | {
        c: f
        for c, f in _TEAM_COLUMNS.items()
        if c not in ("team", "conference", "beat", "early", "late")
    }
    print(f"\nmost under-rated {args.by}-seasons:")
    print(_show(table.head(args.top), columns))
    print(f"\nmost over-rated {args.by}-seasons:")
    print(_show(table.tail(args.top).iloc[::-1], columns))


def _print_team(scored: pd.DataFrame, args: argparse.Namespace) -> None:
    games = team_games(scored)
    teams = set(games["team"])
    if args.team not in teams:
        close = sorted(t for t in teams if args.team.lower() in t.lower())
        raise SystemExit(f"no team named {args.team!r}; close: {close[:10]}")
    log = game_log(games, args.team, args.season)
    if args.since is not None:
        log = log[log["year"] >= args.since]
    columns = {
        "year": "{}",
        "game_number": "{}",
        "week_number": "{}",
        "date": "{:%Y-%m-%d}",
        "site": "{}",
        "opponent": "{:<24.24}",
        "predicted": "{:+.1f}",
        "market": "{:+.1f}",
        "actual": "{:+.0f}",
        "residual": "{:+.1f}",
        "market_residual": "{:+.1f}",
    }
    print(f"\n{args.team}: margins from its own side (model, closing line, result)")
    print(_show(log, columns))
    by_year = log.groupby("year")["residual"].agg(["size", "mean"])
    print(
        "\n  season means: "
        + ",  ".join(
            f"{y} {r['mean']:+.1f} over {r['size']:.0f}" for y, r in by_year.iterrows()
        )
    )


async def _main(args: argparse.Namespace) -> None:
    replay = await load_replay(
        args.league,
        args.model,
        refresh=args.refresh,
        max_age=timedelta(hours=args.max_age_hours),
    )
    print(replay.describe())
    scored = add_residuals(replay.predictions)
    try:
        if args.team is not None:
            _print_team(scored, args)
        elif args.by != "team":
            _print_groups(scored, args)
        else:
            _print_team_seasons(scored, args)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--league", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--season", type=int, help="only this season")
    parser.add_argument("--since", type=int, help="only this season and later")
    parser.add_argument("--until", type=int, help="only this season and earlier")
    parser.add_argument(
        "--division",
        help="only team-seasons in this division (FBS, FCS, d2, d3, or the full label)",
    )
    parser.add_argument(
        "--by",
        choices=("team", "conference", "division"),
        default="team",
        help="what to estimate a season's error for (default: team)",
    )
    parser.add_argument("--team", help="print this team's game log instead")
    parser.add_argument("--top", type=int, default=15, help="rows from each end")
    parser.add_argument(
        "--min-games", type=int, default=1, help="leave out shorter team-seasons"
    )
    parser.add_argument("--null-draws", type=int, default=DEFAULT_NULL_DRAWS)
    parser.add_argument(
        "--refresh", action="store_true", help="replay even if a cached one is current"
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE.total_seconds() / 3600,
        help="replay again when the cached one is older than this",
    )
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()

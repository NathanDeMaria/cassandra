"""Test an idea against a model's residuals before building it.

    python evidence.py --league ncaafb --model glicko_margin_units --offseason \\
        --division FBS --since 2015 --until 2025
    python evidence.py --league ncaafb --model glicko_full --team-seasons ideas.csv
    python evidence.py --league ncaafb --model glicko_full --games travel.csv

State the idea as a table and hand it over:

- `--team-seasons FILE`: a csv with `team` (the model's name for it, as
  `team_seasons.py` prints it) or `espn_id`, `year`, and one column per
  feature. Numbers are fit as slopes, true/false as a difference, text one
  level at a time.
- `--games FILE`: a csv with `game_id` and one column per feature, signed so
  positive favours the home team.
- `--offseason`: the facts `cassandra.offseason` reads -- why last season's
  coach is gone, whether the week-one quarterback is new, and his quality
  change per 0.1 EPA/attempt. Built in because they're the facts the
  offseason model was built on, which makes them the check that this
  script reproduces a known answer: against `glicko_margin_units`, FBS
  2015-2025, a coach who left for a job is about -2.5 early and a fired
  one nothing; against `glicko_margin_units_offseason`, which shifts for
  them, both should be near zero.

Read the per-feature table first -- `slope` with its cluster-robust `t` and
the permutation `perm_p` beside it -- then the joint fit if two features
travel together, then the cross-validated gain, which is the number to
weigh against a search's. See `cassandra.evidence` for the details.
"""

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path

import pandas as pd
from call_it_what_you_want import TeamNamer

from cassandra.evidence import (
    DEFAULT_PERMUTATIONS,
    cross_validated_gain,
    feature_effects,
    game_feature_effects,
    joint_fits,
    prepare_features,
)
from cassandra.offseason import OffseasonFacts
from cassandra.replay_cache import DEFAULT_MAX_AGE, load_replay
from cassandra.residuals import add_residuals, team_tiers
from cassandra.team_seasons import EARLY_GAMES, label_tiers, select, team_games


def _offseason_features(league: str, years: list[int]) -> pd.DataFrame:
    facts = OffseasonFacts.for_league(league)
    rows = [
        {
            "team": team,
            "year": year,
            "coach": fact.coach_departure,
            "new_qb": float("nan")
            if fact.new_quarterback is None
            else float(fact.new_quarterback),
            # Per 0.1 EPA per attempt, the unit `cassandra.offseason` reports in.
            "qb_quality_per_0.1": fact.quality_change * 10,
        }
        for year in years
        for team, fact in facts.seasons(year)
    ]
    if not rows:
        raise SystemExit(f"--offseason: no offseason facts on file for {league}")
    return pd.DataFrame(rows)


def _read_team_seasons(path: Path, league: str, teams: set[str]) -> pd.DataFrame:
    features = pd.read_csv(path)
    if "team" not in features.columns:
        if "espn_id" not in features.columns:
            raise SystemExit(f"{path}: needs a `team` or an `espn_id` column")
        namer = TeamNamer.for_league(league)
        by_id = {str(namer.espn_id(team)): team for team in teams}
        features["team"] = features.pop("espn_id").astype(str).map(by_id)
    unknown = features["team"].isna() | ~features["team"].isin(teams)
    if unknown.any():
        print(
            f"  ({unknown.sum()} of {len(features)} rows name no team the replay has)"
        )
    return features[~unknown]


def _print_effects(effects: list, title: str) -> None:
    if not effects:
        print(f"\n{title}: nothing to fit (no feature varies in the sample)")
        return
    table = pd.DataFrame([e._asdict() for e in effects])
    table["market"] = [
        "-" if pd.isna(s) else f"{s:+.2f} ({s / se:+.1f}t)" if se > 0 else f"{s:+.2f}"
        for s, se in zip(table["market_slope"], table["market_se"])
    ]
    shown = table[
        [
            "feature",
            "window",
            "team_seasons",
            "games",
            "nonzero",
            "feature_mean",
            "slope",
            "se",
            "t",
            "perm_p",
            "n_lined",
            "market",
        ]
    ].rename(
        columns={"team_seasons": "seasons", "feature_mean": "mean", "n_lined": "lined"}
    )
    print(
        f"\n{title}   (slope: points of own-side residual per unit; market: slope of market - model)"
    )
    text = shown.to_string(
        index=False,
        formatters={
            "mean": "{:.3f}".format,
            "slope": "{:+.2f}".format,
            "se": "{:.2f}".format,
            "t": "{:+.1f}".format,
            "perm_p": "{:.3f}".format,
        },
    )
    print("\n".join(f"  {line}" for line in text.splitlines()))


async def _main(args: argparse.Namespace) -> None:
    replay = await load_replay(
        args.league,
        args.model,
        refresh=args.refresh,
        max_age=timedelta(hours=args.max_age_hours),
    )
    print(replay.describe())
    scored = add_residuals(replay.predictions)

    if args.games is not None:
        features = pd.read_csv(args.games)
        if "game_id" not in features.columns:
            raise SystemExit(f"{args.games}: needs a `game_id` column")
        years = scored[["game_id", "year"]].assign(
            game_id=scored["game_id"].astype(str)
        )
        in_span = select(years, since=args.since, until=args.until)
        features = features[
            features["game_id"].astype(str).isin(set(in_span["game_id"]))
        ]
        _print_effects(
            game_feature_effects(scored, features, permutations=args.permutations),
            f"{len(features)} games",
        )
        return

    games = label_tiers(team_games(scored), team_tiers(scored, args.league))
    try:
        games = select(games, args.division, args.since, args.until)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.offseason:
        raw = _offseason_features(args.league, sorted(games["year"].unique()))
    elif args.team_seasons is not None:
        raw = _read_team_seasons(args.team_seasons, args.league, set(games["team"]))
    else:
        raise SystemExit(
            "say what to test: --offseason, --team-seasons FILE or --games FILE"
        )
    features = prepare_features(raw)
    in_sample = features.merge(
        games[["team", "year"]].drop_duplicates(), on=["team", "year"]
    )
    print(
        f"{len(in_sample)} team-seasons with features in the sample "
        f"({games[['team', 'year']].drop_duplicates().shape[0]} team-seasons, "
        f"{len(games)} team-games after the filters)"
    )

    _print_effects(
        feature_effects(games, in_sample, args.early, args.permutations),
        "one feature at a time",
    )
    for fit in joint_fits(games, in_sample, args.early):
        coefficients = ",  ".join(
            f"{row.feature} {row.slope:+.2f} ({row.t:+.1f}t)"
            for row in fit.coefficients.itertuples()
        )
        print(
            f"\njoint, {fit.window} ({fit.team_seasons} team-seasons): {coefficients}"
        )

    gain = cross_validated_gain(scored, games, in_sample, args.early)
    print(
        f"\ncross-validated (odd<->even seasons) shift of predicted margins, "
        f"{gain.touched} of {gain.games} games moved:"
    )
    print(
        f"  margin MAE {gain.mae_before:.4f} -> {gain.mae_change:+.4f};  "
        f"brier {gain.brier_before:.5f} -> {gain.brier_change:+.6f}   (negative is better)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--league", required=True)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--offseason", action="store_true", help="test the offseason facts"
    )
    source.add_argument("--team-seasons", type=Path, help="csv of team-season features")
    source.add_argument("--games", type=Path, help="csv of game features, home-signed")
    parser.add_argument("--division", help="only team-seasons in this division")
    parser.add_argument("--since", type=int, help="only this season and later")
    parser.add_argument("--until", type=int, help="only this season and earlier")
    parser.add_argument(
        "--early",
        type=int,
        default=EARLY_GAMES,
        help=f"a team's first N games are its early window (default {EARLY_GAMES})",
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
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

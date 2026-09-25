"""Ask a model where it is wrong, so the next signal can be chosen.

    python diagnose.py --league nfl --model margin_blend

`evaluate_models.py` scores every model with one number per league, which
picks between models and says nothing about what to build. This runs the same
replay and slices the residuals instead -- see `cassandra.residuals` for what
the columns mean and, more to the point, for what they don't.

The expensive part is the replay, and it is the same replay `evaluate` does,
so this is a separate entry point rather than more output on that one: a
question about which signal to add next is asked a few times a season, and
paying for it on every scheduled evaluate would be paying for it daily.

What to read, in order:

- **`mae ceiling`.** What a *perfect* correction of this axis would recover,
  which is the budget any parameter built on it competes for. Read it first:
  the payoff is quadratic in the bias, so an axis can be several sigma from
  its null and still have a ceiling of a thousandth of a point. Three
  matchup terms were built against axes whose ceilings were 0.0002 to
  0.0017, and none of them moved the model.
- **`signal_points` against the model's margin MAE.** An axis worth building
  for is one where the structure is a real fraction of the error. Against
  ncaafb around 13 and nfl around 10, a tenth of a point is not a project
  however many sigma it is.
- **`sigma`.** Whether the structure is there at all. Under about 3 the axis
  is flat and the slices that look interesting are what noise does.
- **`market_gap` across the slices.** Where the model loses to the closing
  line, which is the closest thing available to a list of what the model
  doesn't know. Its shape across an axis is worth more than its level.
- **`slope`**, where a slice's bias is flat and something still looks
  wrong. The actual margin on the predicted one, within the slice: under 1
  the predictions there are spread too wide, which a mean can't see because
  the modest favorites' shortfall and the big ones' excess cancel. Cross-
  division games are where it bites -- 0.85 for FBS hosting FCS on ncaafb.

Nothing here is a decision. An axis with structure on it says the error is
predictable *after* the game; whether it is predictable before one is the
question a feature has to answer, and the only one that pays.
"""

import argparse
import asyncio
from datetime import datetime

import pandas as pd

from cassandra.constants import CASSANDRA_HOME
from cassandra.replay_cache import load_replay
from cassandra.residuals import (
    AxisReport,
    add_residuals,
    axis_report,
    classification_axes,
    home_field_by,
    home_field_report,
    standard_axes,
)

_DIAGNOSTICS_DIR = CASSANDRA_HOME / "diagnostics"


def _print_report(report: AxisReport, margin_mae: float) -> None:
    share = report.signal_points / margin_mae if margin_mae else float("nan")
    print(f"\n{report.axis}   (bias {report.overall_bias:+.3f} overall)")
    print(
        f"  dispersion {report.dispersion:6.3f}   null {report.null_dispersion:6.3f}"
        f"   sigma {report.sigma:7.2f}"
        f"   signal {report.signal_points:6.3f} pts ({share:5.2%} of MAE)"
    )
    # The number that decides whether to build: what a perfect correction of
    # this axis would actually recover. Usually far smaller than `signal`,
    # because the payoff is quadratic in the bias -- see `AxisReport`.
    print(
        f"  mae ceiling {report.mae_ceiling:7.5f} pts"
        f"   ({report.mae_ceiling / margin_mae if margin_mae else float('nan'):6.3%} of MAE)"
    )
    # `home_field` reuses SliceStats to stay one type, but its two live
    # columns are a team's home edge and its rating error rather than a
    # slice's bias and noise. Printing them under the generic headings is how
    # a reader concludes the model is 4 points off on Air Force.
    if report.axis == "home_field":
        print(f"  {'team':<24}{'games':>8}{'home_excess':>13}{'rating_err':>12}")
        for s in report.slices:
            print(
                f"  {s.label[:24]:<24}{s.n:>8}"
                f"{s.margin_bias:>13.3f}{s.win_prob_bias:>12.3f}"
            )
        return
    print(
        f"  {'slice':<24}{'n':>8}{'bias':>9}{'mae':>8}{'slope':>7}"
        f"{'n_lined':>9}{'mkt_gap':>9}"
    )
    for s in report.slices:
        print(
            f"  {s.label[:24]:<24}{s.n:>8}{s.margin_bias:>9.3f}{s.margin_mae:>8.2f}"
            f"{s.margin_slope:>7.3f}{s.n_lined:>9}{s.market_gap:>9.3f}"
        )


def _print_home_field_by(scored: pd.DataFrame, division: pd.Series) -> None:
    """Home advantage pooled by the home team's division.

    The one theme every classified league can be pooled on without a second
    data source, and on ncaafb the one that decides whether a single
    `home_advantage` is the right shape: FBS sits +0.8 over the constant
    and D-II/D-III -0.4 under it. A team is filed under the division it
    played most of its games in, so a program that moved up is counted once.
    """
    by_team = (
        pd.DataFrame({"team": scored["home_team"], "division": division})
        .groupby("team")["division"]
        .agg(lambda s: s.mode().iloc[0])
    )
    rows = home_field_by(
        scored, {str(team): str(label) for team, label in by_team.items()}
    )
    if not rows:
        return
    print("\nhome_field_by division   (excess = own residual at home minus away)")
    print(
        f"  {'division':<24}{'home':>7}{'away':>7}{'at_home':>9}{'away':>8}"
        f"{'excess':>8}{'se':>6}"
    )
    for t in rows:
        print(
            f"  {t.label[:24]:<24}{t.home_games:>7}{t.away_games:>7}{t.at_home:>9.3f}"
            f"{t.away:>8.3f}{t.excess:>8.3f}{t.se:>6.2f}"
        )


async def _main(
    league: str, model: str, permutations: int, top_teams: int, refresh: bool
) -> None:
    # Under the search's own priors, like `evaluate_models` -- `load_replay`
    # replays that way. Without it a diagnostic reads whichever priors file
    # the machine happened to have, which on one laptop meant
    # `GlickoPredictor` replayed warm and `CompoundGlickoPredictor` cold, and
    # the residuals were compared across a difference in starting
    # information rather than in modelling.
    replay = await load_replay(league, model, refresh=refresh)
    print(replay.describe())
    scored = add_residuals(replay.predictions)
    margin_mae = scored["margin_residual"].abs().mean()
    print(f"{league}/{model}: {len(scored)} games, margin MAE {margin_mae:.4f}")

    # `classification_axes` is empty for a league nobody has filed divisions
    # for -- every professional one -- and said so out loud rather than
    # silently, since a missing axis looks exactly like an axis with nothing
    # on it once the table is printed.
    classified = classification_axes(scored, league)
    if not classified:
        print(f"  (no division/conference on file for {league}; those axes skipped)")
    axes = {**standard_axes(scored), **classified}

    reports = [
        axis_report(scored, labels, name, permutations=permutations)
        for name, labels in axes.items()
    ]
    reports.append(home_field_report(scored, permutations=permutations))
    if "division" in classified:
        _print_home_field_by(scored, classified["division"])

    rows = []
    for report in reports:
        shown = report
        if report.axis in ("home_team", "home_field", "conference"):
            # A row per team is hundreds of lines and the tails are the only
            # part anybody reads. The csv keeps all of them; the terminal
            # gets the ends, which is where a real effect would show.
            ends = report.slices[:top_teams] + report.slices[-top_teams:]
            shown = report._replace(slices=ends)
        _print_report(shown, margin_mae)
        for s in report.slices:
            rows.append(
                {
                    "league": league,
                    "model": model,
                    "axis": report.axis,
                    "overall_bias": report.overall_bias,
                    "dispersion": report.dispersion,
                    "null_dispersion": report.null_dispersion,
                    "sigma": report.sigma,
                    "signal_points": report.signal_points,
                    "margin_mae_overall": margin_mae,
                    **s._asdict(),
                }
            )

    _DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = _DIAGNOSTICS_DIR / f"{timestamp}_{league}_{model}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--league", required=True)
    parser.add_argument(
        "--model",
        default="margin_blend",
        help="model name, without the _result suffix (default: margin_blend)",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=400,
        help="relabelings behind each null (default: 400)",
    )
    parser.add_argument(
        "--top-teams",
        type=int,
        default=10,
        help="how many teams from each end of a per-team table to print",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="replay even if the cached replay is current",
    )
    args = parser.parse_args()
    asyncio.run(
        _main(args.league, args.model, args.permutations, args.top_teams, args.refresh)
    )

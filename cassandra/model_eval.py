from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from endgame.types import Season

from .brier import brier_score_df
from .columns import GameDfColumns
from .odds import OddsDatabase
from .predictor import OptimizationConfig, Predictor, load_predictor
from .predictor import frame as frames
from .predictor.config import load_predictor_class
from .predictor.opponent_prior import OpponentPriorManager
from .prob_to_margin import (
    BaseProbToMarginFitter,
    BaseProbToMarginPredictor,
    IsotonicProbToMarginFitter,
    LogisticProbToMarginFitter,
    MaeLogisticProbToMarginFitter,
)
from .save_predictions import build_predictions_df, join_with_odds, read_league

DEFAULT_FITTERS: dict[str, BaseProbToMarginFitter] = {
    "isotonic": IsotonicProbToMarginFitter(),
    "logistic": LogisticProbToMarginFitter(),
    # The least-squares logistic's sibling, fit on the loss `_best_fit` and
    # the `margin_mae` objective actually judge by. Added rather than
    # replacing "logistic": the two answer different questions -- expected
    # margin vs. the margin to bet -- and a release that wants the mean can
    # still be pointed at the fit that estimates it.
    "logistic_mae": MaeLogisticProbToMarginFitter(),
}


#: How far a league's lined-game count may fall between evaluations before
#: the run is treated as broken rather than quiet. Generous, because the
#: number legitimately moves: a replay picks up games as a season goes on,
#: and the odds database only covers the seasons a book was quoted for.
#: Losing a third of a league's lines between two runs is not that.
SPREAD_COVERAGE_FLOOR = 0.67

#: Below this a league has too few lines to say anything. ncaafb sat at 184
#: of 75,111 games for most of a season; a handful either way there is the
#: odds history growing, not a pipeline breaking.
SPREAD_COVERAGE_MIN_GAMES = 20


class SpreadCoverageDropped(Exception):
    """A league lost most of the lines it had at the last evaluation.

    The backstop, and deliberately the least specific check in the stack. It
    knows nothing about ESPN, horizons, or page limits -- only that cassandra
    scored fewer games against the market than it did last time, which is
    what *every* upstream odds failure eventually looks like from here,
    including the ones nobody has thought of yet.

    It is the last line rather than the first because it is also the slowest
    to fire: a truncated pull is visible in the odds job within minutes, and
    only reaches this once an evaluate has replayed every model. The value
    is that it catches breakage originating in a repo cassandra doesn't
    control, which is where this class of bug has actually come from.
    """


def spread_coverage_drops(
    previous: pd.DataFrame, current: pd.DataFrame
) -> list[str]:
    """Leagues whose lined-game count collapsed since the last evaluation.

    Per league rather than overall: one league losing its odds is invisible
    in a total dominated by another, and the leagues are fetched by separate
    jobs that fail separately.

    Compared at the maximum over a league's models, not the mean. Models
    disagree about `n_spread_games` when one of them replays a shorter
    history, and the question here is "did the odds database lose games",
    which the best-covered model answers.
    """
    if previous.empty or current.empty:
        return []

    def by_league(frame: pd.DataFrame) -> Mapping[str, int]:
        if not {"league", "n_spread_games"} <= set(frame.columns):
            return {}
        counts = frame.groupby("league")["n_spread_games"].max()
        return {str(k): int(v) for k, v in counts.items() if pd.notna(v)}

    was, now = by_league(previous), by_league(current)
    problems = []
    for league, had in sorted(was.items()):
        if had < SPREAD_COVERAGE_MIN_GAMES:
            continue
        # A league that dropped out of this run entirely is not a coverage
        # problem -- `--league` scopes an evaluate, and an unscored league
        # has no number to compare.
        if league not in now:
            continue
        has = now[league]
        if has < had * SPREAD_COVERAGE_FLOOR:
            problems.append(
                f"{league}: {had} lined games at the last evaluation, {has} now"
            )
    return problems


def prior_path(predictor_class: type[Predictor], league: str) -> Path | None:
    """Where `predictor_class` stashes its opponent priors for `league`.

    `None` for a class that builds none, which is every predictor outside
    the Glicko family. Read off a throwaway instance because the path
    belongs to the manager and the manager is built in `__init__`;
    `cassandra.batch.manifest` finds it the same way, and this is the copy
    both callers share.
    """
    manager = getattr(predictor_class(league), "_prior_manager", None)
    return None if manager is None else manager._prior_path


def _prior_override(league: str, priors_path: Path | None) -> dict[str, Any]:
    """Constructor arguments that point a predictor's priors at `priors_path`.

    Empty for None, which leaves the class's shared default in place. Only
    called for a class `prior_path` says has priors: the others don't take
    the argument.
    """
    if priors_path is None:
        return {}
    return {"opponent_prior_manager": OpponentPriorManager(league, path=priors_path)}


def rebuild_priors(
    predictor_class: type[Predictor],
    league: str,
    params: Mapping[str, float | str],
    seasons: Sequence[Season],
    odds_db: OddsDatabase,
    priors_path: Path | None = None,
) -> bool:
    """Replay once with callbacks on, so the priors a fit started from exist.

    `GlickoPredictor.__init__` seeds its ratings from
    `{league}_{class}_priors.json`, and `postrun_callback` is the only thing
    that writes one. A search gets that file because `optimize.py` makes this
    pass before it starts; a *scoring* replay never did, so a model was fit
    from a warm start and then scored from a cold one, and the two numbers
    differed by 0.0007 brier on ncaafb/glicko_full with nothing saying so.

    `params` is the pins alone, for the same reason `optimize.py` passes
    those: the priors a search started from were built before it had fitted
    anything, so rebuilding them from the *fitted* values would produce a
    different file than the one the fit actually used and leave the gap
    open in the other direction.

    Returns whether anything was built, so a caller can skip the second
    replay for a class that has no priors to build.

    `priors_path` builds the file there instead of at the class's shared
    location under ~/.cassandra/predictor/data. A Batch container has that
    location to itself; a laptop running several sessions does not, and a
    rebuild there deletes the file under every other replay reading it.
    """
    path = prior_path(predictor_class, league)
    if path is None:
        return False
    path = priors_path or path
    # `OpponentPriorManager.save` refuses to overwrite, so a rerun in a warm
    # container -- or a second model of the same class in one evaluate --
    # is an instant ValueError unless the old file goes first. `jobs.py`
    # clears it for the same reason before an optimize child runs.
    path.unlink(missing_ok=True)
    predictor = predictor_class(
        league, **params, **_prior_override(league, priors_path)
    )
    for _ in join_with_odds(predictor, seasons, odds_db, post_callbacks=True):
        pass
    return True


async def get_predictions(
    predictor_config_path: Path,
    league: str,
    state_path: Path,
    priors_from: Path | None = None,
    priors_path: Path | None = None,
) -> pd.DataFrame:
    """Run a predictor over a league's games. The expensive, once-per-predictor step.

    `state_path` is explicit because the config can be a checked-in baseline,
    and the state it produces is generated output that doesn't belong there.

    `priors_from` is the *optimization* config -- `models/<league>/<model>.
    json`, not the result -- and asks for the replay to run under the same
    opponent priors the search did. Without it the replay is cold, which is
    what every scoring path did before and what makes a Glicko model's
    evaluate number disagree with the target its own fit reported. `None`
    keeps that old behaviour, for a caller replaying a config that no
    search produced.

    `priors_path` keeps the warm-up's file there rather than at the shared
    default -- see `rebuild_priors`.
    """
    if priors_from is None:
        predictor = load_predictor(predictor_config_path)
        df = await build_predictions_df(predictor, league, post_callbacks=False)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_state(state_path)
        return df

    # Two passes over one league, so the seasons and the odds are read once
    # rather than per pass.
    seasons, odds_db = await read_league(league)
    config = OptimizationConfig.model_validate_json(priors_from.read_text())
    weeks = frames.weeks_per_season([len(season.weeks) for season in seasons])
    built = rebuild_priors(
        load_predictor_class(config.predictor_class),
        league,
        frames.to_params(config.frame, config.fixed, weeks),
        seasons,
        odds_db,
        priors_path=priors_path,
    )
    # Constructed *after* the warm-up, because the priors are read in
    # `__init__` and a predictor built before it would hold the old file.
    predictor = load_predictor(
        predictor_config_path,
        **(_prior_override(league, priors_path) if built else {}),
    )
    df = pd.DataFrame(
        [
            asdict(prediction)
            for prediction in join_with_odds(
                predictor, seasons, odds_db, post_callbacks=False
            )
        ]
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_state(state_path)
    return df


class ScoredPredictions(NamedTuple):
    """Metrics, plus the fit they were computed with.

    The fit used to be discarded once against_spread_accuracy was out of it,
    which left no way to ask what margin a given win probability implies. It
    comes back out so callers can serialize it next to the model's ratings.
    """

    metrics: dict[str, float]
    margin_predictor: BaseProbToMarginPredictor


def score_predictions(
    df: pd.DataFrame, fitter: BaseProbToMarginFitter
) -> ScoredPredictions:
    """Score a predictor's predictions against a single prob-to-margin fitter.

    Cheap relative to get_predictions, so callers comparing multiple fitters
    should call this once per fitter on the same df rather than re-fetching
    predictions each time.
    """
    if df.empty:
        # The same guard `brier_score_df` has, and it has to be here too:
        # that one is called further down, and the assign below reaches for
        # columns an empty frame doesn't have. Without this a caller gets
        # `AttributeError: 'DataFrame' object has no attribute 'home_score'`,
        # which reads like a schema bug rather than "there were no games" --
        # publish spent a run failing that way on a league whose season had
        # been uploaded with nothing in it.
        raise ValueError("No games to score")
    games = df.assign(team1_mov=lambda x: x.home_score - x.away_score)
    # Fit against the margin every game actually finished at, so the fitter
    # trains on the whole schedule instead of only the games a book put a
    # line on. The betting metrics below still need the line, and still only
    # cover that subset.
    margin_predictor = fitter.fit_df(games)
    scored = games.assign(
        predicted_margin=lambda x: margin_predictor.predict_margins(
            x[GameDfColumns.TEAM1_WIN_PROB].to_numpy()
        )
    )
    metrics = {
        "brier_score": brier_score_df(df),
        "margin_mae": (scored["predicted_margin"] - scored["team1_mov"]).abs().mean(),
        "n_games": len(scored),
    }

    with_spread = scored[scored[GameDfColumns.SPREAD].notna()]
    if with_spread.empty:
        # A league the odds database doesn't cover at all. Everything above
        # still holds -- the margin fit never needed a line -- but there's
        # nothing to bet into or compare against.
        return ScoredPredictions(
            metrics={
                **metrics,
                "against_spread_accuracy": float("nan"),
                "spread_game_margin_mae": float("nan"),
                "market_margin_mae": float("nan"),
                "n_spread_games": 0,
            },
            margin_predictor=margin_predictor,
        )

    with_spread = with_spread.assign(
        # A spread is quoted from team1's side and team1 covers when
        # spread + mov > 0, so the margin a line implies is its negation.
        # Getting this backwards still produces a plausible-looking MAE.
        market_margin=lambda x: -x.spread,
        bet_team1=lambda x: x.predicted_margin > x.market_margin,
        team1_covered=lambda x: x.spread + x.team1_mov > 0,
        correct_bet=lambda x: x.bet_team1 == x.team1_covered,
    )
    return ScoredPredictions(
        metrics={
            **metrics,
            "against_spread_accuracy": with_spread["correct_bet"].mean(),
            # margin_mae over the lined games only, which is the slice
            # market_margin_mae covers. Margin MAE means little on its own,
            # since game-to-game noise sets a floor no model gets under; the
            # number that carries information is the gap between these two.
            "spread_game_margin_mae": (
                with_spread["predicted_margin"] - with_spread["team1_mov"]
            )
            .abs()
            .mean(),
            "market_margin_mae": (
                with_spread["market_margin"] - with_spread["team1_mov"]
            )
            .abs()
            .mean(),
            "n_spread_games": len(with_spread),
        },
        margin_predictor=margin_predictor,
    )

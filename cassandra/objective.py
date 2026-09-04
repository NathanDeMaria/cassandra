"""What an optimization run is trying to maximize.

The search sees one number per probe, and which number that is decides what
the model becomes. Brier score asks the model to get *win probabilities*
right; `margin_mae` asks it to get *margins* right, which is the question a
point spread actually poses. They are not the same ask -- a rating system
tuned to call favorites correctly can be badly scaled, and a scaling error
costs nothing in brier and everything against a line.

Everything here is phrased as "higher is better", because
`BayesianOptimization` maximizes. Losses are negated on the way out, which is
why the brier objective is called `_negative_brier` and the margin one
negates an error.

Deliberately light on imports: `cassandra.predictor.config` validates an
objective name against this registry, and `cassandra.predictor` is the half
of the package a webapp installs without the `fit` group. So this reads
`prob_to_margin` (numpy, and sklearn only inside a fit) and never
`model_eval`, which would pull s3 in behind it.
"""

from collections.abc import Callable, Mapping
from functools import partial

import numpy as np
import pandas as pd

from .brier import brier_score_df
from .columns import GameDfColumns
from .prob_to_margin import (
    BaseProbToMarginFitter,
    IsotonicProbToMarginFitter,
    MaeLogisticProbToMarginFitter,
)

type Objective = Callable[[pd.DataFrame], float]


def _negative_brier(df: pd.DataFrame) -> float:
    return -brier_score_df(df)


def _negative_margin_mae(df: pd.DataFrame, fitter: BaseProbToMarginFitter) -> float:
    """Minus the MAE of the margins these predictions imply.

    The same quantity `score_predictions` reports as `margin_mae`, computed
    the same way -- fit prob->margin on the games, then score the fitted
    margins against the ones the games finished at. It is recomputed here
    rather than imported so that this module stays out of `model_eval`'s
    dependency tree; `objective_test.py` holds the two definitions together.

    The fit is in-sample, which for the logistic fitters is one parameter
    against tens of thousands of games -- not a way for a model to cheat.
    `IsotonicProbToMarginFitter` has a knot per distinct probability and can
    flatter a model with erratic ones, which is why the default objective is
    the logistic.
    """
    if df.empty:
        raise ValueError("No games to score")
    games = df.assign(team1_mov=lambda x: x.home_score - x.away_score)
    margin_predictor = fitter.fit_df(games)
    predicted = margin_predictor.predict_margins(
        games[GameDfColumns.TEAM1_WIN_PROB].to_numpy()
    )
    return -float(np.abs(predicted - games[GameDfColumns.TEAM1_MOV].to_numpy()).mean())


_OBJECTIVES: Mapping[str, Objective] = {
    "brier": _negative_brier,
    # The margin objective and the fit it scores through both minimize
    # absolute error, so the search is charged for its ratings rather than
    # for a least-squares transform that runs systematically wide.
    "margin_mae": partial(_negative_margin_mae, fitter=MaeLogisticProbToMarginFitter()),
    # Available for a model whose prob->margin relationship is genuinely not
    # logistic. It fits many more knots than there are constraints holding
    # them down, so a run that only wins under this one has probably found
    # the fitter's flexibility rather than a better model.
    "margin_mae_isotonic": partial(
        _negative_margin_mae, fitter=IsotonicProbToMarginFitter()
    ),
}

#: Every objective a config may name. `DEFAULT_OBJECTIVE` is what a config
#: that names none gets, which keeps every config written before objectives
#: existed scoring exactly as it did.
OBJECTIVE_NAMES = tuple(_OBJECTIVES)
DEFAULT_OBJECTIVE = "brier"


def get_objective(name: str) -> Objective:
    """The scoring function `name` refers to, higher-is-better.

    Raises rather than falling back to brier: a typo that silently optimized
    a different metric than the config asked for would show up only as a
    `target` that can't be compared with anything.
    """
    objective = _OBJECTIVES.get(name)
    if objective is None:
        raise ValueError(
            f"unknown objective: {name!r}; expected one of {', '.join(OBJECTIVE_NAMES)}"
        )
    return objective

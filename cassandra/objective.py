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
`model_eval`, which would pull s3 in behind it. call-it-what-you-want, which
`brier_fbs` reads divisions from, is a serving dependency already and pure
python.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, partial

import numpy as np
import pandas as pd
from call_it_what_you_want import (
    Classifications,
    TeamNamer,
    default_classifications,
    registry_league,
)

from .brier import brier_score_df
from .columns import GameDfColumns
from .prob_to_margin import (
    BaseProbToMarginFitter,
    IsotonicProbToMarginFitter,
    MaeLogisticProbToMarginFitter,
)

type Scorer = Callable[[pd.DataFrame], float]


@dataclass(frozen=True)
class Objective:
    """A number to maximize, and what it has to read to produce one.

    `reads` is the part that isn't obvious. A probe's frame used to carry
    every column a prediction has -- twelve of them -- because the replay
    produced them and nothing said which were wanted. Two were expensive:
    the kickoff date, which pandas converts to datetime64 for nobody, and
    `spread`, which only exists because the odds were read out of s3 first.
    None of the three objectives here reads either. On ncaafb that was
    1.0s of a 4.5s probe, a thousand times a search, plus ~1,800 s3 objects
    and 72,000 parsed odds snapshots before the first probe -- for 450
    priced ncaafb games the search never looks at.

    So an objective says what it reads and the search builds that. The
    declaration is load-bearing in two places: `save_predictions.
    predictions_frame` builds these columns and no others, and `optimize.py`
    reads the odds *only* when `spread` is among them. A market objective
    added later declares `spread` and the odds come back on their own.

    Both halves are tested, because a wrong declaration is the kind of bug
    that reads as a model result: `objective_test` scores every objective
    from a frame holding only what it declared, which raises a KeyError on
    anything undeclared, and checks the names against `_Prediction`'s
    fields so a typo can't quietly ask for a column that never existed.
    """

    score: Scorer
    #: `_Prediction` field names -- what the frame must carry. Not
    #: `GameDfColumns`: `team1_mov` is derived inside the objective from the
    #: two scores, so what the *frame* needs is the scores.
    reads: frozenset[str]

    def __call__(self, df: pd.DataFrame) -> float:
        return self.score(df)

    @property
    def needs_odds(self) -> bool:
        """Whether scoring this needs the odds database read at all.

        `spread` is the only column that comes from it; everything else on a
        prediction comes out of the replay.
        """
        return GameDfColumns.SPREAD in self.reads


#: What `brier_score_df` reads.
_BRIER_READS = frozenset({GameDfColumns.TEAM1_WIN_PROB, GameDfColumns.TEAM1_WIN})

#: What the margin objectives read: the probability, and the two scores the
#: margin they are graded against is the difference of. The fitters read
#: `team1_mov` and `team1_win_prob` and nothing else -- see
#: `prob_to_margin.base_fit.fit_df` -- and `team1_mov` is derived here.
_MARGIN_READS = frozenset({GameDfColumns.TEAM1_WIN_PROB, "home_score", "away_score"})


#: What `_negative_fbs_brier` reads: brier's columns, and who played when.
_FBS_BRIER_READS = _BRIER_READS | {"home_team", "away_team", "year"}

#: The league and division `brier_fbs` scores.
_FBS_LEAGUE = "ncaafb"
_FBS = "FBS"


def _negative_brier(df: pd.DataFrame) -> float:
    return -brier_score_df(df)


@cache
def _fbs_lookup() -> tuple[TeamNamer, Classifications, str | None]:
    """The namer and classification table `_in_fbs` reads, built once."""
    return (
        TeamNamer.for_league(_FBS_LEAGUE),
        default_classifications(),
        registry_league(_FBS_LEAGUE),
    )


@cache
def _in_fbs(team: str, year: int) -> bool:
    """Whether call-it-what-you-want has `team` in FBS in `year`.

    The lookup `cassandra.residuals.team_tiers` makes, minus its handling of
    the lumped lower divisions, which can't make a team FBS. Names are
    canonical already -- a replay renames before the predictor sees a game --
    so the id is read straight off the name. Cached per team-season: a
    search scores the same 150,000 sides every probe.
    """
    namer, classifications, registry = _fbs_lookup()
    espn_id = namer.espn_id(team)
    if espn_id is None or registry is None:
        return False
    found = classifications.classification_in(espn_id, year, registry)
    return found is not None and found.division == _FBS


def _negative_fbs_brier(df: pd.DataFrame) -> float:
    """Minus the brier score of the games between two FBS teams.

    Pooled brier is mostly games below FBS: ncaafb's replay is 75,882 games,
    17,072 of them between two FBS teams. A knob that only touches FBS teams
    -- roster talent is the first -- can move the pooled number by 0.00001
    while moving these games by 0.000065, and a search on pooled brier can't
    tell its settings apart.
    """
    if df.empty:
        raise ValueError("No games to score")
    years = df["year"].to_numpy()
    fbs = np.fromiter(
        (
            _in_fbs(home, int(year)) and _in_fbs(away, int(year))
            for home, away, year in zip(df["home_team"], df["away_team"], years)
        ),
        dtype=bool,
        count=len(df),
    )
    if not fbs.any():
        raise ValueError("No games between two FBS teams to score")
    return -brier_score_df(df[fbs])


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
    "brier": Objective(_negative_brier, _BRIER_READS),
    # The margin objective and the fit it scores through both minimize
    # absolute error, so the search is charged for its ratings rather than
    # for a least-squares transform that runs systematically wide.
    "margin_mae": Objective(
        partial(_negative_margin_mae, fitter=MaeLogisticProbToMarginFitter()),
        _MARGIN_READS,
    ),
    # Available for a model whose prob->margin relationship is genuinely not
    # logistic. It fits many more knots than there are constraints holding
    # them down, so a run that only wins under this one has probably found
    # the fitter's flexibility rather than a better model.
    "margin_mae_isotonic": Objective(
        partial(_negative_margin_mae, fitter=IsotonicProbToMarginFitter()),
        _MARGIN_READS,
    ),
    # Brier on the games between two FBS teams, for a college football knob
    # the pooled number can't see. Only meaningful on ncaafb: any other
    # league has no FBS games and raises.
    "brier_fbs": Objective(_negative_fbs_brier, _FBS_BRIER_READS),
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

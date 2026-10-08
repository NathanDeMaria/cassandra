import numpy as np
import pandas as pd
import pytest

from .brier import brier_score_df
from .model_eval import score_predictions
from .objective import (
    DEFAULT_OBJECTIVE,
    OBJECTIVE_NAMES,
    Objective,
    _negative_margin_mae,
    get_objective,
)
from .prob_to_margin import MaeLogisticProbToMarginFitter
from .save_predictions import _PREDICTION_FIELDS

#: Two FBS teams, then an FCS team against one of them, round and round.
_MATCHUPS = [
    ("Ohio State Buckeyes", "Michigan Wolverines"),
    ("Troy Trojans", "Ohio State Buckeyes"),
    ("North Dakota State Bison", "Troy Trojans"),
]


def _games(win_probs: np.ndarray, margins: np.ndarray) -> pd.DataFrame:
    """A predictions frame in the shape `join_with_odds` produces one."""
    teams = [_MATCHUPS[i % len(_MATCHUPS)] for i in range(len(margins))]
    return pd.DataFrame(
        {
            "year": np.full(len(margins), 2020),
            "home_score": np.maximum(margins, 0) + 60,
            "away_score": 60 - np.minimum(margins, 0),
            "team1_win": margins > 0,
            "team1_win_prob": win_probs,
            "spread": np.full(len(margins), np.nan),
            "home_team": [home for home, _ in teams],
            "away_team": [away for _, away in teams],
        }
    )


def _sample_games() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    win_probs = rng.uniform(0.05, 0.95, 400)
    margins = np.round(
        11 * np.log(win_probs / (1 - win_probs)) + rng.normal(0, 10, 400)
    )
    return _games(win_probs, margins)


def test_the_default_objective_is_the_one_every_config_already_searched() -> None:
    """Every checked-in config predates objectives and was scored on brier.

    If this changes, those configs quietly start searching for something
    else, and their `target`s stop being comparable with the results already
    published under the same names.
    """
    assert DEFAULT_OBJECTIVE == "brier"


def test_brier_objective_is_the_brier_score_negated() -> None:
    """Higher is better for the optimizer; lower is better for brier."""
    games = _sample_games()

    score = get_objective("brier")(games)

    assert score == pytest.approx(
        -score_predictions(games, MaeLogisticProbToMarginFitter()).metrics[
            "brier_score"
        ]
    )
    assert score < 0


def test_margin_objective_is_the_margin_mae_evaluate_reports() -> None:
    """The search and the evaluations csv have to mean the same thing by it.

    `objective.py` recomputes margin_mae rather than importing
    `score_predictions`, to keep s3 out of `cassandra.predictor`'s import
    graph. This is the test that keeps the copy honest.
    """
    games = _sample_games()
    fitter = MaeLogisticProbToMarginFitter()

    objective_score = _negative_margin_mae(games, fitter)

    assert objective_score == pytest.approx(
        -score_predictions(games, fitter).metrics["margin_mae"]
    )


def test_margin_objective_prefers_the_better_scaled_of_two_models() -> None:
    """What brier can't see, and the whole reason this objective exists.

    Both models rank the games identically -- one just calls them with
    probabilities twice as far from a coin flip. The prob->margin fit
    absorbs a *constant* rescaling, so this pair is a sanity check that the
    objective is finite and comparable rather than a claim about which wins;
    what it does hold is that a model whose probabilities carry no ordering
    at all scores worse than one whose do.
    """
    rng = np.random.default_rng(3)
    margins = rng.normal(0, 12, 500).round()
    informative = 1 / (1 + np.exp(-margins / 11))
    uninformative = np.full(len(margins), 0.5)

    objective = get_objective("margin_mae")

    assert objective(_games(informative, margins)) > objective(
        _games(uninformative, margins)
    )


def test_fbs_brier_scores_only_the_games_between_two_fbs_teams() -> None:
    games = _sample_games()
    fbs = games.home_team != "North Dakota State Bison"

    score = get_objective("brier_fbs")(games)

    assert score == pytest.approx(-brier_score_df(games[fbs]))
    assert score != pytest.approx(get_objective("brier")(games))


def test_fbs_brier_reads_the_division_of_the_season_played() -> None:
    """North Dakota State is FBS from 2026, and FCS in every season before."""
    games = _sample_games().assign(
        home_team="North Dakota State Bison", away_team="Troy Trojans"
    )
    with pytest.raises(ValueError, match="two FBS teams"):
        get_objective("brier_fbs")(games)

    assert np.isfinite(get_objective("brier_fbs")(games.assign(year=2026)))


def test_an_unknown_objective_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="margin_mae"):
        get_objective("margin_rmse")


def test_every_registered_objective_scores_a_frame_of_games() -> None:
    games = _sample_games()

    for name in OBJECTIVE_NAMES:
        assert np.isfinite(get_objective(name)(games))


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_no_objective_scores_an_empty_schedule(name: str) -> None:
    """The message a league with nothing uploaded should get.

    Not an AttributeError from reaching for `home_score` on a frame that has
    no columns, which is what an empty frame does to every one of these.
    """
    with pytest.raises(ValueError, match="No games to score"):
        get_objective(name)(pd.DataFrame())


# --- what an objective says it reads -----------------------------------------
#
# A search builds the frame from `reads` and skips the odds read unless
# `spread` is in it, so an under-declaration is not a missing column at the
# end of a run -- it is a search that scored something else for six hours.


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_an_objective_asks_for_columns_a_prediction_has(name: str) -> None:
    """`reads` is `_Prediction` field names, checked against the real ones.

    A typo here would be invisible: the frame would simply not carry the
    column, and the objective would fail on a frame that was built exactly
    as asked.
    """
    assert get_objective(name).reads <= set(_PREDICTION_FIELDS)


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_an_objective_scores_from_what_it_declared_and_nothing_more(
    name: str,
) -> None:
    """The declaration is complete, not just correct.

    Scored twice: from the whole frame, and from one narrowed to `reads` the
    way `_score_probe` narrows it. An undeclared column raises a KeyError
    here rather than being quietly available -- which is what makes the
    narrowing safe to do in the search.
    """
    objective = get_objective(name)
    games = _sample_games()
    narrowed = games[[c for c in games.columns if c in objective.reads]]

    assert objective(narrowed) == objective(games)


def test_the_objectives_that_exist_score_against_results_not_lines() -> None:
    """Which is why a search can skip the odds read entirely.

    Not a property of objectives in general -- it is a fact about the ones
    there are, and the one `optimize.py` acts on. A market objective added
    later declares `spread`, `needs_odds` goes true, and the read comes
    back without anybody remembering to re-enable it.
    """
    for name in OBJECTIVE_NAMES:
        assert not get_objective(name).needs_odds


def test_an_objective_that_wants_a_line_says_so() -> None:
    """The other side of `needs_odds`, so it isn't vacuously true."""
    against_the_line = Objective(lambda df: 0.0, frozenset({"spread"}))

    assert against_the_line.needs_odds

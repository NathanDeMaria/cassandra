from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from endgame.types import Season

from .conftest import season_for
from .model_eval import (
    prior_path,
    rebuild_priors,
    score_predictions,
    spread_coverage_drops,
)
from .odds import OddsDatabase
from .predictor import (
    CompoundGlickoPredictor,
    EloPredictor,
    FlatPredictor,
    GlickoPredictor,
    opponent_prior,
)
from .prob_to_margin import BaseProbToMarginFitter, BaseProbToMarginPredictor


class _FixedMarginFitter(BaseProbToMarginFitter):
    """Predicts one margin for every game, so the metrics are hand-checkable."""

    def __init__(self, margin: float) -> None:
        self._margin = margin
        self.fit_win_probs: np.ndarray | None = None

    def fit(
        self, win_probs: np.ndarray, margins: np.ndarray
    ) -> BaseProbToMarginPredictor:
        self.fit_win_probs = win_probs
        return _FixedMarginPredictor(self._margin)


class _FixedMarginPredictor(BaseProbToMarginPredictor):
    # No `kind`, so this stays out of the from_dict registry -- a test double
    # has no business being rehydratable by name.
    def __init__(self, margin: float) -> None:
        self._margin = margin

    def predict_margins(self, win_probs: np.ndarray) -> np.ndarray:
        return np.full(len(win_probs), self._margin)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fixed", "margin": self._margin}


def _games() -> pd.DataFrame:
    # Game 3 has no line, so it's still training data and still counts toward
    # brier_score and margin_mae, but not toward any of the betting metrics.
    return pd.DataFrame(
        [
            {
                "home_score": 24,
                "away_score": 20,  # team1_mov = +4
                "team1_win": True,
                "team1_win_prob": 0.7,
                "spread": -3.0,  # team1 favored by 3, and covers
            },
            {
                "home_score": 10,
                "away_score": 30,  # team1_mov = -20
                "team1_win": False,
                "team1_win_prob": 0.4,
                "spread": 6.0,  # team1 getting 6, and does not cover
            },
            {
                "home_score": 14,
                "away_score": 7,  # team1_mov = +7
                "team1_win": True,
                "team1_win_prob": 0.6,
                "spread": None,
            },
        ]
    )


def test_fits_on_every_game_not_just_the_lined_ones() -> None:
    """The whole point of predicting margin instead of the spread.

    Three games, one of which no book put a line on, all reach the fitter --
    six rows once fit_df mirrors them onto the away side.
    """
    fitter = _FixedMarginFitter(0.0)

    score_predictions(_games(), fitter)

    assert fitter.fit_win_probs is not None
    assert sorted(fitter.fit_win_probs) == pytest.approx([0.3, 0.4, 0.4, 0.6, 0.6, 0.7])


def test_margin_mae_is_zero_for_a_perfect_prediction() -> None:
    game = _games().iloc[[0]]  # team1 wins by 4

    metrics = score_predictions(game, _FixedMarginFitter(4.0)).metrics

    assert metrics["margin_mae"] == pytest.approx(0.0)


def test_scores_margin_against_the_market() -> None:
    """The market's side of the comparison is where the sign is a trap.

    A spread is quoted from team1's side, so the margin it implies is its
    negation. Flipping it leaves an MAE that still looks like a plausible
    number of points.
    """
    # Predicted margin +5 against actual +4, -20 and +7: errors of 1, 25, 2.
    # The market's -3 and +6 imply +3 and -6: errors of 1 and 14.
    metrics = score_predictions(_games(), _FixedMarginFitter(5.0)).metrics

    assert metrics["margin_mae"] == pytest.approx(28 / 3)
    assert metrics["spread_game_margin_mae"] == pytest.approx(13.0)
    assert metrics["market_margin_mae"] == pytest.approx(7.5)


def test_against_spread_accuracy_and_counts() -> None:
    # Predicting +5 beats the -3 line on game 1 (bet team1, and team1 covers)
    # and also beats the +6 line on game 2 (bet team1, but team1 doesn't).
    metrics = score_predictions(_games(), _FixedMarginFitter(5.0)).metrics

    assert metrics["against_spread_accuracy"] == pytest.approx(0.5)
    assert metrics["n_games"] == 3
    assert metrics["n_spread_games"] == 2


def test_scores_a_league_with_no_lines_at_all() -> None:
    """A league the odds database doesn't cover still gets margin metrics.

    Fitting on final scores rather than on closing lines means these leagues
    are no longer a special case -- only the metrics that need a line to
    compare against drop out.
    """
    no_lines = _games().assign(spread=None)

    metrics = score_predictions(no_lines, _FixedMarginFitter(5.0)).metrics

    assert metrics["brier_score"] == pytest.approx((0.3**2 + 0.4**2 + 0.4**2) / 3)
    assert metrics["margin_mae"] == pytest.approx(28 / 3)
    assert metrics["n_games"] == 3
    assert metrics["n_spread_games"] == 0
    assert np.isnan(metrics["spread_game_margin_mae"])
    assert np.isnan(metrics["against_spread_accuracy"])
    assert np.isnan(metrics["market_margin_mae"])


def test_brier_score_covers_games_without_a_line() -> None:
    metrics = score_predictions(_games(), _FixedMarginFitter(5.0)).metrics

    expected = (0.3**2 + 0.4**2 + 0.4**2) / 3
    assert metrics["brier_score"] == pytest.approx(expected)


def test_a_league_with_no_games_says_so() -> None:
    """
    A season can exist and hold nothing -- an upload made before the league
    started playing. That used to surface as an AttributeError about
    `home_score`, from the assign reaching for a column an empty frame has
    no room for, which reads like a schema bug rather than empty input.
    """
    with pytest.raises(ValueError, match="No games to score"):
        score_predictions(pd.DataFrame([]), _FixedMarginFitter(5.0))


def test_an_empty_frame_fails_the_same_way_the_brier_does() -> None:
    """Both entry points into scoring agree about what empty means."""
    from .brier import brier_score_df

    with pytest.raises(ValueError, match="No games to score"):
        brier_score_df(pd.DataFrame([]))


def _priors_dir(monkeypatch, tmp_path) -> Path:
    """Point the prior manager at a temp directory instead of ~/.cassandra."""
    directory = tmp_path / "predictor" / "data"
    directory.mkdir(parents=True)
    monkeypatch.setattr(opponent_prior, "_PREDICTOR_DATA_DIR", directory)
    return directory


def _one_season(n_games: int) -> list[Season]:
    """One season of `n_games` between the same two teams.

    Fifty or more because `OpponentPriorManager.save` drops a team with
    fewer than that -- the guard against a small school that only ever
    played a handful of games against good ones -- so a shorter season
    writes a file with nothing in it and a warm start that isn't one.
    """
    return [season_for(2024, *(f"g{i}" for i in range(n_games)))]


def test_prior_path_is_none_for_a_predictor_that_builds_none() -> None:
    assert prior_path(FlatPredictor, "test_league") is None
    assert prior_path(EloPredictor, "test_league") is None


def test_prior_path_is_per_class_which_is_why_a_class_can_be_missing_one(
    monkeypatch, tmp_path
) -> None:
    """The filename carries the class, so classes never share a warm start.

    This is the shape of the ncaafb surprise: `GlickoPredictor` had a priors
    file on disk and `CompoundGlickoPredictor` did not, so one replayed warm
    and the other cold and the brier gap between them read as a modelling
    result. Nothing is wrong with keying by class -- the ratings a compound
    fit produces are not the ones a plain one does -- but a caller has to
    know the files are separate, and that only the class that ran gets one.
    """
    _priors_dir(monkeypatch, tmp_path)
    glicko = prior_path(GlickoPredictor, "test_league")
    compound = prior_path(CompoundGlickoPredictor, "test_league")
    assert glicko is not None and compound is not None
    assert glicko != compound
    assert "GlickoPredictor" in glicko.name
    assert "CompoundGlickoPredictor" in compound.name


def test_rebuild_priors_skips_a_class_with_nothing_to_build(
    monkeypatch, tmp_path
) -> None:
    directory = _priors_dir(monkeypatch, tmp_path)
    built = rebuild_priors(
        FlatPredictor, "test_league", {}, _one_season(4), OddsDatabase({})
    )
    assert built is False
    # And no replay was paid for: nothing landed in the directory.
    assert list(directory.iterdir()) == []


def test_rebuild_priors_leaves_a_warm_start_behind(monkeypatch, tmp_path) -> None:
    """The point of the whole thing: a predictor built after this starts warm."""
    _priors_dir(monkeypatch, tmp_path)
    assert GlickoPredictor("test_league").ratings == {}

    built = rebuild_priors(
        GlickoPredictor, "test_league", {}, _one_season(60), OddsDatabase({})
    )

    assert built is True
    path = prior_path(GlickoPredictor, "test_league")
    assert path is not None and path.exists()
    assert GlickoPredictor("test_league").ratings != {}


def test_rebuild_priors_clears_the_file_save_refuses_to_overwrite(
    monkeypatch, tmp_path
) -> None:
    """Twice in one process has to work, and used to be a ValueError.

    `OpponentPriorManager.save` raises rather than overwrite, which is why
    `jobs.py` unlinks before an optimize child runs. An evaluate replays
    every model in turn and several of them are the same class, so the
    second one hits the same guard inside a single process.
    """
    _priors_dir(monkeypatch, tmp_path)
    seasons = _one_season(60)
    rebuild_priors(GlickoPredictor, "test_league", {}, seasons, OddsDatabase({}))
    rebuild_priors(GlickoPredictor, "test_league", {}, seasons, OddsDatabase({}))

    path = prior_path(GlickoPredictor, "test_league")
    assert path is not None and path.exists()


def _eval_rows(league: str, n_spread: int, models=("a", "b")) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"league": league, "model": m, "n_spread_games": n_spread}
            for m in models
        ]
    )


def test_a_league_losing_most_of_its_lines_is_reported() -> None:
    """Every upstream odds failure eventually looks like this from here."""
    drops = spread_coverage_drops(_eval_rows("ncaafb", 184), _eval_rows("ncaafb", 40))
    assert len(drops) == 1
    assert "ncaafb" in drops[0] and "184" in drops[0] and "40" in drops[0]


def test_coverage_growing_is_not_a_drop() -> None:
    assert spread_coverage_drops(_eval_rows("ncaafb", 184), _eval_rows("ncaafb", 260)) == []


def test_a_league_with_barely_any_lines_is_not_evidence() -> None:
    """ncaafb sat at a handful of lines for a season; that is history, not a bug."""
    assert spread_coverage_drops(_eval_rows("nfl", 4), _eval_rows("nfl", 0)) == []


def test_a_league_left_out_of_this_run_is_not_a_drop() -> None:
    """`--league` scopes an evaluate, and an unscored league has no number."""
    assert spread_coverage_drops(_eval_rows("ncaafb", 184), _eval_rows("nfl", 90)) == []


def test_the_best_covered_model_answers_for_the_league() -> None:
    """
    Models disagree when one replays a shorter history; the question is
    whether the odds database lost games, not which model saw fewest.
    """
    previous = pd.DataFrame(
        [
            {"league": "ncaafb", "model": "a", "n_spread_games": 184},
            {"league": "ncaafb", "model": "b", "n_spread_games": 20},
        ]
    )
    assert spread_coverage_drops(previous, _eval_rows("ncaafb", 180)) == []


def test_an_empty_table_has_nothing_to_compare() -> None:
    assert spread_coverage_drops(pd.DataFrame([]), _eval_rows("ncaafb", 1)) == []
    assert spread_coverage_drops(_eval_rows("ncaafb", 184), pd.DataFrame([])) == []


def test_rebuild_priors_can_keep_its_file_out_of_the_shared_directory(
    monkeypatch, tmp_path
) -> None:
    """A local replay's warm-up must not delete the file other replays read."""
    shared = _priors_dir(monkeypatch, tmp_path)
    (shared / "test_league_GlickoPredictor_priors.json").write_text('{"x": 1600.0}')
    private = tmp_path / "mine" / "priors.json"

    built = rebuild_priors(
        GlickoPredictor,
        "test_league",
        {},
        _one_season(60),
        OddsDatabase({}),
        priors_path=private,
    )

    assert built is True
    assert private.exists()
    assert (shared / "test_league_GlickoPredictor_priors.json").read_text() == (
        '{"x": 1600.0}'
    )
    manager = opponent_prior.OpponentPriorManager("test_league", path=private)
    warm = GlickoPredictor("test_league", opponent_prior_manager=manager)
    # Warm from the file it built, and not from the shared one's "x".
    assert warm.ratings != {}
    assert "x" not in warm.ratings

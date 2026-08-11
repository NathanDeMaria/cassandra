import json
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from cassandra.model_eval import score_predictions
from cassandra.prob_to_margin import (
    IsotonicProbToMarginFitter,
    LogisticProbToMarginFitter,
)

from .release import (
    IsotonicMarginCalibration,
    LogisticMarginCalibration,
    MarginCalibration,
    Metrics,
    ModelRelease,
    TrainedThrough,
    calibration_from_predictor,
    metrics_from_scored,
)


def _games() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    win_probs = rng.uniform(0.1, 0.9, 200)
    return pd.DataFrame(
        {
            "team1_win_prob": win_probs,
            "team1_mov": 30 * (win_probs - 0.5) + rng.normal(0, 7, 200),
        }
    )


def _release(margin_calibration: MarginCalibration | None = None) -> ModelRelease:
    return ModelRelease(
        run_id="2026-08-10T09:00:12Z",
        league="mens",
        model="glicko_tuned",
        predictor_class="GlickoPredictor",
        margin_calibration=margin_calibration,
        metrics=Metrics(brier_score=0.19, margin_mae=9.1, n_games=100),
        trained_through=TrainedThrough(season_year=2026),
        created_at=datetime(2026, 8, 10),
        created_by="evaluate",
    )


def test_carries_an_isotonic_fit_through_json_and_back() -> None:
    """The end-to-end path: fit here, serialize, rehydrate over there.

    A consumer holds only the parsed JSON -- no fitter, no estimator, no
    games -- and has to get the same margins out.
    """
    predictor = IsotonicProbToMarginFitter().fit_df(_games())
    release = _release(margin_calibration=calibration_from_predictor(predictor))

    rehydrated = ModelRelease.model_validate_json(release.model_dump_json())

    margin_predictor = rehydrated.margin_predictor()
    assert margin_predictor is not None
    win_probs = np.linspace(0.05, 0.95, 50)
    assert margin_predictor.predict_margins(win_probs) == pytest.approx(
        predictor.predict_margins(win_probs)
    )


def test_carries_a_logistic_fit_too() -> None:
    """Both fitters are scoreable, so both have to be shippable."""
    predictor = LogisticProbToMarginFitter().fit_df(_games())
    release = _release(margin_calibration=calibration_from_predictor(predictor))

    rehydrated = ModelRelease.model_validate_json(release.model_dump_json())

    margin_predictor = rehydrated.margin_predictor()
    assert margin_predictor is not None
    win_probs = np.linspace(0.05, 0.95, 50)
    assert margin_predictor.predict_margins(win_probs) == pytest.approx(
        predictor.predict_margins(win_probs)
    )


def test_calibration_dispatches_on_kind() -> None:
    isotonic = calibration_from_predictor(IsotonicProbToMarginFitter().fit_df(_games()))
    logistic = calibration_from_predictor(LogisticProbToMarginFitter().fit_df(_games()))

    assert isinstance(isotonic, IsotonicMarginCalibration)
    assert isinstance(logistic, LogisticMarginCalibration)


def test_rejects_a_calibration_kind_it_doesnt_know() -> None:
    payload = _release().model_dump(mode="json")
    payload["margin_calibration"] = {"kind": "quadratic", "coefficients": [1.0]}

    with pytest.raises(ValidationError):
        ModelRelease.model_validate(payload)


def test_margin_predictor_is_none_when_nothing_was_fit() -> None:
    assert _release().margin_predictor() is None


def test_metrics_map_nan_to_null_not_to_the_literal_nan() -> None:
    """A league with no lines can't be allowed to emit invalid JSON.

    score_predictions returns nan for every betting metric there, and
    json.dumps writes nan as a bare `NaN` token that isn't valid JSON and that
    browsers refuse to parse.
    """
    games = _games().assign(home_score=100, away_score=90, team1_win=True, spread=None)
    scored = score_predictions(games, IsotonicProbToMarginFitter())

    metrics = metrics_from_scored(scored.metrics)

    assert metrics.against_spread_accuracy is None
    assert metrics.market_margin_mae is None
    assert metrics.spread_game_margin_mae is None
    assert metrics.n_spread_games == 0
    payload = json.loads(metrics.model_dump_json())
    assert payload["against_spread_accuracy"] is None
    assert "NaN" not in metrics.model_dump_json()


def test_metrics_keep_the_numbers_when_there_are_lines() -> None:
    games = _games().assign(home_score=100, away_score=90, team1_win=True, spread=-3.0)
    scored = score_predictions(games, IsotonicProbToMarginFitter())

    metrics = metrics_from_scored(scored.metrics)

    assert metrics.against_spread_accuracy is not None
    assert metrics.n_spread_games == len(games)
    assert metrics.brier_score == pytest.approx(scored.metrics["brier_score"])


def test_reading_a_release_never_imports_sklearn() -> None:
    """The property the whole serialization approach exists for.

    A subprocess, because sklearn is already in sys.modules by the time the
    rest of this file has run. If this ever fails, something moved an import
    to module scope and the web image just grew scikit-learn.
    """
    source = """
import json, sys
from cassandra.serving import ModelRelease

release = ModelRelease.model_validate(json.load(sys.stdin))
predictor = release.margin_predictor()
assert predictor is not None
assert float(predictor.predict_margins([0.62])[0]) > 0
print("sklearn" in sys.modules)
"""
    payload = _release(
        margin_calibration=IsotonicMarginCalibration(
            kind="isotonic",
            x_thresholds=[0.2, 0.5, 0.8],
            y_thresholds=[-10.0, 0.0, 10.0],
        )
    ).model_dump_json()

    result = subprocess.run(
        [sys.executable, "-c", source],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_a_release_is_plain_json_all_the_way_down() -> None:
    """No numpy scalars anywhere, or the artifact can't be written out."""
    predictor = IsotonicProbToMarginFitter().fit_df(_games())
    release = _release(margin_calibration=calibration_from_predictor(predictor))

    payload = json.loads(release.model_dump_json())

    knots = payload["margin_calibration"]
    assert knots["kind"] == "isotonic"
    # `type(x) is float`, not isinstance: np.float64 subclasses float, so
    # isinstance is exactly the check that would miss what this is guarding.
    assert all(type(x) is float for x in knots["x_thresholds"])
    assert knots["x_thresholds"] == sorted(knots["x_thresholds"])
    # Ascending margins: a higher win probability means home wins by more.
    assert knots["y_thresholds"] == sorted(knots["y_thresholds"])
    assert "NaN" not in release.model_dump_json()

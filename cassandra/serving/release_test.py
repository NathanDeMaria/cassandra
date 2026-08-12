import json
import subprocess
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from endgame.types import Game
from pydantic import ValidationError

from cassandra.model_eval import score_predictions
from cassandra.predictor import (
    Elo538Predictor,
    EloPredictor,
    FlatPredictor,
    GlickoPredictor,
    Predictor,
    RatingsUnsupported,
    UnknownPredictorClass,
    predict_matchup,
)
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
    TeamRating,
    TrainedThrough,
    calibration_from_predictor,
    metrics_from_scored,
    ratings_from_predictor,
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


def _release(
    margin_calibration: MarginCalibration | None = None,
    predictor_class: str = "GlickoPredictor",
    params: dict[str, float | str] | None = None,
    ratings: dict[str, TeamRating] | None = None,
    league: str = "mens",
) -> ModelRelease:
    return ModelRelease(
        run_id="2026-08-10T09:00:12Z",
        league=league,
        model="glicko_tuned",
        predictor_class=predictor_class,
        params=params or {},
        ratings=ratings or {},
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


def _game(home: str, away: str, home_score: int, away_score: int) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=False,
        completed=True,
        date=datetime(2026, 1, 1),
        game_id=f"{away}@{home}",
    )


_SEASON = [
    _game("Team A", "Team B", 80, 61),
    _game("Team C", "Team A", 55, 70),
    _game("Team B", "Team C", 66, 65),
    _game("Team A", "Team C", 90, 51),
    _game("Team C", "Team B", 72, 70),
]


def _trained(predictor: Predictor) -> Predictor:
    for game in _SEASON:
        predictor.update_game(game)
    return predictor


def _snapshot(
    predictor: Predictor, predictor_class: str, params: dict[str, Any]
) -> ModelRelease:
    """Everything that happens to a predictor on its way to a consumer.

    Through the JSON, not just through the schema: the consumer holds parsed
    bytes off a bucket, never the object the release was built from.
    """
    release = _release(
        predictor_class=predictor_class,
        params=params,
        ratings=ratings_from_predictor(predictor),
        league="test_league",
    )
    return ModelRelease.model_validate_json(release.model_dump_json())


def test_rebuilds_a_glicko_predictor_that_predicts_the_same() -> None:
    """The property that matters: same matchup, same number.

    Field-by-field equality can pass while the probability is wrong -- a
    dropped home advantage doesn't show up in the ratings at all -- so this
    asserts on what a consumer actually asks for.
    """
    params: dict[str, Any] = {
        "home_advantage": 88.0,
        "k": 61.0,
        "initial_rd": 205.0,
        "scoring_method": "binary",
    }
    original = _trained(GlickoPredictor("test_league", **params))

    rebuilt = _snapshot(original, "GlickoPredictor", params).rating_predictor()

    for home, away in (("Team A", "Team B"), ("Team C", "Team A")):
        for neutral_site in (False, True):
            assert predict_matchup(
                rebuilt, home, away, neutral_site=neutral_site
            ).team1_win_prob == pytest.approx(
                predict_matchup(
                    original, home, away, neutral_site=neutral_site
                ).team1_win_prob
            )


def test_a_rebuilt_glicko_keeps_going_the_same_way() -> None:
    """rd survives the round trip.

    predict_game never reads a rating deviation, so a dropped or defaulted rd
    is invisible until the rebuilt predictor updates on a game and moves by a
    different amount than the original would have.
    """
    original = _trained(GlickoPredictor("test_league", initial_rd=205))
    rebuilt = _snapshot(original, "GlickoPredictor", {"initial_rd": 205.0})
    predictor = rebuilt.rating_predictor()

    later = _game("Team B", "Team A", 71, 70)
    original.update_game(later)
    predictor.update_game(later)

    assert predict_matchup(
        predictor, "Team A", "Team B"
    ).team1_win_prob == pytest.approx(
        predict_matchup(original, "Team A", "Team B").team1_win_prob
    )


@pytest.mark.parametrize(
    ("predictor_class", "name"),
    [(EloPredictor, "EloPredictor"), (Elo538Predictor, "Elo538Predictor")],
)
def test_rebuilds_the_elo_family_too(
    predictor_class: type[Predictor], name: str
) -> None:
    """The classes with no rd, which is where defaulting one goes wrong."""
    params: dict[str, Any] = {"home_advantage": 82.0, "k": 17.0}
    original = _trained(predictor_class("test_league", **params))

    release = _snapshot(original, name, params)
    assert all(rating.rd is None for rating in release.ratings.values())

    rebuilt = release.rating_predictor()
    assert isinstance(rebuilt, predictor_class)
    assert predict_matchup(rebuilt, "Team A", "Team C").team1_win_prob == pytest.approx(
        predict_matchup(original, "Team A", "Team C").team1_win_prob
    )


def test_an_unrated_team_is_the_same_stranger_on_both_sides() -> None:
    """A team the release doesn't carry falls back, rather than erroring."""
    original = _trained(GlickoPredictor("test_league"))
    rebuilt = _snapshot(original, "GlickoPredictor", {}).rating_predictor()

    assert predict_matchup(rebuilt, "Team A", "Nobody").team1_win_prob == pytest.approx(
        predict_matchup(original, "Team A", "Nobody").team1_win_prob
    )


def test_a_predictor_class_this_build_doesnt_have_raises_for_that_reason() -> None:
    """A release from a newer cassandra. Downstream turns this into a 502."""
    release = _release(predictor_class="MarkovChainPredictor")

    with pytest.raises(UnknownPredictorClass, match="MarkovChainPredictor"):
        release.rating_predictor()


def test_a_name_that_isnt_a_predictor_at_all_raises_the_same_thing() -> None:
    """`predictor_class` is untrusted input, and the module has other names."""
    with pytest.raises(UnknownPredictorClass):
        _release(predictor_class="load_predictor").rating_predictor()


def test_a_flat_model_says_it_has_no_ratings() -> None:
    """Better than a predictor that answers 0.5 to everything in silence."""
    with pytest.raises(RatingsUnsupported):
        _release(predictor_class="FlatPredictor").rating_predictor()

    with pytest.raises(RatingsUnsupported):
        ratings_from_predictor(FlatPredictor("test_league"))


def test_rebuilding_a_predictor_never_imports_sklearn() -> None:
    """The same property margin_predictor has, for the other half.

    A subprocess because sklearn is in sys.modules by the time this file has
    finished importing. If this fails, a convenience import somewhere under
    `serving` or `predictor` just put scikit-learn in the web image of a
    consumer that installed cassandra without the fit group.
    """
    source = """
import json, sys
from cassandra.serving import ModelRelease

release = ModelRelease.model_validate(json.load(sys.stdin))
predictor = release.rating_predictor()
from cassandra.predictor import predict_matchup
assert 0.5 < predict_matchup(predictor, "Team A", "Team B").team1_win_prob < 1
print("sklearn" in sys.modules)
"""
    payload = _release(
        predictor_class="GlickoPredictor",
        params={"home_advantage": 95.0},
        ratings={
            "Team A": TeamRating(rating=1600, rd=180),
            "Team B": TeamRating(rating=1500, rd=200),
        },
    ).model_dump_json()

    result = subprocess.run(
        [sys.executable, "-c", source],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"

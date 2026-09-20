"""Tests for the vector-observation margin model.

What is checked: that with nothing to read it *is* the margin model, game by
game; that readings are ignored until their noise has been measured; that a
measured reading pulls the step the way a second opinion should; and that
the measurement survives a round trip.
"""

import math
from typing import Any

import numpy as np
import pytest

from .conftest import GameFactory
from .epa import EpaIndex
from .game_control import GameControlIndex
from .glicko import _Rating
from .margin_glicko import MarginGlickoPredictor
from .types import GameControl, GameEpa
from .vector_glicko import (
    MIN_COVARIANCE_GAMES,
    VectorMarginGlickoPredictor,
    _Readings,
)

_KNOBS: dict[str, Any] = dict(
    home_advantage=0, points_per_rating=0.1, initial_rd=100, obs_sd=10
)


def _vector(**kwargs: Any) -> VectorMarginGlickoPredictor:
    # Both readings on, so the tests exercise the full vector; the class's
    # own default leaves EPA off (see the module docstring for why).
    kwargs.setdefault("epa_scale", 1.24)
    kwargs.setdefault("game_control", GameControlIndex())
    kwargs.setdefault("game_epa", EpaIndex())
    return VectorMarginGlickoPredictor("test_league", **{**_KNOBS, **kwargs})


def _margin_only(**kwargs: Any) -> MarginGlickoPredictor:
    return MarginGlickoPredictor("test_league", **{**_KNOBS, **kwargs})


def _control(share: float) -> GameControl:
    return GameControl(home=share, seconds=3600)


def _epa(margin: float) -> GameEpa:
    # 70 snaps a side; the home offense's per-play number carries the margin.
    return GameEpa(home=margin / 70, away=0.0, home_plays=70, away_plays=70)


def _measured(
    n: int = MIN_COVARIANCE_GAMES + 50,
) -> tuple[VectorMarginGlickoPredictor, list[tuple[str, str, float, str]]]:
    """A model with indexes for `n` games, and the games, so its covariance can be measured.

    The readings are built to be honest but noisy copies of the margin, so the
    measured covariance is well-behaved and the intercepts are near zero.
    """
    rng = np.random.default_rng(3)
    control, epa = {}, {}
    games = []
    for i in range(n):
        margin = float(rng.normal(0, 14))
        logit = margin / 17 + rng.normal(0, 0.6)
        control[str(i)] = _control(1 / (1 + math.exp(-logit)))
        epa[str(i)] = _epa(margin / 1.24 + rng.normal(0, 8))
        games.append((f"A{i % 20}", f"B{i % 23}", margin, str(i)))
    predictor = _vector(game_control=GameControlIndex(control), game_epa=EpaIndex(epa))
    return predictor, games


def _play(predictor, games, game: GameFactory) -> None:
    for home, away, margin, game_id in games:
        home_score = max(0, int(round(margin))) if margin >= 0 else 0
        away_score = 0 if margin >= 0 else int(round(-margin))
        predictor.update_game(game(home, away, home_score, away_score, game_id=game_id))


def test_with_nothing_to_read_it_is_the_margin_model(game: GameFactory) -> None:
    vector, margin = _vector(), _margin_only()
    for g in (("A", "B", 20, 3), ("C", "A", 7, 14), ("B", "C", 0, 31)):
        vector.update_game(game(*g))
        margin.update_game(game(*g))
    for team in "ABC":
        assert vector.get_rating(team) == pytest.approx(margin.get_rating(team))
    assert vector.predict_game(game("A", "C")) == margin.predict_game(game("A", "C"))


def test_both_scales_off_is_the_margin_model_whatever_the_indexes_hold(
    game: GameFactory,
) -> None:
    vector = _vector(
        control_scale=None,
        epa_scale=None,
        game_control=GameControlIndex({"1": _control(0.95)}),
        game_epa=EpaIndex({"1": _epa(40)}),
    )
    margin = _margin_only()
    vector.update_game(game("A", "B", 20, 3, game_id="1"))
    margin.update_game(game("A", "B", 20, 3, game_id="1"))
    assert vector.get_rating("A") == pytest.approx(margin.get_rating("A"))


def test_readings_are_ignored_until_their_noise_is_measured(game: GameFactory) -> None:
    """A control share of 0.95 says a rout; with no covariance yet it says nothing."""
    vector = _vector(
        game_control=GameControlIndex({"1": _control(0.95)}),
        game_epa=EpaIndex({"1": _epa(40)}),
    )
    margin = _margin_only()
    vector.update_game(game("A", "B", 20, 3, game_id="1"))
    margin.update_game(game("A", "B", 20, 3, game_id="1"))
    assert vector.get_rating("A") == pytest.approx(margin.get_rating("A"))
    assert vector.observation_stats.count == 1


def test_a_measured_reading_that_agrees_pulls_harder_and_one_that_doesnt_pulls_less(
    game: GameFactory,
) -> None:
    predictor, games = _measured()
    _play(predictor, games, game)
    assert predictor.observation_stats.count > MIN_COVARIANCE_GAMES
    r = predictor.noise_covariance()
    assert np.all(np.isfinite(r))
    assert r[0, 0] == pytest.approx(10**2)

    def move(control_share: float, epa_margin: float) -> float:
        p = VectorMarginGlickoPredictor(
            "test_league",
            **_KNOBS,
            epa_scale=1.24,
            observation_stats=list(predictor.state_dict()["observation_stats"]),
            game_control=GameControlIndex({"x": _control(control_share)}),
            game_epa=EpaIndex({"x": _epa(epa_margin)}),
        )
        p.update_game(game("New", "Other", 20, 0, game_id="x"))
        return p.get_rating("New").rating - 1500

    alone = _margin_only()
    alone.update_game(game("New", "Other", 20, 0))
    margin_move = alone.get_rating("New").rating - 1500
    # Readings that say the same 20-point win: a firmer step.
    agree = move(1 / (1 + math.exp(-20 / 17)), 20 / 1.24)
    # Readings that say the game was even: the scoreboard flattered them.
    disagree = move(0.5, 0.0)
    assert agree > margin_move > disagree > 0


def test_the_step_is_the_solve_it_stands_in_for(game: GameFactory) -> None:
    """The Sherman-Morrison form agrees with `S^-1 1` solved outright, every subset."""
    predictor, games = _measured()
    _play(predictor, games, game)
    r, b = predictor.noise_covariance(), predictor.intercepts()
    c = predictor.points_per_rating
    my, opp = _Rating(1530.0, 80.0), _Rating(1490.0, 120.0)
    for readings in (
        _Readings(20.0, 10.0, 24.0),
        _Readings(20.0, 10.0, None),
        _Readings(20.0, None, 24.0),
        _Readings(20.0, None, None),
    ):
        present = [i for i, v in enumerate(readings) if v is not None]
        expected = c * (my.rating + 2.0 - opp.rating)
        v = np.array([-(readings[i] - b[i]) - expected for i in present])
        ones = np.ones(len(present))
        s = (
            c**2
            * (my.rating_deviation**2 + opp.rating_deviation**2)
            * np.outer(ones, ones)
            + r[np.ix_(present, present)]
        )
        weights = np.linalg.solve(s, ones)
        gain = my.rating_deviation**2 * c * weights
        shrink = my.rating_deviation**2 * c**2 * float(ones @ weights)
        stepped = predictor._vector_step(my, opp, readings, 2.0, -1)
        assert stepped.rating == pytest.approx(my.rating + float(gain @ v))
        assert stepped.rating_deviation == pytest.approx(
            math.sqrt(my.rating_deviation**2 * (1 - shrink))
        )


def test_the_measurement_round_trips(game: GameFactory) -> None:
    predictor, games = _measured()
    _play(predictor, games, game)
    state = predictor.state_dict()
    assert "nu" not in state
    restored = VectorMarginGlickoPredictor.from_state_dict(state)
    assert restored.observation_stats == predictor.observation_stats
    np.testing.assert_allclose(
        restored.noise_covariance(), predictor.noise_covariance()
    )
    np.testing.assert_allclose(restored.intercepts(), predictor.intercepts())
    assert restored.state_dict() == state


def test_epa_is_off_unless_asked_for(game: GameFactory) -> None:
    """Measured to hurt, so the default reads control only."""
    predictor = VectorMarginGlickoPredictor(
        "test_league", **_KNOBS, game_control=GameControlIndex(), game_epa=EpaIndex()
    )
    assert predictor.epa_scale is None
    assert predictor.control_scale is not None
    assert predictor.state_dict()["epa_scale"] is None


def test_a_scale_of_zero_or_less_is_refused() -> None:
    with pytest.raises(ValueError, match="control_scale"):
        _vector(control_scale=0)
    with pytest.raises(ValueError, match="epa_scale"):
        _vector(epa_scale=-1)


def test_the_smoother_re_reads_the_same_game(game: GameFactory) -> None:
    """Two passes see each game's readings again, and still move only the means."""
    predictor, games = _measured()
    smoothed = VectorMarginGlickoPredictor(
        "test_league",
        **_KNOBS,
        epa_scale=1.24,
        passes=3,
        game_control=predictor._game_control,
        game_epa=predictor._game_epa,
    )
    for p in (predictor, smoothed):
        _play(p, games[:40], game)
        p.pass_week()
        _play(p, games[40:80], game)
        p.pass_week()
    for team in {g[0] for g in games[:80]}:
        assert smoothed.get_rating(team).rating_deviation == pytest.approx(
            predictor.get_rating(team).rating_deviation
        )
    assert any(
        smoothed.get_rating(t).rating != pytest.approx(predictor.get_rating(t).rating)
        for t in {g[0] for g in games[:80]}
    )

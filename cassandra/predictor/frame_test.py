"""The points frame is a change of coordinates, not a change of model."""

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from ..box import continuous_range, is_integer
from ..scoring import DEFAULT_SIGMOID_SCALE
from . import frame
from .config import OptimizationConfig

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# nfl/glicko_full's 2026-09-13 fit, in the constructor's units, and what it
# reads as in points: a 2.95-point home edge, a 3.6-point quarterback, and a
# 10.4-point favourite at 73%. The numbers the frame was built to expose.
_NFL_FIT: dict[str, float] = {
    "travel_advantage": 0.0,
    "home_advantage": 60.19362742298362,
    "initial_rd": 251.58686450937677,
    "weekly_rd_increase": 31.107848063696725,
    "season_rd_increase": 143.2340294711692,
    "season_regression": 0.1151108598194098,
    "sigmoid_scale": 8.500938713470973,
    "prediction_scale": 489.5527220376322,
    "rest_advantage": 1.7167215593418694,
    "qb_out_penalty": 74.1597426963626,
}
_NFL_WEEKS = 22


def _nfl_knobs() -> dict[str, float | str]:
    per_point = frame.rating_units_per_point(_NFL_FIT["sigmoid_scale"])
    total = (
        _NFL_WEEKS * _NFL_FIT["weekly_rd_increase"] ** 2
        + _NFL_FIT["season_rd_increase"] ** 2
    )
    return {
        "travel_pts": 0.0,
        "initial_rd": _NFL_FIT["initial_rd"],
        "season_regression": _NFL_FIT["season_regression"],
        "sigmoid_scale": _NFL_FIT["sigmoid_scale"],
        "hfa_pts": _NFL_FIT["home_advantage"] / per_point,
        "rest_pts": _NFL_FIT["rest_advantage"] / per_point,
        "qb_pts": _NFL_FIT["qb_out_penalty"] / per_point,
        "pred_margin_scale": _NFL_FIT["sigmoid_scale"]
        * _NFL_FIT["prediction_scale"]
        / 400,
        "rd_total": math.sqrt(total),
        "rd_offseason_share": _NFL_FIT["season_rd_increase"] ** 2 / total,
    }


def test_the_rating_frame_is_the_identity() -> None:
    assert frame.to_params(frame.RATING, _NFL_FIT, weeks_per_season=22) == _NFL_FIT
    assert frame.knobs_of(frame.RATING) == frozenset()
    assert frame.searched_params(frame.RATING, {"home_advantage"}) == {"home_advantage"}


def test_the_points_frame_round_trips_a_real_fit() -> None:
    knobs = _nfl_knobs()
    assert knobs["hfa_pts"] == pytest.approx(2.95, abs=0.01)
    assert knobs["qb_pts"] == pytest.approx(3.63, abs=0.01)
    assert knobs["pred_margin_scale"] == pytest.approx(10.4, abs=0.05)

    params = frame.to_params(frame.POINTS, knobs, weeks_per_season=_NFL_WEEKS)

    assert set(params) == set(_NFL_FIT)
    for name, value in _NFL_FIT.items():
        assert params[name] == pytest.approx(value), name


def test_a_real_fit_reads_back_into_the_knobs_it_was_searched_in() -> None:
    """`to_knobs` is `to_params` backwards, on the nfl fit both ways round."""
    knobs = frame.to_knobs(frame.POINTS, _NFL_FIT, weeks_per_season=_NFL_WEEKS)

    assert set(knobs) == set(_nfl_knobs())
    for name, value in _nfl_knobs().items():
        assert knobs[name] == pytest.approx(value), name
    params = frame.to_params(frame.POINTS, knobs, weeks_per_season=_NFL_WEEKS)
    for name, value in _NFL_FIT.items():
        assert params[name] == pytest.approx(value), name

    assert frame.to_knobs(frame.RATING, _NFL_FIT, weeks_per_season=1) == _NFL_FIT


def test_a_fit_with_no_deviation_increase_reads_back_as_no_budget() -> None:
    """Two zeros have no split, so the share is 0 rather than 0/0."""
    knobs = frame.to_knobs(
        frame.POINTS,
        {"sigmoid_scale": 10.0, "weekly_rd_increase": 0.0, "season_rd_increase": 0.0},
        weeks_per_season=17,
    )

    assert knobs["rd_total"] == 0.0
    assert knobs["rd_offseason_share"] == 0.0
    assert frame.to_params(frame.POINTS, knobs, weeks_per_season=17) == {
        "sigmoid_scale": 10.0,
        "weekly_rd_increase": 0.0,
        "season_rd_increase": 0.0,
    }


def test_a_point_is_priced_off_the_update_scale() -> None:
    """Why the exchange rate is what it is.

    A team that wins by `m` scores `sigmoid(m / ss)` a game, and the update
    moves its rating until the expected score at the 400 scale matches, so
    the gap is `400 * m / (ss * ln 10)`. One point at scale 10 is 17.4 units.
    """
    assert frame.rating_units_per_point(10.0) == pytest.approx(
        400 / (10 * math.log(10))
    )
    params = frame.to_params(frame.POINTS, {"sigmoid_scale": 10.0, "hfa_pts": 3.0}, 17)
    assert params["home_advantage"] == pytest.approx(3 * 17.3718, abs=0.01)


def test_the_deviation_budget_is_spread_over_the_season() -> None:
    params = frame.to_params(
        frame.POINTS,
        {"sigmoid_scale": 10.0, "rd_total": 100.0, "rd_offseason_share": 0.36},
        weeks_per_season=16,
    )
    # 36% of the variance at the rollover, the rest over sixteen weeks.
    assert params["season_rd_increase"] == pytest.approx(60.0)
    assert params["weekly_rd_increase"] == pytest.approx(math.sqrt(6400 / 16))
    assert "rd_total" not in params and "rd_offseason_share" not in params


def test_pins_alone_derive_what_they_fix_and_no_more() -> None:
    """What `optimize.py`'s priors pass and `sync_pins.py` hand over.

    A point term with no `sigmoid_scale` beside it is priced at the scale the
    constructor would default to, which is the scale that replay runs at; a
    share with no budget to split says nothing and is dropped.
    """
    params = frame.to_params(
        frame.POINTS,
        {"scoring_method": "sigmoid", "travel_pts": 0, "rd_offseason_share": 0.1},
        weeks_per_season=17,
    )
    assert params == {"scoring_method": "sigmoid", "travel_advantage": 0.0}

    priced = frame.to_params(frame.POINTS, {"hfa_pts": 2.0}, 17)
    assert priced["home_advantage"] == pytest.approx(
        2 * frame.rating_units_per_point(DEFAULT_SIGMOID_SCALE)
    )


def test_a_budget_without_a_split_is_an_error() -> None:
    with pytest.raises(ValueError, match="rd_offseason_share"):
        frame.to_params(frame.POINTS, {"sigmoid_scale": 10.0, "rd_total": 100.0}, 17)


def test_a_knob_and_the_argument_it_derives_cannot_both_be_given() -> None:
    with pytest.raises(ValueError, match="hfa_pts derives home_advantage"):
        frame.to_params(
            frame.POINTS,
            {"sigmoid_scale": 10.0, "hfa_pts": 3.0, "home_advantage": 50.0},
            17,
        )


def test_an_argument_the_frame_does_not_know_passes_through() -> None:
    params = frame.to_params(frame.POINTS, {"sigmoid_scale": 10.0, "k": 65.0}, 17)
    assert params == {"sigmoid_scale": 10.0, "k": 65.0}


def test_searching_the_scale_moves_every_priced_argument() -> None:
    """What a pin copied from a framed fit is checked against.

    `sync_pins.py` asks which constructor arguments a search moves. In this
    frame `home_advantage` moves whenever `sigmoid_scale` does, even with
    `hfa_pts` pinned -- the pin is in points, the argument is in rating units.
    """
    moved = frame.searched_params(frame.POINTS, {"sigmoid_scale", "rd_total"})
    assert moved == {
        "sigmoid_scale",
        "home_advantage",
        "travel_advantage",
        "rest_advantage",
        "qb_out_penalty",
        "prediction_scale",
        "weekly_rd_increase",
        "season_rd_increase",
    }
    # A pinned scale moves only what its own knobs move.
    assert frame.searched_params(frame.POINTS, {"hfa_pts", "initial_rd"}) == {
        "home_advantage",
        "initial_rd",
    }


def test_weeks_per_season_is_the_median_season() -> None:
    assert frame.weeks_per_season([17, 22, 22, 21, 18]) == 21
    assert frame.weeks_per_season([16, 17]) == 16.5
    with pytest.raises(ValueError):
        frame.weeks_per_season([])


def test_an_unknown_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown frame"):
        frame.to_params("furlongs", {}, 17)


# -- the config side ---------------------------------------------------------


def test_a_config_in_the_points_frame_names_its_scale() -> None:
    with pytest.raises(ValidationError, match="sigmoid_scale"):
        OptimizationConfig(
            predictor_class="GlickoPredictor",
            league="nfl",
            frame="points",
            parameters={"hfa_pts": (0, 8)},
        )


def test_a_config_seed_has_to_sit_in_its_own_box() -> None:
    """Caught with the other config errors, before a launcher submits it."""

    def config(seed: dict[str, float]) -> OptimizationConfig:
        return OptimizationConfig.model_validate(
            {
                "predictor_class": "GlickoPredictor",
                "league": "nfl",
                "frame": frame.POINTS,
                "parameters": {"sigmoid_scale": (2, 30), "hfa_pts": (0, 8)},
                "seeds": [seed],
            }
        )

    config({"sigmoid_scale": 21.0, "hfa_pts": 2.8})

    with pytest.raises(ValidationError, match=r"hfa_pts=9\.0 is outside \[0, 8\]"):
        config({"sigmoid_scale": 21.0, "hfa_pts": 9.0})
    with pytest.raises(ValidationError, match="missing hfa_pts"):
        config({"sigmoid_scale": 21.0})


def test_a_config_cannot_search_a_knob_and_pin_what_it_derives() -> None:
    with pytest.raises(ValidationError, match="hfa_pts derives home_advantage"):
        OptimizationConfig(
            predictor_class="GlickoPredictor",
            league="nfl",
            frame="points",
            parameters={"hfa_pts": (0, 8), "sigmoid_scale": (2, 30)},
            fixed={"home_advantage": 60.0},
        )


def test_a_config_reports_what_it_searches_in_constructor_terms() -> None:
    config = OptimizationConfig(
        predictor_class="GlickoPredictor",
        league="nfl",
        frame="points",
        parameters={"sigmoid_scale": (2, 30), "hfa_pts": (0, 8)},
        fixed={"rd_offseason_share": 0.5},
    )
    assert "home_advantage" in config.searched_params()
    assert "hfa_pts" not in config.searched_params()
    assert "weekly_rd_increase" not in config.searched_params()


def test_the_default_frame_is_the_constructors_own() -> None:
    config = OptimizationConfig(
        predictor_class="EloPredictor", league="nfl", parameters={"k": (1, 60)}
    )
    assert config.frame == frame.RATING
    assert config.searched_params() == {"k"}


@pytest.mark.parametrize(
    "config_path",
    sorted(
        path
        for path in _MODELS_DIR.glob("*/*.json")
        if "_result" not in path.stem
        and json.loads(path.read_text()).get("frame", "rating") != "rating"
    ),
    ids=lambda path: f"{path.parent.name}/{path.stem}",
)
def test_every_framed_config_derives_a_full_probe(config_path: Path) -> None:
    """A framed config's knobs, at their box midpoints, build the predictor.

    The constructor is the last line of defence: a knob the frame passes
    through under a name the class doesn't take fails here, before Batch.
    """
    config = OptimizationConfig.model_validate_json(config_path.read_text())
    from .config import load_predictor_class

    probe: dict[str, float | str] = {}
    for name, bounds in config.parameters.items():
        span = continuous_range(bounds)
        if span is None:
            probe[name] = bounds[0]
        elif is_integer(bounds):
            probe[name] = int(sum(span) // 2)
        else:
            probe[name] = sum(span) / 2
    params = frame.to_params(config.frame, {**config.fixed, **probe}, 17)
    load_predictor_class(config.predictor_class)(config.league, **params)

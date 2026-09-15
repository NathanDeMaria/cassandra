import re
from typing import cast

import pytest

import optimize
from cassandra.predictor import (
    OptimizationConfig,
    Predictor,
    PredictorConfig,
    SearchRecord,
    frame,
)
from cassandra.save_predictions import OddsDatabase
from optimize import _pinned_notice

# The pattern `.claude/skills/run-report/summarize_run.py` matches to pull a
# line out of a container log and into the report's tuning diagnostics. Copied
# rather than imported: the skill isn't an importable package, and the point of
# the test is that the two stay in step even so.
_REPORT_DIAGNOSTIC = re.compile(r"^\[optimize\] (?P<message>.+)$")


def test_pinned_notice__none_when_nothing_is_pinned():
    assert _pinned_notice({}, "elo.json") is None


def test_pinned_notice__names_every_pin_and_the_config_to_edit():
    notice = _pinned_notice(
        {"scoring_method": "sigmoid", "season_regression": 0}, "glicko_full.json"
    )

    assert notice is not None
    assert "season_regression=0" in notice
    assert "scoring_method=sigmoid" in notice
    # Without the filename the reader has a question and no way to act on it.
    assert "glicko_full.json" in notice


def test_pinned_notice__is_picked_up_by_the_run_report():
    """The whole mechanism: a pin has to survive into the weekly report.

    A pinned parameter emits no probes and so no bound-hit diagnostic, so this
    line is the only thing that carries the decision forward. If the prefix
    drifts from what `summarize_run.py` matches, the reminder disappears
    silently -- which is the failure this is here to prevent.
    """
    notice = _pinned_notice({"season_regression": 0}, "elo.json")

    assert notice is not None
    assert _REPORT_DIAGNOSTIC.match(notice)


def test_a_probe_reaches_the_constructor_in_its_own_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The search moves knobs; the predictor takes arguments.

    `_score_probe` is the one place the two meet, so a framed config's probe
    -- points, a deviation budget -- has to arrive at the constructor as the
    rating-unit arguments it derives, with the pins merged in the same way.
    """
    seen: dict[str, object] = {}

    class Fake:
        def __init__(self, league: str, **kwargs: object) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(
        optimize,
        "join_with_odds",
        lambda predictor, seasons, odds_db, post_callbacks: [],
    )

    value = optimize._score_probe(
        "nfl",
        [],
        cast(OddsDatabase, None),
        cast(type[Predictor], Fake),
        lambda df: 1.0,
        frame=frame.POINTS,
        weeks_per_season=20,
        fixed={"scoring_method": "sigmoid", "travel_pts": 0},
        sigmoid_scale=10.0,
        hfa_pts=3.0,
    )

    assert value == 1.0
    assert seen["home_advantage"] == pytest.approx(
        3 * frame.rating_units_per_point(10.0)
    )
    assert seen["travel_advantage"] == 0.0
    assert seen["scoring_method"] == "sigmoid"
    assert "hfa_pts" not in seen and "travel_pts" not in seen


# --- seeds -------------------------------------------------------------------


def _points_config(**overrides: object) -> OptimizationConfig:
    fields: dict[str, object] = {
        "predictor_class": "GlickoPredictor",
        "league": "nfl",
        "frame": frame.POINTS,
        "parameters": {
            "sigmoid_scale": (2, 30),
            "hfa_pts": (0, 8),
            "rd_total": (20, 800),
            "rd_offseason_share": (0, 1),
        },
        "fixed": {"scoring_method": "sigmoid", "initial_rd": 250.0},
    }
    return OptimizationConfig.model_validate({**fields, **overrides})


def _fit(**overrides: object) -> PredictorConfig:
    fields: dict[str, object] = {
        "predictor_class": "GlickoPredictor",
        "league": "nfl",
        "target": -0.22,
        "params": {
            "scoring_method": "sigmoid",
            "initial_rd": 250.0,
            "sigmoid_scale": 10.0,
            "home_advantage": 3 * frame.rating_units_per_point(10.0),
            "weekly_rd_increase": 30.0,
            "season_rd_increase": 0.0,
        },
    }
    return PredictorConfig.model_validate({**fields, **overrides})


def test_seeds__the_previous_fit_comes_first_in_the_knobs_it_was_searched_in():
    """A result with a record in this frame is handed over as it was found."""
    knobs = {
        "scoring_method": "sigmoid",
        "initial_rd": 250.0,
        "sigmoid_scale": 10.0,
        "hfa_pts": 3.0,
        "rd_total": 100.0,
        "rd_offseason_share": 0.25,
    }
    previous = _fit(
        search=SearchRecord(frame=frame.POINTS, weeks_per_season=22, knobs=knobs)
    )

    seeds, lines = optimize._seeds(_points_config(), previous, weeks_per_season=22)

    # Only the searched names: the pins are the config's, not the fit's.
    assert seeds == [
        {
            "sigmoid_scale": 10.0,
            "hfa_pts": 3.0,
            "rd_total": 100.0,
            "rd_offseason_share": 0.25,
        }
    ]
    assert len(lines) == 1 and lines[0].startswith(
        "[optimize] seeded with the previous fit: "
    )
    assert all(_REPORT_DIAGNOSTIC.match(line) for line in lines)


def test_seeds__a_rating_unit_fit_is_read_into_a_points_box():
    """Every result from before frames existed, and the one ncaafb lost."""
    seeds, lines = optimize._seeds(_points_config(), _fit(), weeks_per_season=22)

    assert len(seeds) == 1
    assert seeds[0]["sigmoid_scale"] == 10.0
    assert seeds[0]["hfa_pts"] == pytest.approx(3.0)
    assert seeds[0]["rd_total"] == pytest.approx(30.0 * 22**0.5)
    assert seeds[0]["rd_offseason_share"] == 0.0
    assert "seeded with the previous fit" in lines[0]


def test_seeds__a_previous_fit_outside_the_box_is_left_out_and_said_so():
    """Not clamped: a box drawn to exclude the old fit meant to."""
    seeds, lines = optimize._seeds(
        _points_config(parameters={"sigmoid_scale": (2, 30), "hfa_pts": (0, 2)}),
        _fit(),
        weeks_per_season=22,
    )

    assert seeds == []
    assert lines == [
        "[optimize] not seeded with the previous fit: hfa_pts=3.0 is outside [0, 2]"
    ]


def test_seeds__a_fit_of_another_class_is_not_a_point_in_this_box():
    seeds, lines = optimize._seeds(
        _points_config(),
        _fit(predictor_class="EloPredictor", params={"k": 20.0}),
        weeks_per_season=22,
    )

    assert seeds == []
    assert lines == [
        "[optimize] not seeded with the previous fit: it is a EloPredictor, not a GlickoPredictor"
    ]


def test_seeds__the_configs_own_follow_the_previous_fit_without_repeating_it():
    """The point a config seed is for is one the previous result is not.

    Once a search keeps its seed, the result *is* the seed, and the next run
    would probe the same point twice -- bayes_opt scores the second from its
    cache and logs nothing, so it is dropped here and said so.
    """
    kept = {
        "sigmoid_scale": 10.0,
        "hfa_pts": 3.0,
        "rd_total": 100.0,
        "rd_offseason_share": 0.25,
    }
    lost = {
        "sigmoid_scale": 20.0,
        "hfa_pts": 1.0,
        "rd_total": 300.0,
        "rd_offseason_share": 0.5,
    }
    previous = _fit(
        search=SearchRecord(frame=frame.POINTS, weeks_per_season=22, knobs=kept)
    )

    seeds, lines = optimize._seeds(
        _points_config(seeds=[lost, dict(kept)]), previous, weeks_per_season=22
    )

    assert seeds == [kept, lost]
    assert lines[1].startswith("[optimize] seeded with seed 1 of the config: ")
    assert lines[2] == "[optimize] not seeded with seed 2 of the config: already a seed"


def test_seeds__nothing_to_start_from_is_an_ordinary_search():
    assert optimize._seeds(_points_config(), None, weeks_per_season=22) == ([], [])


def test_a_previous_result_is_read_from_where_the_search_writes(tmp_path):
    path = tmp_path / "glicko_full_result.json"
    assert optimize._previous_result(path) is None

    path.write_text(_fit().model_dump_json())
    previous = optimize._previous_result(path)

    assert previous is not None and previous.target == -0.22


def test_seeds__a_knob_the_previous_fit_predates_takes_the_constructor_default():
    """`passes` was 1 for every fit before it existed, so the fit is still a point."""
    config = _points_config(
        parameters={
            "sigmoid_scale": (2, 30),
            "hfa_pts": (0, 8),
            "passes": (1, 4, "int"),
        }
    )

    seeds, lines = optimize._seeds(config, _fit(), weeks_per_season=22)

    assert seeds == [
        {"sigmoid_scale": 10.0, "hfa_pts": pytest.approx(3.0), "passes": 1}
    ]
    assert lines[0].startswith("[optimize] seeded with the previous fit")

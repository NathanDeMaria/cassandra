"""The re-pin follows the run report's rule, and touches nothing else."""

import json
from pathlib import Path

import pytest

import sync_pins


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=4) + "\n")


@pytest.fixture
def league(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A league with a source model, its fit, and two configs pinned from it."""
    models = tmp_path / "models"
    home = tmp_path / "home"
    monkeypatch.setattr(sync_pins, "MODELS_DIR", models)
    monkeypatch.setattr(sync_pins, "CASSANDRA_HOME", home)
    _write(
        models / "mens" / "glicko_full.json",
        {
            "predictor_class": "GlickoPredictor",
            "league": "mens",
            "parameters": {"home_advantage": [0, 200], "initial_rd": [80, 620]},
            "fixed": {"scoring_method": "sigmoid", "season_regression": 0},
        },
    )
    _write(
        home / "models" / "mens" / "glicko_full_result.json",
        {
            "predictor_class": "GlickoPredictor",
            "league": "mens",
            "target": -0.17,
            "params": {
                "home_advantage": 61.5,
                "initial_rd": 300.0,
                "scoring_method": "sigmoid",
                "season_regression": 0,
            },
        },
    )
    _write(
        models / "mens" / "glicko_blend.json",
        {
            "predictor_class": "BlendedGlickoPredictor",
            "league": "mens",
            "parameters": {"play_weight": [0, 1], "initial_rd": [80, 620]},
            "fixed": {
                "home_advantage": 46.0,
                "scoring_method": "binary",
                "mov_scale": 10.0,
            },
            "fixed_from": "glicko_full",
        },
    )
    _write(
        models / "mens" / "elo.json",
        {
            "predictor_class": "EloPredictor",
            "league": "mens",
            "parameters": {"k": [1, 100]},
            "fixed": {"home_advantage": 46.0},
        },
    )
    return models / "mens"


def test_a_searched_pin_takes_the_fit_and_a_fixed_pin_takes_the_fixed_value(
    league: Path,
) -> None:
    lines = sync_pins.sync("mens", "glicko_full")

    blend = json.loads((league / "glicko_blend.json").read_text())
    assert blend["fixed"]["home_advantage"] == 61.5
    assert blend["fixed"]["scoring_method"] == "sigmoid"
    assert lines == [
        "mens/glicko_blend: home_advantage 46.0 -> 61.5, "
        "scoring_method 'binary' -> 'sigmoid'"
    ]


def test_the_targets_own_pins_and_searches_are_left_alone(league: Path) -> None:
    """`mov_scale` is not the source's to set, and `initial_rd` is searched here."""
    sync_pins.sync("mens", "glicko_full")

    blend = json.loads((league / "glicko_blend.json").read_text())
    assert blend["fixed"]["mov_scale"] == 10.0
    assert "initial_rd" not in blend["fixed"]
    assert blend["parameters"]["initial_rd"] == [80, 620]


def test_an_aliased_pin_takes_the_source_value_under_its_other_name(
    league: Path, tmp_path: Path
) -> None:
    """`mov_scale` is the blend's `sigmoid_scale`; nfl's stayed at a stale
    20.99 for two re-pins because nothing matched it by name."""
    _framed_source(league, tmp_path, with_search_record=True)
    child = json.loads((league / "glicko_blend.json").read_text())
    child["fixed_aliases"] = {"mov_scale": "sigmoid_scale"}
    _write(league / "glicko_blend.json", child)

    sync_pins.sync("mens", "glicko_full")

    blend = json.loads((league / "glicko_blend.json").read_text())
    assert blend["fixed"]["mov_scale"] == 8.5
    assert "sigmoid_scale" not in blend["fixed"]


def test_a_config_without_fixed_from_is_not_a_target(league: Path) -> None:
    sync_pins.sync("mens", "glicko_full")

    assert json.loads((league / "elo.json").read_text())["fixed"] == {
        "home_advantage": 46.0
    }


def test_a_second_run_changes_nothing(league: Path) -> None:
    sync_pins.sync("mens", "glicko_full")
    before = (league / "glicko_blend.json").read_text()

    assert sync_pins.sync("mens", "glicko_full") == [
        "mens/glicko_blend: already in sync"
    ]
    assert (league / "glicko_blend.json").read_text() == before


def test_a_missing_fit_says_what_to_do(league: Path, tmp_path: Path) -> None:
    (tmp_path / "home" / "models" / "mens" / "glicko_full_result.json").unlink()

    with pytest.raises(FileNotFoundError, match="--download"):
        sync_pins.sync("mens", "glicko_full")


def test_a_pin_the_fit_predates_waits_for_the_next_search(league: Path) -> None:
    """A dimension added to the source's search, pinned in the child at the
    default, before the search has run: the fit on disk has no value for it.
    Leave the pin alone and say so, rather than crash on every child until
    the next Batch run lands."""
    config = json.loads((league / "glicko_full.json").read_text())
    config["parameters"]["prediction_scale"] = [200, 600]
    _write(league / "glicko_full.json", config)
    child = json.loads((league / "glicko_blend.json").read_text())
    child["fixed"]["prediction_scale"] = 400
    _write(league / "glicko_blend.json", child)

    lines = sync_pins.sync("mens", "glicko_full")

    assert lines == [
        "mens/glicko_blend: home_advantage 46.0 -> 61.5, scoring_method 'binary' -> 'sigmoid'",
        "mens/glicko_blend: prediction_scale not in the fit yet; "
        "re-run after the next glicko_full search",
    ]
    after = json.loads((league / "glicko_blend.json").read_text())
    assert after["fixed"]["prediction_scale"] == 400


def _framed_source(league: Path, tmp_path: Path, *, with_search_record: bool) -> None:
    """Move the source into the points frame; its fit is still in rating units."""
    _write(
        league / "glicko_full.json",
        {
            "predictor_class": "GlickoPredictor",
            "league": "mens",
            "frame": "points",
            "parameters": {
                "sigmoid_scale": [2, 30],
                "hfa_pts": [0, 8],
                "rd_total": [20, 800],
            },
            "fixed": {
                "scoring_method": "sigmoid",
                "travel_pts": 0,
                "rd_offseason_share": 0.25,
                "initial_rd": 300.0,
            },
        },
    )
    result = {
        "predictor_class": "GlickoPredictor",
        "league": "mens",
        "target": -0.17,
        "params": {
            "home_advantage": 61.5,
            "initial_rd": 300.0,
            "scoring_method": "sigmoid",
            "sigmoid_scale": 8.5,
            "travel_advantage": 0.0,
            "weekly_rd_increase": 30.0,
            "season_rd_increase": 80.0,
        },
    }
    if with_search_record:
        result["search"] = {
            "frame": "points",
            "weeks_per_season": 20,
            "knobs": {"sigmoid_scale": 8.5, "hfa_pts": 3.0, "rd_total": 170.0},
        }
    _write(tmp_path / "home" / "models" / "mens" / "glicko_full_result.json", result)


def test_a_pin_of_what_a_framed_source_derives_takes_the_fit(
    league: Path, tmp_path: Path
) -> None:
    """The child pins `home_advantage`; the source searches `hfa_pts` and
    `sigmoid_scale`, which is what moves it. Same rule, read in the
    constructor's terms rather than by name."""
    _framed_source(league, tmp_path, with_search_record=True)
    child = json.loads((league / "glicko_blend.json").read_text())
    child["fixed"]["weekly_rd_increase"] = 10.0
    child["fixed"]["travel_advantage"] = 5.0
    _write(league / "glicko_blend.json", child)

    lines = sync_pins.sync("mens", "glicko_full")

    after = json.loads((league / "glicko_blend.json").read_text())["fixed"]
    assert after["home_advantage"] == 61.5
    assert after["weekly_rd_increase"] == 30.0
    # Fixed through the frame (`travel_pts` 0), so the pin takes what the
    # frame makes of it, not the knob's name.
    assert after["travel_advantage"] == 0.0
    assert after["mov_scale"] == 10.0
    assert lines == [
        "mens/glicko_blend: home_advantage 46.0 -> 61.5, "
        "scoring_method 'binary' -> 'sigmoid', "
        "weekly_rd_increase 10.0 -> 30.0, travel_advantage 5.0 -> 0.0"
    ]


def test_a_fit_from_before_the_frame_leaves_derived_pins_waiting(
    league: Path, tmp_path: Path
) -> None:
    """The source moved to the points frame but the result on disk was
    searched in rating units: what its frame derives is not in that fit."""
    _framed_source(league, tmp_path, with_search_record=False)

    lines = sync_pins.sync("mens", "glicko_full")

    after = json.loads((league / "glicko_blend.json").read_text())["fixed"]
    assert after["home_advantage"] == 46.0
    assert lines == [
        "mens/glicko_blend: scoring_method 'binary' -> 'sigmoid'",
        "mens/glicko_blend: home_advantage not in the fit yet; "
        "re-run after the next glicko_full search",
    ]

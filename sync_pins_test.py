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

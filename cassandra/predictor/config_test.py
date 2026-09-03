import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from .config import (
    OptimizationConfig,
    PredictorConfig,
    load_predictor,
    load_predictor_class,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _REPO_ROOT / "models"


def test_nothing_is_pinned_unless_a_config_says_so() -> None:
    config = OptimizationConfig(
        predictor_class="FlatPredictor", league="mens", parameters={"k": (1.0, 2.0)}
    )

    assert config.fixed == {}


def test_a_parameter_cannot_be_searched_and_pinned_at_once() -> None:
    with pytest.raises(ValidationError, match="scoring_method"):
        OptimizationConfig(
            predictor_class="GlickoPredictor",
            league="nfl",
            parameters={"scoring_method": ["binary", "sigmoid"]},
            fixed={"scoring_method": "sigmoid"},
        )


def test_a_pinned_argument_rebuilds_the_predictor_it_was_fit_with(
    tmp_path: Path,
) -> None:
    """The reason `optimize.py` merges `fixed` into the result rather than
    only recording it beside one: `load_predictor` reads `params` and nothing
    else, so a pinned argument missing from there is a published model
    running on the constructor default -- `binary`, not the `sigmoid` whose
    score got written next to it.
    """
    result = tmp_path / "glicko_full_result.json"
    result.write_text(
        PredictorConfig(
            predictor_class="GlickoPredictor",
            league="nfl",
            target=-0.22,
            params={"home_advantage": 50.0, "scoring_method": "sigmoid"},
        ).model_dump_json()
    )

    assert load_predictor(result).state_dict()["scoring_method"] == "sigmoid"


@pytest.mark.parametrize(
    "config_path",
    sorted(path for path in _MODELS_DIR.glob("*/*.json") if "_result" not in path.stem),
    ids=lambda path: f"{path.parent.name}/{path.stem}",
)
def test_every_checked_in_config_pins_arguments_its_predictor_takes(
    config_path: Path,
) -> None:
    """A pinned name is never probed, so nothing else would catch a typo in
    one until a container had downloaded a league's seasons and started a
    search -- and then it fails identically for every one of them.
    """
    config = OptimizationConfig.model_validate_json(config_path.read_text())

    load_predictor_class(config.predictor_class)(config.league, **config.fixed)

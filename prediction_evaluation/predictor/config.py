import importlib
from pathlib import Path

from pydantic import BaseModel

from .base_predictor import Predictor


def load_predictor_class(name: str) -> type[Predictor]:
    module = importlib.import_module("prediction_evaluation.predictor")
    return getattr(module, name)


class PredictorConfig(BaseModel):
    predictor_class: str
    league: str
    target: float
    params: dict[str, float | str]


def load_predictor(config_path: Path | str) -> Predictor:
    config = PredictorConfig.model_validate_json(Path(config_path).read_text())
    predictor_class = load_predictor_class(config.predictor_class)
    return predictor_class(config.league, **config.params)

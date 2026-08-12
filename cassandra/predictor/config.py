import importlib
from pathlib import Path

from pydantic import BaseModel

from .base_predictor import Predictor


class UnknownPredictorClass(LookupError):
    """A predictor class name this build of cassandra doesn't have.

    Its own type because the case that produces it is ordinary: a release
    written by a newer cassandra names a class that didn't exist when the
    consumer was built. A consumer can catch this and say so, instead of
    catching AttributeError and hiding real bugs alongside it.
    """


def load_predictor_class(name: str) -> type[Predictor]:
    module = importlib.import_module("cassandra.predictor")
    # Checked rather than returned straight from getattr: the name comes off
    # an artifact, and every other module-level name -- `json`, `Prediction`,
    # `load_predictor` -- would otherwise come back as a "predictor class"
    # that fails much further downstream.
    predictor_class = getattr(module, name, None)
    if not (
        isinstance(predictor_class, type) and issubclass(predictor_class, Predictor)
    ):
        raise UnknownPredictorClass(f"unknown predictor class: {name!r}")
    return predictor_class


class PredictorConfig(BaseModel):
    predictor_class: str
    league: str
    target: float
    params: dict[str, float | str]


def load_predictor(config_path: Path | str) -> Predictor:
    config = PredictorConfig.model_validate_json(Path(config_path).read_text())
    predictor_class = load_predictor_class(config.predictor_class)
    return predictor_class(config.league, **config.params)

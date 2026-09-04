import importlib
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

from cassandra.objective import DEFAULT_OBJECTIVE, get_objective

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


class OptimizationConfig(BaseModel):
    """A checked-in `models/<league>/<name>.json`: what to search, and how hard.

    The input side of the pair `PredictorConfig` completes. It lives here
    rather than in `optimize.py` because two callers need to read one
    without running a search -- the batch manifest, and `publish.py`'s
    "that's not a result config" error -- and importing a root-level script
    to get at a schema is backwards.
    """

    predictor_class: str
    league: str
    parameters: dict[str, tuple[float, float] | list[str]]
    # Constructor arguments held at one value rather than searched, and
    # written into the result so the published predictor is rebuilt with
    # them. A categorical whose answer is already known belongs here and not
    # in `parameters` as a one-element list: bayes_opt rejects that outright
    # ("At least two categories are required"), and even where it didn't, a
    # search dimension with one value costs a probe budget it can't spend.
    fixed: dict[str, float | str] = {}
    n_iter: int = 100
    # Which number the search maximizes; see `cassandra.objective`. Defaulted
    # to brier so every config written before this existed keeps searching
    # for exactly what it used to.
    objective: str = DEFAULT_OBJECTIVE

    @field_validator("objective")
    @classmethod
    def _known_objective(cls, name: str) -> str:
        """Reject a name nothing can score, while a config is being read.

        The batch manifest loads every checked-in config before it launches
        anything, so a typo here fails the launcher rather than one array
        child an hour in.
        """
        get_objective(name)
        return name

    @model_validator(mode="after")
    def _no_parameter_is_both(self) -> "OptimizationConfig":
        """A name can be searched or pinned, not both.

        Nothing downstream would raise: `optimize.py` merges the two, so the
        search would run over a parameter whose value the constructor had
        already been given, and the result would record the pinned value
        beside a probe log that varied it. That is a config nobody can read
        the output of, so it doesn't get to be written.
        """
        both = sorted(set(self.parameters) & set(self.fixed))
        if both:
            raise ValueError(
                f"{', '.join(both)} appears in both `parameters` and `fixed`; "
                "a parameter is either searched or pinned"
            )
        return self


class PredictorConfig(BaseModel):
    predictor_class: str
    league: str
    target: float
    params: dict[str, float | str]
    # Which objective `target` is a score on, so two results are only
    # compared when they mean the same thing -- a brier target and a margin
    # target are both "higher is better" and are otherwise unrelated
    # numbers. Defaulted for the releases published before objectives
    # existed, all of which were brier.
    objective: str = DEFAULT_OBJECTIVE


def load_predictor(config_path: Path | str) -> Predictor:
    config = PredictorConfig.model_validate_json(Path(config_path).read_text())
    predictor_class = load_predictor_class(config.predictor_class)
    return predictor_class(config.league, **config.params)

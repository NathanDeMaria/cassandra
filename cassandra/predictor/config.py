import importlib
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

from cassandra.box import ParameterBound, misplaced
from cassandra.objective import DEFAULT_OBJECTIVE, get_objective

from . import frame as frames
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
    # A numeric range, a numeric range that searches whole numbers
    # (`[1, 4, "int"]`; see `cassandra.box.INTEGER`), or the choices of a
    # categorical.
    parameters: dict[str, ParameterBound]
    # Constructor arguments held at one value rather than searched, and
    # written into the result so the published predictor is rebuilt with
    # them. A categorical whose answer is already known belongs here and not
    # in `parameters` as a one-element list: bayes_opt rejects that outright
    # ("At least two categories are required"), and even where it didn't, a
    # search dimension with one value costs a probe budget it can't spend.
    fixed: dict[str, float | str] = {}
    # Which model's fit the values in `fixed` were copied from, when they were
    # copied from one rather than chosen.
    #
    # A pin produces no probes, so it produces no diagnostic either, and a
    # value copied out of another model's result is the kind that goes stale
    # without anything saying so -- that model gets re-searched every week and
    # its answer moves. `models/{nfl,ncaafb}/glicko_blend.json` pins the whole
    # Glicko machinery at `glicko_full`'s fit precisely so that its
    # `play_weight` 0 reproduces that model, which stops being true the moment
    # the two disagree.
    #
    # Naming the source is what lets the run report check it. Comparing pins
    # against every model that happens to search the same name does not work:
    # `home_advantage` is points in `margin_elo` and rating units in the
    # Glicko family, so the cross-family pairs are noise that buries the one
    # comparison that means anything.
    fixed_from: str | None = None
    # A pin that copies a source value under another name: pin -> the name in
    # the source's fit. `BlendedGlickoPredictor`'s `mov_scale` is what
    # `sigmoid_scale` is to `glicko_full` -- the blend drops the parent's
    # scorer and reads a final score with its own -- so `play_weight` 0 is
    # `glicko_full` only while the two agree. Matched by name, nothing checked
    # it, and nfl's sat at the old 20.99 while the fit moved to 10.3.
    fixed_aliases: dict[str, str] = {}
    # The units `parameters` and `fixed` are written in; see
    # `cassandra.predictor.frame`. "rating" is the constructor's own, and
    # what every config before frames existed searched in. "points" is the
    # Glicko family's better-conditioned box: matchup terms in points of
    # margin, the prediction scale as a margin, the deviation increases as a
    # per-season budget and its offseason share.
    frame: str = frames.RATING
    # Points to score before the random start, in the frame's terms: one
    # value per searched parameter. A seed is a probe the search can't lose
    # -- the fit can't end below it -- and the surrogate starts out knowing
    # where it is. `optimize.py` adds the model's previous result to these
    # on its own, so a seed written here is for a point that is *not* the
    # previous result: a fit an earlier search found and a later one lost,
    # which is how ncaafb/glicko_full's 2026-09-12 point came to be one.
    seeds: list[dict[str, float | str]] = []
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
    def _fixed_from_needs_fixed(self) -> "OptimizationConfig":
        """A source for pins that don't exist is a claim about nothing.

        Caught here rather than left to the report, which would silently have
        nothing to compare and say the pins were fine.
        """
        if self.fixed_from and not self.fixed:
            raise ValueError(
                f"fixed_from={self.fixed_from!r} names where pins came from, "
                "but there are no pinned parameters"
            )
        if self.fixed_aliases and not self.fixed_from:
            raise ValueError("fixed_aliases maps pins onto a source; name it in fixed_from")
        unpinned = sorted(set(self.fixed_aliases) - set(self.fixed))
        if unpinned:
            raise ValueError(
                f"fixed_aliases names {', '.join(unpinned)}, which is not pinned here"
            )
        return self

    @field_validator("frame")
    @classmethod
    def _known_frame(cls, name: str) -> str:
        frames.knobs_of(name)
        return name

    @model_validator(mode="after")
    def _frame_is_consistent(self) -> "OptimizationConfig":
        """A framed config names each quantity once, in the frame's terms.

        Caught at load for the same reason a bad objective is: the manifest
        reads every config before anything is launched. The points frame
        needs `sigmoid_scale` to price a point, and a constructor argument it
        derives -- `home_advantage` next to `hfa_pts` -- would be set twice.
        """
        named = set(self.parameters) | set(self.fixed)
        if self.frame == frames.POINTS and "sigmoid_scale" not in named:
            raise ValueError(
                "frame 'points' needs sigmoid_scale in `parameters` or `fixed`"
            )
        # A probe is a full set of knobs, so the derivation's own checks run
        # on a representative one; any conflict shows up here, not an hour in.
        probe = {name: 1.0 for name in self.parameters if name != "sigmoid_scale"}
        if "sigmoid_scale" in self.parameters:
            probe["sigmoid_scale"] = 1.0
        if "rd_offseason_share" in probe:
            probe["rd_offseason_share"] = 0.5
        frames.to_params(self.frame, {**self.fixed, **probe}, weeks_per_season=1)
        return self

    @model_validator(mode="after")
    def _seeds_are_in_the_box(self) -> "OptimizationConfig":
        """A seed the search can't probe fails at load, with the others.

        `optimize` would refuse it too, but an hour into the array rather
        than before the launcher submits anything.
        """
        for seed in self.seeds:
            problem = misplaced(seed, self.parameters)
            if problem is not None:
                raise ValueError(f"seed {seed}: {problem}")
        return self

    def searched_params(self) -> frozenset[str]:
        """The constructor arguments this search moves; see `frame.searched_params`."""
        return frames.searched_params(self.frame, set(self.parameters))

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


class SearchRecord(BaseModel):
    """How a result's `params` were searched, when not in their own units.

    `knobs` are the searched values plus the pins, in the frame's terms, and
    `weeks_per_season` is what `frame.to_params` turned them into `params`
    with. Recorded so a fit can be read in the units it was found in, and so
    a re-search can start from where the last one ended.
    """

    frame: str
    weeks_per_season: float
    knobs: dict[str, float | str]


class PredictorConfig(BaseModel):
    predictor_class: str
    league: str
    target: float
    # Always constructor arguments, whatever the search moved: this is what
    # `load_predictor` rebuilds from and what a pin copies.
    params: dict[str, float | str]
    search: SearchRecord | None = None
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

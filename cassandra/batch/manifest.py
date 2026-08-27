"""The work list: every optimization config in the repo, in the order to run them.

This is the one definition of "what is there to optimize", shared by three
callers that used to answer it separately: `run_models.sh`, the batch launcher
that sizes the array job, and the array child that has to turn an integer index
back into a config. Those last two *must* agree -- a child that resolves index
3 differently than the launcher intended optimizes the wrong model and writes
the wrong result file -- so the ordering here is a contract, not a convenience.

Cheap configs sort first, matching what `run_models.sh` has always done: a
mistake surfaces in the low-`n_iter` runs before the expensive ones burn an
hour. Ties break on league then path so the order is total and stable.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from cassandra.predictor import OptimizationConfig, load_predictor_class

# The checked-in configs, resolved the same way `evaluate_models.py` resolves
# them: relative to the repo root, which in the deployed image is wherever the
# source was copied to.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _REPO_ROOT / "models"

# Set by the launcher on the array job so children don't recompute the list.
# See `resolve_index`.
MANIFEST_ENV_VAR = "CASSANDRA_BATCH_MANIFEST"
# AWS Batch sets this on each child of an array job.
ARRAY_INDEX_ENV_VAR = "AWS_BATCH_JOB_ARRAY_INDEX"


@dataclass(frozen=True)
class Work:
    """One optimization to run."""

    league: str
    model: str
    config_path: Path
    predictor_class: str
    n_iter: int
    # Where this predictor stashes opponent priors, or None if it doesn't
    # build any. `OpponentPriorManager.save` refuses to overwrite, so a rerun
    # in a container that already has one has to clear it first.
    prior_path: Path | None

    @property
    def name(self) -> str:
        """`league/model` -- how a work item is named on a command line."""
        return f"{self.league}/{self.model}"


class UnknownWork(LookupError):
    """A `league/model` that doesn't match any checked-in config."""


def load_manifest(
    leagues: list[str] | None = None, models: list[str] | None = None
) -> list[Work]:
    """Every optimization config, filtered, in run order.

    `leagues` and `models` are both plain name filters; passing neither means
    everything. An empty result is not an error here -- callers that need work
    to do say so themselves, with the filters in hand to put in the message.
    """
    work = [
        item
        for item in _all_work()
        if (not leagues or item.league in leagues)
        and (not models or item.model in models)
    ]
    return sorted(work, key=lambda w: (w.n_iter, w.league, w.model))


def _all_work() -> list[Work]:
    if not _MODELS_DIR.is_dir():
        raise FileNotFoundError(f"No models directory at {_MODELS_DIR}")

    work = []
    for league_dir in sorted(_MODELS_DIR.iterdir()):
        if not league_dir.is_dir():
            continue
        for path in sorted(league_dir.glob("*.json")):
            # `_result` files are optimizer output that got checked in as a
            # baseline; `_state` files are predictor state. Neither is a
            # searchable config.
            if path.stem.endswith(("_result", "_state")):
                continue
            config = OptimizationConfig.model_validate_json(path.read_text())
            if config.league != league_dir.name:
                raise ValueError(
                    f"{path}: league {config.league!r} does not match directory "
                    f"{league_dir.name!r}; evaluate_models.py labels by directory."
                )
            predictor = load_predictor_class(config.predictor_class)(config.league)
            # Predictors that build opponent priors stash a manager; the rest
            # don't, and asking for the attribute is how run_models.sh has
            # always found out.
            manager = getattr(predictor, "_prior_manager", None)
            work.append(
                Work(
                    league=config.league,
                    model=path.stem,
                    config_path=path,
                    predictor_class=config.predictor_class,
                    n_iter=config.n_iter,
                    prior_path=None if manager is None else manager._prior_path,
                )
            )
    return work


def encode(work: list[Work]) -> str:
    """The launcher's view of an array job, as `league/model` names."""
    return json.dumps([item.name for item in work])


def array_index(explicit: int | None = None) -> int | None:
    """Which child of an array job this is: the flag, else what Batch set.

    `None` means neither -- no `--index` and no `AWS_BATCH_JOB_ARRAY_INDEX`,
    so this is a hand-run process rather than one child of a fan-out. Callers
    read that as "not scoped to one item" and do whatever their whole-run
    default is.

    Here rather than in each stage because every fan-out needs it and only
    one of them can afford to get it wrong quietly: a stage that forgets to
    read the environment doesn't fail, it just has every child do the whole
    job.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get(ARRAY_INDEX_ENV_VAR)
    return None if raw is None else int(raw)


def resolve_index(index: int | None = None) -> Work:
    """The work item this array child is responsible for.

    The launcher pins the exact list it sized the array against into
    `CASSANDRA_BATCH_MANIFEST`, and this resolves against that rather than
    recomputing. The two would normally agree -- same image, same commit --
    but "normally" is doing real work there: a launcher run from a laptop
    against an older image tag would silently hand out indices for a list the
    children don't have. Pinning turns that into a `KeyError` at startup
    instead of a run that optimizes the wrong models.
    """
    index = array_index(index)
    if index is None:
        raise ValueError(
            f"No --index given and {ARRAY_INDEX_ENV_VAR} is not set; this is "
            "meant to run as a child of an array job, or with an explicit "
            "--league/--model."
        )

    pinned = os.environ.get(MANIFEST_ENV_VAR)
    if pinned is None:
        # Standalone run (a hand-submitted job, or a local test): the local
        # list is all there is, and it's the same code that built the array.
        manifest = load_manifest()
    else:
        names = json.loads(pinned)
        by_name = {item.name: item for item in load_manifest()}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise UnknownWork(
                f"{MANIFEST_ENV_VAR} names configs this image doesn't have: "
                f"{', '.join(missing)}. The launcher and the image are on "
                "different commits."
            )
        manifest = [by_name[name] for name in names]

    if not 0 <= index < len(manifest):
        raise IndexError(
            f"Array index {index} is out of range for {len(manifest)} work items"
        )
    return manifest[index]


def find(league: str, model: str) -> Work:
    """One work item by name, for running a single model by hand."""
    for item in load_manifest():
        if item.league == league and item.model == model:
            return item
    known = ", ".join(item.name for item in load_manifest())
    raise UnknownWork(f"No config for {league}/{model}. Known: {known}")

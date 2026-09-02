"""Cassandra: rating models, and the release artifact they publish.

The package splits the same way the dependencies do. `cassandra.serving`,
`cassandra.predictor` and `cassandra.prob_to_margin` are the release-reading
half: they import `endgame.types` and pydantic and nothing that talks to s3,
which is what lets the webapp install cassandra without the `fit` group.
`cassandra.save_predictions` and `cassandra.odds` are the fitting half -- they
read the bucket -- and `cassandra.model_eval` drives them.

The four names below belong to the second half, and they are re-exported here
lazily rather than imported at module scope. Importing them eagerly would
undo the split at the import line instead of the install line: `cassandra` is
a package, so `import cassandra.serving` runs this file first, and a
top-level `from .model_eval import ...` here pulls in `save_predictions` and
through it `endgame_aws` -- aiobotocore, and from the rev that adds the
processed play-by-play layer, pyarrow. A consumer that only reads a release
would pay for all of it before reaching the module it asked for.

Deferring rather than deleting the re-export because `publish.py` imports
these from here, and the failure a consumer without the fit group gets is
better landed on the name it used than on an import of a package it has
never heard of. Same reasoning as `IsotonicProbToMarginFitter.fit`'s lazy
sklearn import. `import_boundary_test.py` holds the line.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # For type checkers and editors, which read this branch and never run it,
    # so the names resolve to real definitions without the runtime import.
    from .model_eval import (
        DEFAULT_FITTERS,
        ScoredPredictions,
        get_predictions,
        score_predictions,
    )

__all__ = [
    "DEFAULT_FITTERS",
    "ScoredPredictions",
    "get_predictions",
    "score_predictions",
]


def __getattr__(name: str) -> Any:
    """Resolve the fit-path re-exports on first use (PEP 562).

    Anything not in `__all__` raises AttributeError as it normally would --
    without this, a typo'd name would import `model_eval` before failing,
    which is the cost this module exists to avoid.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import model_eval

    return getattr(model_eval, name)

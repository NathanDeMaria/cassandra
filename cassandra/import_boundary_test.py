"""The install split, asserted at the import line.

`endgame_aws` and scikit-learn are in the `fit` group, not in package
metadata, so a consumer that reads a release -- the webapp -- installs
cassandra without either. That only holds if the modules such a consumer
imports don't reach them, and nothing about the code says so: an ordinary
looking `from .model_eval import ...` in `cassandra/__init__.py` is enough to
undo it, and the dev environment always has both installed, so every other
test in this suite passes either way.

Run in a subprocess because it's the only honest way to ask. By the time
pytest reaches this file, `odds_test` and `save_predictions_test` have
already imported `endgame_aws` into this process; a fresh interpreter per
import is what makes `sys.modules` mean what it looks like it means.
"""

import json
import subprocess
import sys

import pytest

# What a consumer holding a ModelRelease imports. `cassandra` itself is in the
# list because it is a package: importing any of the others runs its
# `__init__` first, so it is on the path whether the consumer names it or not.
_SERVING_MODULES = (
    "cassandra",
    "cassandra.serving",
    "cassandra.predictor",
    "cassandra.prob_to_margin",
)

# The fitting stack, by the top-level package a serving install won't have.
#
# Deliberately not `pyarrow`, even though its ~152MB is what this is really
# protecting: pandas imports pyarrow at import time whenever it's installed,
# so in a dev environment `cassandra.serving` "reaches" it through a package
# that main depends on anyway. That says nothing about the boundary -- a
# `--only main` install has pandas and no pyarrow, and pandas is fine with
# that. The two doors pyarrow can actually come through are the two named
# here, so asserting those is the same assertion without the false positive.
_FIT_ONLY = ("endgame_aws", "lucky_ones", "aiobotocore", "sklearn")


def _imported_by(module: str) -> set[str]:
    """Every module a fresh interpreter loads to `import module`."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}; import json, sys; print(json.dumps(sorted(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(result.stdout))


@pytest.mark.parametrize("module", _SERVING_MODULES)
def test_serving_imports_stay_off_the_fitting_stack(module: str) -> None:
    loaded = _imported_by(module)
    reached = [package for package in _FIT_ONLY if package in loaded]
    assert not reached, (
        f"`import {module}` pulled in {', '.join(reached)}, which the `fit` "
        "group keeps out of a serving install"
    )


def test_fit_path_re_exports_still_resolve() -> None:
    """The lazy names are the same objects `model_eval` defines."""
    import cassandra
    from cassandra import model_eval

    for name in cassandra.__all__:
        assert getattr(cassandra, name) is getattr(model_eval, name)


def test_unknown_attribute_raises() -> None:
    import cassandra

    with pytest.raises(AttributeError):
        cassandra.not_a_real_name
